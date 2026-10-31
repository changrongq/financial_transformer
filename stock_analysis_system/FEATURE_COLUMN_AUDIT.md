# 特征列审计（A 项）：修正与真实发现

> 2026-10-31
> 承接 `FEATURE_SCALE_AUDIT.md`，检查「8 个空列」的实际处理方式
> 脚本：`scripts/check_feature_columns_production.py`、`scripts/quantify_moneyflow_gap.py`

---

## 零、先更正我自己的错误

在 `FEATURE_SCALE_AUDIT.md` 我写了：

> ~~「配置声明 32 个特征，实际只有 24 个有数据」~~
> ~~「`market_cap_ratio` 需市值时序，7 个资金流列需 `stock_moneyflow` 表」~~

**这两句话是错的。**

### 错在哪

我手工构造了一个 DataFrame（只 SELECT 了 OHLCV 等基础列），
直接喂给 `build_layer_features`，然后发现 8 列是 NaN。

**问题是我的脚本没有 merge `stock_moneyflow`，而不是数据不存在。**

走生产路径 `_load_trade_data_with_concepts` 后，真实结果：

| 检查 | 结果 |
|---|---|
| 列不存在 | **0 个** |
| 全 NaN | **0 个** |
| 常数列 | **0 个** |
| **有变化的列** | **32 / 32** |

`stock_moneyflow` 实际有 **2,164,284 行**（5,518 只股票），
`净流入额` **100% 非空**。

**所以"32 个特征都有效"是对的，我之前误判了。**

---

## 一、但发现了真正的问题：时间覆盖缺口

### 1.1 数据表的时间范围

| 表 | 起止 | 行数 |
|---|---|---:|
| `stock_data` | **2021-11-01 ~ 2026-10-29** | 6,369,494 |
| `stock_moneyflow` | **2025-01-02 ~ 2026-10-28** | 2,164,284 |

**训练数据比资金流多 771 个交易日。**

### 1.2 逐年覆盖

| 年 | 交易日 | 样本 | 有资金流天数 | 覆盖率 |
|---|---:|---:|---:|---:|
| 2021 | 45 | 211,287 | 0 | **0.0%** |
| 2022 | 242 | 1,178,820 | 0 | **0.0%** |
| 2023 | 242 | 1,258,502 | 0 | **0.0%** |
| 2024 | 242 | 1,308,411 | 0 | **0.0%** |
| 2025 | 243 | 1,356,182 | 243 | 100.0% |
| 2026 | 192 | 1,056,292 | 192 | 100.0% |

**样本分布：有资金流 37.9%（2,412,474 行），无资金流 62.1%（3,957,020 行）。**

### 1.3 缺口期间发生什么

`build_layer_features` 第 106-109 行：

```python
net_mf_amount_ratio = net_mf_amount_ratio.mask(moneyflow_source_missing, 0.0)
net_lg_amount_ratio  = net_lg_amount_ratio.mask(moneyflow_source_missing, 0.0)
net_elg_amount_ratio = net_elg_amount_ratio.mask(moneyflow_source_missing, 0.0)
net_sm_amount_ratio  = net_sm_amount_ratio.mask(moneyflow_source_missing, 0.0)
```

**缺口期间 395.7 万行的 7 个资金流特征全部被填成常数 0.0。**

---

## 二、为什么这是个严重问题

### 2.1 标准化后会怎样

模拟（3,957,020 个 0 + 2,412,474 个真实值）：

```
全长 mean = 2.2e-07   std = 6.15e-04

标准化后：
  缺口段(0 值)  z = -0.0004   ← 常数，模型无法区分
  真实段        z ∈ [-8.18, +7.79]
```

**缺口期间这 7 列在标准化后是同一个常数** —— 对模型零信息。

### 2.2 更糟的是分布漂移

如果训练/验证切分**跨过 2025-01-02**：

| 分段 | 资金流特征 |
|---|---|
| 训练集（2021-2024） | 恒等于 **0** |
| 验证集（2025+） | 真实波动值 |

**模型在训练时学到"资金流 = 0"，在验证时突然看到真实分布 —— 这是典型的分布漂移。**

默认配置 `train_end: 2025-09-30`：

```
训练集：2021-11 ~ 2025-09  → 含 771 天常数 0 + 约 180 天真实值
验证集：2025-10 ~ 2025-12  → 全真实值
测试集：2026-01 ~ 2026-03  → 全真实值
```

**训练集里 81% 的样本资金流是 0。**

### 2.3 这正好解释了"验证/测试更差"

`MICRO_MODEL_DIAGNOSIS_REPORT.md` 记录的"AUC 0.435，测试比训练差"，
一部分原因可能就是这个：**训练时资金流是常数，测试时是真实值**。

---

## 三、修复建议（按代价排序）

### 方案 A：限制训练窗口到资金流覆盖范围

```yaml
splits:
  train_start: "2025-01-02"     # 新增
  train_end: "2025-09-30"
```

| 优点 | 缺点 |
|---|---|
| 零数据成本 | 训练样本从 637 万降到约 240 万 |
| 分布一致 | 只覆盖 1 年半 |

### 方案 B：从配置移除 7 个资金流特征

```yaml
feature_cols: [ ... 移除 net_mf_* / net_lg_* / net_elg_* / mf_acceleration ]
```

| 优点 | 缺点 |
|---|---|
| 零成本，消除常数污染 | 放弃资金流信息 |

**注意**：如果 8 个资金流特征里只有 2025 年后有效，
那在 2021-2024 训练段它们就是纯噪声，移除反而更干净。

### 方案 C：补 2021-11 ~ 2024-12 的资金流数据

tushare `moneyflow` 接口，按交易日拉，约 771 次调用。

| 优点 | 缺点 |
|---|---|
| 保留全部特征与全部历史 | 需要约 15~20 分钟拉取 |
| 最彻底 | tushare 积分可能不够 |

### 方案 D：missing indicator + mask

```python
df["mf_available"] = moneyflow_present.astype(float)   # 0/1 指示
```

| 优点 | 缺点 |
|---|---|
| 显式告诉模型"今天这个特征不可用" | 需要改模型输入维度 |
| 保留全部历史 | 只在训练集跨缺口时才有意义 |

---

## 四、我的建议

**先做方案 C（补数据），因为它是唯一能同时保留全历史和全特征的做法。**

如果 tushare 积分不够，退而求其次做方案 A（限制窗口），
并在文档中明确"资金流实验只在 2025 年后有效"。

**不建议**直接训练 —— 现在训练等于让模型在 62% 的样本上
把 7 个特征当常数，然后指望它在 38% 的样本上学会用这 7 个特征。

---

## 五、顺带确认的其他事实

### 5.1 `market_cap_ratio` 与 `close` **完全共线**（实测确认）

```
相关系数           : 1.000000000000      ← 精确等于 1
ratio 唯一值数     : 2（跨股票）/ 1（单股票内）
ratio 标准差       : 7.02e-18             ← 数值零

close              唯一值 1387
market_cap_ratio   唯一值 1387            ← 完全相同
```

**`market_cap_ratio = close × 常数`，与 `close` 是同一个信息。**

原因（`features.py:153`）：

```python
market_cap_ratio = _safe_div(df["close"], market_cap_yi)
```

`market_cap_yi` 来自 `_latest_market_caps()`，是**最新一期市值的常数**。

这意味着：

| 场景 | 该列的信息 |
|---|---|
| micro 层内部（单股票） | **零**（就是 close） |
| 跨层训练 | 有「谁市值小」的信息 |

**但更根本的问题是**：最新市值被回填到整段历史，
违反 as-of 原则 —— 这与知微第 6 项是同一个根因。

### 5.2 资金流特征的量级

```
net_mf_amount_ratio: 均值 -3.6e-05, std 6.7e-04
net_lg_amount_ratio: 均值 -1.7e-05, std 3.6e-04
net_elg_amount_ratio: 均值 -1.3e-05, std 2.3e-04
mf_acceleration:     均值 -2.1e-06, std 7.0e-04
```

**量级 1e-04 ~ 1e-06，比 `amount`(1e9) 小 13~15 个数量级。**

**即使补全了数据，这 7 列在原始尺度下也是被淹没的。**
必须配合 `ColumnScaler` 才能生效。

### 5.3 一个真实的数据 bug

`000005` 这只股票走生产路径时抛异常：

```
TypeError: cannot concatenate object of type '<class 'float'>'
```

原因：`stock_moneyflow` 里没有这只股票的记录，
于是 `df.get("净流入额", np.nan)` 返回标量 `np.nan`，
`pd.concat([...])` 无法处理标量。

**这是一个真实的健壮性 bug**，会影响任何"有价格数据但无资金流数据"的股票。
建议修复：`_load_trade_data_with_concepts` 在 `moneyflow_df.empty` 时
仍补全所需列的 NaN，而不是完全不 merge。

---

## 六、复现命令

```powershell
cd d:\qcr\Stock\stock_analysis_system

# 走生产路径确认 32 列真实情况
.venv\Scripts\python.exe scripts\check_feature_columns_production.py

# 量化资金流时间缺口
.venv\Scripts\python.exe scripts\quantify_moneyflow_gap.py

# 验证 market_cap_ratio 共线性
.venv\Scripts\python.exe scripts\verify_market_cap_collinearity.py

# 确认资金流表内容
.venv\Scripts\python.exe scripts\check_moneyflow_data.py
.venv\Scripts\python.exe scripts\check_moneyflow_columns.py
```

---

## 七、汇总：Transformer 线现在已知的工程缺陷

从知微复核到本轮 A 项审计，累计发现 **6 个**问题：

| # | 缺陷 | 级别 | 状态 |
|---|---|---|:--:|
| 1 | 标签跨入验证期 | P1 | ✅ 已修 |
| 2 | 未来不足 45 天当作完整标签 | P1 | ✅ 已修 |
| 3 | **flat 输入尺度淹没 22/24 特征** | P1 | ✅ 已修（`ColumnScaler`）|
| 4 | 回测 T+1 违规 | P1 | ✅ 已修（`T1PositionBook`）|
| 5 | CPU best_state 共享存储 | P2 | ✅ 已修 |
| 6 | 最新市值回填历史（as-of 违规）| P1 | ⚠️ **已确认，未修** |
| 7 | 阈值搜索用测试集 | 方法 | ⚠️ 已确认，未修 |
| **8** | **资金流特征 62% 样本是常数 0** | **P1** | ⚠️ **本轮新发现** |
| **9** | **`market_cap_ratio` 与 close 完全共线** | P2 | ⚠️ **本轮新发现** |
| **10** | **无资金流数据的股票抛 TypeError** | P2 | ⚠️ **本轮新发现** |

### 7.1 对「有效特征数」的最终估计

配置声明 32 维，逐层扣减：

```
32  声明
-1  market_cap_ratio（与 close 完全共线）
-7  资金流（62% 样本是常数 0；剩下 38% 里量级 1e-05，需 log 才能用）
= 24  有独立信息
-22 原始尺度下被 amount/volume 淹没
=  2  真正能被模型"看到"：amount, volume
```

**这就是"输出几乎恒定"最完整的解释。**

### 7.2 与"市场不可预测"的关系

**完全没有关系。**

以上 10 个都是**实现与数据工程问题**，没有一个说明市场有没有规律。


