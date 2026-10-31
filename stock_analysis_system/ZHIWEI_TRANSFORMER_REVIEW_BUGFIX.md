# 知微 Transformer 复核 —— 回应与修复记录

> 审查方：知微 ｜ 日期：2026-10-31
> 被审对象：`changrongq/financial_transformer` commit `63584d2`
> 修复方：Coder ｜ 日期：2026-10-31
> 复现：`scripts/reproduce_transformer_findings.py`（7/7 项确认）
> 验证：`scripts/verify_transformer_fixes.py`（**17/17 PASS**）

---

## 零、总态度

知微的判断是对的：**现有失败记录不足以支持「个股方向本质不可预测」或
「Transformer 已被证明不适用」。**

我逐条复现了他的 7 项发现，**全部成立**，其中第 3 项的证据比他用例更极端。

### 必须降级的结论

| 原结论 | 位置 | 现在 |
|---|---|---|
| "输出必然塌缩" | `MICRO_MODEL_DIAGNOSIS_REPORT.md` | **降级**为「在这组实现的输入尺度下的观察」 |
| "个股方向本质不可预测" | `PROJECT_README.md` | **撤回** |
| "所有可修因素已排除" | 诊断报告 | **撤回**（已找到 4 个未排除的工程因素） |
| "Transformer 已被证明不适用" | 实验记录 | **撤回** |

### 但也要说清楚

**修好这些 bug 不等于一定有效。** 我们同意知微的原话：
"不能把当前 bug 的存在反向包装成'修好就一定有效'"。

所以本文只陈述「已修复什么」「还差什么」，不预测修复后的效果。

---

## 一、七项发现的复现结果

| # | 知微的发现 | 级别 | 复现 |
|---|---|---|:--:|
| 1 | 训练标签跨入验证期 | P1 | ✅ |
| 2 | 未来不足 45 天当作完整标签 | P1 | ✅ |
| 3 | flat 输入尺度掩盖小特征 | P1 | ✅ **更极端** |
| 4 | 回测允许当天买当天卖 | P1 | ✅ |
| 5 | CPU best_state 共享存储 | P2 | ✅ |
| 6 | 用最新市值选历史股票池 | P1 | ✅ |
| 7 | 测试集上挑阈值再切 rolling | 方法 | ✅ |

---

## 二、第 1 项：标签成熟度（已修复）

### 2.1 复现

```python
# 截止日索引 60，当天价格两种场景都是 100
场景A 该日未来5日收益: +0.0000   # 之后价格保持 100
场景B 该日未来5日收益: +0.2000   # 之后价格涨到 120
[PASS] 截止日后的价格能改变截止日训练标签
```

`_assign_split` 的源码证实它**只看当前日期**：

```python
def _assign_split(date_value, config):
    if date_value <= train_end:
        return "train"
    ...
```

**没有任何 label_end / available_at 的概念。**

### 2.2 修复

新增 `scripts/transformer_fixes.py`：

```python
def label_end_from_trade_calendar(df, trade_dates, horizon) -> df:
    """按交易日历精确计算每个样本的标签观察结束日。"""

def label_maturity_mask(df, cutoff, label_end_col="label_end_date"):
    """True = 该样本的标签在 cutoff 之前已成熟。"""
```

关键点：**用交易日历取第 N 个交易日**，不是「加 N 个自然日」。
知微明确要求"不能仅调整 train_end 字符串或按自然日随意留空"。

### 2.3 验证

```
交易日 120 天，训练截止 2025-03-26
成熟样本 56，期望 56（61 天 - 最后 5 天）
[PASS] 【修复验证】未成熟标签被剔除 (56 vs 56)
[PASS] 【缺陷确认】旧逻辑保留了未成熟标签 (61)
[PASS] 【缺陷确认】截止日后价格改变标签 (+0.00 -> +0.20)
[PASS] 【修复验证】截止日样本被剔除
```

---

## 三、第 2 项：不完整标签（已修复）

### 3.1 复现

```python
未来天数: [50, 45, 44, 10, 5, 0]
旧逻辑对 5 天真实数据算出 max=100.00 min=50.00
[PASS] 【缺陷确认】旧逻辑用不足数据算出有限标签
```

原因：`future_window.max(axis=1)` 默认跳过 NaN，所以只有 5 天数据
也能算出"完整的 45 天最大涨幅"。

### 3.2 修复

```python
def incomplete_label_mask(df, n_future, horizon) -> np.ndarray:
    """n_future < horizon 的样本必须剔除。"""
```

### 3.3 验证

```
未来天数: [50, 45, 44, 10, 5, 0]
不完整:   [False, False, True, True, True, True]
[PASS]
```

注意知微的提醒：**"不能因此把尚未完成的最大涨跌幅标签一并判为有效"**。
如果采用"提前命中即可确定"的事件标签（止盈止损），
也要按任务单独定义成熟条件，而不是整体放过。

---

## 四、第 3 项：输入尺度（已修复，证据最强）

### 4.1 复现 —— 比知微给的用例更极端

用原 `FinancialTransformer`（input_dim=32, d_model=64, nhead=4,
num_layers=2, model_variant="flat"），seed=17：

```python
base[:, :, 0] = 1e8      # amount
base[:, :, 1] = 1e6      # volume
x1[:, :, 2] = -0.1       # ret5d
x2[:, :, 2] = +0.1
```

| 检查 | 输出最大差 |
|---|---:|
| **原始尺度** | **0.0000000000** |
| **标准差为 0（完全淹没）** | |
| 逐列标准化后 | **0.0392886065** |

**知微测出 0.0（投影+LayerNorm 后），我们在 logit 层面也是精确的 0。**

这不只是"难以保留"，是**在 float32 里彻底丢失**。

### 4.2 为什么投影后的 LayerNorm 不够

```
flat 路径：x [B,T,32] → Linear(32→64) → LayerNorm → Transformer
                ↑
      这里已经把 1e8 和 0.1 线性混合了
```

`LayerNorm` 作用在**混合后**的 64 维上，无法还原被淹没的成分。

### 4.3 修复

```python
class ColumnScaler:
    """逐列尺度变换，必须随 checkpoint 保存/恢复。"""
    def fit(self, X): ...        # 计算每列 mean/std
    def transform(self, X): ...  # 应用
    def to_dict(self) / from_dict(cls, d)   # 序列化
```

**关键约束（知微强调）**：推理必须加载同一变换，**禁止在全量数据上重新拟合**。

### 4.4 验证

```
原始尺度输出差: 0.0000000000
[PASS] 【缺陷确认】原始尺度下小特征被完全淹没
标准化后输出差: 0.0392886065
[PASS] 【修复验证】标准化后小特征生效
[PASS] 【修复验证】scaler 可随 checkpoint 保存/恢复
```

### 4.5 还没做的

知微建议"应优先检查真实输入各列分位数、单位、梯度和特征扰动响应"。

我们**还没做这个**，因为需要真实的 feature_df。
**下一步应该先跑一次真实数据的列分位数审计，再决定 log 变换用在哪几列。**

---

## 五、第 4 项：T+1 违规（已修复）

### 5.1 复现

`backtest.py:872` 买入并 `open_positions.append(...)`，
`backtest.py:897` 立刻遍历 **所有** `open_positions` 检查止盈止损：

```python
872:                open_positions.append(
877:                        "buy_date": current_date,
897:            for position in open_positions:      ← 包含当天新买的
903:                high_return = bar["high"] / buy_price - 1
914:                if low_return <= position_stop_loss:
915:                    exit_reason = "止损"
```

**没有 `bought_today` / `can_sell` 之类的守卫。**

### 5.2 修复

```python
@dataclass
class T1PositionBook:
    """按批次（lot）管理，每批记录 sellable_from。"""
    def buy(self, code, date, price, qty, t0_ok=False):
        sellable = date if t0_ok else date + timedelta(days=1)
    def can_sell(self, code, date) -> bool
    def sell(self, code, date, price, qty):
        if qty > avail: raise ValueError("T+1 约束被违反")
```

### 5.3 验证

```
2025-02-04 买入 000001 @100 x1000
[PASS] 当天不可卖 (可卖 0)
[PASS] 当天卖出被拒绝
[PASS] 次日可卖 (可卖 1000)
[PASS] 次日卖出成功 (pnl=-6000)
[PASS] T+0 品种可显式放开
```

### 5.4 知微提到的额外静态问题

> "`:1000–1005` 遇某只持仓当天无 bar 直接跳过估值，可能把持仓暂时按零计价"

**这一项我们还没修**，因为它会改变净值曲线的计算方式，需要单独设计
「用上次可信估值」或「明确缺失状态」。已记入待办。

---

## 六、第 5 项：CPU 快照（已修复）

### 6.1 复现

```python
best_state = {k: v.cpu() for k, v in model.state_dict().items()}
# 保存时 weight 首元素: 0.636
model.weight.fill_(7.0)
# best_state 首元素: 7.000   ← 被污染
[PASS] 【缺陷确认】旧写法快照被污染
```

原因：参数已在 CPU 时，`.cpu()` **直接返回同一存储**，不复制。

### 6.2 修复

```python
def snapshot_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```

### 6.3 验证

```
[PASS] 【修复验证】新写法快照不变 (仍是 0.539)
```

### 6.4 范围说明

按知微的提醒，这个缺陷**只在 CPU 分支成立**。
未在 CUDA 上复现，也不声称 CUDA→CPU 拷贝有同样问题。

---

## 七、第 6 项：市值前视（已确认，未修复）

### 7.1 复现

```python
def _latest_market_caps(config):
    df = pd.read_sql_query("SELECT stock_code, total_mv, trade_date ...")
    df = df.sort_values(...).drop_duplicates("stock_code", keep="last")
    #                                                    ^^^^^^^^^ 取最新
```

**函数签名只有 `(config: ExperimentConfig)`，没有 as-of 参数。**

这意味着：用 2026 年的市值决定 2022 年哪些股票属于 micro 层。

### 7.2 为什么还没修

修复需要**历史市值序列**（`market_caps` 表里是否有多期数据待确认）。
如果只有最新一期，就无法做 as-of 筛选，只能：
1. 明确声明这是"今天选出的股票在历史上的回顾研究"
2. 或者补历史市值数据

### 7.3 已采取的措施

在文档中明确标注这一限制（见下文第八节）。
**不能当作"当年可实施的选股策略"。**

---

## 八、第 7 项：阈值搜索（已确认，方法边界）

### 8.1 复现

`threshold_search.py` 直接读取 `test_predictions.csv`，
默认区间正是默认配置的测试日期，然后按收益指标挑最佳组合。

`rolling_backtest.py` 在**已有预测的日期**上划分重叠窗口，
没有生成新的独立训练/选择过程。

### 8.2 结论

- 阈值搜索结果**只能算开发期调参**
- rolling 的多个正窗**不是多个独立实验**
- 高度重叠的窗口不能当作独立样本

### 8.3 需要的登记

知微要求："需要逐报告登记模型选择和阈值选择见过哪些日期"。

**这一点我们还没系统做**，已记入待办。

---

## 九、知微确认的合理部分（我们保留）

按知微的核对，以下几点**没有问题**，不应因这次审查而改动：

| 项目 | 状态 |
|---|---|
| 五个输出头 `squeeze(-1)`，batch=2 时输出 `[2]` | ✅ 无 RL 那种 `[B,1]×[B]` 错误 |
| 每个输入窗口结束于当前预测日 | ✅ 窗口内双向 attention 不构成泄漏 |
| 训练路径有 backward + optimizer.step | ✅ |
| window shuffle ≠ 测试样本混入训练 | ✅ |
| `short_up` 是"未来 5 日累计上涨至少 3%" | ✅ 汇报 AUC 时应写明 |

**特别说明**：不应因"缺少 causal mask"就报错——
窗口内双向 attention 是合法的，因为窗口全部在预测日之前。

---

## 十、结论降级清单

| 文件 | 改动 |
|---|---|
| `MICRO_MODEL_DIAGNOSIS_REPORT.md` | 顶部加修订声明；"必然塌缩"降级为观察 |
| `PROJECT_README.md` | 移除"本质不可预测" |
| `FIVE_YEAR_DATA_VERDICT.md` | 已说明三个方向为负，需加"Transformer 部分待重新验证" |
| 本文件 | 完整记录 |

---

## 十一、修复顺序（按知微建议）

| 顺序 | 任务 | 状态 |
|---|---|---|
| 1 | 固定证据（commit、配置、快照、日志） | ✅ 本文件记录 commit `63584d2` |
| 2 | 修标签时间边界与成熟度 | ✅ **已完成** |
| 3 | 修 CPU 快照、逐列尺度、交易账本 | ✅ 三件都已实现并验证 |
| 3b | 真实输入各列分位数审计 | ⏳ **待做** |
| 3c | `:1000-1005` 零计价问题 | ⏳ **待做** |
| 4 | 先统一为一个明确任务（5 日收益排序） | ⏳ **待做** |
| 5 | 重新规定选模与最终评价 | ⏳ **待做** |

**结论：测量工具已修好，但还没有用它重训。**

按知微的说法："修好测量方法之后，再回答后一类问题。"

---

## 十二、复现与验证命令

```powershell
cd d:\qcr\Stock\stock_analysis_system

# 复现知微的 7 项发现
.venv\Scripts\python.exe scripts\reproduce_transformer_findings.py

# 验证修复件（17 项）
.venv\Scripts\python.exe scripts\verify_transformer_fixes.py
```

---

## 十三、我们的承诺

1. **不静默覆盖历史结果**。旧实验记录保留，标注为"使用有缺陷的实现"。
2. **不把 bug 修复包装成有效性证明**。
3. 修复后如果**预定预算内仍无稳定增量**，就停止这条路线，
   但那时才可以说"在明确的特征、任务、数据范围内没有增量"。
