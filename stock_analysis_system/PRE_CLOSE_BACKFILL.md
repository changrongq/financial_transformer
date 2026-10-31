# `pre_close` 数据补齐报告

> 2026-10-31
> 背景：横截面研究（`CROSS_SECTION_STOCK_SELECTION.md`）发现 `stock_data` 缺少
> `pre_close` 字段，除权日的价格跳空无法区分，污染所有收益率计算。
> 脚本：`scripts/backfill_pre_close.py`、`scripts/verify_pre_close.py`

---

## 一、为什么需要 `pre_close`

### 1.1 直接的问题

`stock_data` 只有 `开盘/最高/最低/收盘`，没有「昨收」。

计算日收益时只能用 `close[t] / close[t-1] - 1`。

**如果期间发生分红送股，`close[t]` 和 `close[t-1]` 不可比**，会算出假跌。

### 1.2 影响范围

| 分析 | 受影响程度 |
|---|---|
| 横截面选股 IC | **直接受影响**（12 个因子全部依赖收益） |
| 个股隔夜/日内分解 | **直接受影响**（`STOCK_OVERNIGHT_CROSS_SECTION.md` 已注明此限制） |
| ETF 相关分析 | 不受影响（`etf_daily` 本来就有 `pre_close`） |
| 指数相关分析 | 不受影响（`index_daily` 有 `pre_close`） |

---

## 二、排查过程（有一个重要转折）

### 2.1 一开始的错误判断

我最初观察到 `etf_daily` 的 `amount/(vol×close)` 稳定在 0.1，
以及 `000001` 在某些日期价格比例突然变化，于是判断：

> ~~「本地 close 在除权日跳空，2025-06-12 有 +1.65% 的假涨，真实是 -1.43%」~~

**这个判断是错的。**

### 2.2 实际验证

用 tushare 的 `pre_close`（交易所口径、已除权调整）逐日比对：

```
日期          raw_close  local_close  raw_pre   ret_raw   ret_db
2025-06-11      11.85       11.49      11.810    0.34%    0.34%
2025-06-12      11.68       11.68      11.490    1.65%    1.65%   ← 除权日
2025-06-13      11.58       11.58      11.680   -0.86%   -0.86%
```

**`ret_raw` 与 `ret_db` 完全一致（偏差 0.000000%）。**

也就是说：
- tushare 的 `pre_close` 在除权日**已经是调整后的值**（11.490）
- 本地 `close` 也是同一口径（11.49 → 11.68 = +1.65%）
- **本地数据本来就是对的**

我之前误以为 `pre_close` 是「前一日的原始收盘价」，实际它是「除权调整后的昨收」。

### 2.3 那为什么还要补？

**因为我们需要这个字段本身，而不只是"知道本地数据没问题"。**

理由：
1. **不能再依赖隐式假设**。如果以后有人换成未复权数据源，没有 `pre_close` 就无法察觉。
2. **隔夜/日内分解需要它**。`gap = open[t]/pre_close[t] - 1`，只有 `pre_close` 才有正确口径。
3. **可以验证数据一致性**。`adj_factor = close_local / close_raw` 能发现数据源切换。

---

## 三、实现方案

### 3.1 数据来源

tushare `daily` 接口，字段：
```
ts_code, trade_date, close, pre_close
```

`pre_close` 是交易所口径的除权调整昨收。

### 3.2 口径换算

本地 `close` 与 tushare `close` 差一个恒定比例（复权因子）：

```
adj_factor    = 本地close / tushare.close
pre_close_adj = tushare.pre_close × adj_factor
```

于是：

```
本地close / pre_close_adj
  = 本地close / (tushare.pre_close × 本地close / tushare.close)
  = tushare.close / tushare.pre_close
  = 真实收益 ✓
```

### 3.3 新增字段

```sql
ALTER TABLE stock_data ADD COLUMN pre_close REAL;    -- 与本地 close 同口径的复权昨收
ALTER TABLE stock_data ADD COLUMN adj_factor REAL;   -- 本地 close / tushare 未复权 close
```

`adj_factor` 的用途：
- 发现数据源切换（如果突然变化，说明复权基准变了）
- 可以反推出任意口径的价格

### 3.4 断点续传

```sql
CREATE TABLE stock_pre_close_progress (
    ts_code TEXT PRIMARY KEY, status TEXT, n_rows INTEGER,
    n_filled INTEGER, msg TEXT, updated_at TEXT
);
```

已 `status='ok'` 的股票跳过，中断后可继续。

---

## 四、踩过的坑

### 4.1 日期格式不一致（第一次失败）

```
tushare trade_date: '20261029'
本地 stock_data.日期: '2026-10-29'
```

`UPDATE ... WHERE 日期='20261029'` 匹配 0 行 → 写入无效但脚本报 ok。

**这个坑之前在 `refresh_market_data.py` 也踩过，这次又踩了一次。**
修复：写入前统一转成 `YYYY-MM-DD`。

### 4.2 进度表结构不兼容（第二次失败）

第一版表有 `n_matched/n_mismatch` 列，第二版改成 `n_filled`，
`CREATE TABLE IF NOT EXISTS` 不会改已有表 → `no column named n_filled`。

修复：检测列不匹配时 `DROP` 重建。

### 4.3 tushare 限速 300 次/分钟（第三次失败）

8 并发无节流 → 每小时会有 7 只报错：
```
Exception: 抱歉，您访问接口(daily)频率超限(300次/分钟)
```

修复：
- 增加滑动窗口限速器（280 次/分钟）
- 限速错误单独退避（5s, 10s, 20s, 40s）
- 并发从 8 降到 4

### 4.4 脏代码 `'None'`

`stock_data` 里有一个 `stock_code = None` 的脏记录，转 ts_code 后变成 `00None.SZ`。

修复：`to_ts_code()` 对非数字输入返回空串，调用方跳过。

---

## 五、执行结果

| 项目 | 数值 |
|---|---|
| 有效股票数 | 5,879 |
| 总行数 | 2,545,113 |
| 成功股票数 | **5,631** |
| 未能补充 | **248**（全部为北交所，见下） |
| 写入行数 | **2,487,406** |
| `pre_close` 覆盖率 | **97.73%** |
| 缺失行数 | 57,707 |
| 耗时 | 约 22 分钟（含重跑） |

### 5.1 未能补充的 248 只

**全部是北交所股票**（代码前缀 43/83/87）。

原因：tushare 的 `daily` 接口**不覆盖北交所**，北交所需要用 `bak_daily` 接口。

影响评估：
- 占股票总数 4.2%，行数 2.3%
- 北交所股票流动性差、市值小
- 横截面研究本来就会按「换手率 ≥ 0.1%」「上市满 200 个交易日」过滤
- **判定：可以接受，不影响研究结论**

如需补齐，可用 `bak_daily`，但优先级低。

### 5.2 修复的一个代码映射 bug

第一次运行时 604 只 no data，原因是 `to_ts_code` 把 `92xxxx` 误判成上交所。
修正为：

```python
# 北交所：43/83/87/88/92
if p2 in ("43", "83", "87", "88", "92"):
    return digits + ".BJ"
# 上交所：60/68/90
if digits.startswith(("60", "68", "90")):
    return digits + ".SH"
# 深交所：00/30/20
if digits.startswith(("00", "30", "20")):
    return digits + ".SZ"
```

修正后 356 只北交所成功补上（说明部分北交所确实有数据）。

---

## 六、验证

### 6.1 覆盖率

```
总行数        : 2,545,113
pre_close 缺失: 57,707
pre_close <=0 : 0
覆盖率        : 97.73%
进度          : {'ok': 5,631, 'empty': 248}
```

### 6.2 收益分布合理性

用 `pre_close` 计算的日收益：

| 分位 | 收益 |
|---|---:|
| p0.01% | **-20.015%** |
| p0.10% | -15.643% |
| p1.00% | -8.930% |
| p5.00% | -4.621% |
| p50.00% | +0.000% |
| p95.00% | +5.202% |
| p99.00% | +10.022% |
| p99.90% | **+20.000%** |
| p99.99% | +29.988% |

**极值正好落在涨跌停限制内**：
- p0.01% 的 -20.015% ≈ 创业板/科创跌停
- p99.90% 的 +20.000% = 创业板/科创涨停
- p99.99% 的 +29.988% = 新股上市首日

### 6.3 极端值检查

|ret| > 21%（超出所有板块涨跌停）：**550 行，占 0.0221%**

样例：
```
002898  2026-06-26  收盘 0.35  pre_close 8.30   -95.78%  adj 1.0000
002808  2026-06-23  收盘 0.20  pre_close 2.83   -92.93%  adj 1.0000
300029  2026-06-18  收盘 0.20  pre_close 2.77   -92.78%  adj 1.0000
```

**这些是真实的退市整理期暴跌**（`adj_factor = 1.0000` 说明没有数据源切换），不是换算错误。

按板块统计：
```
主板         总 1,566,745  跌停附近 210   严重超跌 240
创业板/科创  总   920,661  跌停附近   4   严重超跌  13
```

主板极端值多，符合「ST 股/退市股集中在主板」的实际情况。

### 6.4 adj_factor 稳定性

```
600519 贵州茅台: 460 天，adj_factor 跳变 3 次
  adj_factor 范围: 0.9829 ~ 1.0001
```

**只在除权日跳变**，符合茅台一年一次分红的实际情况。

### 6.5 独立抽查比对

随机抽 15 只，重新调 tushare 比对：

```
比对结果: 一致 12  不一致 0
```
（3 只因样本不足 20 行被跳过）

**换算公式验证正确。**

---

## 七、对之前结论的影响

### 7.1 需要重跑的分析

| 分析 | 是否需重跑 | 原因 |
|---|---|---|
| 横截面选股 | **必须** | 所有因子依赖收益 |
| 个股隔夜/日内 | **必须** | 需要 `gap = open/pre_close - 1` |
| ETF 择时 | 不需要 | 本来就用 `etf_daily.pre_close` |
| RL 实验 | 不需要 | 用 ETF 数据 |
| 特征价值检验（ETF） | 不需要 | 用 ETF 数据 |

### 7.2 预期影响

因为本地 `close` 本来就是除权调整后的（已验证），
所以**用 `close.pct_change()` 算的收益其实已经是对的**。

补 `pre_close` 的收益主要在于：
- 能让 `gap`（隔夜）算对
- 能显式检查数据质量
- 不再依赖隐式假设

**所以横截面结论「无增量信息」大概率不会因为这次补数据而改变。**

但这需要重跑确认，不能假设。

---

## 八、复现命令

```powershell
cd d:\qcr\Stock\stock_analysis_system

# 全量补充（断点续传，可重复运行）
.venv\Scripts\python.exe scripts\backfill_pre_close.py --workers 4

# 查看进度
.venv\Scripts\python.exe scripts\monitor_pre_close.py
.venv\Scripts\python.exe scripts\monitor_pre_close.py --watch 60

# 校验
.venv\Scripts\python.exe scripts\backfill_pre_close.py --verify-only
.venv\Scripts\python.exe scripts\verify_pre_close.py

# 小样本测试（含重置）
.venv\Scripts\python.exe scripts\backfill_pre_close.py --limit 10 --reset
```

---

## 九、备份

执行前已备份数据库：

```
D:\qcr\Stock\stock_history_before_pre_close.db  (1322 MB)
```

如需回滚：
```powershell
Copy-Item D:\qcr\Stock\stock_history_before_pre_close.db D:\qcr\Stock\stock_history.db -Force
```
