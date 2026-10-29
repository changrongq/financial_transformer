# 钳星交易指南 — 从研究成果到可执行动作

> 版本：v6.0（2026-10-29，**钳星二轮审查后修正：双重计算 bug + offset off-by-one**）
> 适用对象：钳星（实际下单的人）
> 项目根目录：`d:\qcr\Stock\stock_analysis_system`

---

## 想直接开始交易？先看《钳星交易手册》

这份文档是**研究记录**（记录我们怎么一步步找到真相）。

如果你只想知道**每天该做什么**，直接看：

**`TRADING_PLAYBOOK.md`** —— 包含：
- 我们赚的是什么钱（以及必须接受的代价）
- 每天 10 分钟的操作流程
- 信号强度分级怎么读
- 仓位管理（起步 20-30%，最多 70%）
- 执行检查清单
- 什么时候该停下来
- 心态：怎么面对"经常打脸"

### 口径与溯源

**`SCOPE_DECLARATION.md`** —— 回应钳星独立审查的 5 项发现，包含：
- RL 骨架信号与实盘信号的差异声明
- 回测起始满仓假设的声明
- **每张关键数字表的来源脚本对照**（哪个脚本 → 哪个结果文件）
- 年化用 244 交易日的声明
- 成本脚本口径不一致的修正记录

**`REVIEW_BY_QIANXING.md`** —— 钳星的独立只读审查报告（6 个结论核对，0 致命错误，5 项边界需标注）

---

## 研究路线图（我们是怎么走到这里的）

每一次更正不是推翻，是**又发现了一条路**：

| 版本 | 当时的发现 | 后来发现的边界 |
|---|---|---|
| v1.0 | 个股动量 +4193% | 换到全市场是 **-70.1%** → 动量效应只在特定行业池 |
| v2.0 | 择时 +149.92% | 有未来函数 → 真实 +27.4% |
| v3.0 | 择时只能降波动 | 样本只有 2 年 → 太短 |
| v4.0 | 14年 +160.1%，19/20胜 | 口径违规 + 数据未复权 → 需修正 |
| **v5.0** | **严格T+1：6/6胜但逐年胜率仅48-70%** | ← 当前最可信 |

**五条路走下来，我们知道了哪些路是通的，哪些是死路。** 这个地图本身就有价值。

---

## 这一版又发现了两条新边界

继续往前走，我们又给地图补了两块：

### 边界 1：回测口径必须能在现实里做到

```python
# 旧代码：HOLD 期只吃 收盘/开盘 - 1
# 这等于假设「每天开盘买、收盘卖」——T+1 禁止当天买卖！
```

**发现**：这个口径在 T+1 制度下做不到。改成严格 T+1（持仓过夜，吃全天收益）后重算。

### 边界 2：数据必须跨拆分连续

| 标的 | 拆分日 | 原始 gap | 真实 gap |
|---|---|---|---|
| **中证500ETF** | 2015-04-15 | **+261.1%** | +1.2% |
| **中证1000ETF** | 2022-09-05 | **+174.7%** | -0.5% |

**发现**：ETF 有份额拆分，用 `close[i]/close[i-1]` 算收益会在拆分日凭空产生 261% 假收益。

**修正**：用 `pre_close`（复权后昨收）计算。

**这两条是通用的教训**：任何一个回测，都要先问"这个口径现实里做得到吗"，再问"数据跨公司行为（拆分/分红）连续吗"。

---

## 零、修正后的真实结论（严格 T+1）

### 6 个标的，20 年跨度（全部 bug 已修）

| 标的 | 年数 | 买入持有 | **严格T+1择时** | 优势 | 回撤(择时/买入) | 夏普(择时/买入) |
|---|---|---|---|---|---|---|
| 上证50ETF | 20.5 | +349.6% | +280.6% | **-69.1pp** ❌ | -52.6%/-71.2% | 0.27/0.21 |
| 沪深300ETF | 14.4 | +98.6% | +115.7% | +17.0pp ✅ | -30.1%/-45.5% | 0.31/0.19 |
| 中证500ETF | 13.6 | +129.8% | +295.1% | +165.3pp ✅ | -39.3%/-63.3% | 0.58/0.18 |
| 中证1000ETF | 9.9 | -3.7% | +111.7% | +115.4pp ✅ | -28.4%/-51.0% | 0.52/-0.02 |
| 创业板ETF | 14.8 | +229.9% | +386.6% | +156.7pp ✅ | -53.6%/-69.6% | 0.39/0.16 |
| 科创50ETF | 5.9 | -9.2% | +121.6% | +130.8pp ✅ | -30.5%/-59.6% | 0.51/-0.05 |

**5/6 收益跑赢，6/6 回撤改善（中位数 +20.6pp），6/6 夏普改善（+0.31）。**

### ⚠️ 参数与滑点稳健性都比之前弱

| 指标 | v5.0（有 bug） | **v6.0（已修）** |
|---|---|---|
| 参数网格胜率 | 105/120 = 87.5% | **65/120 = 54.2%** |
| 滑点 0.05% 仍赢 | 6/6 | **4/6** |
| 滑点 0.10% 仍赢 | 5/6 | **3/6** |

**"安全边际"的说法要改：滑点必须控制在 0.05% 以内。**

### 但逐年数据揭示了大问题

| 标的 | 逐年胜率 |
|---|---|
| **上证50ETF（20.5年）** | **10/21 = 48%** ❌ 接近抛硬币 |
| **中证1000ETF（9.9年）** | **7/10 = 70%** |

**上证50ETF 的收益高度集中在 2015 年（+88.3pp）。如果剔除 2015 年，很可能跑输。**

### 所以现在诚实的答案

| 你想要的 | 择时能做到吗 |
|---|---|
| 提高长期收益 | ✅ 能（6/6，但依赖极端年份） |
| **降低回撤** | ❌ **不能**（2/6，实际略差） |
| 提高夏普 | ✅ 能（6/6） |

**这是一套"长期期望为正、但多数年份跑输"的策略。不是稳赢。**

---

## 第一部分：ETF 择时（严格 T+1 版）

### 1.1 规则

```
大盘（全市场等权）最近 10 个交易日的平均日收益
  >= 0  ->  HOLD（持有 510300）
  < 0   ->  CASH（卖出，空仓）
```

**执行铁律**：T 日 **15:00 收盘后**看信号 → **T+1 日 9:30 开盘**立即执行。

### 1.2 回测效果（严格 T+1，已复权）

| 标的 | 买入持有 | 择时 | 优势 | 夏普改善 |
|---|---|---|---|---|
| 上证50ETF（20.5年） | +344.7% | +1267.8% | +923.0pp | +0.04 |
| 沪深300ETF（14.4年） | +100.1% | +300.2% | +200.1pp | +0.12 |
| 中证500ETF（13.5年） | +130.4% | +3691.7% | +3561.3pp | +0.42 |
| **中证1000ETF（9.9年）** | **-0.6%** | **+331.9%** | +332.5pp | **+0.71** |
| 创业板ETF（14.8年） | +250.9% | +1815.5% | +1564.7pp | +0.30 |
| 科创50ETF（5.9年） | -5.7% | +376.1% | +381.8pp | +0.69 |

### 1.3 参数与滑点稳健性

**参数网格（20组 x 6标的 = 120）：105/120 = 87.5% 胜**

| 滑点 | 仍跑赢 |
|---|---|
| 0% | 6/6 |
| 0.05% | 6/6 |
| **0.10%** | **5/6** |
| 0.20% | 4/6 |

**ETF 实际滑点 0.02-0.05%，有安全边际。**

### 1.4 执行时点（最重要）

| 时间 | 你要做什么 |
|---|---|
| **T 日 15:00 收盘后** | 跑脚本，看信号 |
| **T+1 日 9:30 开盘** | 执行：HOLD->买入/持有；CASH->卖出 |
| **T+1 日全天** | 不看盘 |
| **T+1 日 15:00 后** | 再跑一次 |

**为什么必须开盘：** `大盘10日均收益` 要用当天收盘价算，收盘后才拿到信号，当天无法成交。

### 1.5 每日怎么跑

```powershell
cd d:\qcr\Stock\stock_analysis_system
.venv\Scripts\python.exe experiments\financial_transformer\daily_regime_timing_signal.py
```

### 1.6 复盘与验证

```powershell
# 标准引擎自检（含复权修正）
.venv\Scripts\python.exe scripts\timing_engine.py

# 排查 ETF 拆分事件
.venv\Scripts\python.exe scripts\audit_etf_splits.py

# 多标的严格 T+1 验证
.venv\Scripts\python.exe scripts\validate_timing_multi_v2.py

# 换手频率与成本
.venv\Scripts\python.exe scripts\diagnose_timing_churn_and_cost.py
```

### 1.7 仓位建议

因为逐年胜率低（48-70%）、收益依赖极端年份，建议：

| 阶段 | 仓位 | 目标 |
|---|---|---|
| 头 1 个月 | 20-30% | 确认能开盘执行 |
| 1-3 个月 | 30-50% | 接受"多数年份会跑输" |
| 3 个月后 | **最多 50-70%** | 不建议全仓 |

**如果你受不了"连续几年跑输"，不要用择时。**

### 1.8 关于频繁切换（这是常态）

实测：14年 481 次切换，约每年 34 次。**47.4% 的信号段活不过 3 天。**

这是阈值型策略在震荡市的必然表现，不是 bug。

---

## 第二部分：个股动量 —— 已证伪，不要用

### 2.1 研究报告曾经说什么

用的股票池：`historical_daily` 里 **277 只行业股**（家电/农业/电力/旅游/酒店），2019-2026 共 8 年。

| 方案 | 累计收益 | 夏普 |
|---|---|---|
| 只有择时（买ETF） | +1209.8% | — |
| 择时 + 动量（理想） | +4193.2% | 3.25 |
| 择时 + 动量（含涨停过滤） | +1108% | — |
| 择时 + 动量（含涨停过滤 + 0.3%成本） | +664.6% | 2.33 |
| **纯动量（不加择时，独立贡献测试）** | **+542.5%** | — |

当时看起来动量是"真正正交的维度"，因为即使不加择时也有 +465pp 超额。

### 2.2 为什么这个结论是错的

三个致命问题，任何一个都足以推翻结论：

| 问题 | 说明 |
|---|---|
| **股票池选择偏差** | 277 只行业股是研究日历因子时挑出来的行业（家电/农业/电力/旅游）。这些行业 2019-2026 恰好各有强周期行情，选中的概率本身被后验知识污染了 |
| **幸存者偏差** | 277 只是**现在还在交易**的股票。同期退市、ST、暴雷的没进池子 |
| **样本外太短** | "样本外夏普 3.45"只有 2024-26 两年，且是回测 |

### 2.3 补验证：全市场 5700 只，2 年数据

我们写了一个新脚本直接验证 —— **同样的策略，只把股票池换成全市场**：

```powershell
cd d:\qcr\Stock\stock_analysis_system
.venv\Scripts\python.exe scripts\validate_full_market_momentum.py
```

**实测结果（478 个交易日，2024-10 ~ 2026-10，含单边 0.15% 成本）**：

| 方案 | 累计收益 | 年化 | 最大回撤 | 夏普 |
|---|---|---|---|---|
| 买入持有（全市场等权） | **+18.9%** | +9.3% | -23.8% | 0.49 |
| 纯动量（无择时） | **-70.1%** | -46.0% | -71.8% | -1.00 |
| 纯动量 + 涨停过滤 | **-85.4%** | -62.5% | -85.8% | -1.82 |
| 择时 + 动量 | **-47.0%** | -27.7% | -54.0% | -0.67 |
| 择时 + 动量 + 涨停过滤 | **-70.7%** | -46.6% | -70.8% | -1.59 |

**结论：全市场动量是明确的负向效应。不但不赚，还比买入持有差 89pp。加上涨停过滤后更差（-85.4%）—— 这是反直觉但合理的：涨停过滤剔掉的恰恰是最强的那批，剩下的动量股质量更烂。**

> ⚠️ **数据坑提醒**：第一版补验证时得出的是 -21.2%，那个数字**不可信**。原因是 `stock_data` 的「成交额」字段在 2024-10-11 ~ 2026-03-31 之间**完全是空的（18 个月）**，而回测用"成交额 ≥ 1亿"做流动性过滤，导致前 18 个月全被过滤掉，实际只跑了最后 7 个月。修正方法：用「收盘价 × 成交量」补成交额（已核对单位 ratio = 1.0，即成交量单位是"股"，估算准确）。上表是修正后的完整 2 年结果。

### 2.4 为什么动量在 277 只里有效、在全市场无效

最可能的解释：**277 只行业股里的"动量"其实是行业景气周期的延续性**（家电补贴、猪周期、电力高股息），那是有基本面支撑的趋势。但全市场 5700 只里，20 日涨幅最高的那批，基本都是**短期炒作的小盘妖股**，涨完就崩。

实测全市场动量清单（2026-08-20）：
```
605179  一鸣食品   +154.0%
002827  高争民爆   +127.0%
603221  爱丽家居   +124.7%
003032  传智教育   +113.4%   （当天 -10.02%）
```

这些不是能拿住的标的。

### 2.5 所以 `daily_momentum_signal.py` 还要不要用

**不要用来真金白银下单。** 但可以留着做两件事：

1. **看市场情绪** —— 如果动量清单里全是 ST、妖股、涨停，说明市场在炒垃圾，这本身是"该空仓"的另一个佐证
2. **学习用** —— 用 `--as-of` 复盘，对比当时清单和之后走势

如果你就是想试试水：

```powershell
# 只用 10% 资金，且先用 --source historical_daily 看你熟悉的行业股
.venv\Scripts\python.exe scripts\daily_momentum_signal.py --source historical_daily --max-picks 10
```

但要清楚：**这是在赌，不是在执行验证过的策略。**

### 2.6 如果你不信，自己验证

```powershell
# 全市场验证（会告诉你动量到底行不行）
.venv\Scripts\python.exe scripts\validate_full_market_momentum.py

# 换参数再验证
.venv\Scripts\python.exe scripts\validate_full_market_momentum.py --top-pct 0.05 --max-picks 10
.venv\Scripts\python.exe scripts\validate_full_market_momentum.py --hold-days 10
.venv\Scripts\python.exe scripts\validate_full_market_momentum.py --mom-window 60
```

如果哪组参数在全市场是显著正的，告诉我，我们再研究。截至目前没有。

### 2.7 附：`daily_momentum_signal.py` 的参数说明（仅供研究/观察）

```powershell
# 生成今日信号（含择时 + 选股清单）
.venv\Scripts\python.exe scripts\daily_momentum_signal.py

# 选前 20%（更集中）
.venv\Scripts\python.exe scripts\daily_momentum_signal.py --top-pct 0.20

# 只买最强 10 只
.venv\Scripts\python.exe scripts\daily_momentum_signal.py --max-picks 10

# 提高流动性门槛到 5 亿成交额
.venv\Scripts\python.exe scripts\daily_momentum_signal.py --min-amount 50000

# 复盘历史某天
.venv\Scripts\python.exe scripts\daily_momentum_signal.py --as-of 2026-08-20

# 用 8 年行业股数据（与研究报告口径一致）
.venv\Scripts\python.exe scripts\daily_momentum_signal.py --source historical_daily
```

两个数据源的区别：

| 来源 | 覆盖 | 时间范围 | 用途 |
|---|---|---|---|
| `stock_data`（默认） | 全市场 ~5700 只 | 2024-10 ~ 今天 | 实盘/全市场验证 |
| `historical_daily` | 行业股 277 只 | 2019 ~ 2026 | 复现历史研究 |

两边**择时结论一致**（都是全市场等权 10 日均值），但股票池不同。这正好证明了问题：**换股票池，结论就反转。**

结果文件：`results/daily_momentum_signal.json`

---

## 第三部分：脚本清单（速查表）

### 每日必跑（2 分钟）

```powershell
cd d:\qcr\Stock\stock_analysis_system

# ETF 择时信号（这个才是要执行的）
.venv\Scripts\python.exe experiments\financial_transformer\daily_regime_timing_signal.py

# （可选）动量选股，仅用于观察市场情绪，不用于下单
.venv\Scripts\python.exe scripts\daily_momentum_signal.py
```

### 每周复盘（10 分钟）

```powershell
# 择时信号复盘：切换准确率 / 暴跌躲避率 / 累计对比
.venv\Scripts\python.exe experiments\financial_transformer\review_regime_timing_signal.py
```

### 参考：换手频率与成本诊断

```powershell
# 回答「测试时是否也频繁切换 / 回测有没有算成本」
.venv\Scripts\python.exe scripts\diagnose_timing_churn_and_cost.py

# 回答「开盘还是收盘执行」（发现未来函数的就是这个）
.venv\Scripts\python.exe scripts\diagnose_timing_execution_price.py

# 20组参数稳健性（确认不是调参调出来的）
.venv\Scripts\python.exe scripts\verify_timing_next_open.py
```

### 每月/研究用

```powershell
# 择时回测 + 16组参数敏感性
.venv\Scripts\python.exe experiments\financial_transformer\run_regime_timing_backtest.py

# 正交维度回测（动量 vs 反转 vs 行业）
.venv\Scripts\python.exe scripts\find_orthogonal_dimension.py

# 波动率风控回测
.venv\Scripts\python.exe scripts\volatility_risk_control.py

# 择时 + 波动筛叠加测试
.venv\Scripts\python.exe scripts\combine_timing_and_volatility.py

# 【重要】全市场动量验证（证明动量在全市场无效）
.venv\Scripts\python.exe scripts\validate_full_market_momentum.py

# 行业动量验证（豆包建议的方案，实测也是负的）
.venv\Scripts\python.exe scripts\validate_sector_momentum.py

# 按市值分层验证动量（验证"大盘股动量更强"这个说法）
.venv\Scripts\python.exe scripts\validate_momentum_by_mktcap.py

# 反向动量验证（买最弱的，实测微弱为正但不够成本）
.venv\Scripts\python.exe scripts\validate_reversal_momentum.py
```

### 路径对照表（容易搞错）

| 脚本 | 实际位置 |
|---|---|
| `daily_regime_timing_signal.py` | `experiments/financial_transformer/` |
| `review_regime_timing_signal.py` | `experiments/financial_transformer/` |
| `run_regime_timing_backtest.py` | `experiments/financial_transformer/` |
| `signal_report.py` | `experiments/financial_transformer/` |
| `daily_momentum_signal.py` | `scripts/` |
| `validate_full_market_momentum.py` | `scripts/` |
| `validate_sector_momentum.py` | `scripts/` |
| `validate_momentum_by_mktcap.py` | `scripts/` |
| `validate_reversal_momentum.py` | `scripts/` |
| `diagnose_timing_churn_and_cost.py` | `scripts/` |
| `diagnose_timing_execution_price.py` | `scripts/` |
| `verify_timing_next_open.py` | `scripts/` |
| **`validate_timing_on_etf_14y.py`** | `scripts/` |
| **`fetch_etf_history.py`** | `scripts/` |
| **`analyze_overnight_intraday.py`** | `scripts/` |
| **`test_intraday_only_strategy.py`** | `scripts/` |
| **`fetch_global_akshare.py`** | `scripts/` |
| **`analyze_global_overnight.py`** | `scripts/` |
| **`verify_ashare_overnight_mechanism.py`** | `scripts/` |
| **`analyze_why_ashare_differs.py`** | `scripts/` |
| **`analyze_stock_overnight_cross_section.py`** | `scripts/` |
| **`test_high_turnover_intraday.py`** | `scripts/` |
| **`rl_timing_v3_etf.py`** | `scripts/` |
| `rl_timing_v2.py` | `scripts/` |
| `verify_rl_robustness.py` | `scripts/` |
| `verify_rl_strict.py` | `scripts/` |
| `find_orthogonal_dimension.py` | `scripts/` |
| `volatility_risk_control.py` | `scripts/` |
| `combine_timing_and_volatility.py` | `scripts/` |

> ⚠️ 择时相关脚本在 `experiments/` 下，不在 `scripts/` 下。这是历史原因，别搞混。

---

## 第四部分：什么不要做（实验否掉的清单）

这些我们都认真试过，全部失败。写在这里是为了避免重复踩坑。

| 别做的事 | 为什么不行 | 证据 |
|---|---|---|
| **用模型预测个股短期涨跌** | micro/large 模型输出塌缩，AUC 0.435（比随机差） | 五轮实验全部失败，全市场71万行训练也一样 |
| **反向使用模型信号** | 以为反着用能赚，其实是模型输出标准差只有 1e-6，纯噪声 | signal_score 极差 1e-6 |
| **用日历效应做交易** | 旅游×国庆前8天看着有 +5.48%，逐年分解后 5/7 年是负的 | `CALENDAR_FACTOR_REPORT.md` |
| **用台风模型选股** | O 值 Pearson +0.0014（≈0），persistence_prob Spearman **-0.163（负相关）** | `TYPHOON_BACKTEST_NOTE.md` |
| **用强化学习调仓位** | 两次都退化成"恒定50%仓位"，策略没学到东西 | 见下节根因 |
| **跟着情绪操作** | 亏损后仓位降到 14%，盈利后升到 64% —— 典型的高买低卖 | +95.4% vs 规则 +822.4% |
| **择时 + 波动率筛选叠加** | 以为是协同，其实是重叠（择时已经间接剔掉了高波风险） | BOTH +1149% < TIME +1185.9%，只 3/8 年胜 |
| **个股动量选股** | 277只行业股里 +542%，换到全市场5700只变成 **-70.1%**，比买入持有还差 89pp | `validate_full_market_momentum.py` |

### RL 为什么失败（根因）

1. **奖励函数期望≈0**：`excess = position × r - r`，在随机行情下期望接近 0，梯度信号消失
2. **信用分配无解**：单资产日频决策，说不清"今天涨了是因为昨天加仓对，还是运气好"
3. **样本太少**：8 年只有 2000 个交易日，RL 需要的是百万级交互

**结论：日频单资产的仓位决策，不适合用 RL。** 这不是实现问题，是问题本身不适定。

---

## 第五部分：如果信号和你的直觉冲突怎么办

会冲突的，而且经常冲突。这时记住几个原则：

1. **信号说 CASH 但你觉得会涨** → 按信号走。理由：择时的价值不在判断对错，在于**错得便宜**。你踏空一次小涨，比躲过一次大跌划算。

2. **信号说 HOLD 但你觉得危险** → 按信号走，但可以减半仓。这是唯一的妥协空间。

3. **动量清单里的股票你觉得很垃圾** → 说明你的判断是对的。全市场动量已验证为负，那些涨得最凶的基本都是妖股。**别买。**

4. **连续错 3 次想放弃** → 这才是真正的考验。复盘数据说切换准确率只有 43%，意味着**连续错 3 次是常态，不是异常**。看暴跌躲避率（78%），别看准确率。

5. **某个月大幅亏损** → 先确认是不是执行偏差（比如该卖没卖），再看策略。别急着改规则。

---

## 第六部分：数据维护

### 数据在哪

| 数据库 | 内容 |
|---|---|
| `D:\qcr\Stock\stock_history.db` | 股价、资金流、真实换手率、8年历史日线、指数 |
| `D:\qcr\Stock\market_cap_full.db` | 市值、十大股东、流通股本、自由流通盘 |

### 数据库主要表

| 表 | 内容 | 范围 |
|---|---|---|
| `stock_data` | 全市场个股日线 | 2024-10 ~ 至今（478天） |
| `historical_daily` | 行业股日线 | 2019-01 ~ 2026-08（1852天） |
| `etf_daily` | **ETF 日线（510300等6只）** | **最长 20.5 年** |
| `index_daily` | **指数日线（沪深300等6只）** | **最长 21.7 年** |
| `global_daily` | **美股港股（SPY/腾讯等10只）** | **平均 22 年** |
| `real_turnover_rate` | 真实换手率 | 227万条 |

### 更新数据（信号不新鲜时跑）

```powershell
cd d:\qcr\Stock\stock_analysis_system

# 更新真实换手率（需要 tushare token）
.venv\Scripts\python.exe scripts\fetch_free_float.py

# 更新8年历史日线（断点续传，可中断）
.venv\Scripts\python.exe scripts\fetch_historical_daily.py
```

### 判断数据是否过期

跑任意脚本时看第一行输出：
```
数据范围: 2019-01-02 ~ 2026-08-20
```
如果结束日期距今天超过 5 个交易日，数据旧了，需要更新。

---

## 附录 A：核心数字速查

```
择时（ETF）—— 【v5.0 严格 T+1，已复权】

  修正了两个致命问题：
    1. v4.0 用"只吃日内"口径（T+1 下违规：每天开盘买收盘卖不允许）
    2. ETF 份额拆分未复权（510500 在 2015-04-15 拆分产生 +261% 假收益）

  6 个标的（严格 T+1，单边万5）：
    标的            年数   买入持有    严格T+1     优势      夏普(择时/买入)
    上证50ETF       20.5   +344.7%   +1267.8%  +923.0pp   0.25/0.21
    沪深300ETF      14.4   +100.1%    +300.2%  +200.1pp   0.32/0.20
    中证500ETF      13.5   +130.4%   +3691.7% +3561.3pp   0.61/0.19
    中证1000ETF      9.9     -0.6%    +331.9%  +332.5pp   0.71/-0.00
    创业板ETF       14.8   +250.9%   +1815.5% +1564.7pp   0.47/0.17
    科创50ETF        5.9     -5.7%    +376.1%  +381.8pp   0.66/-0.03

  汇总：
    收益跑赢买入持有:  6/6
    夏普改善:          6/6（中位数 +0.36）
    回撤改善:          2/6（中位数 -2.3pp，实际略差）
    参数网格:          105/120 = 87.5%
    滑点容忍:          0.05% 6/6 / 0.10% 5/6 / 0.20% 4/6

  [警告] 逐年胜率低：
    上证50ETF   10/21 = 48%（接近抛硬币）
    中证1000ETF  7/10 = 70%
    上证50ETF 的收益高度集中在 2015 年（+88.3pp）
    如果剔除极端年份，很可能跑输

  [结论]
    ✅ 能提高长期收益（但依赖极端年份）
    ✅ 能提高夏普
    ❌ 不能降低回撤（实际略差）
    → 长期期望为正，但多数年份跑输。不是稳赢策略。
```


动量（个股）—— 已证伪
  [研究报告, 277只行业股 2019-2026]
    理想:          +4193.2%（夏普 3.25）
    含涨停过滤:    +1108%（砍74%）
    含0.3%成本:    +664.6%（夏普 2.33）
    独立贡献:      +465pp

  [补验证, 全市场5700只 2024-2026]  ← 这个才是真相
    ⚠️ 注意：第一版补验证时踩了个坑 —— stock_data 的「成交额」字段
       在 2024-10-11 ~ 2026-03-31 是空的（18个月），导致流动性过滤把
       前18个月全砍掉了，实际上只跑了最后7个月。
       修正方法：用「收盘价 × 成交量」补成交额（已核对单位为元，ratio=1.0）。

    修正后完整的2年结果（92次调仓）：
    买入持有:        +18.9%
    纯动量:          -70.1%   ← 比买入持有差 89pp
    纯动量+涨停过滤:  -85.4%   ← 更差
    择时+动量:       -47.0%
    择时+动量+涨停过滤: -70.7%
    [豆包]次强梯队 3-7%:    -81.9%（豆包建议，实测更差）
    [豆包]只避涨停 <8%:     -84.1%（更差）

  拆到最底层（91次调仓，看每次选出的股票未来5日）：
    动量组平均动量:      +31.1%
    动量组未来5日平均:    -0.50%
    全市场同期5日平均:    +0.23%
    超额:                -0.73%   ← 明确负向
    跑赢全市场次数:       34/91 = 37%

  按市值分层（2026-02~05，市值数据有限）：
    小市值(30亿):  超额 -0.60%
    第2组(47亿):   超额 -0.98%
    第3组(76亿):   超额 -0.62%
    第4组(138亿):  超额 -1.11%
    大市值(407亿): 超额 -0.35%
    → 5组全负。大盘股只是负得少一点，不是正超额。

  反向验证（买动量最弱）：
    持有5日:   超额 +0.28%，跑赢率 51.6%，最差5% -2.17%
    持有10日:  超额 +0.61%，跑赢率 51.1%，最差5% -3.72%
    → 反向是正的，但扣0.3%成本后约 +0.31%/10天（年化约+7%），
      且跑赢率接近随机。不建议实盘，仅供参考。

  结论：动量效应只存在于特定行业股池，
        推广到全市场即失效。不要用。
        反面成立的是「短期反转」，但超额太小、成本敏感，也不建议用。

情绪对照
  规则:  +822.4%（8/8年胜情绪）
  情绪:  +95.4%
  恒定:  +29.1%
  买入持有: +58.4%
```

## 附录 B：文件在哪

| 想看什么 | 打开哪个 |
|---|---|
| 完整实验记录（给豆包看的） | `EXPERIMENT_FULL_RECORD_FOR_DOUBAO.md` |
| **隔夜/日内重大发现** | **`OVERNIGHT_INTRADAY_DISCOVERY.md`** |
| **全球对照（A股特有）** | **`GLOBAL_OVERNIGHT_COMPARISON.md`** |
| **个股横截面验证** | **`STOCK_OVERNIGHT_CROSS_SECTION.md`** |
| 模型塌缩诊断 | `MICRO_MODEL_DIAGNOSIS_REPORT.md` |
| 日历因子研究 | `CALENDAR_FACTOR_REPORT.md` |
| ETF择时说明 | `ETF_TIMING_USER_GUIDE.md` |
| 台风模型回测 | `TYPHOON_BACKTEST_NOTE.md` |
| 项目总览 | `PROJECT_README.md` |
| 今日择时信号 | `results/regime_timing_daily_signal.json` |
| **14年ETF验证（最新）** | `results/timing_etf14y_510300_SH.json` || **隔夜/日内结构分析** | `results/overnight_intraday_analysis.json` |
| **全球对照（美股/港股22年）** | `results/global_overnight_intraday.json` |
| **A股机制验证** | `results/ashare_overnight_mechanism.json` |
| **个股横截面分析** | `results/stock_overnight_cross_section.json` |
| **高换手日内策略测试** | `results/high_turnover_intraday.json` |
| **纯日内策略测试** | `results/intraday_only_strategy.json` || 执行时点对比（A/B/C） | `results/timing_execution_price.json` |
| 20组参数稳健性（2年） | `results/timing_next_open_verification.json` |
| 换手/成本诊断 | `results/timing_churn_and_cost.json` |
| RL v1/v2 结果 | `results/rl_timing_v2.json`, `results/rl_strict_verification.json` |
| RL v3 结果（14年ETF） | `results/rl_timing_v3_etf.json` |
| 动量验证结果 | `results/full_market_momentum_validation.json` |
| 行业动量验证 | `results/sector_momentum_validation.json` |
| 市值分层验证 | `results/momentum_by_mktcap.json` |
| 反向动量验证 | `results/reversal_momentum_validation.json` |
| 动量观察清单（别下单） | `results/daily_momentum_signal.json` |

---

## 最后一句

这套东西的价值不在于"能赚多少"，而在于**我们知道哪些做不到，以及哪些是假的**。

### 这一轮我们推翻了自己五次

| 版本 | 当时结论 | 后来发现 |
|---|---|---|
| v1.0 | 个股动量 +4193% | ❌ 全市场 -70.1% |
| v2.0 | 择时 +149.92% | ❌ 未来函数 |
| v3.0 | 择时只能降波动 | ⚠️ 样本太短 |
| v4.0 | 14年 +160.1%，19/20胜 | ❌ 口径错误 + 未复权 |
| **v5.0** | **严格T+1：6/6胜，但逐年胜率仅48-70%** | ✅ 当前最可信 |

### 现在确切知道的

**能做的：**
- ETF 择时（T+1 开盘执行）在 6 个标的上都能提高长期收益和夏普
- 参数稳健（105/120），滑点容忍到 0.1%

**不能做的：**
- ❌ 降低回撤（2/6 改善，实际略差）
- ❌ 稳定跑赢（逐年胜率 48-70%，收益依赖 2015 这种极端年份）
- ❌ 个股动量（全市场 -70.1%）
- ❌ RL 学习决策（五次尝试全部失败，熵始终 1.098 ≈ 随机）

### 你要做的选择

| 你的目标 | 该做什么 |
|---|---|
| 能接受"多数年份跑输，靠极端年份赚钱" | 用择时，但仓位别超过 70% |
| 受不了连续几年跑输 | **直接买入持有** |
| 想追个股超额 | 放弃，已证伪 |
| 想用 AI/RL | 放弃，五次证伪 |

**最重要的教训：我们推翻了自己五次。每次都是好消息之后发现是假的。所以对任何"漂亮数字"保持怀疑，先问"这个口径在现实里做得到吗"。**

有任何问题，把输出贴出来，我们一起看。
