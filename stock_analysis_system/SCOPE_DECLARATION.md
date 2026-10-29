# 口径声明（审计回应）
> 2026-10-29
> 回应钳星独立审查报告的 5 项发现。
> 每一项都核对了代码，确认成立，并做了修正或显式声明。

---

## 审查结论回顾

钳星核了 6 个关键结论，**0 个致命错误**，5 个"边界需更诚实标注"。5 项全部成立。

| # | 钳星的发现 | 我的核实 | 处理 |
|---|---|---|---|
| 1 | RL 骨架信号 ≠ 实盘信号 | ✅ 成立 | 代码加声明 |
| 2 | 回测起始硬编码满仓 | ✅ 成立 | 代码加声明 |
| 3 | 结论表来源未标注 | ✅ 成立 | **本文档填补** |
| 4 | 年化用 244 未声明 | ✅ 成立 | 常量+声明 |
| 5 | 成本脚本"择时对照"是旧口径 | ✅ 成立 | **改了代码** |

---

## 问题 1：RL 骨架信号 ≠ 实盘信号

### 事实

| 脚本 | 信号计算方式 |
|---|---|
| **实盘** `daily_regime_timing_signal.py` | `stock_data` 全市场等权日收益的 10 日均 |
| **RL** `rl_overlay_on_timing.py` | **单个 ETF 自身**日收益的 10 日均 |
| **回测** `validate_timing_multi_v2.py` | **单个 ETF 自身**日收益的 10 日均 |

### 这意味着什么

- RL 的结论（"未学到有效策略"）**只在"ETF 自身信号"上成立**
- 实盘用的是全市场等权信号，**两者不完全等价**
- 全市场等权信号更平滑（5000+ 只股票平均），ETF 自身信号更嘈杂

### 会影响结论吗

**不会推翻，但边界要说清：**

1. RL 失败的原因是"动作空间小到 3 个、样本 463 次仍然散乱"——这是**问题本身性质**，与信号来源无关
2. 但严格说，我们**没有在"全市场等权信号"上验证过 RL**
3. 如果未来要重新试 RL，应该用实盘的信号口径

### 已做的修正

`rl_overlay_on_timing.py` 的 `load_etf` 加了口径声明注释。

---

## 问题 2：回测起始硬编码满仓

### 事实

`rl_overlay_on_timing.py` 第 141 行：
```python
pos = 1.0  # 起始假设满仓
```

回测第一天假设"昨天就已满仓"。

### 影响分析

| 影响 | 说明 |
|---|---|
| 首日收益 | 若首日信号是 CASH，会多一次"卖出"成本（约 0.1%） |
| 对基线 | 同样作用（基线也是 `pos=1.0`） |
| 对 RL | 同样作用 |
| **净影响** | **方向性可抵消**，不改变结论 |

### 为什么不改

改成"起始空仓"会走向另一个极端（首日如果 HOLD 会多一次买入成本）。**两种假设都是一种选择，关键是声明出来。**

已加注释。

---

## 问题 3：结论表来源未标注 ⭐

### 这是最需要补的

本项目的每张关键数字表，都应该能追溯到具体脚本。以下是完整对照：

### v5.0 核心结论表（6 标的）的来源

**脚本**：`scripts/validate_timing_multi_v2.py`
**命令**：`.venv\Scripts\python.exe scripts\validate_timing_multi_v2.py`
**结果文件**：`results/timing_multi_v2_strict.json`
**核心函数**：`timing_engine.run_strict_t1()`

### 完整溯源表

| 文档中的数字 | 来源脚本 | 结果文件 |
|---|---|---|
| **6标的严格T+1收益/回撤/夏普** | `validate_timing_multi_v2.py` | `timing_multi_v2_strict.json` |
| 参数网格 105/120 | `validate_timing_multi_v2.py` | `timing_multi_v2_strict.json` |
| 滑点敏感性 | `validate_timing_multi_v2.py` | `timing_multi_v2_strict.json` |
| 逐年胜率 48% / 70% | `validate_timing_multi_v2.py` | `timing_multi_v2_strict.json` |
| 理想化虚高 154.7pp | `validate_timing_multi_v2.py` | `timing_multi_v2_strict.json` |
| **隔夜/日内 11/11 标的** | `analyze_overnight_intraday.py` | `overnight_intraday_analysis.json` |
| 逐年隔夜为负 19/21 | `analyze_overnight_intraday.py` | `overnight_intraday_analysis.json` |
| **美股港股对照** | `analyze_global_overnight.py` | `global_overnight_intraday.json` |
| ETF vs 指数复核 | `verify_ashare_overnight_mechanism.py` | `ashare_overnight_mechanism.json` |
| 分时段稳定性 | `verify_ashare_overnight_mechanism.py` | `ashare_overnight_mechanism.json` |
| **个股横截面 73.8%** | `analyze_stock_overnight_cross_section.py` | `stock_overnight_cross_section.json` |
| 换手率分层 5倍 | `analyze_stock_overnight_cross_section.py` | `stock_overnight_cross_section.json` |
| 高换手日内临界 0.114% | `test_high_turnover_intraday.py` | `high_turnover_intraday.json` |
| **ETF拆分事件** | `audit_etf_splits.py` | `etf_split_audit.json` |
| **成本临界 0.032%** | `test_intraday_only_strategy.py` | `intraday_only_strategy.json` |
| **RL 熵 1.098 / 种子散乱** | `verify_grokking_vs_noise.py` | `grokking_vs_noise.json` |
| RL v1-v5 各结果 | `rl_timing_v2/v3/v4/overlay.py` | 各自 json |
| 动量全市场 -70.1% | `validate_full_market_momentum.py` | `full_market_momentum_validation.json` |
| 行业动量 | `validate_sector_momentum.py` | `sector_momentum_validation.json` |
| 市值分层 | `validate_momentum_by_mktcap.py` | `momentum_by_mktcap.json` |
| 反向动量 | `validate_reversal_momentum.py` | `reversal_momentum_validation.json` |
| **换手频率 481次/34次/年** | `diagnose_timing_churn_and_cost.py` | `timing_churn_and_cost.json` |
| 执行时点对比 | `diagnose_timing_execution_price.py` | `timing_execution_price.json` |
| T+1可交易性 | `clarify_t1_intraday.py` | `t1_intraday_clarification.json` |

### 为什么 `timing_engine.py` 自检只测 2 个标的

`timing_engine.py` 的 `__main__` 是**引擎自检**，只需 1-2 个标的验证"三种口径差异"是否正确。
**6 标的的完整表在 `validate_timing_multi_v2.py`**。

两者分工：
- `timing_engine.py`：提供正确实现 + 最小自检
- `validate_timing_multi_v2.py`：全部 6 标的 × 20 参数 × 4 滑点

---

## 问题 4：年化用 244 未声明

### 事实

所有脚本统一用 `244` 作为年化交易日数。

**A 股实际约 243 个交易日/年**（2020-2026 平均 242.8）。

### 影响量化

年化收益 = `(1 + 累计收益)^(244 / 天数) - 1`

用 244 比 243，会让年化**略微偏高**：

| 真实年化 | 用244显示 |
|---|---|
| 5% | 5.02% |
| 10% | 10.04% |
| 20% | 20.08% |

**偏差约 0.4%（相对值）。**

### 结论

- **等比例缩放**，不改变方向
- **不改变排序**（所有标的用同一个常数）
- **不改变胜负判定**（跑赢/跑输的关系不变）
- 若要精确，改为 243 需重跑全部结果，但结论不变

### 已做的修正

`timing_engine.py` 顶部加了常量声明：
```python
TRADING_DAYS_PER_YEAR = 244
TRADING_DAYS_ACTUAL = 243  # A股实际值，供对照
```

---

## 问题 5：成本脚本"择时对照"是旧口径 ⭐

### 事实

`test_intraday_only_strategy.py` 原来这样算"对比择时"：
```python
r = float(d.loc[i, "intraday"]) if target > 0 else 0.0  # 只吃日内
```
**这是 v5.0 已废弃的理想化口径。**

导致该脚本打印的"对比择时 +160.1%"与
`validate_timing_multi_v2.py` 的"+300.2%"对不上。

### 修正

改为调用标准引擎：
```python
import timing_engine as _TE
_strict = _TE.run_strict_t1(_etf_df, cost=0.0005, slip=0.0, window=10, threshold=0.0)
```

### 修正后的对照

| | 修正前（旧口径） | 修正后（严格T+1） |
|---|---|---|
| 对比择时（沪深300） | +160.1% | **+300.2%** ✅ |
| 与 validate_multi_v2 一致 | ❌ 不一致 | ✅ 一致 |

**核心结论（日内交易成本太高，临界 0.032%）不受影响**，因为那部分算的是"每天开盘买收盘卖"的成本敏感性，本来就是这个口径。

---

## 附：审查中提到的"守据肌肉"

钳星说我们长出了 5 块肌肉。我把它们的具体实现在这里列出，方便以后复用：

| 肌肉 | 实现位置 |
|---|---|
| 看全体分布，不看最好种子 | `verify_grokking_vs_noise.py`（20种子 + 五项判据） |
| 逐年检验，不看累计均值 | `validate_timing_multi_v2.py`（逐年胜率表） |
| 用指数复核，排除 ETF 特有因素 | `verify_ashare_overnight_mechanism.py`（T1检验） |
| 自曝数据缺陷 | `TRADING_GUIDE_FOR_QIANXING.md`（成交额空18个月） |
| 检查口径能否在现实中实现 | `timing_engine.py`（严格T+1 vs 理想化对比） |

---

## 一句话

**这 5 项没有一项能推翻结论，它们让"边界"标得更准。**
**而"标边界"这件事，正是这套代码里最该继续做的事。**
