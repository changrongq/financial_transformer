# 钳星交易指南 — 从研究成果到可执行动作

> 版本：v2.0（2026-10-28，含全市场动量验证的重大修正）
> 适用对象：钳星（实际下单的人）
> 项目根目录：`d:\qcr\Stock\stock_analysis_system`

---

## 零、先看这一段（5 分钟读完，这段最重要）

**这份指南的 v1.0 曾说"个股动量有效（+4193%）"。v2.0 必须更正：那个结论站不住。**

写 v1.0 之后我们立刻做了一次补验证，结果推翻了动量策略：

| 版本 | 股票池 | 纯动量（不加择时）表现 |
|---|---|---|
| 研究报告 | 277 只行业股，2019-2026 | **+542.5%**（看起来有效） |
| **实盘补验证** | **5700 只全市场，2024-2026** | **-21.2%**（比买入持有 -14.0% 还差） |

**同一个策略，换个股票池就从 +542% 变成 -21%。** 这说明之前那个漂亮数字来自 **股票池选择偏差 + 幸存者偏差**，不是市场规律。

所以现在能确认的只有一件事：

### ✅ 唯一可以用的：ETF 择时

```
大盘（全市场等权）最近 10 个交易日的平均日收益
  ≥ 0  →  HOLD（持有 510300）
  < 0  →  CASH（空仓）
```

- 8 年回测：**+149.92% vs 买入持有 +22.83%**，最大回撤 **-7.4% vs -23.9%**
- **暴跌躲避率 78%**（37 个暴跌日躲过 29 个）
- 16/16 组参数全部跑赢
- 切换准确率只有 43% —— **但它错得便宜**（错是踏空小涨，对是躲过大跌）

### ❌ 不能用的：个股动量选股

即使和择时叠加，全市场 2 年最好也只有 +12.3%，远不如只做 ETF 择时。原因见第二部分。

### 研究结论压缩成三句话

1. **预测个股涨跌做不到**（方向 IC ≈ 0.07）。micro、large、台风、日历因子全部失败，重训五轮也一样。
2. **大盘什么时候该空仓能做到**（78% 暴跌躲避）。原因是恐慌是非理性的，市场不会提前定价它。
3. **"买最强股票"看着有效、实际无效**。全市场验证为负，之前的正收益是幸存者偏差。

**所以钳星只需执行一件事：ETF 择时。别做个股动量。**

---

## 第一部分：唯一可执行的策略 —— ETF 择时

### 1.1 它做什么

每天收盘后判断：**明天该持有沪深300ETF（510300），还是空仓拿现金。**

规则只有一条：

```
大盘（全市场等权）最近10个交易日的平均日收益
  ≥ 0  →  HOLD（持有 510300）
  < 0  →  CASH（卖出，空仓）
```

### 1.2 回测效果（2019-2026，8年）

| 方案 | 累计收益 | 最大回撤 |
|---|---|---|
| 一直持有沪深300ETF | **+22.83%** | -26.40% |
| 按择时信号 | **+149.92%** | **-7.35%** |

- 16/16 组参数组合全部跑赢买入持有（不是调参调出来的）
- **暴跌躲避率 78%**（37 个暴跌日中躲过 29 个）
- 但**切换准确率只有 43%**——它经常错，只是错得便宜（错的时候是踏空小涨，对的时候是躲过大跌）

> 这是择时策略的典型特征：**不追求对得多，追求错得便宜。**

### 1.3 每日怎么跑

**第一步：生成今日信号**

```powershell
cd d:\qcr\Stock\stock_analysis_system
.venv\Scripts\python.exe experiments\financial_transformer\daily_regime_timing_signal.py
```

看看输出的最后几行，会告诉你 `HOLD` 还是 `CASH`。

**第二步：按信号操作**

- 出现 `CASH` → 明天开盘卖出 510300，钱放货币基金
- 出现 `HOLD` → 保持持有，或把空仓的钱买回 510300

**第三步（可选）：看完整报告**

```powershell
.venv\Scripts\python.exe experiments\financial_transformer\signal_report.py --signal-profile all_split --skip-refresh-data --skip-predict-latest
```

报告里会有 `regime_timing` 段落，显示当前信号和历史准确率。

### 1.4 结果文件在哪

| 文件 | 内容 |
|---|---|
| `results/regime_timing_daily_signal.json` | 今日信号（HOLD/CASH）+ 大盘10日均值 |
| `results/regime_timing_signal_history.csv` | 每天的信号历史，可画图 |

打开 `regime_timing_daily_signal.json`，看这两个字段：
```json
{
  "signal": "CASH",            // 或 "HOLD"
  "regime_mean": -0.01125       // 大盘10日均收益
}
```

> **当前实况（2026-10-28）**：大盘10日均收益 **-1.125%**，当日全市场等权 **-4.898%**，信号 **CASH（空仓）**。
> 这正是策略该起作用的时候 —— 大盘在跌，我们不在场。

### 1.5 复盘：验证信号到底准不准

```powershell
.venv\Scripts\python.exe experiments\financial_transformer\review_regime_timing_signal.py
```

输出三个维度：
1. **切换准确率**（预期 43% 左右，低是正常的）
2. **暴跌躲避率**（这才是重点，预期 78%）
3. **累计收益对比**（择时 vs 一直持有）

### 1.6 仓位建议

| 阶段 | 仓位 |
|---|---|
| 头 1 个月 | 20% 资金试水，观察信号和你手动判断的差异 |
| 1-3 个月 | 如果信号逻辑你能接受（哪怕它经常错），加到 50% |
| 3 个月后 | 全仓执行 |

> **别一上来就全仓。** 不是因为策略不好，是因为你要先确认自己能不能执行——**在它连续错 3 次的时候还能按信号操作**，这才是最难的部分。

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
| 买入持有（全市场等权） | **-14.0%** | -7.4% | -23.7% | -0.58 |
| 纯动量（无择时） | **-21.2%** | -11.4% | -48.5% | -0.36 |
| 纯动量 + 涨停过滤 | **-33.8%** | -19.0% | -55.8% | -0.71 |
| 择时 + 动量 | **+12.3%** | 6.1% | -21.5% | 0.52 |
| 择时 + 动量 + 涨停过滤 | **-7.5%** | -3.9% | -25.7% | -0.22 |

**结论：全市场动量是负的。不但不赚，还比买入持有更差（-21.2% vs -14.0%）。加上涨停过滤后更差（-33.8%）—— 这是反直觉但合理的：涨停过滤剔掉的恰恰是最强的那批，剩下的动量股质量更烂。**

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
| **个股动量选股** | 277只行业股里 +542%，换到全市场5700只变成 **-21.2%**，比买入持有还差 | `validate_full_market_momentum.py` |

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
择时（ETF）
  全期:      +149.92%  vs 买入持有 +22.83%
  最大回撤:  -7.4%  vs -23.9%
  暴跌躲避:  78% (29/37)
  切换准确:  43%
  参数稳健:  16/16 组全胜

动量（个股）—— 已证伪
  [研究报告, 277只行业股 2019-2026]
    理想:          +4193.2%（夏普 3.25）
    含涨停过滤:    +1108%（砍74%）
    含0.3%成本:    +664.6%（夏普 2.33）
    独立贡献:      +465pp

  [补验证, 全市场5700只 2024-2026]  ← 这个才是真相
    买入持有:      -14.0%
    纯动量:        -21.2%   ← 比买入持有还差
    纯动量+涨停过滤: -33.8%   ← 更差
    择时+动量:     +12.3%
    择时+动量+涨停过滤: -7.5%

  结论：动量效应只存在于特定行业股池，
        推广到全市场即失效。不要用。

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
| 模型塌缩诊断 | `MICRO_MODEL_DIAGNOSIS_REPORT.md` |
| 日历因子研究 | `CALENDAR_FACTOR_REPORT.md` |
| ETF择时说明 | `ETF_TIMING_USER_GUIDE.md` |
| 台风模型回测 | `TYPHOON_BACKTEST_NOTE.md` |
| 项目总览 | `PROJECT_README.md` |
| 今日择时信号 | `results/regime_timing_daily_signal.json` |
| 动量验证结果 | `results/full_market_momentum_validation.json` |
| 动量观察清单（别下单） | `results/daily_momentum_signal.json` |

---

## 最后一句

这套东西的价值不在于"能赚多少"，而在于**我们知道哪些做不到**。

我们花了很长时间证明：预测个股涨跌做不到、日历效应不存在、台风信号没用、RL 学不会、动量选股在全市场无效。这些"负结论"和正结论一样重要——它们让你不会在错误的方向上亏钱。

**所以只做一件事：ETF 择时。** 用 1 个月确认自己能执行（在它连续错 3 次时还能按信号操作），再考虑加仓位。

个股动量不要碰。如果你真的手痒想试，用 10% 资金，并且清楚那是在赌。

有任何问题，把输出贴出来，我们一起看。
