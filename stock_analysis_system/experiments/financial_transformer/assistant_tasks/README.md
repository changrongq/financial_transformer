# 助手工作区 — 主线 vs Filter 试验线对比任务

> 欢迎！这份 README 会帮你快速了解项目背景、当前任务，以及如何上手。
>
> **!!! 最重要的规则（请先读这个）!!!**
>
> 1. **禁止编造任何数字。** 报告中的每一个数字，都必须来自你亲手运行的程序输出。如果你没有运行某段代码，你就没有那个数字。
> 2. **每一步都必须真的执行。** 不要「假设」某条命令会成功，不要「推测」某个文件的内容。用 terminal 跑命令，把输出截图/粘贴下来。
> 3. **每完成一步，在本目录下留一个 checkpoint 文件。** 格式：`checkpoint_<步骤号>.txt`，里面写你跑了什么命令、输出是什么。
> 4. **如果你遇到错误或阻塞，停下来报告，不要跳过它。** 不要用一个看起来合理的假结果绕过去。

---

## 0. 当前系统状态（已帮你探查好，不用猜）

| 项目 | 状态 |
|---|---|
| 主线 checkpoint | **不存在** — `artifacts/two_stage_rerank_best_v1/` 目录为空 |
| Filter checkpoint | **存在** — `results/formal_filtered_v5_e80_bsstop_seed33/artifacts/financial_transformer.pt` |
| 主线 test_predictions.csv | 存在，但数据只到 **2026-03-04**，没有 5-6 月数据 |
| Filter test_predictions.csv | 存在，但数据也只到 **2026-03-04** |
| Filter latest_predictions.csv | 存在，日期是 **2026-06-04**（今天的！） |
| OHLCV 数据库 | `stock_history.db` 存在 |
| 市值数据库 | `market_cap_full.db` 存在 |

> **关键结论：主线模型 checkpoint 丢了，无法直接对主线跑 5-6 月预测。你需要先解决这个问题。**

---

## 1. 背景知识

### 1.1 项目是什么？

这是一个 **A 股短线选股系统**，核心用 Transformer 多任务模型预测股票未来 5 天的涨跌信号，并通过回测验证策略。

### 1.2 当前有两条活跃的信号线

| 信号线 | 配置 | 股票池 | 说明 |
|---|---|---|---|
| **主线 (micro)** | `config_two_stage_rerank_leader_tiered_v1.yaml` | 全市场 micro 层 | 默认生产线，G1-G5 全部通过 |
| **Filter 试验线 (filtered_v5)** | `config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml` | 固定 54 只 | 试验线，严格阈值 |

### 1.3 统一执行参数（两条线必须用相同的！）

```
short_up_threshold = 0.254
take_profit_threshold = 0.572
stop_loss_threshold = 0.50
top_k_per_day = 2
hold_days = 35
take_profit = 0.14   (即 +14%)
stop_loss = -0.06    (即 -6%)
```

---

## 2. 你的任务

用 **2026-05-01 到 2026-06-04** 的数据窗口，对比主线和 Filter 试验线的回测表现。

---

## 3. 执行步骤（每一步都必须真的跑！）

### 步骤 0：创建 checkpoint

在本目录下创建 `checkpoint_0_env.txt`，写入：

```
环境验证时间: <当前时间>
Python路径: <粘贴 python --version 输出>
工作目录: <粘贴 pwd 输出>
```

### 步骤 1：检查并确认主线 checkpoint 问题

主线 checkpoint 不存在。你有两个选择：

**方案 A（推荐）：** 用 `signal_report.py` 已有机制生成主线 latest_predictions，然后用它来回测 5-6 月。实际上主线 daily 已经在跑了——`results/two_stage_rerank_best_v1/latest_predictions.csv` 如果有 5 月的数据就可以用。

**方案 B：** 重新训练主线模型。这需要跑 `train.py --config config_two_stage_rerank_leader_tiered_v1.yaml`，可能需要几小时。

**请先运行以下诊断命令，然后把输出贴到 `checkpoint_1_diagnosis.txt`：**

```powershell
cd d:\qcr\Stock\stock_analysis_system

# 1. 主线 latest_predictions 的日期范围
d:\qcr\Stock\stock_analysis_system\.venv\Scripts\python.exe -c "
import pandas as pd; from pathlib import Path
p = Path('experiments/financial_transformer/results/two_stage_rerank_best_v1/latest_predictions.csv')
if p.exists():
    df = pd.read_csv(p, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print('latest_predictions date range:', df['date'].min(), '~', df['date'].max())
    print('rows:', len(df))
else:
    print('FILE NOT FOUND')
"

# 2. 主线 test_predictions 的日期范围
d:\qcr\Stock\stock_analysis_system\.venv\Scripts\python.exe -c "
import pandas as pd; from pathlib import Path
p = Path('experiments/financial_transformer/results/two_stage_rerank_best_v1/test_predictions.csv')
df = pd.read_csv(p, dtype={'stock_code': str})
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print('test_predictions date range:', df['date'].min(), '~', df['date'].max())
print('rows:', len(df))
"

# 3. Filter test_predictions 的日期范围
d:\qcr\Stock\stock_analysis_system\.venv\Scripts\python.exe -c "
import pandas as pd; from pathlib import Path
p = Path('experiments/financial_transformer/results/formal_filtered_v5_e80_bsstop_seed33/test_predictions.csv')
df = pd.read_csv(p, dtype={'stock_code': str})
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print('filter test_predictions date range:', df['date'].min(), '~', df['date'].max())
print('rows:', len(df))
"

# 4. Filter latest_predictions 的日期范围
d:\qcr\Stock\stock_analysis_system\.venv\Scripts\python.exe -c "
import pandas as pd; from pathlib import Path
p = Path('experiments/financial_transformer/results/formal_filtered_v5_e80_bsstop_seed33/latest_predictions.csv')
df = pd.read_csv(p, dtype={'stock_code': str})
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print('filter latest date:', df['date'].max())
print('rows:', len(df))
"
```

**把上面 4 条命令的完整输出贴到 `checkpoint_1_diagnosis.txt`，然后根据结果决定下一步。**

### 步骤 2：获取/生成 Filter 试验线的 5-6 月预测

Filter 的 checkpoint 存在，所以可以跑 predict。但 `predict.py` 只有 `predict_latest()` 函数（只预测最新一天），没有 `predict_historical()`。

这意味着要对 Filter 做 5-6 月全量回测，你有两个选择：

**选择 2A：用 test_predictions.csv 回测 1-3 月（作为对照基准），然后用 latest_predictions 逐日推算。**

**选择 2B：写一个循环脚本，用 predict_latest 的逻辑对 5-6 月每个交易日生成预测。** 伪代码：

```python
# 对每个交易日，修改数据库中该日之后的特征不可见，然后调用 predict_latest
# 这比较麻烦，需要操作 build_feature_frame 的日期截断
```

**选择 2C（最简单）：直接用已有的 daily latest_predictions。**

Filter 线的 `latest_predictions.csv` 已经有 6 月 4 日的数据（54 行）。主线如果在跑 daily signal_report，也应该有。检查一下主线 latest 的日期。

**考虑到复杂度，建议先用 Filter 线作为主力验证对象，主线如果 checkpoint 丢了就先对比"有数据的最大窗口"。**

创建 `checkpoint_2_plan.txt`，写清楚你选择的方案和理由。

### 步骤 3：执行回测

**这一步你只能通过在 terminal 里跑 Python 脚本来完成！不能凭空写数字！**

创建一个脚本 `assistant_tasks/run_backtest.py`，内容如下（复制粘贴，不要自己改参数）：

```python
"""主线 vs Filter 试验线 5-6月对比回测"""
import sys, json
from pathlib import Path
from datetime import date

ROOT = Path("d:/qcr/Stock/stock_analysis_system")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/financial_transformer/src"))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.backtest import TransformerStrategyBacktester

TODAY = "2026-06-04"
FAIR = {
    "short_up_threshold": 0.254,
    "take_profit_threshold": 0.572,
    "stop_loss_threshold": 0.50,
    "top_k_per_day": 2,
    "hold_days": 35,
    "take_profit": 0.14,
    "stop_loss": -0.06,
}

results = {}

for label, config_path in [
    ("micro_mainline", "experiments/financial_transformer/config_two_stage_rerank_leader_tiered_v1.yaml"),
    ("filtered_v5", "experiments/financial_transformer/config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml"),
]:
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    try:
        config = load_config(config_path)
        bt = TransformerStrategyBacktester(config)
        result = bt.run(
            start_date="2026-05-01",
            end_date=TODAY,
            **FAIR,
        )
        results[label] = {
            "status": "ok",
            "total_signals": result.total_signals,
            "total_trades": result.total_trades,
            "total_return": round(result.total_return, 6),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "win_rate": round(result.win_rate, 4),
            "profit_trades": result.profit_trades,
        }
        print(f"OK: signals={result.total_signals}, trades={result.total_trades}, "
              f"return={result.total_return:.4%}, sharpe={result.sharpe_ratio:.2f}, "
              f"max_dd={result.max_drawdown:.4%}, win_rate={result.win_rate:.2%}")
    except Exception as e:
        results[label] = {"status": "error", "error": str(e)}
        print(f"ERROR: {e}")

# 输出 JSON
output_path = Path(__file__).parent / f"backtest_results_{TODAY}.json"
output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"\nResults saved to: {output_path}")
```

然后在 terminal 中运行：

```powershell
cd d:\qcr\Stock\stock_analysis_system
d:\qcr\Stock\stock_analysis_system\.venv\Scripts\python.exe experiments\financial_transformer\assistant_tasks\run_backtest.py
```

**把完整的 terminal 输出保存到 `checkpoint_3_backtest_output.txt`。**

### 步骤 4：生成对比报告

**报告里的每一个数字，都必须来自 `backtest_results_2026-06-04.json`！** 不能自己编。

创建 `comparison_report_2026-06-04.md`，格式：

```markdown
# 主线 vs Filter 试验线对比报告

- 数据窗口：2026-05-01 ~ 2026-06-04
- 报告生成时间：（你运行的时间）
- 统一执行参数：short_up>0.254, take_profit>0.572, stop_loss<0.50, top_k=2, hold=35d

## 对比表格

| 指标 | 主线 (micro) | Filter (filtered_v5) |
|---|---|---|
| 信号数 | （来自 JSON） | （来自 JSON） |
| 交易数 | （来自 JSON） | （来自 JSON） |
| 总收益 | （来自 JSON） | （来自 JSON） |
| Sharpe | （来自 JSON） | （来自 JSON） |
| 最大回撤 | （来自 JSON） | （来自 JSON） |
| 胜率 | （来自 JSON） | （来自 JSON） |

## 分析

（根据实际数字来写，不要套模板）
```

### 步骤 5：最终检查

在提交前，确认以下文件都存在：

- [ ] `checkpoint_0_env.txt`
- [ ] `checkpoint_1_diagnosis.txt`
- [ ] `checkpoint_2_plan.txt`
- [ ] `run_backtest.py`
- [ ] `checkpoint_3_backtest_output.txt`
- [ ] `backtest_results_2026-06-04.json`
- [ ] `comparison_report_2026-06-04.md`

---

## 附录 A：关键路径速查

| 用途 | 路径 |
|---|---|
| 项目根 | `d:\qcr\Stock\stock_analysis_system` |
| 主线配置 | `experiments\financial_transformer\config_two_stage_rerank_leader_tiered_v1.yaml` |
| Filter 配置 | `experiments\financial_transformer\config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml` |
| 主线结果 | `experiments\financial_transformer\results\two_stage_rerank_best_v1\` |
| Filter 结果 | `experiments\financial_transformer\results\formal_filtered_v5_e80_bsstop_seed33\` |
| 回测模块 | `experiments\financial_transformer\src\financial_transformer_experiment\backtest.py` |
| 数据库 | `stock_history.db` / `market_cap_full.db` |

## 附录 B：3 月份基准数据（不要编，这里是准确的）

来自 `experiment_report_2026-04-15.md`：
- 验证窗口：`2026-01-01 ~ 2026-03-04`
- 主线 `moneyflow_v1`：总收益 `+15.75%`，Sharpe `5.52`
- 第一版 trade-quality filter 被证伪

---

如果你卡住了，把 checkpoint 文件给 Coder 看，他会帮你。

