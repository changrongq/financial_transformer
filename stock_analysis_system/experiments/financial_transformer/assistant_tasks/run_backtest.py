"""主线 vs Filter 试验线 5-6月对比回测

用法：
    cd d:\qcr\Stock\stock_analysis_system
    .venv\Scripts\python.exe experiments\financial_transformer\assistant_tasks\run_backtest.py

输出：
    - assistant_tasks/backtest_results_<日期>.json  原始回测数据
    - terminal 上打印的对比摘要
"""

import json
import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPERIMENT_ROOT = ROOT / "experiments" / "financial_transformer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from financial_transformer_experiment.config import ExperimentConfig, load_config  # noqa: E402
from financial_transformer_experiment.backtest import TransformerStrategyBacktester  # noqa: E402

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

LINES = [
    (
        "micro_mainline",
        ROOT / "experiments" / "financial_transformer" / "config_two_stage_rerank_leader_tiered_v1.yaml",
    ),
    (
        "filtered_v5",
        ROOT / "experiments" / "financial_transformer" / "config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml",
    ),
]

results = {}

for label, config_path in LINES:
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"  Config:  {config_path}")
    print(f"{'=' * 60}")

    if not Path(config_path).exists():
        results[label] = {"status": "error", "error": f"config not found: {config_path}"}
        print(f"  ERROR: config file not found")
        continue

    try:
        config = load_config(str(config_path))
        bt = TransformerStrategyBacktester(config)
        result = bt.run(
            start_date="2026-05-01",
            end_date=TODAY,
            **FAIR,
        )
        results[label] = {
            "status": "ok",
            "total_signals": int(result.total_signals),
            "total_trades": int(result.total_trades),
            "total_return": round(float(result.total_return), 6),
            "sharpe_ratio": round(float(result.sharpe_ratio), 4),
            "max_drawdown": round(float(result.max_drawdown), 4),
            "win_rate": round(float(result.win_rate), 4),
            "profit_trades": int(result.profit_trades),
        }
        print(f"  OK")
        print(f"    signals:   {result.total_signals}")
        print(f"    trades:    {result.total_trades}")
        print(f"    return:    {result.total_return:.4%}")
        print(f"    sharpe:    {result.sharpe_ratio:.2f}")
        print(f"    max_dd:    {result.max_drawdown:.4%}")
        print(f"    win_rate:  {result.win_rate:.2%}")
    except FileNotFoundError as e:
        results[label] = {"status": "error", "error": str(e)}
        print(f"  ERROR (FileNotFound): {e}")
    except RuntimeError as e:
        results[label] = {"status": "error", "error": str(e)}
        print(f"  ERROR (RuntimeError): {e}")
    except Exception as e:
        results[label] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        print(f"  ERROR: {type(e).__name__}: {e}")

# --- 输出 JSON ---
output_dir = Path(__file__).parent
output_path = output_dir / f"backtest_results_{TODAY}.json"
output_path.write_text(
    json.dumps(results, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
print(f"\n{'=' * 60}")
print(f"  Results saved to: {output_path}")
print(f"{'=' * 60}")

# --- 打印对比摘要 ---
print(f"\n{'=' * 60}")
print(f"  对比摘要 (2026-05-01 ~ {TODAY})")
print(f"{'=' * 60}")

ok_lines = {k: v for k, v in results.items() if v.get("status") == "ok"}
error_lines = {k: v for k, v in results.items() if v.get("status") == "error"}

if ok_lines:
    header = f"  {'指标':<16} " + " ".join(f"{k:<22}" for k in ok_lines)
    print(header)
    print(f"  {'-' * (16 + 23 * len(ok_lines))}")

    for metric_key, metric_label in [
        ("total_signals", "信号数"),
        ("total_trades", "交易数"),
        ("total_return", "总收益"),
        ("sharpe_ratio", "Sharpe"),
        ("max_drawdown", "最大回撤"),
        ("win_rate", "胜率"),
    ]:
        vals = []
        for v in ok_lines.values():
            raw = v.get(metric_key, "N/A")
            if metric_key in ("total_return", "max_drawdown", "win_rate"):
                vals.append(f"{float(raw):.2%}" if raw != "N/A" else "N/A")
            elif metric_key == "sharpe_ratio":
                vals.append(f"{float(raw):.2f}" if raw != "N/A" else "N/A")
            else:
                vals.append(str(raw))
        print(f"  {metric_label:<16} " + " ".join(f"{val:<22}" for val in vals))

if error_lines:
    print(f"\n  错误:")
    for label, info in error_lines.items():
        print(f"    {label}: {info.get('error', 'unknown')}")
