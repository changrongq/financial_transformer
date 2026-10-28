"""主线 vs Filter 试验线 完整回测对比"""
import sys, json
from pathlib import Path

ROOT = Path("d:/qcr/Stock/stock_analysis_system")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/financial_transformer/src"))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.backtest import TransformerStrategyBacktester

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

for label, config_rel in [
    ("micro_mainline", "experiments/financial_transformer/config_two_stage_rerank_leader_tiered_v1.yaml"),
    ("filtered_v5", "experiments/financial_transformer/config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml"),
]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        config = load_config(str(ROOT / config_rel))
        bt = TransformerStrategyBacktester(config)
        result = bt.run(**FAIR)
        s = result.summary
        results[label] = {
            "status": "ok",
            "signal_count": int(s.get("signal_count", 0)),
            "trade_count": int(s.get("trade_count", 0)),
            "total_return": round(float(s.get("total_return", 0)), 6),
            "annualized_return": round(float(s.get("annualized_return", 0)), 4),
            "sharpe_ratio": round(float(s.get("sharpe_ratio", 0)), 4),
            "max_drawdown": round(float(s.get("max_drawdown", 0)), 4),
            "win_rate": round(float(s.get("win_rate", 0)), 4),
            "profit_factor": round(float(s.get("profit_factor", 0)), 4),
            "avg_hold_days": round(float(s.get("avg_hold_days", 0)), 1),
        }
        print(f"  signals={s['signal_count']}, trades={s['trade_count']}")
        print(f"  return={s['total_return']:.4%}, sharpe={s['sharpe_ratio']:.2f}")
        print(f"  max_dd={s['max_drawdown']:.4%}, win_rate={s['win_rate']:.2%}")
        print(f"  profit_factor={s['profit_factor']:.2f}, avg_hold={s['avg_hold_days']:.1f}d")
    except Exception as e:
        results[label] = {"status": "error", "error": str(e)}
        print(f"  ERROR: {e}")

# Comparison table
print(f"\n\n{'='*60}")
print(f"  COMPARISON TABLE")
print(f"{'='*60}")
ok = {k: v for k, v in results.items() if v["status"] == "ok"}
if ok:
    print(f"  {'Metric':<22} {'micro_mainline':<18} {'filtered_v5':<18} {'delta':<14}")
    print(f"  {'-'*22} {'-'*18} {'-'*18} {'-'*14}")
    metrics = [
        ("signal_count", "d"),
        ("trade_count", "d"),
        ("total_return", ".4%"),
        ("annualized_return", ".2f"),
        ("sharpe_ratio", ".2f"),
        ("max_drawdown", ".4%"),
        ("win_rate", ".2%"),
        ("profit_factor", ".2f"),
        ("avg_hold_days", ".1f"),
    ]
    for key, fmt in metrics:
        m = float(ok["micro_mainline"][key])
        f = float(ok["filtered_v5"][key])
        if fmt == "d":
            print(f"  {key:<22} {int(m):<18} {int(f):<18} {int(f)-int(m):<+14}")
        elif fmt.endswith("%"):
            print(f"  {key:<22} {m:<18{fmt}} {f:<18{fmt}} {f-m:<+14{fmt}}")
        else:
            print(f"  {key:<22} {m:<18{fmt}} {f:<18{fmt}} {f-m:<+14{fmt}}")

for label, info in results.items():
    if info["status"] == "error":
        print(f"\n  {label}: ERROR - {info['error']}")

# Save JSON
out = Path(__file__).parent / "comparison_results.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved to: {out}")
