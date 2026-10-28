"""用Filter完整预测序列做回测，分三段窗口对比主线"""
import sys, json
from pathlib import Path

import pandas as pd

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

# ---- Filter 完整区间回测（用 backfill_predictions.csv） ----
print("=" * 60)
print("  Filter: full backfill (2026-01-05 ~ 2026-06-04)")
print("=" * 60)
try:
    config = load_config(str(ROOT / "experiments/financial_transformer/config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml"))

    # 临时修改配置，让回测器读 backfill_predictions.csv 而非 test_predictions.csv
    import financial_transformer_experiment.backtest as bt_module

    original_load = bt_module.TransformerStrategyBacktester._load_predictions

    def patched_load(self, start_date, end_date, short_up_threshold, take_profit_threshold,
                     stop_loss_threshold, top_k_per_day=None, use_calibration=False,
                     min_secondary_primary_ratio=0.0, min_score_ratio_to_leader=0.0,
                     min_filter_column=None, min_filter_value=None):
        pred_path = self.config.results_dir / "backfill_predictions.csv"
        df = pd.read_csv(pred_path, dtype={"stock_code": str})
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"])
        required_cols = ["pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown"]
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", *required_cols]).copy()
        if start_date is not None:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        if df.empty:
            raise RuntimeError("指定区间内没有可用于回测的 Transformer 多任务预测。")
        if "layer" not in df.columns:
            df["layer"] = self.config.experiment.layer_filter or "unknown"
        from financial_transformer_experiment.scoring import strategy_score_frame
        df["signal_score"] = strategy_score_frame(df, exit_label_mode=self.config.multitask.exit_label_mode)
        df["strategy_a_pass"] = (
            (df["pred_short_up"] > short_up_threshold)
            & (df["pred_take_profit"] > take_profit_threshold)
            & (df["pred_stop_loss"] < stop_loss_threshold)
        )
        df = df[df["strategy_a_pass"]].copy()
        if min_score_ratio_to_leader > 0.0 and not df.empty:
            leader_score = df.groupby("date")["signal_score"].transform("max")
            df["score_ratio_to_leader"] = df["signal_score"].div(leader_score.where(leader_score.abs() > 1e-9))
            df = df[df["score_ratio_to_leader"] >= min_score_ratio_to_leader]
        if top_k_per_day is not None and top_k_per_day > 0 and not df.empty:
            df = (df.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
                   .groupby("date", group_keys=False)
                   .head(top_k_per_day)
                   .reset_index(drop=True))
        return df.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)

    bt_module.TransformerStrategyBacktester._load_predictions = patched_load

    bt = TransformerStrategyBacktester(config)

    # 3 个窗口分别回测
    windows = [
        ("full", None, None),
        ("old (Jan-Mar)", "2026-01-01", "2026-03-04"),
        ("recent (Mar-Jun)", "2026-03-05", "2026-06-04"),
    ]

    for label, start, end in windows:
        try:
            result = bt.run(start_date=start, end_date=end, **FAIR)
            s = result.summary
            results[f"filter_{label}"] = {
                "status": "ok",
                "signal_count": int(s.get("signal_count", 0)),
                "trade_count": int(s.get("trade_count", 0)),
                "total_return": round(float(s.get("total_return", 0)), 6),
                "sharpe_ratio": round(float(s.get("sharpe_ratio", 0)), 4),
                "max_drawdown": round(float(s.get("max_drawdown", 0)), 4),
                "win_rate": round(float(s.get("win_rate", 0)), 4),
                "profit_factor": round(float(s.get("profit_factor", 0)), 4),
            }
            print(f"  {label}: signals={s['signal_count']}, trades={s['trade_count']}, "
                  f"return={s['total_return']:.4%}, sharpe={s['sharpe_ratio']:.2f}, "
                  f"max_dd={s['max_drawdown']:.4%}, win_rate={s['win_rate']:.2%}")
        except Exception as e:
            results[f"filter_{label}"] = {"status": "error", "error": str(e)}
            print(f"  {label}: ERROR - {e}")

    bt_module.TransformerStrategyBacktester._load_predictions = original_load

except Exception as e:
    print(f"  FATAL: {e}")
    results["filter_fatal"] = {"status": "error", "error": str(e)}

# ---- 主线 1-3月回测（复用之前结果） ----
print("\n" + "=" * 60)
print("  micro_mainline old (Jan-Mar)")
print("=" * 60)
try:
    config = load_config(str(ROOT / "experiments/financial_transformer/config_two_stage_rerank_leader_tiered_v1.yaml"))
    bt = TransformerStrategyBacktester(config)
    result = bt.run(**FAIR)
    s = result.summary
    results["micro_old"] = {
        "status": "ok",
        "signal_count": int(s.get("signal_count", 0)),
        "trade_count": int(s.get("trade_count", 0)),
        "total_return": round(float(s.get("total_return", 0)), 6),
        "sharpe_ratio": round(float(s.get("sharpe_ratio", 0)), 4),
        "max_drawdown": round(float(s.get("max_drawdown", 0)), 4),
        "win_rate": round(float(s.get("win_rate", 0)), 4),
        "profit_factor": round(float(s.get("profit_factor", 0)), 4),
    }
    print(f"  signals={s['signal_count']}, trades={s['trade_count']}, "
          f"return={s['total_return']:.4%}, sharpe={s['sharpe_ratio']:.2f}, "
          f"max_dd={s['max_drawdown']:.4%}, win_rate={s['win_rate']:.2%}")
except Exception as e:
    results["micro_old"] = {"status": "error", "error": str(e)}
    print(f"  ERROR: {e}")

# ---- 对比表格 ----
print("\n\n" + "=" * 70)
print("  FINAL COMPARISON")
print("=" * 70)

rows_spec = [
    ("micro_old", "主线 (Jan-Mar)"),
    ("filter_old (Jan-Mar)", "Filter (Jan-Mar)"),
    ("filter_recent (Mar-Jun)", "Filter (Mar-Jun)"),
    ("filter_full", "Filter (Full: Jan-Jun)"),
]

ok = {}
for key, label in rows_spec:
    if key in results and results[key]["status"] == "ok":
        ok[label] = results[key]
    elif f"filter_{key}" in results:
        pass

# re-collect
ok = {}
for key, display in rows_spec:
    if key in results and results[key]["status"] == "ok":
        ok[display] = results[key]

if ok:
    headers = ["Metric"] + [f"{k:<26}" for k in ok]
    print("  " + "  ".join(headers))
    print("  " + "-" * (10 + 28 * len(ok)))

    for metric_key, metric_label, fmt in [
        ("signal_count", "信号数", "d"),
        ("trade_count", "交易数", "d"),
        ("total_return", "总收益", ".4%"),
        ("sharpe_ratio", "Sharpe", ".2f"),
        ("max_drawdown", "最大回撤", ".4%"),
        ("win_rate", "胜率", ".2%"),
        ("profit_factor", "Profit Factor", ".2f"),
    ]:
        vals = []
        for v in ok.values():
            val = v.get(metric_key, 0)
            if fmt == "d":
                vals.append(f"{int(val):<26}")
            elif fmt.endswith("%"):
                vals.append(f"{float(val):<26{fmt}}")
            else:
                vals.append(f"{float(val):<26{fmt}}")
        print(f"  {metric_label:<10}  " + "  ".join(vals))

for key, display in [("micro_old", "主线"), ("filter_recent (Mar-Jun)", "Filter Mar-Jun")]:
    if key not in results or results[key]["status"] != "ok":
        print(f"\n  {display}: 无可用结果 (可能主线checkpoint丢失)")

# Save
out = Path(__file__).parent / "comparison_full_results.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved to: {out}")
