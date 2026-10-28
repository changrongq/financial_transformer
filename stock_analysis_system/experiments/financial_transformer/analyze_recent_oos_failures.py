from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config


DEFAULT_CONFIGS = [
    "config_moneyflow_v1_recent_oos_v1.yaml",
    "config_moneyflow_v1_concept_sentiment_manual_micro_recent_oos_v1.yaml",
    "config_moneyflow_v1_concept_sentiment_manual_micro_exposure_smoke_recent_oos_v1.yaml",
]


def _strategy_score(df: pd.DataFrame) -> pd.Series:
    return 0.45 * df["pred_short_up"] + 0.35 * df["pred_take_profit"] - 0.20 * df["pred_stop_loss"]


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _analyze_config(config_path: Path) -> dict:
    config = load_config(str(config_path))
    results_dir = config.results_dir
    predictions_path = results_dir / "test_predictions.csv"
    summary_path = results_dir / "transformer_micro_multitask_backtest_summary.json"
    signals_path = results_dir / "transformer_micro_multitask_backtest_signals.csv"
    trades_path = results_dir / "transformer_micro_multitask_backtest_trades.csv"
    metrics_path = results_dir / "metrics.json"
    universe_path = results_dir / "selected_stock_universe.csv"

    predictions = _load_csv(predictions_path, dtype={"stock_code": str})
    if predictions.empty:
        raise RuntimeError(f"missing predictions: {predictions_path}")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    for column in [
        "future_return",
        "pred_short_up",
        "pred_take_profit",
        "pred_stop_loss",
        "label_short_up",
        "label_take_profit",
        "label_stop_loss",
    ]:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    predictions = predictions.dropna(subset=["date", "future_return", "pred_short_up", "pred_take_profit", "pred_stop_loss"]).copy()
    predictions["signal_score"] = _strategy_score(predictions)
    predictions["pass"] = (
        (predictions["pred_short_up"] > 0.5)
        & (predictions["pred_take_profit"] > 0.46)
        & (predictions["pred_stop_loss"] < 0.58)
    )

    passed = predictions[predictions["pass"]].copy()
    top1 = (
        passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
        .groupby("date", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )

    signals = _load_csv(signals_path, dtype={"stock_code": str})
    trades = _load_csv(trades_path, dtype={"stock_code": str})
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    universe = _load_csv(universe_path, dtype={"stock_code": str})

    if not trades.empty:
        for column in ["profit", "profit_pct", "max_drawdown", "max_gain", "signal_score"]:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")

    top1_rows = top1[
        ["date", "stock_code", "future_return", "pred_short_up", "pred_take_profit", "pred_stop_loss", "signal_score"]
    ].copy()
    if not top1_rows.empty:
        top1_rows["date"] = top1_rows["date"].dt.strftime("%Y-%m-%d")

    trade_failure_by_stock: list[dict] = []
    if not trades.empty:
        trade_failure_by_stock = (
            trades.groupby("stock_code", as_index=False)
            .agg(
                trade_count=("stock_code", "size"),
                total_profit=("profit", "sum"),
                avg_profit_pct=("profit_pct", "mean"),
                avg_signal_score=("signal_score", "mean"),
            )
            .sort_values(["total_profit", "trade_count"], ascending=[True, False])
            .head(10)
            .to_dict(orient="records")
        )

    top1_stock_frequency = top1["stock_code"].value_counts().head(10).to_dict() if not top1.empty else {}

    return {
        "config_name": config_path.name,
        "results_dir": str(results_dir),
        "summary": summary,
        "offline_metrics": {
            "business_score": _to_float(metrics.get("business_score")),
            "auc_short_up": _to_float(metrics.get("auc_short_up")),
            "auc_take_profit": _to_float(metrics.get("auc_take_profit")),
            "auc_stop_loss": _to_float(metrics.get("auc_stop_loss")),
        },
        "dataset": {
            "test_size": int(metrics.get("test_size", 0)) if metrics else 0,
            "stock_universe_size": int(metrics.get("stock_universe_size", len(universe))) if metrics else int(len(universe)),
        },
        "prediction_quality": {
            "all_avg_future_return": _to_float(predictions["future_return"].mean()),
            "pass_rows": int(len(passed)),
            "pass_dates": int(passed["date"].nunique()),
            "pass_avg_future_return": _to_float(passed["future_return"].mean()),
            "pass_short_up_hit": _to_float(passed["label_short_up"].mean()) if not passed.empty else None,
            "pass_take_profit_hit": _to_float(passed["label_take_profit"].mean()) if not passed.empty else None,
            "pass_stop_loss_hit": _to_float(passed["label_stop_loss"].mean()) if not passed.empty else None,
            "signal_future_pearson": _to_float(predictions["signal_score"].corr(predictions["future_return"], method="pearson")),
            "signal_future_spearman": _to_float(predictions["signal_score"].corr(predictions["future_return"], method="spearman")),
        },
        "top1_failure": {
            "top1_count": int(len(top1)),
            "unique_top1_stocks": int(top1["stock_code"].nunique()) if not top1.empty else 0,
            "negative_top1_ratio": _to_float((top1["future_return"] < 0).mean()) if not top1.empty else None,
            "top1_avg_future_return": _to_float(top1["future_return"].mean()) if not top1.empty else None,
            "top1_take_profit_hit": _to_float(top1["label_take_profit"].mean()) if not top1.empty else None,
            "top1_stock_frequency": top1_stock_frequency,
            "worst_three": top1_rows.nsmallest(3, "future_return").to_dict(orient="records") if not top1_rows.empty else [],
            "best_three": top1_rows.nlargest(3, "future_return").to_dict(orient="records") if not top1_rows.empty else [],
            "full_sequence": top1_rows.to_dict(orient="records"),
        },
        "trade_failure": {
            "trade_count": int(len(trades)),
            "exit_reason_counts": trades["exit_reason"].value_counts().to_dict() if not trades.empty and "exit_reason" in trades.columns else {},
            "worst_stocks": trade_failure_by_stock,
        },
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _build_markdown(report: dict) -> str:
    lines: list[str] = ["# Recent-OOS Failure Report", ""]
    for item in report["configs"]:
        summary = item["summary"]
        quality = item["prediction_quality"]
        top1 = item["top1_failure"]
        trade_failure = item["trade_failure"]
        lines.append(f"## {item['config_name']}")
        lines.append("")
        lines.append(f"- Trading: total_return={_format_pct(summary.get('total_return'))}, sharpe={_format_float(summary.get('sharpe_ratio'))}, max_drawdown={_format_pct(summary.get('max_drawdown'))}, trades={summary.get('trade_count', 0)}")
        lines.append(f"- Offline: business_score={_format_float(item['offline_metrics'].get('business_score'))}, auc_short_up={_format_float(item['offline_metrics'].get('auc_short_up'))}, auc_take_profit={_format_float(item['offline_metrics'].get('auc_take_profit'))}")
        lines.append(f"- Prediction quality: pass_avg_future_return={_format_pct(quality.get('pass_avg_future_return'))}, signal_future_pearson={_format_float(quality.get('signal_future_pearson'))}, signal_future_spearman={_format_float(quality.get('signal_future_spearman'))}")
        lines.append(f"- Top1 failure: avg_future_return={_format_pct(top1.get('top1_avg_future_return'))}, negative_ratio={_format_pct(top1.get('negative_top1_ratio'))}, unique_top1_stocks={top1.get('unique_top1_stocks', 0)}")
        lines.append(f"- Top1 concentration: {json.dumps(top1.get('top1_stock_frequency', {}), ensure_ascii=False)}")
        lines.append(f"- Exit reasons: {json.dumps(trade_failure.get('exit_reason_counts', {}), ensure_ascii=False)}")
        if top1.get("worst_three"):
            lines.append("- Worst Top1 rows:")
            for row in top1["worst_three"]:
                lines.append(
                    f"  - {row['date']} {row['stock_code']} future_return={_format_pct(row.get('future_return'))} score={_format_float(row.get('signal_score'))} pred_su={_format_float(row.get('pred_short_up'))} pred_tp={_format_float(row.get('pred_take_profit'))} pred_sl={_format_float(row.get('pred_stop_loss'))}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze recent-OOS failures for one or more experiment configs")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-prefix", default="recent_oos_failure_report")
    args = parser.parse_args()

    config_paths = [(ROOT / config_name).resolve() for config_name in args.configs]
    report = {"configs": [_analyze_config(path) for path in config_paths]}

    output_json = ROOT / "results" / f"{args.output_prefix}.json"
    output_md = ROOT / "results" / f"{args.output_prefix}.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "config_count": len(report["configs"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()