from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

DEFAULT_WINDOWS = [
    {
        "label": "rolling_window_202506",
        "results_dir": "moneyflow_v1_rolling_window_202506",
        "train_end": "2025-06-30",
        "val_end": "2025-09-30",
        "test_end": "2025-12-31",
        "notes": "远期窗口：训练至 2025-06，外推至 2025-12",
    },
    {
        "label": "rolling_window_202509",
        "results_dir": "moneyflow_v1_rolling_window_202509",
        "train_end": "2025-09-30",
        "val_end": "2025-12-31",
        "test_end": "2026-03-04",
        "notes": "中期窗口：训练至 2025-09，外推至 2026-03",
    },
    {
        "label": "rolling_window_202512",
        "results_dir": "moneyflow_v1_rolling_window_202512",
        "train_end": "2025-12-31",
        "val_end": "2026-03-04",
        "test_end": "2026-05-25",
        "notes": "近期窗口：训练至 2025-12，外推至 2026-05",
    },
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_window_entry(spec: dict) -> dict:
    result_dir = RESULTS / spec["results_dir"]
    metrics = _load_json(result_dir / "metrics.json")
    split_summary = _load_optional_json(result_dir / "split_prediction_summary.json")
    backtest = _load_optional_json(result_dir / "transformer_micro_multitask_backtest_summary.json") or {}
    return {
        "label": spec["label"],
        "window": {
            "train_end": spec["train_end"],
            "val_end": spec["val_end"],
            "test_end": spec["test_end"],
            "backtest_start_date": backtest.get("requested_start_date"),
            "backtest_end_date": backtest.get("requested_end_date"),
        },
        "dataset": {
            "train_size": int(metrics.get("train_size", 0)),
            "val_size": int(metrics.get("val_size", 0)),
            "test_size": int(metrics.get("test_size", 0)),
            "val_rows": int(split_summary.get("val_rows", 0)) if split_summary else 0,
            "test_rows": int(split_summary.get("test_rows", 0)) if split_summary else 0,
            "val_dates": int(split_summary.get("val_dates", 0)) if split_summary else 0,
            "test_dates": int(split_summary.get("test_dates", 0)) if split_summary else 0,
        },
        "offline_metrics": {
            "business_score": float(metrics.get("business_score", 0.0)),
            "auc_short_up": float(metrics.get("auc_short_up", 0.0)),
            "auc_take_profit": float(metrics.get("auc_take_profit", 0.0)),
            "auc_stop_loss": float(metrics.get("auc_stop_loss", 0.0)),
            "composite": float(metrics.get("composite", 0.0)),
            "signal_future_spearman": float(metrics.get("signal_future_spearman", 0.0)),
            "top1_mean_future_return": float(metrics.get("top1_mean_future_return", 0.0)),
        },
        "split_test_metrics": dict(split_summary.get("test_metrics", {})) if split_summary else {
            "signal_future_spearman": float(metrics.get("signal_future_spearman", 0.0)),
            "top1_mean_future_return": float(metrics.get("top1_mean_future_return", 0.0)),
            "business_score": float(metrics.get("business_score", 0.0)),
        },
        "backtest_metrics": {
            "signal_count": int(backtest.get("signal_count", 0)),
            "trade_count": int(backtest.get("trade_count", 0)),
            "total_return": float(backtest.get("total_return", 0.0)),
            "annualized_return": float(backtest.get("annualized_return", 0.0)),
            "sharpe_ratio": float(backtest.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(backtest.get("max_drawdown", 0.0)),
            "win_rate": float(backtest.get("win_rate", 0.0)),
            "avg_hold_days": float(backtest.get("avg_hold_days", 0.0)),
        },
        "notes": spec["notes"],
        "sources": {
            "metrics": str(result_dir / "metrics.json"),
            "split_summary": str(result_dir / "split_prediction_summary.json") if split_summary else None,
            "backtest_summary": str(result_dir / "transformer_micro_multitask_backtest_summary.json") if backtest else None,
        },
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Rolling Window Training Summary",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Interpretation: {payload['interpretation']}",
        f"- Window3 minus Window1 business_score: {payload['comparisons']['business_score_recent_minus_remote']:.4f}",
        f"- Window3 minus Window1 top1_mean_future_return: {payload['comparisons']['top1_return_recent_minus_remote']:.4f}",
        f"- Window3 minus Window1 total_return: {payload['comparisons']['backtest_return_recent_minus_remote']:.4f}",
        "",
        "| Window | Train End | Test End | business_score | test_spearman | top1_mean_future_return | Total Return | Sharpe | Max DD |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in payload["entries"]:
        split_test_metrics = entry["split_test_metrics"]
        backtest = entry["backtest_metrics"]
        lines.append(
            f"| {entry['label']} | {entry['window']['train_end']} | {entry['window']['test_end']} | {entry['offline_metrics']['business_score']:.4f} | {float(split_test_metrics.get('signal_future_spearman', 0.0)):.4f} | {float(split_test_metrics.get('top1_mean_future_return', 0.0)):.4f} | {backtest['total_return']:.4f} | {backtest['sharpe_ratio']:.2f} | {backtest['max_drawdown']:.4f} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Main decision signal: compare recent-window versus remote-window business_score, signal_future_spearman, and top1_mean_future_return.",
        "- Backtest metrics are included to align the rolling-window conclusion with the existing unified trading validation lens.",
        "- This script summarizes the time-axis rolling windows only; the stock-code cross-sectional split remains a separate follow-up validation.",
        "",
        "## Sources",
        "",
    ])
    for entry in payload["entries"]:
        lines.append(f"- {entry['label']}: {json.dumps(entry['sources'], ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rolling-window moneyflow_v1 training results")
    parser.add_argument("--output-name", default="rolling_window_training_summary")
    args = parser.parse_args()

    entries = [_build_window_entry(spec) for spec in DEFAULT_WINDOWS]
    remote = entries[0]
    recent = entries[-1]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "interpretation": "rolling-window training comparison for time-drift validation",
        "entries": entries,
        "comparisons": {
            "business_score_recent_minus_remote": float(recent["offline_metrics"]["business_score"] - remote["offline_metrics"]["business_score"]),
            "top1_return_recent_minus_remote": float(recent["split_test_metrics"].get("top1_mean_future_return", 0.0) - remote["split_test_metrics"].get("top1_mean_future_return", 0.0)),
            "backtest_return_recent_minus_remote": float(recent["backtest_metrics"]["total_return"] - remote["backtest_metrics"]["total_return"]),
        },
    }

    output_json = RESULTS / f"{args.output_name}.json"
    output_md = RESULTS / f"{args.output_name}.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()