from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_window(*, label: str, trade_summary_path: Path, split_summary_path: Path) -> dict:
    trade_summary = _load_json(trade_summary_path)
    split_summary = _load_json(split_summary_path)
    test_metrics = split_summary.get("test_metrics", {})
    return {
        "label": label,
        "window": {
            "start_date": trade_summary.get("requested_start_date"),
            "end_date": trade_summary.get("requested_end_date"),
            "test_dates": int(split_summary.get("test_dates", 0)),
            "test_rows": int(split_summary.get("test_rows", 0)),
        },
        "trading": {
            "signal_count": int(trade_summary.get("signal_count", 0)),
            "trade_count": int(trade_summary.get("trade_count", 0)),
            "total_return": float(trade_summary.get("total_return", 0.0)),
            "sharpe_ratio": float(trade_summary.get("sharpe_ratio", 0.0)),
            "win_rate": float(trade_summary.get("win_rate", 0.0)),
            "max_drawdown": float(trade_summary.get("max_drawdown", 0.0)),
        },
        "prediction": {
            "business_score": float(test_metrics.get("business_score", 0.0)),
            "signal_future_spearman": float(test_metrics.get("signal_future_spearman", 0.0)),
            "pass_signal_future_spearman": float(test_metrics.get("pass_signal_future_spearman", 0.0)),
            "pass_mean_future_return": float(test_metrics.get("pass_mean_future_return", 0.0)),
            "top1_mean_future_return": float(test_metrics.get("top1_mean_future_return", 0.0)),
        },
        "sources": {
            "trade_summary": str(trade_summary_path),
            "split_summary": str(split_summary_path),
        },
    }


def _load_rerank_probe(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    rerank_summary = payload.get("rerank_summary", {})
    if not rerank_summary:
        return None

    if "best_rules" in payload:
        return {
            "available": True,
            "label": "latest_incremental_v3_rule_rerank_grid",
            "window": {
                "start_date": rerank_summary.get("requested_start_date"),
                "end_date": rerank_summary.get("requested_end_date"),
            },
            "metrics": {
                "signal_count": int(rerank_summary.get("signal_count", 0)),
                "trade_count": int(rerank_summary.get("trade_count", 0)),
                "total_return": float(rerank_summary.get("total_return", 0.0)),
                "sharpe_ratio": float(rerank_summary.get("sharpe_ratio", 0.0)),
                "win_rate": float(rerank_summary.get("win_rate", 0.0)),
                "max_drawdown": float(rerank_summary.get("max_drawdown", 0.0)),
            },
            "model": {
                "rules": payload.get("best_rules", {}),
                "grid_search": bool(payload.get("grid_search", False)),
            },
            "source": str(path),
        }

    return {
        "available": True,
        "label": "latest_incremental_v3_core4_rerank",
        "window": {
            "start_date": rerank_summary.get("requested_start_date"),
            "end_date": rerank_summary.get("requested_end_date"),
        },
        "metrics": {
            "signal_count": int(rerank_summary.get("signal_count", 0)),
            "trade_count": int(rerank_summary.get("trade_count", 0)),
            "total_return": float(rerank_summary.get("total_return", 0.0)),
            "sharpe_ratio": float(rerank_summary.get("sharpe_ratio", 0.0)),
            "win_rate": float(rerank_summary.get("win_rate", 0.0)),
            "max_drawdown": float(rerank_summary.get("max_drawdown", 0.0)),
        },
        "model": {
            "features": list((payload.get("model") or {}).get("features", [])),
            "blend_weight": float((payload.get("model") or {}).get("blend_weight", 1.0)),
        },
        "source": str(path),
    }


def _resolve_status(entries: list[dict]) -> tuple[str, str, list[str]]:
    if not entries:
        return "unknown", "no incremental windows found", []

    latest = entries[-1]
    trailing = entries[-2:] if len(entries) >= 2 else entries
    trailing_avg_return = sum(item["trading"]["total_return"] for item in trailing) / len(trailing)
    non_positive_trailing = all(item["trading"]["total_return"] <= 0 for item in trailing)
    latest_return = latest["trading"]["total_return"]
    latest_sharpe = latest["trading"]["sharpe_ratio"]
    latest_pass_mean = latest["prediction"]["pass_mean_future_return"]
    latest_top1 = latest["prediction"]["top1_mean_future_return"]

    reasons = [
        f"latest total_return={latest_return:.4f}",
        f"latest sharpe={latest_sharpe:.2f}",
        f"latest pass_mean_future_return={latest_pass_mean:.4f}",
        f"latest top1_mean_future_return={latest_top1:.4f}",
        f"trailing2 avg_return={trailing_avg_return:.4f}",
    ]

    if non_positive_trailing and trailing_avg_return < -0.005 and latest_return < -0.01:
        return "pause", "recent incremental prediction health is weak; pause new default trading and require manual review", reasons
    if latest_return <= 0 or trailing_avg_return < 0:
        return "caution", "recent incremental prediction health is deteriorating; reduce trust and review before normal trading", reasons
    return "trade", "recent incremental prediction health is acceptable; normal trading can continue", reasons


def _resolve_execution_plan(status: str) -> dict:
    if status == "pause":
        return {
            "mode": "no_new_trades",
            "capital_fraction_multiplier": 0.0,
            "description": "pause new default trades; observation only",
        }
    if status == "caution":
        return {
            "mode": "reduced_risk",
            "capital_fraction_multiplier": 0.5,
            "description": "allow only reduced-size trial trading",
        }
    return {
        "mode": "normal",
        "capital_fraction_multiplier": 1.0,
        "description": "normal trading can continue",
    }


def _render_markdown(payload: dict) -> str:
    status = payload["health_status"]
    lines = [
        "# Mainline Prediction Health",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Current status: {status['status']}",
        f"- Recommendation: {status['recommendation']}",
        f"- Execution mode: {payload['execution_plan']['mode']} / capital x{payload['execution_plan']['capital_fraction_multiplier']:.2f}",
        "",
        "## Trend",
        "",
        "| Window | Dates | Test Dates | Trading Return | Sharpe | Signals | Trades | Pass Mean Return | Top1 Mean Return |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in payload["entries"]:
        lines.append(
            f"| {entry['label']} | {entry['window']['start_date']} ~ {entry['window']['end_date']} | {entry['window']['test_dates']} | {entry['trading']['total_return']:.4f} | {entry['trading']['sharpe_ratio']:.2f} | {entry['trading']['signal_count']} | {entry['trading']['trade_count']} | {entry['prediction']['pass_mean_future_return']:.4f} | {entry['prediction']['top1_mean_future_return']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    for reason in status.get("reasons", []):
        lines.append(f"- {reason}")
    rerank_probe = payload.get("repair_probe") or {}
    if rerank_probe.get("available"):
        metrics = rerank_probe.get("metrics", {})
        lines.extend([
            "",
            "## Repair Probe",
            "",
            f"- Latest narrow rerank repair: total_return={metrics.get('total_return', 0.0):.4f}, sharpe={metrics.get('sharpe_ratio', 0.0):.2f}, signals={int(metrics.get('signal_count', 0))}, trades={int(metrics.get('trade_count', 0))}",
            f"- Current interpretation: the latest weakness looks partly repairable by ranking/filter correction, but the default line itself is not in a clean strong-health state.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recent mainline prediction-health trend from latest incremental windows")
    parser.add_argument("--output-name", default="mainline_prediction_health")
    args = parser.parse_args()

    entries = [
        _load_window(
            label="latest_incremental_v1",
            trade_summary_path=RESULTS / "two_stage_rerank_leader_tiered_latest_incremental_v1" / "transformer_micro_multitask_backtest_summary.json",
            split_summary_path=RESULTS / "moneyflow_v1_latest_incremental_v1" / "split_prediction_summary.json",
        ),
        _load_window(
            label="latest_incremental_v2",
            trade_summary_path=RESULTS / "two_stage_rerank_leader_tiered_latest_incremental_v2" / "transformer_micro_multitask_backtest_latest_incremental_v2_summary.json",
            split_summary_path=RESULTS / "moneyflow_v1_latest_incremental_v2" / "split_prediction_summary.json",
        ),
        _load_window(
            label="latest_incremental_v3",
            trade_summary_path=RESULTS / "two_stage_rerank_leader_tiered_latest_incremental_v3" / "transformer_micro_multitask_backtest_latest_incremental_v3_summary.json",
            split_summary_path=RESULTS / "moneyflow_v1_latest_incremental_v3" / "split_prediction_summary.json",
        ),
    ]
    status, recommendation, reasons = _resolve_status(entries)
    rerank_probe = _load_rerank_probe(
        RESULTS / "two_stage_rerank_leader_tiered_latest_incremental_v3_rule_rerank_probe_latest_incremental_v3_grid" / "rule_rerank_probe.json"
    ) or _load_rerank_probe(
        RESULTS / "two_stage_rerank_leader_tiered_latest_incremental_v3_two_stage_oldplusv2_to_latest_incremental_v3_core4probe" / "lightweight_rerank_probe.json"
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": "micro",
        "health_status": {
            "status": status,
            "recommendation": recommendation,
            "reasons": reasons,
        },
        "execution_plan": _resolve_execution_plan(status),
        "entries": entries,
        "repair_probe": rerank_probe or {"available": False},
    }

    output_json = RESULTS / f"{args.output_name}.json"
    output_md = RESULTS / f"{args.output_name}.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "status": status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()