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


def _build_entry(
    *,
    label: str,
    profile_type: str,
    summary_path: Path,
    split_summary_path: Path | None = None,
    notes: str | None = None,
) -> dict:
    summary = _load_json(summary_path)
    split_summary = _load_json(split_summary_path) if split_summary_path is not None else None
    return {
        "label": label,
        "profile_type": profile_type,
        "window": {
            "start_date": summary.get("requested_start_date"),
            "end_date": summary.get("requested_end_date"),
            "test_dates": int(split_summary.get("test_dates", 0)) if split_summary else None,
            "test_rows": int(split_summary.get("test_rows", 0)) if split_summary else None,
        },
        "metrics": {
            "signal_count": int(summary.get("signal_count", 0)),
            "trade_count": int(summary.get("trade_count", 0)),
            "total_return": float(summary.get("total_return", 0.0)),
            "annualized_return": float(summary.get("annualized_return", 0.0)),
            "sharpe_ratio": float(summary.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(summary.get("max_drawdown", 0.0)),
            "win_rate": float(summary.get("win_rate", 0.0)),
            "avg_hold_days": float(summary.get("avg_hold_days", 0.0)),
        },
        "notes": notes,
        "sources": {
            "summary": str(summary_path),
            "split_summary": str(split_summary_path) if split_summary_path else None,
        },
    }


def _build_rerank_entry(*, label: str, probe_path: Path, notes: str | None = None) -> dict:
    probe = _load_json(probe_path)
    base_summary = probe["base_summary"]
    rerank_summary = probe["rerank_summary"]
    return {
        "label": label,
        "profile_type": "large-rerank-candidate",
        "window": {
            "start_date": rerank_summary.get("requested_start_date"),
            "end_date": rerank_summary.get("requested_end_date"),
            "test_dates": None,
            "test_rows": None,
        },
        "metrics": {
            "signal_count": int(rerank_summary.get("signal_count", 0)),
            "trade_count": int(rerank_summary.get("trade_count", 0)),
            "total_return": float(rerank_summary.get("total_return", 0.0)),
            "annualized_return": float(rerank_summary.get("annualized_return", 0.0)),
            "sharpe_ratio": float(rerank_summary.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(rerank_summary.get("max_drawdown", 0.0)),
            "win_rate": float(rerank_summary.get("win_rate", 0.0)),
            "avg_hold_days": float(rerank_summary.get("avg_hold_days", 0.0)),
        },
        "base_reference": {
            "signal_count": int(base_summary.get("signal_count", 0)),
            "trade_count": int(base_summary.get("trade_count", 0)),
            "total_return": float(base_summary.get("total_return", 0.0)),
            "sharpe_ratio": float(base_summary.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(base_summary.get("max_drawdown", 0.0)),
        },
        "rerank_model": {
            "blend_weight": float(probe["model"].get("blend_weight", 1.0)),
            "features": list(probe["model"].get("features", [])),
            "validation_dates": int(probe["model"].get("validation_dates", 0)),
        },
        "notes": notes,
        "sources": {
            "probe": str(probe_path),
            "summary": str(Path(probe["detail_outputs"]["summary"])),
        },
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Latest Incremental Validation Summary",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Window: {payload['window']['start_date']} ~ {payload['window']['end_date']}",
        f"- Interpretation: {payload['interpretation']}",
        "",
        "| Profile | Type | Signals | Trades | Total Return | Sharpe | Max Drawdown | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for entry in payload["entries"]:
        metrics = entry["metrics"]
        lines.append(
            f"| {entry['label']} | {entry['profile_type']} | {metrics['signal_count']} | {metrics['trade_count']} | {metrics['total_return']:.4f} | {metrics['sharpe_ratio']:.2f} | {metrics['max_drawdown']:.4f} | {entry.get('notes') or ''} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This is a tiny latest incremental validation window and should be read as a sanity check, not as a replacement for the longer recent-OOS slice.",
        "- The mainline and both Large single-model branches are positive on this latest slice.",
        "- The current Large rerank candidate remains a targeted repair for the earlier weak recent-OOS zone rather than a universal upgrade, because it underperforms raw large_experimental on this newest slice.",
        "",
        "## Sources",
        "",
    ])

    for entry in payload["entries"]:
        lines.append(f"- {entry['label']}: {json.dumps(entry['sources'], ensure_ascii=False)}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize current latest incremental validation results across the mainline and Large branches")
    parser.add_argument("--output-name", default="latest_incremental_validation_summary")
    args = parser.parse_args()

    entries = [
        _build_entry(
            label="two_stage_mainline",
            profile_type="micro-two-stage-mainline",
            summary_path=RESULTS / "two_stage_rerank_leader_tiered_latest_incremental_v1" / "transformer_micro_multitask_backtest_summary.json",
            split_summary_path=RESULTS / "moneyflow_v1_latest_incremental_v1" / "split_prediction_summary.json",
            notes="current production-like leader-tiered two-stage mainline",
        ),
        _build_entry(
            label="plain_large",
            profile_type="large-single-model",
            summary_path=RESULTS / "moneyflow_v1_large_latest_incremental_v1" / "transformer_micro_multitask_backtest_summary.json",
            split_summary_path=RESULTS / "moneyflow_v1_large_latest_incremental_v1" / "split_prediction_summary.json",
            notes="frozen plain-large baseline on the newest unseen slice",
        ),
        _build_entry(
            label="large_experimental",
            profile_type="large-single-model-experimental",
            summary_path=RESULTS / "moneyflow_selective_large_latest_incremental_v1" / "transformer_micro_multitask_backtest_summary.json",
            split_summary_path=RESULTS / "moneyflow_selective_large_latest_incremental_v1" / "split_prediction_summary.json",
            notes="raw selective-large base on the newest unseen slice",
        ),
        _build_rerank_entry(
            label="large_experimental_rerank_blend060",
            probe_path=RESULTS / "moneyflow_selective_large_latest_incremental_v1_selective_large_oldboth_to_latest_incremental_core4_blend060_v1" / "lightweight_rerank_probe.json",
            notes="current reduced-head anchored rerank compromise candidate",
        ),
    ]

    start_date = min(entry["window"]["start_date"] for entry in entries if entry["window"]["start_date"])
    end_date = max(entry["window"]["end_date"] for entry in entries if entry["window"]["end_date"])
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "interpretation": "latest incremental sanity check across mainline and Large branches",
        "entries": entries,
    }

    output_json = RESULTS / f"{args.output_name}.json"
    output_md = RESULTS / f"{args.output_name}.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()