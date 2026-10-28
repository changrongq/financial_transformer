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


def _gate_positive(summary: dict, label: str) -> tuple[str, str]:
    total_return = float(summary.get("total_return", 0.0))
    sharpe_ratio = float(summary.get("sharpe_ratio", 0.0))
    signal_count = int(summary.get("signal_count", 0))
    trade_count = int(summary.get("trade_count", 0))
    if total_return > 0 and signal_count > 0:
        return "pass", f"{label} total_return={total_return:.4f}, sharpe={sharpe_ratio:.2f}, signals={signal_count}, trades={trade_count}"
    return "fail", f"{label} total_return={total_return:.4f}, sharpe={sharpe_ratio:.2f}, signals={signal_count}, trades={trade_count}"


def _gate_rolling(rolling: dict) -> tuple[str, str]:
    aggregate = rolling.get("aggregate", {})
    positive_windows = int(aggregate.get("positive_windows", 0))
    window_count = int(rolling.get("window_count", 0))
    avg_total_return = float(aggregate.get("avg_total_return", 0.0))
    if window_count <= 0:
        return "unknown", "rolling window_count=0"
    if positive_windows == window_count and avg_total_return > 0:
        return "pass", f"positive_windows={positive_windows}/{window_count}, avg_total_return={avg_total_return:.4f}"
    if positive_windows / window_count >= 0.80 and avg_total_return > 0:
        return "partial", f"positive_windows={positive_windows}/{window_count}, avg_total_return={avg_total_return:.4f}"
    return "fail", f"positive_windows={positive_windows}/{window_count}, avg_total_return={avg_total_return:.4f}"


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Mainline Acceptance Scorecard",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Decision summary: {payload['summary']}",
        "",
        f"## {payload['candidate']['label']}",
        "",
        f"- Config: {payload['candidate']['config']}",
        f"- Decision: {payload['candidate']['decision']}",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for gate in payload["candidate"]["gates"]:
        lines.append(f"| {gate['id']} | {gate['status']} | {gate['evidence']} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- This scorecard answers a narrower question than the Large scorecard: whether the current production-like leader-tiered two-stage mainline still deserves to remain the default execution route.",
        "- The newest incremental slice remains a tiny sanity check and should not be treated as a full replacement for the recent-OOS window.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an automated acceptance scorecard for the current two-stage leader-tiered mainline")
    parser.add_argument("--output-name", default="mainline_acceptance_scorecard")
    args = parser.parse_args()

    latest_summary = _load_json(RESULTS / "two_stage_rerank_best_v1" / "latest_predictions_filtered_two_stage_rerank_leader_tiered_v1_summary.json")
    old_summary = _load_json(RESULTS / "two_stage_rerank_best_v1" / "execution_search_mt5_ms10_leader_tiered_refine_v2_summary.json")
    rolling_summary = _load_json(RESULTS / "two_stage_rerank_best_v1" / "rolling_backtest_leader_tiered_baseline_20x5.json")
    recent_oos_summary = _load_json(RESULTS / "two_stage_rerank_best_recent_oos_v1" / "transformer_micro_multitask_backtest_summary.json")
    latest_incremental_summary = _load_json(RESULTS / "two_stage_rerank_leader_tiered_latest_incremental_v1" / "transformer_micro_multitask_backtest_summary.json")

    g2_status, g2_evidence = _gate_positive(old_summary, "historical_old_window")
    g3_status, g3_evidence = _gate_rolling(rolling_summary)
    g4_status, g4_evidence = _gate_positive(recent_oos_summary, "recent_oos")
    g5_status, g5_evidence = _gate_positive(latest_incremental_summary, "latest_incremental")

    latest_date = latest_summary.get("date")
    filtered_count = int(latest_summary.get("filtered_count", 0))
    prediction_count = int(latest_summary.get("prediction_count", 0))
    if filtered_count > 0 and prediction_count > 0:
        g1_status = "pass"
        g1_evidence = f"latest summary date={latest_date}, filtered_count={filtered_count}, prediction_count={prediction_count}"
    else:
        g1_status = "fail"
        g1_evidence = f"latest summary date={latest_date}, filtered_count={filtered_count}, prediction_count={prediction_count}"

    gates = [
        {"id": "G1", "status": g1_status, "evidence": g1_evidence},
        {"id": "G2", "status": g2_status, "evidence": g2_evidence},
        {"id": "G3", "status": g3_status, "evidence": g3_evidence},
        {"id": "G4", "status": g4_status, "evidence": g4_evidence},
        {"id": "G5", "status": g5_status, "evidence": g5_evidence},
    ]

    all_pass = all(gate["status"] == "pass" for gate in gates)
    candidate = {
        "label": "two_stage_leader_tiered_mainline",
        "config": "config_two_stage_rerank_leader_tiered_v1.yaml",
        "decision": "keep_default" if all_pass else "re-review_required",
        "gates": gates,
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": "Current two-stage leader-tiered mainline passes all gates" if all_pass else "Current two-stage leader-tiered mainline fails one or more gates",
        "candidate": candidate,
    }

    output_json = RESULTS / f"{args.output_name}.json"
    output_md = RESULTS / f"{args.output_name}.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()