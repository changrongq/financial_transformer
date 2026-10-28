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


def _status_rank(status: str) -> int:
    order = {"pass": 3, "partial": 2, "unknown": 1, "fail": 0, "n/a": 0}
    return order.get(status, 0)


def _gate_g2_status(total_return: float, *, is_repair: bool = False, old_base_total_return: float | None = None) -> tuple[str, str]:
    if total_return <= 0:
        return "fail", f"historical base total_return={total_return:.4f}"
    if is_repair and old_base_total_return and old_base_total_return > 0:
        retention = total_return / old_base_total_return
        if retention >= 0.85:
            return "pass", f"old-window retention={retention:.2%} vs raw base"
        if retention >= 0.50:
            return "partial", f"old-window retention={retention:.2%} vs raw base"
        return "fail", f"old-window retention={retention:.2%} vs raw base"
    if total_return >= 0.10:
        return "pass", f"historical base total_return={total_return:.4f}"
    return "partial", f"historical base total_return={total_return:.4f}"


def _gate_g3_status(rolling: dict | None) -> tuple[str, str]:
    if not rolling:
        return "unknown", "rolling summary missing"
    aggregate = rolling.get("aggregate", {})
    positive_windows = int(aggregate.get("positive_windows", 0))
    window_count = int(rolling.get("window_count", 0))
    avg_total_return = float(aggregate.get("avg_total_return", 0.0))
    if window_count <= 0:
        return "unknown", "rolling window_count=0"
    ratio = positive_windows / window_count
    if ratio >= 1.0 and avg_total_return > 0:
        return "pass", f"positive_windows={positive_windows}/{window_count}, avg_total_return={avg_total_return:.4f}"
    if ratio >= 0.80 and avg_total_return > 0:
        return "partial", f"positive_windows={positive_windows}/{window_count}, avg_total_return={avg_total_return:.4f}"
    return "fail", f"positive_windows={positive_windows}/{window_count}, avg_total_return={avg_total_return:.4f}"


def _gate_positive_window(summary: dict, label: str) -> tuple[str, str]:
    total_return = float(summary.get("total_return", 0.0))
    trade_count = int(summary.get("trade_count", 0))
    signal_count = int(summary.get("signal_count", 0))
    if total_return > 0 and max(trade_count, signal_count) > 0:
        return "pass", f"{label} total_return={total_return:.4f}, signals={signal_count}, trades={trade_count}"
    return "fail", f"{label} total_return={total_return:.4f}, signals={signal_count}, trades={trade_count}"


def _gate_g6_status(*, old_base: dict, recent_base: dict, latest_base: dict, recent_rerank: dict, latest_rerank: dict) -> tuple[str, str]:
    old_retention = float(recent_rerank["old_summary"]["total_return"]) / float(old_base["total_return"]) if float(old_base["total_return"]) > 0 else 0.0
    recent_improved = float(recent_rerank["recent_summary"]["total_return"]) > float(recent_base["total_return"])
    latest_not_worse = float(latest_rerank["rerank_summary"]["total_return"]) >= float(latest_base["total_return"])
    if recent_improved and latest_not_worse and old_retention >= 0.85:
        return "pass", f"recent improved, latest not worse, old retention={old_retention:.2%}"
    if recent_improved and old_retention >= 0.50:
        return "partial", f"recent improved, latest_not_worse={latest_not_worse}, old retention={old_retention:.2%}"
    return "fail", f"recent improved={recent_improved}, latest_not_worse={latest_not_worse}, old retention={old_retention:.2%}"


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Large Acceptance Scorecard",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Decision summary: {payload['summary']}",
        "",
    ]
    for candidate in payload["candidates"]:
        lines.extend([
            f"## {candidate['label']}",
            "",
            f"- Config: {candidate['config']}",
            f"- Decision: {candidate['decision']}",
            "",
            "| Gate | Status | Evidence |",
            "| --- | --- | --- |",
        ])
        for gate in candidate["gates"]:
            lines.append(f"| {gate['id']} | {gate['status']} | {gate['evidence']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an automated Large-route acceptance scorecard from existing result files")
    parser.add_argument("--output-name", default="large_acceptance_scorecard")
    args = parser.parse_args()

    plain_old = _load_json(RESULTS / "moneyflow_v1_large_v1" / "transformer_micro_multitask_backtest_summary.json")
    plain_recent = _load_json(RESULTS / "moneyflow_v1_large_recent_oos_v1" / "transformer_micro_multitask_backtest_summary.json")
    plain_latest = _load_json(RESULTS / "moneyflow_v1_large_latest_incremental_v1" / "transformer_micro_multitask_backtest_summary.json")
    plain_rolling = _load_json(RESULTS / "moneyflow_v1_large_v1" / "rolling_backtest_fair_single_large_10x5.json")

    selective_old = _load_json(RESULTS / "moneyflow_selective_large_v1" / "transformer_micro_multitask_backtest_summary.json")
    selective_recent = _load_json(RESULTS / "moneyflow_selective_large_recent_oos_v1" / "transformer_micro_multitask_backtest_summary.json")
    selective_latest = _load_json(RESULTS / "moneyflow_selective_large_latest_incremental_v1" / "transformer_micro_multitask_backtest_summary.json")
    selective_rolling = _load_json(RESULTS / "moneyflow_selective_large_v1" / "rolling_backtest_fair_single_large_experimental_10x5.json")

    rerank_recent_probe = _load_json(RESULTS / "moneyflow_selective_large_recent_oos_v1_selective_large_oldboth_to_recent_core4_blend060_v1" / "lightweight_rerank_probe.json")
    rerank_old_probe = _load_json(RESULTS / "moneyflow_selective_large_v1_selective_large_oldboth_to_old_core4_blend060_v1" / "lightweight_rerank_probe.json")
    rerank_latest_probe = _load_json(RESULTS / "moneyflow_selective_large_latest_incremental_v1_selective_large_oldboth_to_latest_incremental_core4_blend060_v1" / "lightweight_rerank_probe.json")

    plain_g2, plain_g2_evidence = _gate_g2_status(float(plain_old["total_return"]))
    plain_g3, plain_g3_evidence = _gate_g3_status(plain_rolling)
    plain_g4, plain_g4_evidence = _gate_positive_window(plain_recent, "recent_oos")
    plain_g5, plain_g5_evidence = _gate_positive_window(plain_latest, "latest_incremental")

    selective_g2, selective_g2_evidence = _gate_g2_status(float(selective_old["total_return"]))
    selective_g3, selective_g3_evidence = _gate_g3_status(selective_rolling)
    selective_g4, selective_g4_evidence = _gate_positive_window(selective_recent, "recent_oos")
    selective_g5, selective_g5_evidence = _gate_positive_window(selective_latest, "latest_incremental")

    rerank_old_summary = rerank_old_probe["rerank_summary"]
    rerank_recent_summary = rerank_recent_probe["rerank_summary"]
    rerank_latest_summary = rerank_latest_probe["rerank_summary"]
    rerank_g2, rerank_g2_evidence = _gate_g2_status(
        float(rerank_old_summary["total_return"]),
        is_repair=True,
        old_base_total_return=float(selective_old["total_return"]),
    )
    rerank_g4, rerank_g4_evidence = _gate_positive_window(rerank_recent_summary, "recent_oos_rerank")
    latest_vs_base = float(rerank_latest_summary["total_return"]) - float(selective_latest["total_return"])
    if latest_vs_base >= 0:
        rerank_g5, rerank_g5_evidence = "pass", f"latest rerank delta_vs_raw={latest_vs_base:.4f}"
    else:
        rerank_g5, rerank_g5_evidence = "fail", f"latest rerank delta_vs_raw={latest_vs_base:.4f}"
    rerank_g6, rerank_g6_evidence = _gate_g6_status(
        old_base=selective_old,
        recent_base=selective_recent,
        latest_base=selective_latest,
        recent_rerank={"recent_summary": rerank_recent_summary, "old_summary": rerank_old_summary},
        latest_rerank=rerank_latest_probe,
    )

    candidates = [
        {
            "label": "plain_large",
            "config": "config_moneyflow_v1_large_v1.yaml",
            "decision": "not_promotable",
            "gates": [
                {"id": "G1", "status": "fail", "evidence": "legacy default route remains blocked because the underlying artifact path is degraded"},
                {"id": "G2", "status": plain_g2, "evidence": plain_g2_evidence},
                {"id": "G3", "status": plain_g3, "evidence": plain_g3_evidence},
                {"id": "G4", "status": plain_g4, "evidence": plain_g4_evidence},
                {"id": "G5", "status": plain_g5, "evidence": plain_g5_evidence},
                {"id": "G6", "status": "n/a", "evidence": "no promoted repair layer attached"},
            ],
        },
        {
            "label": "large_experimental",
            "config": "config_moneyflow_selective_large_v1.yaml",
            "decision": "experimental_only",
            "gates": [
                {"id": "G1", "status": "pass", "evidence": "explicit report path exists and latest artifacts load cleanly"},
                {"id": "G2", "status": selective_g2, "evidence": selective_g2_evidence},
                {"id": "G3", "status": selective_g3, "evidence": selective_g3_evidence},
                {"id": "G4", "status": selective_g4, "evidence": selective_g4_evidence},
                {"id": "G5", "status": selective_g5, "evidence": selective_g5_evidence},
                {"id": "G6", "status": "n/a", "evidence": "raw base line only"},
            ],
        },
        {
            "label": "large_experimental_rerank_blend060",
            "config": "moneyflow_selective_large_*_blend060",
            "decision": "research_only",
            "gates": [
                {"id": "G1", "status": "partial", "evidence": "probe outputs exist, but this is not a stable reporting profile"},
                {"id": "G2", "status": rerank_g2, "evidence": rerank_g2_evidence},
                {"id": "G3", "status": "unknown", "evidence": "no rolling validation artifact for the blend-0.60 repair candidate"},
                {"id": "G4", "status": rerank_g4, "evidence": rerank_g4_evidence},
                {"id": "G5", "status": rerank_g5, "evidence": rerank_g5_evidence},
                {"id": "G6", "status": rerank_g6, "evidence": rerank_g6_evidence},
            ],
        },
    ]

    all_pass = any(all(gate["status"] == "pass" or gate["status"] == "n/a" for gate in candidate["gates"]) for candidate in candidates)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": "At least one Large candidate passes all gates" if all_pass else "No Large candidate passes all gates",
        "candidates": candidates,
    }

    output_json = RESULTS / f"{args.output_name}.json"
    output_md = RESULTS / f"{args.output_name}.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()