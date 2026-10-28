from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_rule_rerank_probe import (  # noqa: E402
    _apply_rule_rerank,
    _resolve_rule_probe_fair,
)
from analyze_lightweight_rerank_probe import (  # noqa: E402
    _freeze_to_checkpoint_universe,
    _load_or_generate_split_predictions,
    _prepare_candidates,
    _run_backtest,
)
from financial_transformer_experiment.config import load_config  # noqa: E402


DEFAULT_CONFIGS = [
    "config_two_stage_rerank_leader_tiered_latest_incremental_v1.yaml",
    "config_two_stage_rerank_leader_tiered_latest_incremental_v2.yaml",
    "config_two_stage_rerank_leader_tiered_latest_incremental_v3.yaml",
]


def _load_rules(path: Path) -> tuple[dict, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_rules = payload.get("rules") or {}
    best_rules = payload.get("best_rules") or {}
    rules = {**base_rules, **best_rules}
    if not rules:
        raise RuntimeError(f"no rules found in {path}")
    candidate_pool = int(payload.get("candidate_pool", 8))
    return rules, candidate_pool


def _run_for_config(config_name: str, rules: dict, candidate_pool: int) -> dict:
    config_path = (ROOT / config_name).resolve()
    config = _freeze_to_checkpoint_universe(load_config(str(config_path)))
    args = argparse.Namespace(
        short_up_threshold=None,
        take_profit_threshold=None,
        stop_loss_threshold=None,
        top_k_per_day=None,
        hold_days=None,
        take_profit=None,
        stop_loss=None,
    )
    fair = _resolve_rule_probe_fair(config_path, args)
    _, test_predictions = _load_or_generate_split_predictions(config, require_val=False, require_test=True)
    base_candidates = _prepare_candidates(test_predictions, config, fair, candidate_pool)
    if base_candidates.empty:
        raise RuntimeError(f"candidate pool is empty for {config_name}")
    rule_candidates = _apply_rule_rerank(
        base_candidates,
        score_mode=str(rules.get("score_mode", "additive_gate")),
        high_score_percentile=float(rules["high_score_percentile"]),
        max_drawdown_cutoff=float(rules["max_drawdown_cutoff"]),
        drawdown_penalty_multiplier=float(rules["drawdown_penalty_multiplier"]),
        reward_gap=float(rules["reward_gap"]),
        reward_multiplier=float(rules["reward_multiplier"]),
        reward_score_bonus=float(rules.get("reward_score_bonus", 1.0)),
        reward_quality_score_percentile=float(rules.get("reward_quality_score_percentile", 0.10)),
        reward_safe_drawdown_cutoff=rules.get("reward_safe_drawdown_cutoff"),
        reward_top_score_percentile=float(rules.get("reward_top_score_percentile", 0.05)),
        reward_top_score_bonus=float(rules.get("reward_top_score_bonus", 1.6)),
        reward_mid_score_percentile=float(rules.get("reward_mid_score_percentile", 0.10)),
        reward_mid_score_bonus=float(rules.get("reward_mid_score_bonus", 0.4)),
        penalty_score_cost=float(rules.get("penalty_score_cost", 1.0)),
        gate_score_scale=float(rules.get("gate_score_scale", 0.05)),
    )
    base_result = _run_backtest(config, base_candidates, score_col="signal_score", fair=fair)
    rule_result = _run_backtest(config, rule_candidates, score_col="rule_rerank_score", fair=fair)
    return {
        "config": config_name,
        "window": {
            "start_date": base_result.summary.get("requested_start_date"),
            "end_date": base_result.summary.get("requested_end_date"),
        },
        "rule_stats": {
            "candidate_rows": int(len(rule_candidates)),
            "candidate_dates": int(rule_candidates["date"].nunique()),
            "penalty_hits": int(rule_candidates["rule_penalty_flag"].sum()),
            "reward_hits": int(rule_candidates["rule_reward_flag"].sum()),
            "changed_score_rows": int((rule_candidates["rule_rerank_score"] != rule_candidates["signal_score"]).sum()),
        },
        "base_summary": base_result.summary,
        "rerank_summary": rule_result.summary,
        "delta": {
            "total_return": float(rule_result.summary["total_return"] - base_result.summary["total_return"]),
            "sharpe_ratio": float(rule_result.summary["sharpe_ratio"] - base_result.summary["sharpe_ratio"]),
            "trade_count": int(rule_result.summary["trade_count"] - base_result.summary["trade_count"]),
            "signal_count": int(rule_result.summary["signal_count"] - base_result.summary["signal_count"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one fixed B2 rule set across incremental two-stage windows")
    parser.add_argument(
        "--rules-json",
        default="results/two_stage_rerank_leader_tiered_latest_incremental_v3_rule_rerank_probe_latest_incremental_v3_grid/rule_rerank_probe.json",
    )
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--output-name", default="rule_rerank_window_series")
    args = parser.parse_args()

    rules_path = (ROOT / args.rules_json).resolve()
    rules, candidate_pool = _load_rules(rules_path)
    config_names = [item.strip() for item in args.configs.split(",") if item.strip()]
    rows = [_run_for_config(config_name, rules, candidate_pool) for config_name in config_names]

    summary = {
        "rules_source": str(rules_path),
        "rules": rules,
        "candidate_pool": candidate_pool,
        "window_count": int(len(rows)),
        "improved_return_windows": int(sum(1 for row in rows if row["delta"]["total_return"] > 0)),
        "improved_sharpe_windows": int(sum(1 for row in rows if row["delta"]["sharpe_ratio"] > 0)),
        "avg_delta_total_return": float(sum(row["delta"]["total_return"] for row in rows) / len(rows)),
        "avg_delta_sharpe_ratio": float(sum(row["delta"]["sharpe_ratio"] for row in rows) / len(rows)),
        "rows": rows,
    }

    output_json = ROOT / "results" / f"{args.output_name}.json"
    output_md = ROOT / "results" / f"{args.output_name}.md"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Rule Rerank Window Series", ""]
    lines.append(f"- Rules source: {rules_path}")
    lines.append(f"- Candidate pool: {candidate_pool}")
    lines.append(
        f"- Improved return windows: {summary['improved_return_windows']} / {summary['window_count']}, improved sharpe windows: {summary['improved_sharpe_windows']} / {summary['window_count']}"
    )
    lines.append(f"- Avg delta total return: {summary['avg_delta_total_return'] * 100:.2f}%")
    lines.append(f"- Avg delta sharpe: {summary['avg_delta_sharpe_ratio']:.4f}")
    lines.append("")
    lines.append("## Windows")
    lines.append("")
    for row in rows:
        lines.append(
            f"- {row['config']}: {row['window']['start_date']} ~ {row['window']['end_date']}, delta_return={row['delta']['total_return'] * 100:.2f}%, delta_sharpe={row['delta']['sharpe_ratio']:.4f}, delta_trades={row['delta']['trade_count']}, delta_signals={row['delta']['signal_count']}, penalties={row['rule_stats']['penalty_hits']}, rewards={row['rule_stats']['reward_hits']}"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "window_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()