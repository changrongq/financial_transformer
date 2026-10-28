from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_lightweight_rerank_probe import (  # noqa: E402
    _freeze_to_checkpoint_universe,
    _load_or_generate_split_predictions,
    _prepare_candidates,
    _resolve_fair_params,
    _run_backtest,
)
from financial_transformer_experiment.backtest import write_backtest_outputs  # noqa: E402
from financial_transformer_experiment.config import load_config  # noqa: E402
from financial_transformer_experiment.two_stage_execution import resolve_two_stage_execution_spec  # noqa: E402


def _resolve_rule_probe_fair(config_path: Path, args) -> dict:
    spec = resolve_two_stage_execution_spec(str(config_path))
    if spec is not None:
        fair = {
            "short_up_threshold": float(spec.default_short_up_threshold),
            "take_profit_threshold": float(spec.default_take_profit_threshold),
            "stop_loss_threshold": float(spec.default_stop_loss_threshold),
            "top_k_per_day": int(spec.default_top_k_per_day),
            "hold_days": int(spec.default_execution_hold_days),
            "take_profit": float(spec.default_execution_take_profit),
            "stop_loss": float(spec.default_execution_stop_loss),
        }
        override_map = {
            "short_up_threshold": args.short_up_threshold,
            "take_profit_threshold": args.take_profit_threshold,
            "stop_loss_threshold": args.stop_loss_threshold,
            "top_k_per_day": args.top_k_per_day,
            "hold_days": args.hold_days,
            "take_profit": args.take_profit,
            "stop_loss": args.stop_loss,
        }
        for key, value in override_map.items():
            if value is not None:
                fair[key] = value
        return fair
    return _resolve_fair_params(config_path.stem, args)


def _score_percentile_desc(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["signal_score"].rank(method="average", ascending=False, pct=True)


def _score_rank_desc(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["signal_score"].rank(method="first", ascending=False)


def _apply_rule_rerank(
    candidates: pd.DataFrame,
    *,
    score_mode: str,
    high_score_percentile: float,
    max_drawdown_cutoff: float,
    drawdown_penalty_multiplier: float,
    reward_gap: float,
    reward_multiplier: float,
    reward_score_bonus: float,
    reward_quality_score_percentile: float,
    reward_safe_drawdown_cutoff: float | None,
    reward_top_score_percentile: float,
    reward_top_score_bonus: float,
    reward_mid_score_percentile: float,
    reward_mid_score_bonus: float,
    penalty_score_cost: float,
    gate_score_scale: float,
) -> pd.DataFrame:
    working = candidates.copy()
    working["score_percentile_desc"] = _score_percentile_desc(working)
    working["score_rank_desc"] = _score_rank_desc(working)
    working["daily_candidate_count"] = working.groupby("date")["stock_code"].transform("count")
    working["tp_sl_gap"] = working["pred_take_profit"] - working["pred_stop_loss"]

    high_score_risk = (
        (working["score_percentile_desc"] <= float(high_score_percentile))
        & (working["pred_max_drawdown"] <= float(max_drawdown_cutoff))
    )
    asym_reward = (
        (working["pred_take_profit"] > working["pred_stop_loss"])
        & (working["tp_sl_gap"] > float(reward_gap))
    )
    reward_safe_drawdown_value = float(max_drawdown_cutoff if reward_safe_drawdown_cutoff is None else reward_safe_drawdown_cutoff)
    reward_quality_rank_limit = (working["daily_candidate_count"] * float(reward_quality_score_percentile)).apply(
        lambda value: max(1, int(value) if float(value).is_integer() else int(value) + 1)
    )
    reward_quality_gate = (
        (working["pred_max_drawdown"] > reward_safe_drawdown_value)
        & (working["score_rank_desc"] <= reward_quality_rank_limit)
    )
    reward_eligible = asym_reward & reward_quality_gate

    working["rule_penalty_flag"] = high_score_risk.astype(int)
    working["rule_reward_flag"] = reward_eligible.astype(int)
    working["rule_reward_quality_flag"] = reward_quality_gate.astype(int)
    working["rule_penalty_score"] = -float(penalty_score_cost) * working["rule_penalty_flag"]
    reward_score = pd.Series(0.0, index=working.index)
    reward_top_rank_limit = (working["daily_candidate_count"] * float(reward_top_score_percentile)).apply(lambda value: max(1, int(value) if float(value).is_integer() else int(value) + 1))
    reward_mid_rank_limit = (working["daily_candidate_count"] * float(reward_mid_score_percentile)).apply(lambda value: max(1, int(value) if float(value).is_integer() else int(value) + 1))
    reward_top_bucket = reward_eligible & (working["score_rank_desc"] <= reward_top_rank_limit)
    reward_mid_bucket = (
        reward_eligible
        & ~reward_top_bucket
        & (working["score_rank_desc"] <= reward_mid_rank_limit)
    )
    reward_other_bucket = reward_eligible & ~(reward_top_bucket | reward_mid_bucket)
    reward_score.loc[reward_top_bucket] = float(reward_top_score_bonus)
    reward_score.loc[reward_mid_bucket] = float(reward_mid_score_bonus)
    reward_score.loc[reward_other_bucket] = float(reward_score_bonus)
    working["rule_reward_score"] = reward_score
    working["rule_reward_top_bucket"] = reward_top_bucket.astype(int)
    working["rule_reward_mid_bucket"] = reward_mid_bucket.astype(int)
    working["rule_gate_score"] = working["rule_reward_score"] + working["rule_penalty_score"]
    working["rule_multiplier"] = 1.0
    working.loc[high_score_risk, "rule_multiplier"] *= float(drawdown_penalty_multiplier)
    working.loc[reward_eligible, "rule_multiplier"] *= float(reward_multiplier)
    if score_mode == "multiplier":
        working["rule_rerank_score"] = working["signal_score"] * working["rule_multiplier"]
    elif score_mode == "additive_gate":
        working["rule_rerank_score"] = working["signal_score"] + float(gate_score_scale) * working["rule_gate_score"]
    else:
        raise RuntimeError(f"unsupported score_mode: {score_mode}")
    return working


def _top_selection_compare(base_candidates: pd.DataFrame, rule_candidates: pd.DataFrame) -> pd.DataFrame:
    base_top = (
        base_candidates.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
        .groupby("date", group_keys=False)
        .head(1)[["date", "stock_code", "day_rank", "future_return", "signal_score"]]
        .rename(
            columns={
                "stock_code": "base_stock_code",
                "day_rank": "base_rank",
                "future_return": "base_future_return",
                "signal_score": "base_score",
            }
        )
    )
    rule_top = (
        rule_candidates.sort_values(["date", "rule_rerank_score", "pred_take_profit"], ascending=[True, False, False])
        .groupby("date", group_keys=False)
        .head(1)[
            [
                "date",
                "stock_code",
                "day_rank",
                "future_return",
                "rule_rerank_score",
                "rule_multiplier",
                "rule_penalty_flag",
                "rule_reward_flag",
                "rule_gate_score",
                "tp_sl_gap",
                "pred_max_drawdown",
            ]
        ]
        .rename(
            columns={
                "stock_code": "rule_stock_code",
                "day_rank": "rule_rank",
                "future_return": "rule_future_return",
            }
        )
    )
    compare = base_top.merge(rule_top, on="date", how="outer").sort_values("date")
    compare["date"] = pd.to_datetime(compare["date"]).dt.strftime("%Y-%m-%d")
    return compare


def _parse_float_grid(raw_values: str) -> list[float]:
    values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
    if not values:
        raise RuntimeError("grid is empty")
    return sorted(set(values))


def _result_sort_key(summary: dict) -> tuple[float, float, int, int]:
    return (
        float(summary["total_return"]),
        float(summary["sharpe_ratio"]),
        int(summary["trade_count"]),
        int(summary["signal_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe simple rule-based rerank overlays on existing candidate pools")
    parser.add_argument("--config", default="config_two_stage_rerank_leader_tiered_latest_incremental_v3.yaml")
    parser.add_argument("--output-tag", default="rule_rerank_probe")
    parser.add_argument("--state-profile", default=None)
    parser.add_argument("--candidate-pool", type=int, default=8)
    parser.add_argument("--score-mode", choices=["multiplier", "additive_gate"], default="additive_gate")
    parser.add_argument("--high-score-percentile", type=float, default=0.20)
    parser.add_argument("--max-drawdown-cutoff", type=float, default=-0.15)
    parser.add_argument("--drawdown-penalty-multiplier", type=float, default=0.80)
    parser.add_argument("--reward-gap", type=float, default=0.10)
    parser.add_argument("--reward-multiplier", type=float, default=1.20)
    parser.add_argument("--reward-score-bonus", type=float, default=0.0)
    parser.add_argument("--reward-quality-score-percentile", type=float, default=0.10)
    parser.add_argument("--reward-safe-drawdown-cutoff", type=float, default=None)
    parser.add_argument("--reward-top-score-percentile", type=float, default=0.05)
    parser.add_argument("--reward-top-score-bonus", type=float, default=1.6)
    parser.add_argument("--reward-mid-score-percentile", type=float, default=0.10)
    parser.add_argument("--reward-mid-score-bonus", type=float, default=0.4)
    parser.add_argument("--penalty-score-cost", type=float, default=1.0)
    parser.add_argument("--gate-score-scale", type=float, default=0.05)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--high-score-percentiles", default="0.2,0.35,0.5")
    parser.add_argument("--max-drawdown-cutoffs", default="-0.055,-0.05,-0.0475")
    parser.add_argument("--drawdown-penalty-multipliers", default="0.6,0.8,0.9")
    parser.add_argument("--reward-gaps", default="0.08,0.1,0.12,0.14")
    parser.add_argument("--reward-multipliers", default="1.1,1.2,1.35")
    parser.add_argument("--max-grid-results", type=int, default=10)
    parser.add_argument("--short-up-threshold", type=float, default=None)
    parser.add_argument("--take-profit-threshold", type=float, default=None)
    parser.add_argument("--stop-loss-threshold", type=float, default=None)
    parser.add_argument("--top-k-per-day", type=int, default=None)
    parser.add_argument("--hold-days", type=int, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--min-score-ratio-to-leader", type=float, default=0.0)
    parser.add_argument("--leader-max-day-rank", type=int, default=None)
    parser.add_argument("--leader-take-profit", type=float, default=None)
    parser.add_argument("--leader-stop-loss", type=float, default=None)
    parser.add_argument("--leader-hold-days", type=int, default=None)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    config = _freeze_to_checkpoint_universe(load_config(str(config_path)))
    fair = _resolve_rule_probe_fair(config_path, args)

    output_dir = config.root_dir / "results" / f"{config.results_dir.name}_{args.output_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_predictions = _load_or_generate_split_predictions(config, require_val=False, require_test=True)
    base_candidates = _prepare_candidates(test_predictions, config, fair, args.candidate_pool)
    if base_candidates.empty:
        raise RuntimeError("test candidate pool is empty")

    default_rule_candidates = _apply_rule_rerank(
        base_candidates,
        score_mode=args.score_mode,
        high_score_percentile=args.high_score_percentile,
        max_drawdown_cutoff=args.max_drawdown_cutoff,
        drawdown_penalty_multiplier=args.drawdown_penalty_multiplier,
        reward_gap=args.reward_gap,
        reward_multiplier=args.reward_multiplier,
        reward_score_bonus=args.reward_score_bonus,
        reward_quality_score_percentile=args.reward_quality_score_percentile,
        reward_safe_drawdown_cutoff=args.reward_safe_drawdown_cutoff,
        reward_top_score_percentile=args.reward_top_score_percentile,
        reward_top_score_bonus=args.reward_top_score_bonus,
        reward_mid_score_percentile=args.reward_mid_score_percentile,
        reward_mid_score_bonus=args.reward_mid_score_bonus,
        penalty_score_cost=args.penalty_score_cost,
        gate_score_scale=args.gate_score_scale,
    )

    base_result = _run_backtest(
        config,
        base_candidates,
        score_col="signal_score",
        fair=fair,
        state_profile=args.state_profile,
        min_score_ratio_to_leader=args.min_score_ratio_to_leader,
        leader_max_day_rank=args.leader_max_day_rank,
        leader_take_profit=args.leader_take_profit,
        leader_stop_loss=args.leader_stop_loss,
        leader_hold_days=args.leader_hold_days,
    )

    default_rule_result = _run_backtest(
        config,
        default_rule_candidates,
        score_col="rule_rerank_score",
        fair=fair,
        state_profile=args.state_profile,
        min_score_ratio_to_leader=args.min_score_ratio_to_leader,
        leader_max_day_rank=args.leader_max_day_rank,
        leader_take_profit=args.leader_take_profit,
        leader_stop_loss=args.leader_stop_loss,
        leader_hold_days=args.leader_hold_days,
    )

    best_rules = {
        "high_score_percentile": float(args.high_score_percentile),
        "max_drawdown_cutoff": float(args.max_drawdown_cutoff),
        "drawdown_penalty_multiplier": float(args.drawdown_penalty_multiplier),
        "reward_gap": float(args.reward_gap),
        "reward_multiplier": float(args.reward_multiplier),
    }
    best_rule_candidates = default_rule_candidates
    best_rule_result = default_rule_result
    grid_results: list[dict] = []
    searched_grids = {}
    if args.grid_search:
        searched_grids = {
            "high_score_percentile": _parse_float_grid(args.high_score_percentiles),
            "max_drawdown_cutoff": _parse_float_grid(args.max_drawdown_cutoffs),
            "drawdown_penalty_multiplier": _parse_float_grid(args.drawdown_penalty_multipliers),
            "reward_gap": _parse_float_grid(args.reward_gaps),
            "reward_multiplier": _parse_float_grid(args.reward_multipliers),
        }
        for high_score_percentile, max_drawdown_cutoff, drawdown_penalty_multiplier, reward_gap, reward_multiplier in product(
            searched_grids["high_score_percentile"],
            searched_grids["max_drawdown_cutoff"],
            searched_grids["drawdown_penalty_multiplier"],
            searched_grids["reward_gap"],
            searched_grids["reward_multiplier"],
        ):
            rule_candidates = _apply_rule_rerank(
                base_candidates,
                score_mode=args.score_mode,
                high_score_percentile=high_score_percentile,
                max_drawdown_cutoff=max_drawdown_cutoff,
                drawdown_penalty_multiplier=drawdown_penalty_multiplier,
                reward_gap=reward_gap,
                reward_multiplier=reward_multiplier,
                reward_score_bonus=args.reward_score_bonus,
                reward_quality_score_percentile=args.reward_quality_score_percentile,
                reward_safe_drawdown_cutoff=args.reward_safe_drawdown_cutoff,
                reward_top_score_percentile=args.reward_top_score_percentile,
                reward_top_score_bonus=args.reward_top_score_bonus,
                reward_mid_score_percentile=args.reward_mid_score_percentile,
                reward_mid_score_bonus=args.reward_mid_score_bonus,
                penalty_score_cost=args.penalty_score_cost,
                gate_score_scale=args.gate_score_scale,
            )
            rule_result = _run_backtest(
                config,
                rule_candidates,
                score_col="rule_rerank_score",
                fair=fair,
                state_profile=args.state_profile,
                min_score_ratio_to_leader=args.min_score_ratio_to_leader,
                leader_max_day_rank=args.leader_max_day_rank,
                leader_take_profit=args.leader_take_profit,
                leader_stop_loss=args.leader_stop_loss,
                leader_hold_days=args.leader_hold_days,
            )
            rules = {
                "score_mode": args.score_mode,
                "high_score_percentile": float(high_score_percentile),
                "max_drawdown_cutoff": float(max_drawdown_cutoff),
                "drawdown_penalty_multiplier": float(drawdown_penalty_multiplier),
                "reward_gap": float(reward_gap),
                "reward_multiplier": float(reward_multiplier),
                "reward_score_bonus": float(args.reward_score_bonus),
                "reward_quality_score_percentile": float(args.reward_quality_score_percentile),
                "reward_safe_drawdown_cutoff": None if args.reward_safe_drawdown_cutoff is None else float(args.reward_safe_drawdown_cutoff),
                "reward_top_score_percentile": float(args.reward_top_score_percentile),
                "reward_top_score_bonus": float(args.reward_top_score_bonus),
                "reward_mid_score_percentile": float(args.reward_mid_score_percentile),
                "reward_mid_score_bonus": float(args.reward_mid_score_bonus),
                "penalty_score_cost": float(args.penalty_score_cost),
                "gate_score_scale": float(args.gate_score_scale),
            }
            candidate = {
                "rules": rules,
                "summary": rule_result.summary,
                "rule_stats": {
                    "penalty_hits": int(rule_candidates["rule_penalty_flag"].sum()),
                    "reward_hits": int(rule_candidates["rule_reward_flag"].sum()),
                    "changed_score_rows": int((rule_candidates["rule_rerank_score"] != rule_candidates["signal_score"]).sum()),
                },
                "delta": {
                    "total_return": float(rule_result.summary["total_return"] - base_result.summary["total_return"]),
                    "sharpe_ratio": float(rule_result.summary["sharpe_ratio"] - base_result.summary["sharpe_ratio"]),
                    "trade_count": int(rule_result.summary["trade_count"] - base_result.summary["trade_count"]),
                    "signal_count": int(rule_result.summary["signal_count"] - base_result.summary["signal_count"]),
                },
            }
            grid_results.append(candidate)
            if _result_sort_key(rule_result.summary) > _result_sort_key(best_rule_result.summary):
                best_rules = rules
                best_rule_candidates = rule_candidates
                best_rule_result = rule_result
        grid_results.sort(key=lambda item: _result_sort_key(item["summary"]), reverse=True)
    else:
        searched_grids = {
            "high_score_percentile": [float(args.high_score_percentile)],
            "max_drawdown_cutoff": [float(args.max_drawdown_cutoff)],
            "drawdown_penalty_multiplier": [float(args.drawdown_penalty_multiplier)],
            "reward_gap": [float(args.reward_gap)],
            "reward_multiplier": [float(args.reward_multiplier)],
        }

    compare = _top_selection_compare(base_candidates, best_rule_candidates)
    compare_path = output_dir / "selected_compare.csv"
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    best_rule_candidates.to_csv(output_dir / "test_candidate_pool.csv", index=False, encoding="utf-8-sig")
    detail_outputs = write_backtest_outputs(best_rule_result, output_dir, "rule_rerank_best")

    payload = {
        "config": str(config_path),
        "state_profile": args.state_profile,
        "candidate_pool": int(args.candidate_pool),
        "fair": fair,
        "rules": {
            "score_mode": args.score_mode,
            "high_score_percentile": float(args.high_score_percentile),
            "max_drawdown_cutoff": float(args.max_drawdown_cutoff),
            "drawdown_penalty_multiplier": float(args.drawdown_penalty_multiplier),
            "reward_gap": float(args.reward_gap),
            "reward_multiplier": float(args.reward_multiplier),
            "reward_score_bonus": float(args.reward_score_bonus),
            "reward_quality_score_percentile": float(args.reward_quality_score_percentile),
            "reward_safe_drawdown_cutoff": None if args.reward_safe_drawdown_cutoff is None else float(args.reward_safe_drawdown_cutoff),
            "reward_top_score_percentile": float(args.reward_top_score_percentile),
            "reward_top_score_bonus": float(args.reward_top_score_bonus),
            "reward_mid_score_percentile": float(args.reward_mid_score_percentile),
            "reward_mid_score_bonus": float(args.reward_mid_score_bonus),
            "penalty_score_cost": float(args.penalty_score_cost),
            "gate_score_scale": float(args.gate_score_scale),
        },
        "grid_search": bool(args.grid_search),
        "searched_grids": searched_grids,
        "best_rules": best_rules,
        "rule_stats": {
            "candidate_rows": int(len(best_rule_candidates)),
            "candidate_dates": int(best_rule_candidates["date"].nunique()),
            "penalty_hits": int(best_rule_candidates["rule_penalty_flag"].sum()),
            "reward_hits": int(best_rule_candidates["rule_reward_flag"].sum()),
            "changed_score_rows": int((best_rule_candidates["rule_rerank_score"] != best_rule_candidates["signal_score"]).sum()),
        },
        "base_summary": base_result.summary,
        "default_rule_summary": default_rule_result.summary,
        "rerank_summary": best_rule_result.summary,
        "delta": {
            "total_return": float(best_rule_result.summary["total_return"] - base_result.summary["total_return"]),
            "sharpe_ratio": float(best_rule_result.summary["sharpe_ratio"] - base_result.summary["sharpe_ratio"]),
            "trade_count": int(best_rule_result.summary["trade_count"] - base_result.summary["trade_count"]),
            "signal_count": int(best_rule_result.summary["signal_count"] - base_result.summary["signal_count"]),
        },
        "top_grid_results": grid_results[: max(int(args.max_grid_results), 0)],
        "selected_compare": str(compare_path),
        "detail_outputs": detail_outputs,
    }
    (output_dir / "rule_rerank_probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()