from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inverse_rl_market_simulation.config import load_config
from inverse_rl_market_simulation.data import load_stock_window
from inverse_rl_market_simulation.objective import evaluate_simulation
from inverse_rl_market_simulation.simulator import simulate_path


AGENT_COLS = ["slow_demand", "trend_demand", "herd_demand"]


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _classify_day_types(real_frame: pd.DataFrame) -> dict[str, pd.Series]:
    real_return = real_frame["return_1d"].astype(float)
    abs_return = real_return.abs()
    turnover_z = real_frame["turnover_z"].astype(float)

    up_threshold = float(real_return.quantile(0.80))
    down_threshold = float(real_return.quantile(0.20))
    abs_low_threshold = float(abs_return.quantile(0.40))
    turnover_high_threshold = float(turnover_z.quantile(0.80))

    day_types = {
        "all_days": pd.Series(True, index=real_frame.index),
        "strong_up": real_return >= up_threshold,
        "strong_down": real_return <= down_threshold,
        "calm_days": abs_return <= abs_low_threshold,
        "turnover_spike": turnover_z >= turnover_high_threshold,
        "strong_up_with_turnover": (real_return >= up_threshold) & (turnover_z >= turnover_high_threshold),
        "strong_down_with_turnover": (real_return <= down_threshold) & (turnover_z >= turnover_high_threshold),
    }
    return day_types


def _summarize_slice(simulation: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    subset = simulation.loc[mask].copy()
    abs_demands = subset[AGENT_COLS].abs()
    total_abs = abs_demands.sum(axis=1)
    dominant = abs_demands.idxmax(axis=1) if not subset.empty else pd.Series(dtype=object)

    summary: dict[str, object] = {
        "rows": int(len(subset)),
        "mean_real_return": float(subset["real_return"].mean()) if not subset.empty else 0.0,
        "mean_sim_return": float(subset["sim_return"].mean()) if not subset.empty else 0.0,
        "mean_turnover_z": float(subset["real_turnover_z"].mean()) if not subset.empty else 0.0,
        "agent_abs_share_mean": {},
        "agent_signed_mean": {},
        "agent_active_ratio": {},
        "dominant_agent_share": {},
    }
    for agent in AGENT_COLS:
        abs_series = abs_demands[agent] if not subset.empty else pd.Series(dtype=float)
        share = abs_series.div(total_abs.replace(0.0, np.nan)).fillna(0.0) if not subset.empty else pd.Series(dtype=float)
        active = abs_series > 1e-8 if not subset.empty else pd.Series(dtype=bool)
        summary["agent_abs_share_mean"][agent] = float(share.mean()) if not share.empty else 0.0
        summary["agent_signed_mean"][agent] = float(subset[agent].mean()) if not subset.empty else 0.0
        summary["agent_active_ratio"][agent] = float(active.mean()) if not active.empty else 0.0
        summary["dominant_agent_share"][agent] = float((dominant == agent).mean()) if not subset.empty else 0.0
    return summary


def _aggregate_day_type_means(trial_summaries: list[dict[str, object]], day_type_names: list[str]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for day_type in day_type_names:
        rows = [summary["day_type_summary"][day_type] for summary in trial_summaries]
        aggregate[day_type] = {
            "rows_mean": float(np.mean([row["rows"] for row in rows])),
            "mean_real_return": float(np.mean([row["mean_real_return"] for row in rows])),
            "mean_sim_return": float(np.mean([row["mean_sim_return"] for row in rows])),
            "mean_turnover_z": float(np.mean([row["mean_turnover_z"] for row in rows])),
            "agent_abs_share_mean": {
                agent: float(np.mean([row["agent_abs_share_mean"][agent] for row in rows])) for agent in AGENT_COLS
            },
            "agent_active_ratio": {
                agent: float(np.mean([row["agent_active_ratio"][agent] for row in rows])) for agent in AGENT_COLS
            },
            "dominant_agent_share": {
                agent: float(np.mean([row["dominant_agent_share"][agent] for row in rows])) for agent in AGENT_COLS
            },
        }
    return aggregate


def _build_report(
    stock_code: str,
    top_k: int,
    best_trial: dict[str, object],
    aggregate: dict[str, object],
) -> str:
    lines = [
        "# Top-Trial Agent Role Analysis",
        "",
        f"- stock_code: `{stock_code}`",
        f"- top_k: `{top_k}`",
        f"- best_trial_id: `{best_trial['trial_id']}`",
        f"- best_total_loss: `{float(best_trial['total_loss']):.6f}`",
        "",
        "## Reading",
        "",
        "This report is for diagnosis only. It asks where slow, trend, and herd contribute across day types, without adding any structural preference to the loss.",
        "",
    ]
    for day_type, summary in aggregate.items():
        lines.extend(
            [
                f"## {day_type}",
                "",
                f"- rows_mean: `{float(summary['rows_mean']):.2f}`",
                f"- mean_real_return: `{float(summary['mean_real_return']):.6f}`",
                f"- mean_sim_return: `{float(summary['mean_sim_return']):.6f}`",
                f"- mean_turnover_z: `{float(summary['mean_turnover_z']):.6f}`",
                f"- slow_abs_share_mean: `{float(summary['agent_abs_share_mean']['slow_demand']):.4f}`",
                f"- trend_abs_share_mean: `{float(summary['agent_abs_share_mean']['trend_demand']):.4f}`",
                f"- herd_abs_share_mean: `{float(summary['agent_abs_share_mean']['herd_demand']):.4f}`",
                f"- slow_active_ratio: `{float(summary['agent_active_ratio']['slow_demand']):.4f}`",
                f"- trend_active_ratio: `{float(summary['agent_active_ratio']['trend_demand']):.4f}`",
                f"- herd_active_ratio: `{float(summary['agent_active_ratio']['herd_demand']):.4f}`",
                f"- dominant_slow_share: `{float(summary['dominant_agent_share']['slow_demand']):.4f}`",
                f"- dominant_trend_share: `{float(summary['dominant_agent_share']['trend_demand']):.4f}`",
                f"- dominant_herd_share: `{float(summary['dominant_agent_share']['herd_demand']):.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Hint",
            "",
            "If trend share rises specifically on strong up/down or turnover-spike days, then trend is doing conditional work even when its global share stays smaller than herd. If trend stays weak across all high-move regimes in most top trials, then trend weakness is more likely a property of this data window or current objective than a leftover activation bug.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze top-trial agent roles by market day type")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config_single_stock_mvp.yaml"))
    parser.add_argument("--results-dir")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else config.results.results_dir
    real_frame = load_stock_window(config)
    trials = pd.read_csv(results_dir / "calibration_trials.csv")
    trials = trials.sort_values("total_loss", ascending=True).reset_index(drop=True)
    top_trials = trials.head(int(args.top_k)).copy()

    day_types = _classify_day_types(real_frame)
    day_type_names = list(day_types.keys())

    trial_summaries: list[dict[str, object]] = []
    for _, trial in top_trials.iterrows():
        params = {
            key: (int(trial[key]) if key in {"anchor_window", "trend_lookback"} else float(trial[key]))
            for key in [
                "anchor_window",
                "mean_reversion_strength",
                "inventory_inertia",
                "trend_lookback",
                "trend_strength",
                "entry_threshold",
                "herd_strength",
                "attention_to_turnover",
                "panic_asymmetry",
            ]
        }
        simulation = simulate_path(real_frame, params, search_space=config.optimization.search_space)
        evaluated = evaluate_simulation(simulation, label=str(trial["label"]))
        trial_summary = {
            "trial_id": int(trial["trial_id"]),
            "label": str(trial["label"]),
            "total_loss": float(evaluated["total_loss"]),
            "day_type_summary": {
                day_type: _summarize_slice(simulation, mask) for day_type, mask in day_types.items()
            },
        }
        trial_summaries.append(trial_summary)

    aggregate = _aggregate_day_type_means(trial_summaries, day_type_names)
    output = {
        "stock_code": config.experiment.stock_code,
        "top_k": int(len(trial_summaries)),
        "trial_summaries": trial_summaries,
        "aggregate_day_type_summary": aggregate,
    }

    json_path = results_dir / "top_trial_role_analysis.json"
    md_path = results_dir / "top_trial_role_analysis.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(
        _build_report(config.experiment.stock_code, int(len(trial_summaries)), trial_summaries[0], aggregate),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "top_k": int(len(trial_summaries)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()