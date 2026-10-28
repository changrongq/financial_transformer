from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
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

    return {
        "all_days": pd.Series(True, index=real_frame.index),
        "strong_up": real_return >= up_threshold,
        "strong_down": real_return <= down_threshold,
        "calm_days": abs_return <= abs_low_threshold,
        "turnover_spike": turnover_z >= turnover_high_threshold,
        "strong_up_with_turnover": (real_return >= up_threshold) & (turnover_z >= turnover_high_threshold),
        "strong_down_with_turnover": (real_return <= down_threshold) & (turnover_z >= turnover_high_threshold),
    }


def _clip01(values: pd.Series | np.ndarray | float) -> pd.Series:
    return pd.Series(np.clip(np.asarray(values, dtype=float), 0.0, 1.0))


def _compute_same_sign_streak(values: pd.Series) -> pd.Series:
    streak = np.zeros(len(values), dtype=float)
    last_sign = 0
    current = 0
    for idx, value in enumerate(values.to_numpy(dtype=float)):
        sign = 0 if abs(value) <= 1e-12 else (1 if value > 0.0 else -1)
        if sign == 0:
            current = 0
            last_sign = 0
        elif sign == last_sign:
            current += 1
        else:
            current = 1
            last_sign = sign
        streak[idx] = current
    return pd.Series(streak, index=values.index, dtype=float)


def _compute_switch_frame(real_frame: pd.DataFrame, simulation: pd.DataFrame, params: dict[str, float | int]) -> pd.DataFrame:
    frame = simulation.copy()
    abs_demands = frame[AGENT_COLS].abs()
    total_abs = abs_demands.sum(axis=1).replace(0.0, np.nan)
    frame["slow_share"] = abs_demands["slow_demand"].div(total_abs).fillna(0.0)
    frame["trend_share"] = abs_demands["trend_demand"].div(total_abs).fillna(0.0)
    frame["herd_share"] = abs_demands["herd_demand"].div(total_abs).fillna(0.0)

    rolling_peak = frame["sim_close"].cummax().replace(0.0, np.nan)
    frame["underwater_ratio"] = (rolling_peak - frame["sim_close"]).div(rolling_peak).fillna(0.0)
    frame["negative_run_5d"] = (-frame["sim_return"].rolling(5, min_periods=1).sum()).clip(lower=0.0)

    trap_signal = (
        0.65 * np.clip(frame["underwater_ratio"] / 0.08, 0.0, 1.0)
        + 0.35 * np.clip(frame["negative_run_5d"] / 0.05, 0.0, 1.0)
    )
    frame["w_hs"] = (frame["herd_share"] * trap_signal).clip(lower=0.0, upper=1.0)

    trend_lookback = max(int(params["trend_lookback"]), 1)
    frame["trend_persistence"] = frame["real_return"].rolling(trend_lookback, min_periods=1).mean().abs()
    frame["trend_strength_score"] = np.clip(
        frame["trend_persistence"] / max(float(params["entry_threshold"]), 1e-6),
        0.0,
        1.0,
    )
    frame["slow_streak"] = _compute_same_sign_streak(frame["slow_inventory"])
    frame["slow_streak_score"] = np.clip(frame["slow_streak"] / max(trend_lookback, 1), 0.0, 1.0)

    slow_sign = np.sign(frame["slow_inventory"])
    trend_sign = np.sign(frame["real_return"].rolling(trend_lookback, min_periods=1).mean())
    alignment = (slow_sign == trend_sign) & (slow_sign != 0)
    frame["alignment_score"] = alignment.astype(float)
    frame["w_st"] = (
        frame["slow_share"]
        * frame["slow_streak_score"]
        * frame["trend_strength_score"]
        * frame["alignment_score"]
    ).clip(lower=0.0, upper=1.0)
    return frame


def _summarize_mask(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return {
            "rows": 0,
            "w_hs_mean": 0.0,
            "w_hs_p95": 0.0,
            "w_hs_max": 0.0,
            "w_st_mean": 0.0,
            "w_st_p95": 0.0,
            "w_st_max": 0.0,
        }
    return {
        "rows": int(len(subset)),
        "w_hs_mean": float(subset["w_hs"].mean()),
        "w_hs_p95": float(subset["w_hs"].quantile(0.95)),
        "w_hs_max": float(subset["w_hs"].max()),
        "w_st_mean": float(subset["w_st"].mean()),
        "w_st_p95": float(subset["w_st"].quantile(0.95)),
        "w_st_max": float(subset["w_st"].max()),
    }


def _top_events(frame: pd.DataFrame, weight_col: str, top_n: int = 5) -> list[dict[str, object]]:
    top = frame.nlargest(top_n, weight_col)
    events: list[dict[str, object]] = []
    for _, row in top.iterrows():
        events.append(
            {
                "date": str(pd.Timestamp(row["date"]).date()),
                "weight": float(row[weight_col]),
                "sim_return": float(row["sim_return"]),
                "real_return": float(row["real_return"]),
                "herd_share": float(row["herd_share"]),
                "slow_share": float(row["slow_share"]),
                "trend_share": float(row["trend_share"]),
                "underwater_ratio": float(row["underwater_ratio"]),
                "slow_streak": float(row["slow_streak"]),
                "trend_strength_score": float(row["trend_strength_score"]),
            }
        )
    return events


def _diagnostic_flag(mean_value: float, p95_value: float, max_value: float) -> str:
    if max_value >= 0.30 or p95_value >= 0.20 or mean_value >= 0.08:
        return "signal_present"
    if max_value >= 0.12 or p95_value >= 0.08 or mean_value >= 0.03:
        return "weak_signal"
    return "near_silent"


def _build_report(stock_code: str, results_dir: Path, summary: dict[str, object]) -> str:
    overall = summary["overall"]
    by_day_type = summary["by_day_type"]
    lines = [
        "# Empty Role-Switch Diagnostic",
        "",
        f"- stock_code: `{stock_code}`",
        f"- results_dir: `{results_dir}`",
        f"- herd_to_slow_flag: `{overall['w_hs_flag']}`",
        f"- slow_to_trend_flag: `{overall['w_st_flag']}`",
        "",
        "## Reading",
        "",
        "This diagnostic does not fit switch parameters. It only asks whether the fixed-role best path already contains natural trigger conditions that would activate soft switching weights.",
        "",
        "## Overall",
        "",
        f"- w_hs_mean: `{overall['w_hs_mean']:.4f}`",
        f"- w_hs_p95: `{overall['w_hs_p95']:.4f}`",
        f"- w_hs_max: `{overall['w_hs_max']:.4f}`",
        f"- w_st_mean: `{overall['w_st_mean']:.4f}`",
        f"- w_st_p95: `{overall['w_st_p95']:.4f}`",
        f"- w_st_max: `{overall['w_st_max']:.4f}`",
        "",
        "## By Day Type",
        "",
    ]
    for day_type, values in by_day_type.items():
        lines.extend(
            [
                f"### {day_type}",
                "",
                f"- rows: `{int(values['rows'])}`",
                f"- w_hs_mean: `{float(values['w_hs_mean']):.4f}`",
                f"- w_hs_p95: `{float(values['w_hs_p95']):.4f}`",
                f"- w_hs_max: `{float(values['w_hs_max']):.4f}`",
                f"- w_st_mean: `{float(values['w_st_mean']):.4f}`",
                f"- w_st_p95: `{float(values['w_st_p95']):.4f}`",
                f"- w_st_max: `{float(values['w_st_max']):.4f}`",
                "",
            ]
        )
    lines.extend(["## Top herd -> slow dates", ""])
    for event in summary["top_w_hs_events"]:
        lines.append(
            f"- {event['date']}: weight=`{event['weight']:.4f}`, underwater_ratio=`{event['underwater_ratio']:.4f}`, herd_share=`{event['herd_share']:.4f}`"
        )
    lines.extend(["", "## Top slow -> trend dates", ""])
    for event in summary["top_w_st_events"]:
        lines.append(
            f"- {event['date']}: weight=`{event['weight']:.4f}`, slow_streak=`{event['slow_streak']:.1f}`, trend_strength_score=`{event['trend_strength_score']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If both flags are near_silent, the fixed-role best path is not naturally producing switch triggers and parameter search would likely be forcing a silent mechanism. If one or both flags show signal_present, then soft switching is plausibly data-driven and worth deeper investment.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run empty role-switch diagnostics on fixed-role best parameters")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config_single_stock_mvp.yaml"))
    parser.add_argument("--results-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else config.results.results_dir
    real_frame = load_stock_window(config)
    best_params = json.loads((results_dir / "best_params.json").read_text(encoding="utf-8"))
    simulation = simulate_path(real_frame, best_params, search_space=config.optimization.search_space)
    switch_frame = _compute_switch_frame(real_frame, simulation, best_params)

    day_types = _classify_day_types(real_frame)
    summary = {
        "stock_code": config.experiment.stock_code,
        "results_dir": str(results_dir),
        "overall": {
            "w_hs_mean": float(switch_frame["w_hs"].mean()),
            "w_hs_p95": float(switch_frame["w_hs"].quantile(0.95)),
            "w_hs_max": float(switch_frame["w_hs"].max()),
            "w_st_mean": float(switch_frame["w_st"].mean()),
            "w_st_p95": float(switch_frame["w_st"].quantile(0.95)),
            "w_st_max": float(switch_frame["w_st"].max()),
        },
        "by_day_type": {day_type: _summarize_mask(switch_frame, mask) for day_type, mask in day_types.items()},
        "top_w_hs_events": _top_events(switch_frame, "w_hs"),
        "top_w_st_events": _top_events(switch_frame, "w_st"),
    }
    summary["overall"]["w_hs_flag"] = _diagnostic_flag(
        summary["overall"]["w_hs_mean"],
        summary["overall"]["w_hs_p95"],
        summary["overall"]["w_hs_max"],
    )
    summary["overall"]["w_st_flag"] = _diagnostic_flag(
        summary["overall"]["w_st_mean"],
        summary["overall"]["w_st_p95"],
        summary["overall"]["w_st_max"],
    )

    json_path = results_dir / "role_switch_empty_diagnostic.json"
    md_path = results_dir / "role_switch_empty_diagnostic.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(_build_report(config.experiment.stock_code, results_dir, summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "w_hs_flag": summary["overall"]["w_hs_flag"],
                "w_st_flag": summary["overall"]["w_st_flag"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()