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
from inverse_rl_market_simulation.simulator import simulate_path


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


def _compute_probe_frame(real_frame: pd.DataFrame, simulation: pd.DataFrame, params: dict[str, float | int]) -> pd.DataFrame:
    frame = simulation.copy()
    abs_demands = frame[["slow_demand", "trend_demand", "herd_demand"]].abs()
    total_abs = abs_demands.sum(axis=1).replace(0.0, np.nan)
    frame["slow_share"] = abs_demands["slow_demand"].div(total_abs).fillna(0.0)

    trend_lookback = max(int(params["trend_lookback"]), 1)
    trend_signal = frame["real_return"].rolling(trend_lookback, min_periods=1).mean()
    slow_streak = _compute_same_sign_streak(frame["slow_inventory"])
    trend_streak = _compute_same_sign_streak(trend_signal)

    frame["slow_survival_score"] = np.clip(slow_streak / trend_lookback, 0.0, 1.0)
    frame["trend_persistence_score"] = np.clip(trend_streak / trend_lookback, 0.0, 1.0)
    frame["trend_strength_score"] = np.clip(trend_signal.abs() / max(float(params["entry_threshold"]), 1e-6), 0.0, 1.0)

    slow_sign = np.sign(frame["slow_inventory"])
    trend_sign = np.sign(trend_signal)
    frame["sign_consistency_score"] = ((slow_sign == trend_sign) & (slow_sign != 0)).astype(float)
    frame["directional_probe_state"] = (
        frame["slow_survival_score"]
        * frame["trend_persistence_score"]
        * frame["trend_strength_score"]
        * frame["sign_consistency_score"]
    ).clip(lower=0.0, upper=1.0)
    frame["directional_probe_weight"] = (
        frame["slow_share"] * frame["directional_probe_state"]
    ).clip(lower=0.0, upper=1.0)
    return frame


def _summarize_mask(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return {
            "rows": 0,
            "probe_mean": 0.0,
            "probe_p95": 0.0,
            "probe_max": 0.0,
            "state_mean": 0.0,
            "state_p95": 0.0,
            "state_max": 0.0,
        }
    return {
        "rows": int(len(subset)),
        "probe_mean": float(subset["directional_probe_weight"].mean()),
        "probe_p95": float(subset["directional_probe_weight"].quantile(0.95)),
        "probe_max": float(subset["directional_probe_weight"].max()),
        "state_mean": float(subset["directional_probe_state"].mean()),
        "state_p95": float(subset["directional_probe_state"].quantile(0.95)),
        "state_max": float(subset["directional_probe_state"].max()),
    }


def _top_events(frame: pd.DataFrame, top_n: int = 5) -> list[dict[str, object]]:
    top = frame.nlargest(top_n, "directional_probe_weight")
    events: list[dict[str, object]] = []
    for _, row in top.iterrows():
        events.append(
            {
                "date": str(pd.Timestamp(row["date"]).date()),
                "probe_weight": float(row["directional_probe_weight"]),
                "probe_state": float(row["directional_probe_state"]),
                "slow_share": float(row["slow_share"]),
                "slow_survival_score": float(row["slow_survival_score"]),
                "trend_persistence_score": float(row["trend_persistence_score"]),
                "trend_strength_score": float(row["trend_strength_score"]),
                "sign_consistency_score": float(row["sign_consistency_score"]),
                "sim_return": float(row["sim_return"]),
                "real_return": float(row["real_return"]),
            }
        )
    return events


def _build_report(stock_code: str, results_dir: Path, summary: dict[str, object]) -> str:
    overall = summary["overall"]
    lines = [
        "# Slow-To-Trend Directional Persistence Probe",
        "",
        f"- stock_code: `{stock_code}`",
        f"- results_dir: `{results_dir}`",
        "",
        "## Reading",
        "",
        "This is an offline replay diagnostic only. It does not change search or fit new parameters. It asks whether a sparse directional-persistence proxy spikes in the already-known successful windows.",
        "",
        "## Overall",
        "",
        f"- probe_mean: `{overall['probe_mean']:.4f}`",
        f"- probe_p95: `{overall['probe_p95']:.4f}`",
        f"- probe_max: `{overall['probe_max']:.4f}`",
        f"- state_mean: `{overall['state_mean']:.4f}`",
        f"- state_p95: `{overall['state_p95']:.4f}`",
        f"- state_max: `{overall['state_max']:.4f}`",
        "",
        "## By Day Type",
        "",
    ]
    for day_type, values in summary["by_day_type"].items():
        lines.extend(
            [
                f"### {day_type}",
                "",
                f"- rows: `{int(values['rows'])}`",
                f"- probe_mean: `{float(values['probe_mean']):.4f}`",
                f"- probe_p95: `{float(values['probe_p95']):.4f}`",
                f"- probe_max: `{float(values['probe_max']):.4f}`",
                f"- state_mean: `{float(values['state_mean']):.4f}`",
                f"- state_p95: `{float(values['state_p95']):.4f}`",
                f"- state_max: `{float(values['state_max']):.4f}`",
                "",
            ]
        )
    lines.extend(["## Top Probe Dates", ""])
    for event in summary["top_probe_events"]:
        lines.append(
            f"- {event['date']}: weight=`{event['probe_weight']:.4f}`, state=`{event['probe_state']:.4f}`, slow_survival=`{event['slow_survival_score']:.4f}`, trend_persistence=`{event['trend_persistence_score']:.4f}`, sign_consistency=`{event['sign_consistency_score']:.4f}`"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline directional persistence probe for slow->trend")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config_single_stock_mvp.yaml"))
    parser.add_argument("--results-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else config.results.results_dir
    real_frame = load_stock_window(config)
    best_params = json.loads((results_dir / "best_params.json").read_text(encoding="utf-8"))
    simulation = simulate_path(real_frame, best_params, search_space=config.optimization.search_space)
    probe_frame = _compute_probe_frame(real_frame, simulation, best_params)

    day_types = _classify_day_types(real_frame)
    summary = {
        "stock_code": config.experiment.stock_code,
        "results_dir": str(results_dir),
        "overall": _summarize_mask(probe_frame, pd.Series(True, index=probe_frame.index)),
        "by_day_type": {name: _summarize_mask(probe_frame, mask) for name, mask in day_types.items()},
        "top_probe_events": _top_events(probe_frame),
    }

    json_path = results_dir / "slow_to_trend_directional_probe.json"
    md_path = results_dir / "slow_to_trend_directional_probe.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(_build_report(config.experiment.stock_code, results_dir, summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "probe_mean": summary["overall"]["probe_mean"],
                "probe_max": summary["overall"]["probe_max"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()