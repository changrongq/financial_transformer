from __future__ import annotations

from pathlib import Path
import json
import shutil
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_rl_market_simulation.agents import INTEGER_PARAMS, PARAMETER_BOUNDS
from inverse_rl_market_simulation.config import ExperimentConfig
from inverse_rl_market_simulation.objective import evaluate_simulation
from inverse_rl_market_simulation.simulator import simulate_path


AGENT_COLS = ["slow_demand", "trend_demand", "herd_demand"]
SWITCH_COLS = ["herd_to_slow_weight", "slow_to_trend_weight"]


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def run_stability_check(real_frame: pd.DataFrame, params: dict[str, float | int], stability_window_days: int, search_space: str = "fixed_role") -> dict[str, object]:
    n_obs = len(real_frame)
    midpoint = n_obs // 2
    earlier = real_frame.iloc[:midpoint].copy()
    later = real_frame.iloc[midpoint:].copy()
    checks: dict[str, object] = {
        "window_days": int(stability_window_days),
        "split_method": "earlier_vs_later_half",
        "windows": [],
    }
    for label, subset in [("earlier_half", earlier), ("later_half", later)]:
        if len(subset) < max(20, stability_window_days // 2):
            continue
        simulation = simulate_path(subset, params, search_space=search_space)
        result = evaluate_simulation(simulation, label=label)
        checks["windows"].append(
            {
                "label": label,
                "rows": int(len(subset)),
                "start_date": str(subset["date"].iloc[0].date()),
                "end_date": str(subset["date"].iloc[-1].date()),
                "total_loss": float(result["total_loss"]),
                "path_loss": float(result["path_loss"]),
                "vol_loss": float(result["vol_loss"]),
                "drawdown_loss": float(result["drawdown_loss"]),
            }
        )
    losses = [float(item["total_loss"]) for item in checks["windows"]]
    if losses:
        checks["loss_range"] = float(max(losses) - min(losses))
        checks["loss_mean"] = float(sum(losses) / len(losses))
    else:
        checks["loss_range"] = None
        checks["loss_mean"] = None
    return checks


def _write_plot(real_frame: pd.DataFrame, simulation: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(real_frame["date"], simulation["real_normalized_close"], label="real_path", linewidth=2.0)
    axes[0].plot(real_frame["date"], simulation["sim_normalized_close"], label="sim_path", linewidth=2.0)
    axes[0].set_ylabel("normalized close")
    axes[0].legend(loc="upper left")

    axes[1].plot(real_frame["date"], simulation["real_turnover_z"], label="real_turnover_z", linewidth=1.5)
    axes[1].plot(real_frame["date"], simulation["sim_turnover_z"], label="sim_turnover_z", linewidth=1.5)
    axes[1].set_ylabel("turnover z")
    axes[1].legend(loc="upper left")
    axes[1].set_xlabel("date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_markdown(
    output_path: Path,
    config: ExperimentConfig,
    best_result: dict[str, object],
    baseline_results: dict[str, dict[str, object]],
    stability: dict[str, object],
) -> None:
    lines = [
        "# Single-Stock MVP Interpretation",
        "",
        f"- stock_code: `{config.experiment.stock_code}`",
        f"- best_total_loss: `{float(best_result['total_loss']):.6f}`",
        f"- null_baseline_loss: `{float(baseline_results['null_baseline']['total_loss']):.6f}`",
        f"- hand_tuned_loss: `{float(baseline_results['hand_tuned']['total_loss']):.6f}`",
        "",
        "## Reading",
        "",
        "This MVP is successful only if the calibrated loss is clearly below both baselines and the half-window losses do not drift wildly.",
        "",
        "## Stability",
        "",
        f"- loss_range: `{stability.get('loss_range')}`",
        f"- loss_mean: `{stability.get('loss_mean')}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    corr = float(left.corr(right))
    if math.isnan(corr):
        return 0.0
    return corr


def _summarize_switch_slice(simulation: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    subset = simulation.loc[mask].copy()
    if subset.empty:
        return {
            "rows": 0,
            "herd_to_slow_mean": 0.0,
            "herd_to_slow_p95": 0.0,
            "herd_to_slow_max": 0.0,
            "slow_to_trend_mean": 0.0,
            "slow_to_trend_p95": 0.0,
            "slow_to_trend_max": 0.0,
        }
    return {
        "rows": int(len(subset)),
        "herd_to_slow_mean": float(subset["herd_to_slow_weight"].mean()),
        "herd_to_slow_p95": float(subset["herd_to_slow_weight"].quantile(0.95)),
        "herd_to_slow_max": float(subset["herd_to_slow_weight"].max()),
        "slow_to_trend_mean": float(subset["slow_to_trend_weight"].mean()),
        "slow_to_trend_p95": float(subset["slow_to_trend_weight"].quantile(0.95)),
        "slow_to_trend_max": float(subset["slow_to_trend_weight"].max()),
    }


def _top_switch_events(simulation: pd.DataFrame, weight_col: str, top_n: int = 5) -> list[dict[str, object]]:
    top = simulation.nlargest(top_n, weight_col)
    rows: list[dict[str, object]] = []
    for _, item in top.iterrows():
        rows.append(
            {
                "date": str(pd.Timestamp(item["date"]).date()),
                "weight": float(item[weight_col]),
                "real_return": float(item["real_return"]),
                "sim_return": float(item["sim_return"]),
                "trap_state": float(item.get("trap_state", 0.0)),
                "trend_switch_state": float(item.get("trend_switch_state", 0.0)),
            }
        )
    return rows


def summarize_switch_behavior(real_frame: pd.DataFrame, simulation: pd.DataFrame) -> dict[str, object]:
    day_types = _classify_day_types(real_frame)
    return {
        "overall": _summarize_switch_slice(simulation, pd.Series(True, index=simulation.index)),
        "by_day_type": {
            name: _summarize_switch_slice(simulation, mask) for name, mask in day_types.items()
        },
        "top_herd_to_slow_events": _top_switch_events(simulation, "herd_to_slow_weight"),
        "top_slow_to_trend_events": _top_switch_events(simulation, "slow_to_trend_weight"),
    }


def _write_switch_report(output_path: Path, config: ExperimentConfig, switch_summary: dict[str, object]) -> None:
    overall = switch_summary["overall"]
    lines = [
        "# Soft-Switch Behavior Report",
        "",
        f"- stock_code: `{config.experiment.stock_code}`",
        f"- search_space: `{config.optimization.search_space}`",
        "",
        "## Overall",
        "",
        f"- herd_to_slow_mean: `{float(overall['herd_to_slow_mean']):.4f}`",
        f"- herd_to_slow_p95: `{float(overall['herd_to_slow_p95']):.4f}`",
        f"- herd_to_slow_max: `{float(overall['herd_to_slow_max']):.4f}`",
        f"- slow_to_trend_mean: `{float(overall['slow_to_trend_mean']):.4f}`",
        f"- slow_to_trend_p95: `{float(overall['slow_to_trend_p95']):.4f}`",
        f"- slow_to_trend_max: `{float(overall['slow_to_trend_max']):.4f}`",
        "",
    ]
    for day_type, summary in switch_summary["by_day_type"].items():
        lines.extend(
            [
                f"## {day_type}",
                "",
                f"- rows: `{int(summary['rows'])}`",
                f"- herd_to_slow_mean: `{float(summary['herd_to_slow_mean']):.4f}`",
                f"- herd_to_slow_p95: `{float(summary['herd_to_slow_p95']):.4f}`",
                f"- herd_to_slow_max: `{float(summary['herd_to_slow_max']):.4f}`",
                f"- slow_to_trend_mean: `{float(summary['slow_to_trend_mean']):.4f}`",
                f"- slow_to_trend_p95: `{float(summary['slow_to_trend_p95']):.4f}`",
                f"- slow_to_trend_max: `{float(summary['slow_to_trend_max']):.4f}`",
                "",
            ]
        )
    lines.extend(["## Top herd -> slow dates", ""])
    for event in switch_summary["top_herd_to_slow_events"]:
        lines.append(
            f"- {event['date']}: weight=`{event['weight']:.4f}`, trap_state=`{event['trap_state']:.4f}`, sim_return=`{event['sim_return']:.4f}`"
        )
    lines.extend(["", "## Top slow -> trend dates", ""])
    for event in switch_summary["top_slow_to_trend_events"]:
        lines.append(
            f"- {event['date']}: weight=`{event['weight']:.4f}`, trend_switch_state=`{event['trend_switch_state']:.4f}`, sim_return=`{event['sim_return']:.4f}`"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_agent_behavior(simulation: pd.DataFrame) -> dict[str, object]:
    demand_cols = ["slow_demand", "trend_demand", "herd_demand"]
    abs_demands = simulation[demand_cols].abs()
    total_abs = abs_demands.sum(axis=1)
    dominant = abs_demands.idxmax(axis=1)
    summary: dict[str, object] = {
        "rows": int(len(simulation)),
        "agents": {},
        "pairwise_demand_corr": {
            "slow_vs_trend": _safe_corr(simulation["slow_demand"], simulation["trend_demand"]),
            "slow_vs_herd": _safe_corr(simulation["slow_demand"], simulation["herd_demand"]),
            "trend_vs_herd": _safe_corr(simulation["trend_demand"], simulation["herd_demand"]),
        },
        "dominant_agent_share": {
            agent: float((dominant == agent).mean()) for agent in demand_cols
        },
        "inventory": {
            "slow_inventory_abs_mean": float(simulation["slow_inventory"].abs().mean()),
            "trend_inventory_abs_mean": float(simulation["trend_inventory"].abs().mean()),
            "total_inventory_abs_mean": float(simulation["total_inventory"].abs().mean()),
            "slow_inventory_range": float(simulation["slow_inventory"].max() - simulation["slow_inventory"].min()),
            "trend_inventory_range": float(simulation["trend_inventory"].max() - simulation["trend_inventory"].min()),
            "total_inventory_range": float(simulation["total_inventory"].max() - simulation["total_inventory"].min()),
        },
    }
    for agent in demand_cols:
        abs_series = abs_demands[agent]
        active_mask = abs_series > 1e-8
        share = abs_series.div(total_abs.replace(0.0, np.nan)).fillna(0.0)
        summary["agents"][agent] = {
            "active_days": int(active_mask.sum()),
            "active_ratio": float(active_mask.mean()),
            "abs_mean": float(abs_series.mean()),
            "abs_std": float(abs_series.std(ddof=0)),
            "abs_share_mean": float(share.mean()),
            "signed_mean": float(simulation[agent].mean()),
            "signed_std": float(simulation[agent].std(ddof=0)),
            "corr_with_real_return": _safe_corr(simulation[agent], simulation["real_return"]),
            "corr_with_sim_return": _safe_corr(simulation[agent], simulation["sim_return"]),
        }
    return summary


def summarize_trial_convergence(trials: pd.DataFrame, top_k: int = 10) -> dict[str, object]:
    top = trials.head(top_k).copy()
    summary: dict[str, object] = {
        "top_k": int(min(top_k, len(top))),
        "loss_min": float(top["total_loss"].min()),
        "loss_max": float(top["total_loss"].max()),
        "loss_range": float(top["total_loss"].max() - top["total_loss"].min()),
        "parameters": {},
    }
    for name, (low, high) in PARAMETER_BOUNDS.items():
        series = top[name].astype(float)
        span = float(high) - float(low)
        relative_range = 0.0 if span == 0.0 else float((series.max() - series.min()) / span)
        summary["parameters"][name] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "relative_range": relative_range,
        }
    return summary


def _write_agent_plot(simulation: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(simulation["date"], simulation["slow_demand"], label="slow_demand", linewidth=1.5)
    axes[0].plot(simulation["date"], simulation["trend_demand"], label="trend_demand", linewidth=1.5)
    axes[0].plot(simulation["date"], simulation["herd_demand"], label="herd_demand", linewidth=1.5)
    axes[0].set_ylabel("demand")
    axes[0].legend(loc="upper left")

    axes[1].plot(simulation["date"], simulation["slow_inventory"], label="slow_inventory", linewidth=1.5)
    axes[1].plot(simulation["date"], simulation["trend_inventory"], label="trend_inventory", linewidth=1.5)
    axes[1].plot(simulation["date"], simulation["total_inventory"], label="total_inventory", linewidth=1.5)
    axes[1].set_ylabel("inventory")
    axes[1].set_xlabel("date")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_role_activity_vectors(simulation: pd.DataFrame, stock_code: str | None = None) -> pd.DataFrame:
    demand_matrix = simulation[AGENT_COLS].astype(float).abs()
    total_activity = demand_matrix.sum(axis=1)
    normalized = demand_matrix.div(total_activity.replace(0.0, np.nan), axis=0).fillna(0.0)

    dominant_role = demand_matrix.idxmax(axis=1).str.replace("_demand", "", regex=False)
    dominant_role = dominant_role.where(total_activity > 1e-9, "none")

    role_vectors = pd.DataFrame(
        {
            "date": pd.to_datetime(simulation["date"]),
            "stock_code": stock_code or "",
            "herd_act": normalized["herd_demand"],
            "slow_act": normalized["slow_demand"],
            "trend_act": normalized["trend_demand"],
            "total_activity": total_activity,
            "dominant_role": dominant_role,
        }
    )
    for column in SWITCH_COLS:
        if column in simulation.columns:
            role_vectors[column] = simulation[column].astype(float)
    return role_vectors


def _write_role_heatmap_plot(simulation: pd.DataFrame, output_path: Path) -> None:
    prices = simulation["real_close"].astype(float).to_numpy()
    dates = pd.to_datetime(simulation["date"])
    demand_matrix = simulation[AGENT_COLS].astype(float).abs().to_numpy()
    demand_sum = demand_matrix.sum(axis=1, keepdims=True)
    role_share = np.divide(
        demand_matrix,
        np.maximum(demand_sum, 1e-9),
        out=np.zeros_like(demand_matrix),
        where=demand_sum > 0.0,
    )

    total_activity = demand_sum[:, 0]
    activity_scale = np.divide(
        total_activity,
        max(float(np.quantile(total_activity, 0.95)), 1e-9),
        out=np.zeros_like(total_activity),
        where=total_activity >= 0.0,
    )
    activity_scale = np.clip(activity_scale, 0.0, 1.0)
    activity_scale = np.where(activity_scale > 0.0, 0.22 + 0.78 * activity_scale, 0.0)

    price_min = float(prices.min())
    price_max = float(prices.max())
    price_span = max(price_max - price_min, 1e-6)
    padding = 0.05 * price_span
    price_grid = np.linspace(price_min - padding, price_max + padding, 96)
    band_width = max(0.055 * price_span, float(np.median(prices)) * 0.004, 1e-6)

    # RGB channels: herd red, slow blue, trend green.
    role_palette = np.array(
        [
            [0.16, 0.42, 0.86],
            [0.17, 0.72, 0.31],
            [0.88, 0.23, 0.20],
        ],
        dtype=float,
    )
    # Reorder to match AGENT_COLS = slow, trend, herd.
    role_palette = np.array([role_palette[0], role_palette[1], role_palette[2]], dtype=float)

    heatmap = np.zeros((len(price_grid), len(simulation), 3), dtype=float)
    for idx, price in enumerate(prices):
        gaussian_band = np.exp(-0.5 * ((price_grid - price) / band_width) ** 2)
        color = role_share[idx] @ role_palette
        heatmap[:, idx, :] = np.outer(gaussian_band * activity_scale[idx], color)

    heatmap = np.power(np.clip(heatmap, 0.0, 1.0), 0.72)

    w_hs = simulation["herd_to_slow_weight"].astype(float)
    w_st = simulation["slow_to_trend_weight"].astype(float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.5, 1.2]},
    )

    x_min = dates.iloc[0].to_pydatetime()
    x_max = dates.iloc[-1].to_pydatetime()
    axes[0].imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[matplotlib.dates.date2num(x_min), matplotlib.dates.date2num(x_max), price_grid[0], price_grid[-1]],
        interpolation="bilinear",
    )
    axes[0].set_facecolor("#03050A")
    axes[0].plot(dates, prices, color="#F5F7FA", linewidth=1.35, alpha=0.9, label="real_close")
    axes[0].set_ylabel("price")
    axes[0].set_title("Role Heatmap Beta: red=herd, blue=slow, green=trend")
    axes[0].legend(loc="upper left")

    axes[1].plot(dates, w_hs, color="#d14a61", linewidth=1.5, label="w_hs")
    axes[1].plot(dates, w_st, color="#2f7ed8", linewidth=1.5, label="w_st")
    axes[1].set_ylabel("switch")
    axes[1].set_xlabel("date")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_agent_report(
    output_path: Path,
    config: ExperimentConfig,
    best_result: dict[str, object],
    agent_summary: dict[str, object],
    convergence_summary: dict[str, object],
) -> None:
    agent_block = agent_summary["agents"]
    lines = [
        "# Agent Differentiation Report",
        "",
        f"- stock_code: `{config.experiment.stock_code}`",
        f"- best_total_loss: `{float(best_result['total_loss']):.6f}`",
        f"- volatility_ratio: `{float(best_result['volatility_ratio']):.6f}`",
        "",
        "## Agent Activity",
        "",
        f"- slow active_ratio: `{float(agent_block['slow_demand']['active_ratio']):.4f}`",
        f"- trend active_ratio: `{float(agent_block['trend_demand']['active_ratio']):.4f}`",
        f"- herd active_ratio: `{float(agent_block['herd_demand']['active_ratio']):.4f}`",
        f"- dominant slow share: `{float(agent_summary['dominant_agent_share']['slow_demand']):.4f}`",
        f"- dominant trend share: `{float(agent_summary['dominant_agent_share']['trend_demand']):.4f}`",
        f"- dominant herd share: `{float(agent_summary['dominant_agent_share']['herd_demand']):.4f}`",
        "",
        "## Pairwise Demand Correlation",
        "",
        f"- slow_vs_trend: `{float(agent_summary['pairwise_demand_corr']['slow_vs_trend']):.4f}`",
        f"- slow_vs_herd: `{float(agent_summary['pairwise_demand_corr']['slow_vs_herd']):.4f}`",
        f"- trend_vs_herd: `{float(agent_summary['pairwise_demand_corr']['trend_vs_herd']):.4f}`",
        "",
        "## Convergence Read",
        "",
        f"- top_k: `{int(convergence_summary['top_k'])}`",
        f"- loss_range: `{float(convergence_summary['loss_range']):.6f}`",
        "",
        "Interpretation rule:",
        "Different agents should remain active on meaningfully different fractions of days, show different demand magnitudes, and avoid collapsing into near-identical correlations or parameter ranges.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


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


def _summarize_role_slice(simulation: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
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


def _aggregate_role_day_types(trial_summaries: list[dict[str, object]], day_type_names: list[str]) -> dict[str, object]:
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


def summarize_top_trial_roles(real_frame: pd.DataFrame, trials: pd.DataFrame, top_k: int = 10, search_space: str = "fixed_role") -> dict[str, object]:
    top_trials = trials.head(top_k).copy()
    day_types = _classify_day_types(real_frame)
    day_type_names = list(day_types.keys())
    trial_summaries: list[dict[str, object]] = []

    for _, trial in top_trials.iterrows():
        params = {
            key: (int(trial[key]) if key in INTEGER_PARAMS else float(trial[key]))
            for key in PARAMETER_BOUNDS.keys()
        }
        simulation = simulate_path(real_frame, params, search_space=search_space)
        evaluated = evaluate_simulation(simulation, label=str(trial["label"]))
        trial_summaries.append(
            {
                "trial_id": int(trial["trial_id"]),
                "label": str(trial["label"]),
                "total_loss": float(evaluated["total_loss"]),
                "day_type_summary": {
                    day_type: _summarize_role_slice(simulation, mask) for day_type, mask in day_types.items()
                },
            }
        )

    return {
        "top_k": int(len(trial_summaries)),
        "trial_summaries": trial_summaries,
        "aggregate_day_type_summary": _aggregate_role_day_types(trial_summaries, day_type_names),
    }


def _write_top_trial_role_report(
    output_path: Path,
    config: ExperimentConfig,
    best_result: dict[str, object],
    role_summary: dict[str, object],
) -> None:
    aggregate = role_summary["aggregate_day_type_summary"]
    best_trial = role_summary["trial_summaries"][0]
    lines = [
        "# Top-Trial Agent Role Analysis",
        "",
        f"- stock_code: `{config.experiment.stock_code}`",
        f"- top_k: `{int(role_summary['top_k'])}`",
        f"- best_trial_id: `{best_trial['trial_id']}`",
        f"- best_total_loss: `{float(best_result['total_loss']):.6f}`",
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
    output_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_trend_convergence_drift(real_frame: pd.DataFrame, trials: pd.DataFrame, top_ks: tuple[int, ...] = (3, 5, 10), search_space: str = "fixed_role") -> dict[str, object]:
    trend_progression: dict[str, object] = {}
    for top_k in top_ks:
        role_summary = summarize_top_trial_roles(real_frame, trials, top_k=top_k, search_space=search_space)
        aggregate = role_summary["aggregate_day_type_summary"]
        trend_progression[str(top_k)] = {
            "top_k": int(role_summary["top_k"]),
            "all_days": {
                "trend_abs_share_mean": float(aggregate["all_days"]["agent_abs_share_mean"]["trend_demand"]),
                "trend_active_ratio": float(aggregate["all_days"]["agent_active_ratio"]["trend_demand"]),
                "trend_dominant_share": float(aggregate["all_days"]["dominant_agent_share"]["trend_demand"]),
            },
            "turnover_spike": {
                "trend_abs_share_mean": float(aggregate["turnover_spike"]["agent_abs_share_mean"]["trend_demand"]),
                "trend_active_ratio": float(aggregate["turnover_spike"]["agent_active_ratio"]["trend_demand"]),
            },
            "strong_up_with_turnover": {
                "trend_abs_share_mean": float(aggregate["strong_up_with_turnover"]["agent_abs_share_mean"]["trend_demand"]),
                "trend_active_ratio": float(aggregate["strong_up_with_turnover"]["agent_active_ratio"]["trend_demand"]),
            },
            "strong_down_with_turnover": {
                "trend_abs_share_mean": float(aggregate["strong_down_with_turnover"]["agent_abs_share_mean"]["trend_demand"]),
                "trend_active_ratio": float(aggregate["strong_down_with_turnover"]["agent_active_ratio"]["trend_demand"]),
            },
            "loss_window": {
                "min_total_loss": float(trials.head(top_k)["total_loss"].min()),
                "max_total_loss": float(trials.head(top_k)["total_loss"].max()),
                "loss_range": float(trials.head(top_k)["total_loss"].max() - trials.head(top_k)["total_loss"].min()),
            },
        }
    return {
        "top_k_progression": trend_progression,
        "interpretation": "If trend metrics stay high from top-3 to top-10, trend participation is converging robustly. If they decay sharply as top_k widens, the trend role is still fragile across nearby good solutions.",
    }


def _write_trend_convergence_report(output_path: Path, drift_summary: dict[str, object]) -> None:
    lines = [
        "# Trend Convergence Drift Report",
        "",
        "This report compares trend participation across top-3, top-5, and top-10 trials.",
        "",
    ]
    for key, summary in drift_summary["top_k_progression"].items():
        lines.extend(
            [
                f"## top_{key}",
                "",
                f"- all_days trend_abs_share_mean: `{float(summary['all_days']['trend_abs_share_mean']):.4f}`",
                f"- all_days trend_active_ratio: `{float(summary['all_days']['trend_active_ratio']):.4f}`",
                f"- all_days trend_dominant_share: `{float(summary['all_days']['trend_dominant_share']):.4f}`",
                f"- turnover_spike trend_abs_share_mean: `{float(summary['turnover_spike']['trend_abs_share_mean']):.4f}`",
                f"- turnover_spike trend_active_ratio: `{float(summary['turnover_spike']['trend_active_ratio']):.4f}`",
                f"- strong_up_with_turnover trend_abs_share_mean: `{float(summary['strong_up_with_turnover']['trend_abs_share_mean']):.4f}`",
                f"- strong_up_with_turnover trend_active_ratio: `{float(summary['strong_up_with_turnover']['trend_active_ratio']):.4f}`",
                f"- strong_down_with_turnover trend_abs_share_mean: `{float(summary['strong_down_with_turnover']['trend_abs_share_mean']):.4f}`",
                f"- strong_down_with_turnover trend_active_ratio: `{float(summary['strong_down_with_turnover']['trend_active_ratio']):.4f}`",
                f"- loss_range: `{float(summary['loss_window']['loss_range']):.6f}`",
                "",
            ]
        )
    lines.extend(["## Interpretation", "", str(drift_summary["interpretation"])])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    config: ExperimentConfig,
    real_frame: pd.DataFrame,
    best_simulation: pd.DataFrame,
    best_params: dict[str, float | int],
    best_result: dict[str, object],
    baseline_results: dict[str, dict[str, object]],
    trials: pd.DataFrame,
    stability: dict[str, object],
    source_config: Path,
) -> dict[str, str]:
    output_dir = config.results.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_config = output_dir / "config.yaml"
    shutil.copyfile(source_config, copied_config)

    best_params_path = output_dir / "best_params.json"
    fit_summary_path = output_dir / "fit_summary.json"
    simulation_path = output_dir / "simulated_path.csv"
    trials_path = output_dir / "calibration_trials.csv"
    stability_path = output_dir / "stability_check.json"
    plot_path = output_dir / "real_vs_sim_plot.png"
    note_path = output_dir / "interpretation.md"
    agent_summary_path = output_dir / "agent_differentiation_summary.json"
    agent_report_path = output_dir / "agent_differentiation_report.md"
    agent_plot_path = output_dir / "agent_behavior_plot.png"
    role_activity_path = output_dir / "role_activity_vectors.csv"
    role_heatmap_path = output_dir / "role_heatmap_beta.png"
    role_summary_path = output_dir / "top_trial_role_analysis.json"
    role_report_path = output_dir / "top_trial_role_analysis.md"
    trend_drift_summary_path = output_dir / "trend_convergence_drift.json"
    trend_drift_report_path = output_dir / "trend_convergence_drift.md"
    switch_summary_path = output_dir / "soft_switch_behavior_summary.json"
    switch_report_path = output_dir / "soft_switch_behavior_report.md"

    best_params_path.write_text(json.dumps(best_params, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    fit_summary = {
        "stock_code": config.experiment.stock_code,
        "rows": int(len(real_frame)),
        "best_result": best_result,
        "baseline_results": baseline_results,
        "top_trials": trials.head(10).to_dict(orient="records"),
    }
    fit_summary_path.write_text(json.dumps(fit_summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    best_simulation.to_csv(simulation_path, index=False, encoding="utf-8-sig")
    build_role_activity_vectors(best_simulation, config.experiment.stock_code).to_csv(
        role_activity_path,
        index=False,
        encoding="utf-8-sig",
    )
    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    stability_path.write_text(json.dumps(stability, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    agent_summary = {
        "best_agent_behavior": summarize_agent_behavior(best_simulation),
        "top_trial_convergence": summarize_trial_convergence(trials),
    }
    role_summary = summarize_top_trial_roles(real_frame, trials, search_space=config.optimization.search_space)
    trend_drift_summary = summarize_trend_convergence_drift(real_frame, trials, search_space=config.optimization.search_space)
    switch_summary = summarize_switch_behavior(real_frame, best_simulation)
    agent_summary_path.write_text(json.dumps(agent_summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    role_summary_path.write_text(json.dumps(role_summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    trend_drift_summary_path.write_text(json.dumps(trend_drift_summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    switch_summary_path.write_text(json.dumps(switch_summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_plot(real_frame, best_simulation, plot_path)
    _write_markdown(note_path, config, best_result, baseline_results, stability)
    _write_agent_plot(best_simulation, agent_plot_path)
    _write_role_heatmap_plot(best_simulation, role_heatmap_path)
    _write_agent_report(
        agent_report_path,
        config,
        best_result,
        agent_summary["best_agent_behavior"],
        agent_summary["top_trial_convergence"],
    )
    _write_top_trial_role_report(role_report_path, config, best_result, role_summary)
    _write_trend_convergence_report(trend_drift_report_path, trend_drift_summary)
    _write_switch_report(switch_report_path, config, switch_summary)

    return {
        "results_dir": str(output_dir),
        "best_params": str(best_params_path),
        "fit_summary": str(fit_summary_path),
        "simulated_path": str(simulation_path),
        "calibration_trials": str(trials_path),
        "stability_check": str(stability_path),
        "plot": str(plot_path),
        "interpretation": str(note_path),
        "agent_summary": str(agent_summary_path),
        "agent_report": str(agent_report_path),
        "agent_plot": str(agent_plot_path),
        "role_activity_vectors": str(role_activity_path),
        "role_heatmap": str(role_heatmap_path),
        "role_summary": str(role_summary_path),
        "role_report": str(role_report_path),
        "trend_drift_summary": str(trend_drift_summary_path),
        "trend_drift_report": str(trend_drift_report_path),
        "switch_summary": str(switch_summary_path),
        "switch_report": str(switch_report_path),
    }