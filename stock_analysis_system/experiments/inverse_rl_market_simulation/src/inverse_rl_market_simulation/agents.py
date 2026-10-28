from __future__ import annotations

from collections import OrderedDict
import math


PARAMETER_BOUNDS = OrderedDict(
    [
        ("anchor_window", (10, 60)),
        ("mean_reversion_strength", (0.05, 1.50)),
        ("inventory_inertia", (0.00, 1.00)),
        ("trend_lookback", (3, 20)),
        ("trend_strength", (0.05, 1.50)),
        ("entry_threshold", (0.001, 0.040)),
        ("herd_strength", (0.05, 1.50)),
        ("attention_to_turnover", (0.00, 3.00)),
        ("panic_asymmetry", (0.00, 2.00)),
        ("trap_to_slow_strength", (0.00, 1.00)),
        ("trap_to_slow_threshold", (0.05, 0.95)),
        ("slow_to_trend_strength", (0.00, 1.00)),
        ("slow_to_trend_threshold", (0.05, 0.95)),
    ]
)


BASE_PARAMETER_NAMES = (
    "anchor_window",
    "mean_reversion_strength",
    "inventory_inertia",
    "trend_lookback",
    "trend_strength",
    "entry_threshold",
    "herd_strength",
    "attention_to_turnover",
    "panic_asymmetry",
)


SWITCH_PARAMETER_NAMES = (
    "trap_to_slow_strength",
    "trap_to_slow_threshold",
    "slow_to_trend_strength",
    "slow_to_trend_threshold",
)


SEARCH_SPACE_PARAM_NAMES = {
    "fixed_role": BASE_PARAMETER_NAMES,
    "soft_switch_v1": BASE_PARAMETER_NAMES + SWITCH_PARAMETER_NAMES,
    "soft_switch_v2_directional_probe": BASE_PARAMETER_NAMES + SWITCH_PARAMETER_NAMES,
}


INTEGER_PARAMS = {"anchor_window", "trend_lookback"}


DEFAULT_HAND_TUNED_PARAMS = {
    "anchor_window": 30,
    "mean_reversion_strength": 0.55,
    "inventory_inertia": 0.35,
    "trend_lookback": 8,
    "trend_strength": 0.70,
    "entry_threshold": 0.008,
    "herd_strength": 0.50,
    "attention_to_turnover": 1.10,
    "panic_asymmetry": 0.60,
    "trap_to_slow_strength": 0.0,
    "trap_to_slow_threshold": 0.35,
    "slow_to_trend_strength": 0.0,
    "slow_to_trend_threshold": 0.40,
}


def normalize_params(params: dict[str, float]) -> dict[str, float | int]:
    normalized: dict[str, float | int] = {}
    merged_params = dict(DEFAULT_HAND_TUNED_PARAMS)
    merged_params.update(params)
    for name, (low, high) in PARAMETER_BOUNDS.items():
        value = merged_params[name]
        low, high = PARAMETER_BOUNDS[name]
        clipped = min(max(float(value), float(low)), float(high))
        if name in INTEGER_PARAMS:
            normalized[name] = int(round(clipped))
        else:
            normalized[name] = clipped
    return normalized


def get_search_space_param_names(search_space: str) -> tuple[str, ...]:
    try:
        return SEARCH_SPACE_PARAM_NAMES[search_space]
    except KeyError as exc:
        raise ValueError(f"Unsupported optimization.search_space: {search_space}") from exc


def compute_agent_demands(
    anchor_gap: float,
    trend_signal: float,
    previous_return: float,
    turnover_shock: float,
    slow_inventory: float,
    trend_inventory: float,
    params: dict[str, float | int],
) -> dict[str, float]:
    slow_demand = (
        -float(params["mean_reversion_strength"]) * anchor_gap
        - float(params["inventory_inertia"]) * slow_inventory
    )

    trend_lookback = max(int(params["trend_lookback"]), 1)
    threshold = float(params["entry_threshold"]) / math.sqrt(trend_lookback)
    if abs(trend_signal) <= threshold:
        active_trend = 0.0
    else:
        active_trend = float((abs(trend_signal) - threshold) * (1.0 if trend_signal >= 0 else -1.0))
    trend_demand = float(params["trend_strength"]) * active_trend - 0.15 * trend_inventory

    herd_multiplier = 1.0 + float(params["attention_to_turnover"]) * max(turnover_shock, 0.0)
    if previous_return < 0.0:
        herd_multiplier *= 1.0 + float(params["panic_asymmetry"])
    herd_demand = float(params["herd_strength"]) * previous_return * herd_multiplier

    return {
        "slow_demand": slow_demand,
        "trend_demand": trend_demand,
        "herd_demand": herd_demand,
    }