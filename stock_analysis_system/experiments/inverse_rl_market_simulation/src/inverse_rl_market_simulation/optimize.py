from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd

from inverse_rl_market_simulation.agents import DEFAULT_HAND_TUNED_PARAMS, INTEGER_PARAMS, PARAMETER_BOUNDS, get_search_space_param_names, normalize_params
from inverse_rl_market_simulation.config import ExperimentConfig
from inverse_rl_market_simulation.objective import evaluate_simulation
from inverse_rl_market_simulation.simulator import simulate_null_baseline, simulate_path


def _sample_params(rng: np.random.Generator, param_names: tuple[str, ...]) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for name in param_names:
        bounds = PARAMETER_BOUNDS[name]
        low, high = bounds
        if name in INTEGER_PARAMS:
            params[name] = int(rng.integers(int(low), int(high) + 1))
        else:
            params[name] = float(rng.uniform(float(low), float(high)))
    return normalize_params(params)


def _flatten_result(prefix: str, result: Mapping[str, object]) -> dict[str, float | str | int]:
    flat: dict[str, float | str | int] = {}
    for key, value in result.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{prefix}{key}_{sub_key}"] = float(sub_value)
        elif isinstance(value, (int, float, str)):
            flat[f"{prefix}{key}"] = value
    return flat


def run_calibration(config: ExperimentConfig, real_frame: pd.DataFrame) -> dict[str, object]:
    rng = np.random.default_rng(config.optimization.random_seed)
    param_names = get_search_space_param_names(config.optimization.search_space)

    null_simulation = simulate_null_baseline(real_frame)
    null_result = evaluate_simulation(null_simulation, label="null_baseline")

    hand_simulation = simulate_path(real_frame, DEFAULT_HAND_TUNED_PARAMS, search_space=config.optimization.search_space)
    hand_result = evaluate_simulation(hand_simulation, label="hand_tuned")

    best_params = DEFAULT_HAND_TUNED_PARAMS.copy()
    best_simulation = hand_simulation
    best_result = hand_result

    trial_rows: list[dict[str, float | str | int]] = []
    for trial_id in range(1, config.optimization.n_trials + 1):
        params = _sample_params(rng, param_names)
        simulation = simulate_path(real_frame, params, search_space=config.optimization.search_space)
        result = evaluate_simulation(simulation, label=f"trial_{trial_id:04d}")
        row: dict[str, float | str | int] = {"trial_id": trial_id}
        row.update({key: value for key, value in params.items()})
        row.update(_flatten_result("", result))
        trial_rows.append(row)
        if float(result["total_loss"]) < float(best_result["total_loss"]):
            best_params = dict(params)
            best_simulation = simulation
            best_result = result

    trials = pd.DataFrame(trial_rows).sort_values("total_loss").reset_index(drop=True)
    return {
        "best_params": normalize_params(best_params),
        "best_simulation": best_simulation,
        "best_result": best_result,
        "baseline_results": {
            "null_baseline": null_result,
            "hand_tuned": hand_result,
        },
        "trials": trials,
    }