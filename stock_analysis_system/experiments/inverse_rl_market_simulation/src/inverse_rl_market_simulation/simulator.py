from __future__ import annotations

import math
import numpy as np
import pandas as pd

from inverse_rl_market_simulation.agents import compute_agent_demands, normalize_params


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / std


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _switch_weight(signal_score: float, strength: float, threshold: float) -> float:
    if strength <= 0.0:
        return 0.0
    normalized_gap = max(float(signal_score) - float(threshold), 0.0)
    scaled_gap = normalized_gap / max(1.0 - float(threshold), 1e-6)
    return _clip01(float(strength) * scaled_gap)


def _sign(value: float) -> int:
    if abs(value) <= 1e-12:
        return 0
    return 1 if value > 0.0 else -1


def simulate_path(real_frame: pd.DataFrame, params: dict[str, float | int], search_space: str = "fixed_role") -> pd.DataFrame:
    params = normalize_params(params)
    n_obs = len(real_frame)
    sim_close = np.zeros(n_obs, dtype=float)
    sim_return = np.zeros(n_obs, dtype=float)
    sim_turnover_proxy = np.zeros(n_obs, dtype=float)
    slow_demand = np.zeros(n_obs, dtype=float)
    trend_demand = np.zeros(n_obs, dtype=float)
    herd_demand = np.zeros(n_obs, dtype=float)
    base_slow_demand = np.zeros(n_obs, dtype=float)
    base_trend_demand = np.zeros(n_obs, dtype=float)
    base_herd_demand = np.zeros(n_obs, dtype=float)
    herd_to_slow_weight = np.zeros(n_obs, dtype=float)
    slow_to_trend_weight = np.zeros(n_obs, dtype=float)
    trap_state = np.zeros(n_obs, dtype=float)
    trend_switch_state = np.zeros(n_obs, dtype=float)
    slow_survival_score_series = np.zeros(n_obs, dtype=float)
    trend_persistence_score_series = np.zeros(n_obs, dtype=float)
    imbalance = np.zeros(n_obs, dtype=float)
    slow_inventory_series = np.zeros(n_obs, dtype=float)
    trend_inventory_series = np.zeros(n_obs, dtype=float)
    total_inventory_series = np.zeros(n_obs, dtype=float)

    closes = real_frame["close"].to_numpy(dtype=float)
    real_returns = real_frame["return_1d"].to_numpy(dtype=float)
    turnover_shocks = real_frame["turnover_z"].to_numpy(dtype=float)
    sim_close[0] = closes[0]
    sim_turnover_proxy[0] = max(turnover_shocks[0], 0.0)

    slow_inventory = 0.0
    trend_inventory = 0.0
    previous_inventory = 0.0
    previous_slow_sign = 0
    slow_streak = 0.0
    previous_trend_sign = 0
    trend_streak = 0.0

    for idx in range(1, n_obs):
        anchor_window = int(params["anchor_window"])
        anchor_start = max(0, idx - anchor_window)
        anchor_price = float(sim_close[anchor_start:idx].mean())
        anchor_gap = 0.0 if anchor_price == 0.0 else float(sim_close[idx - 1] / anchor_price - 1.0)

        trend_lookback = int(params["trend_lookback"])
        trend_start = max(0, idx - trend_lookback)
        trend_signal = float(real_returns[trend_start:idx].mean()) if idx > trend_start else 0.0
        previous_return = float(real_returns[idx - 1])
        turnover_shock = float(turnover_shocks[idx - 1])
        peak_close = float(sim_close[:idx].max()) if idx > 0 else float(sim_close[0])
        underwater_ratio = 0.0 if peak_close == 0.0 else max(peak_close - float(sim_close[idx - 1]), 0.0) / peak_close
        recent_negative_run = max(-float(sim_return[max(0, idx - 5):idx].sum()), 0.0)
        trap_signal = 0.65 * _clip01(underwater_ratio / 0.08) + 0.35 * _clip01(recent_negative_run / 0.05)

        current_slow_sign = _sign(slow_inventory)
        if current_slow_sign == 0:
            slow_streak = 0.0
        elif current_slow_sign == previous_slow_sign:
            slow_streak += 1.0
        else:
            slow_streak = 1.0
            previous_slow_sign = current_slow_sign
        if current_slow_sign == 0:
            previous_slow_sign = 0

        slow_streak_score = _clip01(slow_streak / max(float(trend_lookback), 1.0))
        trend_strength_score = _clip01(abs(trend_signal) / max(float(params["entry_threshold"]), 1e-6))
        trend_sign = _sign(trend_signal)
        if trend_sign == 0:
            trend_streak = 0.0
        elif trend_sign == previous_trend_sign:
            trend_streak += 1.0
        else:
            trend_streak = 1.0
            previous_trend_sign = trend_sign
        if trend_sign == 0:
            previous_trend_sign = 0

        trend_persistence_score = _clip01(trend_streak / max(float(trend_lookback), 1.0))
        alignment_score = 1.0 if current_slow_sign != 0 and current_slow_sign == trend_sign else 0.0
        if search_space == "soft_switch_v2_directional_probe":
            survival_persistence_score = math.sqrt(slow_streak_score * trend_persistence_score)
            trend_signal_state = survival_persistence_score * trend_strength_score * alignment_score
        else:
            trend_signal_state = slow_streak_score * trend_strength_score * alignment_score

        base_demands = compute_agent_demands(
            anchor_gap=anchor_gap,
            trend_signal=trend_signal,
            previous_return=previous_return,
            turnover_shock=turnover_shock,
            slow_inventory=slow_inventory,
            trend_inventory=trend_inventory,
            params=params,
        )

        w_hs = _switch_weight(
            signal_score=trap_signal,
            strength=float(params["trap_to_slow_strength"]),
            threshold=float(params["trap_to_slow_threshold"]),
        )
        slow_after_hs = base_demands["slow_demand"] + w_hs * base_demands["herd_demand"]
        if search_space == "soft_switch_v2_directional_probe":
            resistance_lookback = max(20, 2 * trend_lookback)
            resistance_start = max(0, idx - resistance_lookback)
            resistance_close_window = closes[resistance_start:idx]
            resistance_turnover_window = turnover_shocks[resistance_start:idx]
            resistance_mask = resistance_turnover_window >= 1.0
            if resistance_mask.any():
                resistance_level = float(resistance_close_window[resistance_mask].max())
            else:
                resistance_level = float(resistance_close_window.max()) if resistance_close_window.size > 0 else float(closes[0])
            breakout_gap = 0.0 if resistance_level <= 0.0 else float(closes[idx - 1] / resistance_level - 1.0)
            overhang_proximity_score = _clip01((breakout_gap + 0.25) / 0.25)
            slow_to_trend_threshold = 0.75 * float(params["slow_to_trend_threshold"]) * (1.0 - 0.15 * overhang_proximity_score)
            if slow_after_hs <= 0.0:
                w_st = 0.0
            else:
                normalized_gap = max(trend_signal_state - slow_to_trend_threshold, 0.0)
                scaled_gap = normalized_gap / max(1.0 - slow_to_trend_threshold, 1e-6)
                w_st = _clip01(float(params["slow_to_trend_strength"]) * math.sqrt(scaled_gap))
        else:
            w_st = _switch_weight(
                signal_score=trend_signal_state,
                strength=float(params["slow_to_trend_strength"]),
                threshold=float(params["slow_to_trend_threshold"]),
            )

        trend_flow = w_st * slow_after_hs
        demands = {
            "slow_demand": slow_after_hs - trend_flow,
            "trend_demand": base_demands["trend_demand"] + trend_flow,
            "herd_demand": (1.0 - w_hs) * base_demands["herd_demand"],
        }

        slow_inventory_carry = 0.60
        slow_inventory = slow_inventory_carry * slow_inventory + demands["slow_demand"]
        trend_inventory = 0.40 * trend_inventory + demands["trend_demand"]
        current_inventory = slow_inventory + trend_inventory
        inventory_delta = current_inventory - previous_inventory
        previous_inventory = current_inventory

        total_imbalance = demands["slow_demand"] + demands["trend_demand"] + demands["herd_demand"]
        raw_return = 0.08 * total_imbalance + 0.03 * abs(total_imbalance) * np.sign(total_imbalance) + 0.02 * inventory_delta
        clipped_return = float(np.clip(raw_return, -0.18, 0.18))

        sim_return[idx] = clipped_return
        sim_close[idx] = sim_close[idx - 1] * (1.0 + clipped_return)
        slow_demand[idx] = demands["slow_demand"]
        trend_demand[idx] = demands["trend_demand"]
        herd_demand[idx] = demands["herd_demand"]
        base_slow_demand[idx] = base_demands["slow_demand"]
        base_trend_demand[idx] = base_demands["trend_demand"]
        base_herd_demand[idx] = base_demands["herd_demand"]
        herd_to_slow_weight[idx] = w_hs
        slow_to_trend_weight[idx] = w_st
        trap_state[idx] = trap_signal
        trend_switch_state[idx] = trend_signal_state
        slow_survival_score_series[idx] = slow_streak_score
        trend_persistence_score_series[idx] = trend_persistence_score
        imbalance[idx] = total_imbalance
        sim_turnover_proxy[idx] = abs(demands["trend_demand"]) + abs(demands["herd_demand"]) + 0.50 * abs(demands["slow_demand"])
        slow_inventory_series[idx] = slow_inventory
        trend_inventory_series[idx] = trend_inventory
        total_inventory_series[idx] = current_inventory

    simulation = pd.DataFrame(
        {
            "date": real_frame["date"],
            "real_close": real_frame["close"],
            "real_return": real_frame["return_1d"],
            "real_turnover_rate": real_frame["turnover_rate"],
            "sim_close": sim_close,
            "sim_return": sim_return,
            "sim_turnover_proxy": sim_turnover_proxy,
            "slow_demand": slow_demand,
            "trend_demand": trend_demand,
            "herd_demand": herd_demand,
            "base_slow_demand": base_slow_demand,
            "base_trend_demand": base_trend_demand,
            "base_herd_demand": base_herd_demand,
            "herd_to_slow_weight": herd_to_slow_weight,
            "slow_to_trend_weight": slow_to_trend_weight,
            "trap_state": trap_state,
            "trend_switch_state": trend_switch_state,
            "slow_survival_score": slow_survival_score_series,
            "trend_persistence_score": trend_persistence_score_series,
            "imbalance": imbalance,
            "slow_inventory": slow_inventory_series,
            "trend_inventory": trend_inventory_series,
            "total_inventory": total_inventory_series,
        }
    )
    simulation["real_normalized_close"] = simulation["real_close"] / float(simulation["real_close"].iloc[0])
    simulation["sim_normalized_close"] = simulation["sim_close"] / float(simulation["sim_close"].iloc[0])
    simulation["real_volatility_10"] = simulation["real_return"].rolling(10, min_periods=2).std(ddof=0).fillna(0.0)
    simulation["sim_volatility_10"] = simulation["sim_return"].rolling(10, min_periods=2).std(ddof=0).fillna(0.0)
    simulation["real_turnover_z"] = _zscore(simulation["real_turnover_rate"].to_numpy(dtype=float))
    simulation["sim_turnover_z"] = _zscore(simulation["sim_turnover_proxy"].to_numpy(dtype=float))
    return simulation


def simulate_null_baseline(real_frame: pd.DataFrame) -> pd.DataFrame:
    real_returns = real_frame["return_1d"].to_numpy(dtype=float)
    mu = float(real_returns.mean())
    if len(real_returns) > 1:
        left = real_returns[1:]
        right = real_returns[:-1]
        left_std = float(left.std(ddof=0))
        right_std = float(right.std(ddof=0))
        if left_std == 0.0 or right_std == 0.0 or np.isnan(left_std) or np.isnan(right_std):
            autocorr = 0.0
        else:
            autocorr = np.corrcoef(left, right)[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0
    sim_return = np.zeros_like(real_returns)
    sim_return[0] = real_returns[0]
    for idx in range(1, len(sim_return)):
        sim_return[idx] = mu + autocorr * (sim_return[idx - 1] - mu)
        sim_return[idx] = float(np.clip(sim_return[idx], -0.18, 0.18))

    sim_close = np.zeros_like(sim_return)
    sim_close[0] = float(real_frame["close"].iloc[0])
    for idx in range(1, len(sim_return)):
        sim_close[idx] = sim_close[idx - 1] * (1.0 + sim_return[idx])

    sim_turnover_proxy = np.full(len(sim_return), float(real_frame["turnover_rate"].mean()), dtype=float)
    simulation = pd.DataFrame(
        {
            "date": real_frame["date"],
            "real_close": real_frame["close"],
            "real_return": real_frame["return_1d"],
            "real_turnover_rate": real_frame["turnover_rate"],
            "sim_close": sim_close,
            "sim_return": sim_return,
            "sim_turnover_proxy": sim_turnover_proxy,
            "slow_demand": 0.0,
            "trend_demand": 0.0,
            "herd_demand": 0.0,
            "base_slow_demand": 0.0,
            "base_trend_demand": 0.0,
            "base_herd_demand": 0.0,
            "herd_to_slow_weight": 0.0,
            "slow_to_trend_weight": 0.0,
            "trap_state": 0.0,
            "trend_switch_state": 0.0,
            "slow_survival_score": 0.0,
            "trend_persistence_score": 0.0,
            "imbalance": 0.0,
        }
    )
    simulation["real_normalized_close"] = simulation["real_close"] / float(simulation["real_close"].iloc[0])
    simulation["sim_normalized_close"] = simulation["sim_close"] / float(simulation["sim_close"].iloc[0])
    simulation["real_volatility_10"] = simulation["real_return"].rolling(10, min_periods=2).std(ddof=0).fillna(0.0)
    simulation["sim_volatility_10"] = simulation["sim_return"].rolling(10, min_periods=2).std(ddof=0).fillna(0.0)
    simulation["real_turnover_z"] = _zscore(simulation["real_turnover_rate"].to_numpy(dtype=float))
    simulation["sim_turnover_z"] = _zscore(simulation["sim_turnover_proxy"].to_numpy(dtype=float))
    return simulation