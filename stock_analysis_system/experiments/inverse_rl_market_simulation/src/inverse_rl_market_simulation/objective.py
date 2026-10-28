from __future__ import annotations

import numpy as np
import pandas as pd


MIN_SIM_VOL_RATIO = 0.20
LOW_VOL_PENALTY_WEIGHT = 10.0
ZERO_DRAWDOWN_PENALTY = 5.0
MEANINGFUL_DRAWDOWN_THRESHOLD = 0.01
ZERO_DRAWDOWN_EPSILON = 1e-8


def _safe_skew(values: np.ndarray) -> float:
    centered = values - float(values.mean())
    std = float(values.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    centered = values - float(values.mean())
    std = float(values.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(np.mean((centered / std) ** 4) - 3.0)


def _safe_autocorr(values: np.ndarray, lag: int = 1) -> float:
    if len(values) <= lag:
        return 0.0
    left = values[lag:]
    right = values[:-lag]
    left_std = float(left.std(ddof=0))
    right_std = float(right.std(ddof=0))
    if left_std == 0.0 or right_std == 0.0 or np.isnan(left_std) or np.isnan(right_std):
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _max_drawdown(path: np.ndarray) -> float:
    running_max = np.maximum.accumulate(path)
    drawdown = path / running_max - 1.0
    return float(np.min(drawdown))


def _recovery_length(path: np.ndarray) -> int:
    running_max = np.maximum.accumulate(path)
    drawdown = path / running_max - 1.0
    trough = int(np.argmin(drawdown))
    peak_value = float(running_max[trough])
    for idx in range(trough, len(path)):
        if path[idx] >= peak_value:
            return idx - trough
    return len(path) - trough


def _distribution_signature(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "skew": _safe_skew(values),
        "kurtosis": _safe_kurtosis(values),
        "lag1_autocorr": _safe_autocorr(values, lag=1),
    }


def evaluate_simulation(simulation: pd.DataFrame, label: str) -> dict[str, float | str | dict[str, float]]:
    real_path = simulation["real_normalized_close"].to_numpy(dtype=float)
    sim_path = simulation["sim_normalized_close"].to_numpy(dtype=float)
    real_returns = simulation["real_return"].to_numpy(dtype=float)
    sim_returns = simulation["sim_return"].to_numpy(dtype=float)
    real_turnover = simulation["real_turnover_z"].to_numpy(dtype=float)
    sim_turnover = simulation["sim_turnover_z"].to_numpy(dtype=float)
    real_vol = simulation["real_volatility_10"].to_numpy(dtype=float)
    sim_vol = simulation["sim_volatility_10"].to_numpy(dtype=float)

    path_loss = float(np.mean((real_path - sim_path) ** 2))

    real_sig = _distribution_signature(real_returns)
    sim_sig = _distribution_signature(sim_returns)
    dist_loss = float(np.mean([(real_sig[key] - sim_sig[key]) ** 2 for key in real_sig]))

    vol_loss = float(np.mean((real_vol - sim_vol) ** 2))
    turnover_loss = float(np.mean((real_turnover - sim_turnover) ** 2))

    real_drawdown = _max_drawdown(real_path)
    sim_drawdown = _max_drawdown(sim_path)
    real_recovery = _recovery_length(real_path)
    sim_recovery = _recovery_length(sim_path)
    drawdown_loss = float(abs(real_drawdown - sim_drawdown) + abs(real_recovery - sim_recovery) / max(len(simulation), 1))

    real_return_std = float(real_sig["std"])
    sim_return_std = float(sim_sig["std"])
    volatility_ratio = 0.0 if real_return_std <= 0.0 else float(sim_return_std / real_return_std)
    low_vol_penalty = 0.0
    min_allowed_sim_std = MIN_SIM_VOL_RATIO * real_return_std
    if real_return_std > 0.0 and sim_return_std < min_allowed_sim_std:
        shortfall_ratio = (min_allowed_sim_std - sim_return_std) / min_allowed_sim_std
        low_vol_penalty = float(LOW_VOL_PENALTY_WEIGHT * shortfall_ratio**2)

    zero_drawdown_penalty = 0.0
    if abs(real_drawdown) > MEANINGFUL_DRAWDOWN_THRESHOLD and abs(sim_drawdown) <= ZERO_DRAWDOWN_EPSILON:
        zero_drawdown_penalty = float(ZERO_DRAWDOWN_PENALTY)

    total_loss = (
        0.35 * path_loss
        + 0.25 * dist_loss
        + 0.15 * vol_loss
        + 0.10 * turnover_loss
        + 0.15 * drawdown_loss
        + low_vol_penalty
        + zero_drawdown_penalty
    )
    return {
        "label": label,
        "total_loss": float(total_loss),
        "path_loss": path_loss,
        "dist_loss": dist_loss,
        "vol_loss": vol_loss,
        "turnover_loss": turnover_loss,
        "drawdown_loss": drawdown_loss,
        "volatility_ratio": volatility_ratio,
        "low_vol_penalty": low_vol_penalty,
        "zero_drawdown_penalty": zero_drawdown_penalty,
        "real_distribution": real_sig,
        "sim_distribution": sim_sig,
        "real_max_drawdown": real_drawdown,
        "sim_max_drawdown": sim_drawdown,
        "real_recovery_length": int(real_recovery),
        "sim_recovery_length": int(sim_recovery),
    }