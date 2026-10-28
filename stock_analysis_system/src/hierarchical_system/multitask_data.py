from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .config import SystemConfig
from .features import LAYER_FEATURES, build_layer_features
from .logging_utils import setup_logger
from .prepare import _latest_market_caps, _load_splits, _load_trade_data

logger = setup_logger("hierarchical.multitask_data")

MULTITASK_TARGET_COLUMNS = [
    "future_return_5d",
    "target_up_5d",
    "target_take_profit_45d",
    "target_stop_loss_45d",
    "max_gain_45d",
    "max_drawdown_45d",
]


def add_multitask_targets(
    df: pd.DataFrame,
    short_horizon: int,
    short_threshold: float,
    hold_days: int,
    take_profit_threshold: float,
    stop_loss_threshold: float,
    exit_label_mode: str = "window_any",
) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]
    result["future_return_5d"] = close.shift(-short_horizon) / close - 1.0
    future_window = pd.concat([close.shift(-step) / close - 1.0 for step in range(1, hold_days + 1)], axis=1)
    result["max_gain_45d"] = future_window.max(axis=1)
    result["max_drawdown_45d"] = future_window.min(axis=1)
    result["target_up_5d"] = (result["future_return_5d"] >= short_threshold).astype(float)
    if exit_label_mode in {"first_hit", "realized_profit"}:
        tp_hits = future_window.ge(take_profit_threshold)
        sl_hits = future_window.le(stop_loss_threshold)
        tp_hit_any = tp_hits.any(axis=1)
        sl_hit_any = sl_hits.any(axis=1)

        tp_hit_array = tp_hits.to_numpy(dtype=bool)
        sl_hit_array = sl_hits.to_numpy(dtype=bool)
        tp_hit_any_array = tp_hit_any.to_numpy(dtype=bool)
        sl_hit_any_array = sl_hit_any.to_numpy(dtype=bool)

        tp_first_idx = np.where(tp_hit_any_array, tp_hit_array.argmax(axis=1) + 1, hold_days + 1)
        sl_first_idx = np.where(sl_hit_any_array, sl_hit_array.argmax(axis=1) + 1, hold_days + 1)

        # Match the backtester's same-day conflict rule: stop-loss wins ties.
        tp_first = tp_hit_any_array & (~sl_hit_any_array | (tp_first_idx < sl_first_idx))
        sl_first = sl_hit_any_array & (~tp_hit_any_array | (sl_first_idx <= tp_first_idx))

        if exit_label_mode == "realized_profit":
            final_returns = future_window.iloc[:, -1].to_numpy(dtype=float)
            realized_exit_return = np.where(
                sl_first,
                stop_loss_threshold,
                np.where(tp_first, take_profit_threshold, final_returns),
            )
            result["target_take_profit_45d"] = (realized_exit_return > 0.0).astype(float)
            result["target_stop_loss_45d"] = (realized_exit_return <= 0.0).astype(float)
            result["realized_exit_return_45d"] = realized_exit_return.astype(float)
        else:
            result["target_take_profit_45d"] = tp_first.astype(float)
            result["target_stop_loss_45d"] = sl_first.astype(float)
    else:
        result["target_take_profit_45d"] = (result["max_gain_45d"] >= take_profit_threshold).astype(float)
        result["target_stop_loss_45d"] = (result["max_drawdown_45d"] <= stop_loss_threshold).astype(float)
    return result


def extract_multitask_features(config: SystemConfig) -> pd.DataFrame:
    market_caps = _latest_market_caps(config.databases.market_cap_db)
    splits = _load_splits(config.databases.features_db)
    merged = splits.merge(market_caps, on="stock_code", how="inner")
    all_frames: list[pd.DataFrame] = []
    min_length = max(90, config.training.lookback_window + config.multitask.hold_days)

    for _, row in merged.iterrows():
        stock_code = str(row["stock_code"])
        split = row["split"]
        layer = row["layer"]
        market_cap_yi = float(row["market_cap_yi"])
        try:
            price_df = _load_trade_data(config.databases.trade_db, stock_code)
            if price_df.empty or len(price_df) < min_length:
                continue
            feats = build_layer_features(price_df, market_cap_yi)
            feats = add_multitask_targets(
                feats,
                short_horizon=config.multitask.short_horizon,
                short_threshold=config.multitask.short_threshold,
                hold_days=config.multitask.hold_days,
                take_profit_threshold=config.multitask.take_profit_threshold,
                stop_loss_threshold=config.multitask.stop_loss_threshold,
                exit_label_mode=config.multitask.exit_label_mode,
            )
            feats["split"] = split
            feats["stock_code"] = stock_code
            keep_cols = [
                "stock_code", "date", "layer", "split", "market_cap_yi",
                *MULTITASK_TARGET_COLUMNS,
                *sorted({col for cols in LAYER_FEATURES.values() for col in cols}),
            ]
            feats = feats[[col for col in keep_cols if col in feats.columns]].replace([np.inf, -np.inf], np.nan)
            feats = feats.dropna().reset_index(drop=True)
            all_frames.append(feats)
        except Exception as exc:
            logger.warning("多任务特征提取失败 %s: %s", stock_code, exc)

    if not all_frames:
        raise RuntimeError("没有提取到任何多任务特征数据。")

    feature_df = pd.concat(all_frames, ignore_index=True)
    Path(config.databases.features_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.databases.features_db)
    try:
        feature_df.to_sql("multitask_features", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    logger.info("多任务特征提取完成，共 %s 行", len(feature_df))
    return feature_df


def summarize_multitask_targets(feature_df: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_df.groupby(["split", "layer"], as_index=False)
        .agg(
            samples=("stock_code", "size"),
            up5_rate=("target_up_5d", "mean"),
            take_profit_45d_rate=("target_take_profit_45d", "mean"),
            stop_loss_45d_rate=("target_stop_loss_45d", "mean"),
            max_gain_45d_mean=("max_gain_45d", "mean"),
            max_gain_45d_p75=("max_gain_45d", lambda x: float(x.quantile(0.75))),
            max_drawdown_45d_mean=("max_drawdown_45d", "mean"),
            max_drawdown_45d_p25=("max_drawdown_45d", lambda x: float(x.quantile(0.25))),
        )
        .sort_values(["split", "layer"])
        .reset_index(drop=True)
    )