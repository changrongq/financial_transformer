from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from inverse_rl_market_simulation.config import ExperimentConfig


QUERY = """
SELECT stock_code, 日期, 开盘, 最高, 最低, 收盘, 成交量, 换手率
FROM stock_data
WHERE stock_code = ?
ORDER BY 日期
"""


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def load_stock_window(config: ExperimentConfig) -> pd.DataFrame:
    with _connect(config.databases.trade_db) as conn:
        frame = pd.read_sql_query(QUERY, conn, params=[config.experiment.stock_code])
    if frame.empty:
        raise ValueError(f"stock_data 中找不到股票 {config.experiment.stock_code}。")

    frame = frame.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "换手率": "turnover_rate",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ["open", "high", "low", "close", "volume", "turnover_rate"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if config.experiment.start_date:
        frame = frame.loc[frame["date"] >= pd.Timestamp(config.experiment.start_date)].copy()
    if config.experiment.end_date:
        frame = frame.loc[frame["date"] <= pd.Timestamp(config.experiment.end_date)].copy()
    if frame.empty:
        raise ValueError("日期过滤后没有可用样本。")

    if not config.experiment.start_date:
        frame = frame.tail(config.experiment.window_days).copy()
    if len(frame) < config.experiment.min_history_days:
        raise ValueError(
            f"样本不足: 需要至少 {config.experiment.min_history_days} 天，实际只有 {len(frame)} 天。"
        )

    frame = frame.reset_index(drop=True)
    frame["return_1d"] = frame["close"].pct_change().fillna(0.0)
    frame["log_return"] = np.log(frame["close"]).diff().fillna(0.0)
    frame["normalized_close"] = frame["close"] / float(frame["close"].iloc[0])
    frame["turnover_rate"] = frame["turnover_rate"].fillna(frame["turnover_rate"].median())
    frame["turnover_z"] = _zscore(frame["turnover_rate"]).fillna(0.0)
    frame["volatility_10"] = frame["return_1d"].rolling(10, min_periods=2).std(ddof=0).fillna(0.0)
    return frame