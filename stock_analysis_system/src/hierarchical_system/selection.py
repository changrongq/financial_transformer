from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_latest_rankings(path: str | Path) -> pd.DataFrame:
    ranking_path = Path(path)
    if not ranking_path.exists():
        raise FileNotFoundError(f"预测结果不存在: {ranking_path}")
    df = pd.read_csv(ranking_path)
    required = {"stock_code", "date", "ensemble_score"}
    if not required.issubset(df.columns):
        raise ValueError(f"预测文件缺少必要列: {ranking_path}")
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["ensemble_score"] = pd.to_numeric(df["ensemble_score"], errors="coerce")
    df = df.dropna(subset=["ensemble_score"]).sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    return df


def select_top_percentile(df: pd.DataFrame, percentile: float = 0.05) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if percentile <= 0 or percentile > 1:
        raise ValueError("percentile 必须在 (0, 1] 区间内。")
    count = max(1, int(np.ceil(len(df) * percentile)))
    result = df.head(count).copy()
    result["selection_percentile"] = percentile
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def write_daily_selection(input_file: str | Path, output_file: str | Path, percentile: float = 0.05) -> pd.DataFrame:
    df = load_latest_rankings(input_file)
    selected = select_top_percentile(df, percentile=percentile)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")
    return selected