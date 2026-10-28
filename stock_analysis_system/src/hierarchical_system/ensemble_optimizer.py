from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import load_config
from .ensemble import build_weighted_ensemble_frame, load_prediction_frames
from .percentile_backtest import _evaluate_percentile_strategy
from .realistic_backtest import RealisticPortfolioBacktester


def _generate_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    combos: list[dict[str, float]] = []
    for micro in range(units + 1):
        for small in range(units + 1 - micro):
            for mid in range(units + 1 - micro - small):
                large = units - micro - small - mid
                combos.append(
                    {
                        "micro": round(micro * step, 4),
                        "small": round(small * step, 4),
                        "mid": round(mid * step, 4),
                        "large": round(large * step, 4),
                    }
                )
    return combos


def _combine_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required_columns = ["stock_code", "date", "prediction_prob", "future_return", "label"]
    combined_frames = []
    for layer, df in frames.items():
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"{layer} 预测文件缺少列: {missing}")
        frame = df[required_columns].copy()
        frame["layer"] = layer
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["prediction_prob"] = pd.to_numeric(frame["prediction_prob"], errors="coerce")
        frame["future_return"] = pd.to_numeric(frame["future_return"], errors="coerce")
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        combined_frames.append(frame.dropna(subset=["prediction_prob", "future_return", "label", "date"]))

    combined = pd.concat(combined_frames, ignore_index=True)
    duplicates = combined.duplicated(["stock_code", "date"], keep=False)
    if duplicates.any():
        raise ValueError("历史预测中存在跨层重复的 stock_code/date，无法直接做单层加权优化。")
    return combined.reset_index(drop=True)


def optimize_ensemble_weights(
    config_path: str | Path,
    percentile: float = 0.05,
    step: float = 0.05,
    top_k: int = 25,
) -> tuple[pd.DataFrame, dict]:
    config = load_config(config_path)
    prediction_files = {
        layer: str(Path(config.results_dir) / f"{layer}_predictions.csv")
        for layer in ["micro", "small", "mid", "large"]
    }
    frames = load_prediction_frames(prediction_files)
    if not frames:
        raise RuntimeError("未找到可用于优化集成权重的历史预测文件。")
    combined = _combine_frames(frames)
    layer_index = combined["layer"].map({"micro": 0, "small": 1, "mid": 2, "large": 3}).to_numpy(dtype=np.int8)
    probabilities = combined["prediction_prob"].to_numpy(dtype=np.float64)
    base_eval_frame = combined[["stock_code", "date", "future_return", "label"]].copy()

    coarse_rows: list[dict] = []
    for weights in _generate_weight_grid(step=step):
        weight_array = np.array([weights["micro"], weights["small"], weights["mid"], weights["large"]], dtype=np.float64)
        eval_frame = base_eval_frame.copy()
        eval_frame["score"] = probabilities * weight_array[layer_index]
        _, _, summary = _evaluate_percentile_strategy("ensemble", eval_frame, percentile, 100000.0)
        coarse_rows.append({**weights, **summary})

    coarse_df = pd.DataFrame(coarse_rows)
    coarse_df = coarse_df.sort_values(["sharpe_ratio", "total_return", "max_drawdown"], ascending=[False, False, False]).reset_index(drop=True)
    shortlisted = coarse_df.head(top_k).copy()

    backtester = RealisticPortfolioBacktester(config_path)
    realistic_rows: list[dict] = []
    for _, row in shortlisted.iterrows():
        weights = {layer: float(row[layer]) for layer in ["micro", "small", "mid", "large"]}
        weight_array = np.array([weights["micro"], weights["small"], weights["mid"], weights["large"]], dtype=np.float64)
        ensemble_df = combined[["stock_code", "date"]].copy()
        ensemble_df["ensemble_score"] = probabilities * weight_array[layer_index]
        realistic = backtester.run_frame(
            predictions=ensemble_df,
            percentile=percentile,
            max_positions=20,
            hold_days=5,
            buy_cost=0.001,
            sell_cost=0.001,
            initial_capital=100000.0,
        )
        realistic_rows.append({**weights, **realistic.summary})

    realistic_df = pd.DataFrame(realistic_rows)
    realistic_df = realistic_df.sort_values(["sharpe_ratio", "total_return", "max_drawdown"], ascending=[False, False, False]).reset_index(drop=True)

    best = realistic_df.iloc[0].to_dict()
    recommendation = {
        "recommended_weights": {layer: float(best[layer]) for layer in ["micro", "small", "mid", "large"]},
        "realistic_top5_summary": {
            key: float(best[key]) if isinstance(best[key], (int, float)) else best[key]
            for key in ["total_return", "max_drawdown", "sharpe_ratio", "win_rate", "profit_factor", "trade_count"]
        },
        "search": {
            "step": step,
            "percentile": percentile,
            "candidate_count": int(len(coarse_df)),
            "shortlist_count": int(len(shortlisted)),
        },
    }

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    coarse_df.to_csv(results_dir / "ensemble_weight_search_coarse.csv", index=False, encoding="utf-8-sig")
    realistic_df.to_csv(results_dir / "ensemble_weight_search_realistic.csv", index=False, encoding="utf-8-sig")
    with (results_dir / "ensemble_weight_recommendation.json").open("w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, ensure_ascii=False, indent=2)

    return realistic_df, recommendation