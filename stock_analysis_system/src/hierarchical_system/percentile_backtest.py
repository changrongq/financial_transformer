from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config


@dataclass
class PercentileBacktestArtifacts:
    summary: pd.DataFrame
    equity: pd.DataFrame
    selections: pd.DataFrame


def _safe_sharpe(returns: pd.Series) -> float:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return 0.0
    std = float(series.std())
    if std <= 0:
        return 0.0
    return float(series.mean() / std * np.sqrt(252))


def _max_drawdown(equity_curve: pd.Series) -> float:
    series = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if series.empty:
        return 0.0
    peak = series.cummax()
    drawdown = series / peak - 1.0
    return float(drawdown.min())


def _load_prediction_frame(path: Path, score_col: str = "prediction_prob") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"stock_code", "date", "future_return", "label", score_col}
    if not required.issubset(df.columns):
        raise ValueError(f"预测文件缺少必要列: {path}")
    frame = df[["stock_code", "date", "future_return", "label", score_col, "layer"]].copy()
    frame = frame.rename(columns={score_col: "score"})
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["future_return"] = pd.to_numeric(frame["future_return"], errors="coerce")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    return frame.dropna(subset=["date", "future_return", "label", "score"]).reset_index(drop=True)


def _build_sources(config_path: str | Path) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    results_dir = Path(config.results_dir)
    layer_frames: dict[str, pd.DataFrame] = {}
    weighted_frames: list[pd.DataFrame] = []

    for layer in ["micro", "small", "mid", "large"]:
        path = results_dir / f"{layer}_predictions.csv"
        frame = _load_prediction_frame(path)
        if frame.empty:
            continue
        frame["source"] = layer
        layer_frames[layer] = frame

        weighted = frame.copy()
        weighted["score"] = weighted["score"] * float(config.ensemble.weights.get(layer, 0.0))
        weighted["source"] = "ensemble"
        weighted_frames.append(weighted)

    if weighted_frames:
        layer_frames["ensemble"] = pd.concat(weighted_frames, ignore_index=True)
    return layer_frames


def _evaluate_percentile_strategy(source_name: str, frame: pd.DataFrame, percentile: float, initial_capital: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pct_label = f"top_{int(percentile * 100)}pct"
    daily_rows: list[dict] = []
    selection_rows: list[dict] = []
    equity = initial_capital

    grouped = frame.sort_values(["date", "score"], ascending=[True, False]).groupby("date", sort=True)
    for date_value, group in grouped:
        pick_count = max(1, int(np.ceil(len(group) * percentile)))
        picks = group.head(pick_count).copy()
        if picks.empty:
            continue
        daily_return = float(picks["future_return"].mean())
        hit_rate = float((picks["label"] > 0).mean())
        equity *= (1.0 + daily_return)
        daily_rows.append(
            {
                "source": source_name,
                "strategy": pct_label,
                "date": str(pd.Timestamp(date_value).date()),
                "pick_count": int(len(picks)),
                "mean_score": float(picks["score"].mean()),
                "mean_future_return": daily_return,
                "hit_rate": hit_rate,
                "equity": equity,
            }
        )
        picks["source"] = source_name
        picks["strategy"] = pct_label
        selection_rows.extend(picks.to_dict(orient="records"))

    daily_df = pd.DataFrame(daily_rows)
    selections_df = pd.DataFrame(selection_rows)
    if daily_df.empty:
        summary = {
            "source": source_name,
            "strategy": pct_label,
            "days": 0,
            "avg_pick_count": 0.0,
            "avg_daily_return": 0.0,
            "hit_rate": 0.0,
            "final_equity": float(initial_capital),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }
        return daily_df, selections_df, summary

    summary = {
        "source": source_name,
        "strategy": pct_label,
        "days": int(len(daily_df)),
        "avg_pick_count": float(daily_df["pick_count"].mean()),
        "avg_daily_return": float(daily_df["mean_future_return"].mean()),
        "hit_rate": float(daily_df["hit_rate"].mean()),
        "final_equity": float(daily_df.iloc[-1]["equity"]),
        "total_return": float(daily_df.iloc[-1]["equity"] / initial_capital - 1.0),
        "max_drawdown": _max_drawdown(daily_df["equity"]),
        "sharpe_ratio": _safe_sharpe(daily_df["mean_future_return"]),
    }
    return daily_df, selections_df, summary


def run_percentile_backtest(
    config_path: str | Path,
    percentiles: list[float] | None = None,
    initial_capital: float = 100000.0,
) -> PercentileBacktestArtifacts:
    percentiles = percentiles or [0.01, 0.05, 0.10, 0.20]
    sources = _build_sources(config_path)
    if not sources:
        raise RuntimeError("未找到可用于分位数回测的历史预测文件。")

    summary_rows: list[dict] = []
    equity_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []

    for source_name, frame in sources.items():
        if frame.empty:
            continue
        for percentile in percentiles:
            equity_df, selections_df, summary = _evaluate_percentile_strategy(source_name, frame, percentile, initial_capital)
            summary_rows.append(summary)
            if not equity_df.empty:
                equity_frames.append(equity_df)
            if not selections_df.empty:
                selection_frames.append(selections_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(["source", "strategy"]).reset_index(drop=True)
    equity_df = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    selections_df = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()

    config = load_config(config_path)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "percentile_backtest_summary.csv"
    equity_path = results_dir / "percentile_backtest_equity.csv"
    selections_path = results_dir / "percentile_backtest_selections.csv"
    json_path = results_dir / "percentile_backtest_summary.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if not equity_df.empty:
        equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")
    if not selections_df.empty:
        selections_df.to_csv(selections_path, index=False, encoding="utf-8-sig")
    summary_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    return PercentileBacktestArtifacts(summary=summary_df, equity=equity_df, selections=selections_df)