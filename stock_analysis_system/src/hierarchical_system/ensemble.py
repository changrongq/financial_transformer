from __future__ import annotations

from pathlib import Path
import pandas as pd

from .logging_utils import setup_logger

logger = setup_logger("hierarchical.ensemble")


def load_prediction_frames(prediction_files: dict[str, str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for layer, file_path in prediction_files.items():
        path = Path(file_path)
        if not path.exists():
            logger.warning("缺少预测文件: %s", path)
            continue
        df = pd.read_csv(path)
        frames[layer] = df
    return frames


def build_weighted_ensemble_frame(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    weighted_frames = []
    for layer, df in frames.items():
        frame = df.copy()
        frame["layer_weight"] = weights.get(layer, 0.0)
        frame["weighted_score"] = pd.to_numeric(frame["prediction_prob"], errors="coerce") * frame["layer_weight"]
        weighted_frames.append(frame)

    if not weighted_frames:
        raise RuntimeError("没有可用于集成的预测文件。")

    merged = pd.concat(weighted_frames, ignore_index=True)
    aggregation: dict[str, tuple[str, str | callable]] = {
        "ensemble_score": ("weighted_score", "sum"),
        "mean_prob": ("prediction_prob", "mean"),
        "layers": ("layer", lambda x: ",".join(sorted(set(x)))),
    }
    for column in ["future_return", "label", "layer"]:
        if column in merged.columns:
            aggregation[column] = (column, "first")

    result = (
        merged.groupby(["stock_code", "date"], as_index=False)
        .agg(**aggregation)
        .sort_values("ensemble_score", ascending=False)
    )
    return result


def weighted_ensemble(prediction_files: dict[str, str], weights: dict[str, float], output_file: str) -> pd.DataFrame:
    frames = load_prediction_frames(prediction_files)
    result = build_weighted_ensemble_frame(frames, weights)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    return result
