from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from hierarchical_system.config import load_config as load_main_config
from hierarchical_system.models import build_model

from .config import ExperimentConfig
from .data import build_feature_frame, _assign_split


def _metrics(labels: list[float], preds: list[float]) -> dict[str, float]:
    if not preds:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0}
    pred_label = [1 if value >= 0.5 else 0 for value in preds]
    return {
        "accuracy": float(accuracy_score(labels, pred_label)),
        "precision": float(precision_score(labels, pred_label, zero_division=0)),
        "recall": float(recall_score(labels, pred_label, zero_division=0)),
        "auc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.0,
    }


def _build_layer_predictions(feature_df: pd.DataFrame, layer: str, checkpoint: dict, device: torch.device, config: ExperimentConfig) -> pd.DataFrame:
    feature_cols = checkpoint.get("feature_cols", [])
    lookback = int(checkpoint.get("lookback", config.model.seq_len))
    layer_df = feature_df[feature_df["layer"] == layer].copy()
    if layer_df.empty:
        return pd.DataFrame()

    model = build_model(layer, len(feature_cols)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict] = []
    with torch.no_grad():
        for stock_code, group in layer_df.groupby("stock_code"):
            group = group.sort_values("date").reset_index(drop=True)
            usable = group.dropna(subset=[*feature_cols, "target_up", "future_return"]).reset_index(drop=True)
            if len(usable) < lookback:
                continue
            values = usable[feature_cols].to_numpy(dtype=np.float32)
            for idx in range(lookback - 1, len(usable)):
                split = _assign_split(pd.Timestamp(usable.loc[idx, "date"]), config)
                if split != "test":
                    continue
                window = values[idx - lookback + 1: idx + 1]
                if not np.isfinite(window).all():
                    continue
                x = torch.tensor(window[None, :, :], dtype=torch.float32, device=device)
                prob = torch.sigmoid(model(x)).item()
                rows.append({
                    "stock_code": stock_code,
                    "date": str(pd.Timestamp(usable.loc[idx, "date"]).date()),
                    "layer": layer,
                    "prediction_prob": float(prob),
                    "label": float(usable.loc[idx, "target_up"]),
                    "future_return": float(usable.loc[idx, "future_return"]),
                    "model_family": "hierarchical",
                })
    return pd.DataFrame(rows)


def compare_with_hierarchical_models(config: ExperimentConfig, project_config_path: str | Path) -> dict:
    project_config = load_main_config(project_config_path)
    feature_df = build_feature_frame(config)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")

    prediction_frames: list[pd.DataFrame] = []
    available_layers: list[str] = []
    missing_layers: list[str] = []

    for layer in ["micro", "small", "mid", "large"]:
        model_path = project_config.models_dir / f"{layer}_model.pt"
        if not model_path.exists():
            missing_layers.append(layer)
            continue
        checkpoint = torch.load(model_path, map_location=device)
        pred_df = _build_layer_predictions(feature_df, layer, checkpoint, device, config)
        if not pred_df.empty:
            prediction_frames.append(pred_df)
            available_layers.append(layer)

    combined_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    hierarchical_metrics = _metrics(
        combined_predictions["label"].tolist() if not combined_predictions.empty else [],
        combined_predictions["prediction_prob"].tolist() if not combined_predictions.empty else [],
    )

    transformer_metrics_path = config.results_dir / "metrics.json"
    transformer_metrics = None
    if transformer_metrics_path.exists():
        with transformer_metrics_path.open("r", encoding="utf-8") as handle:
            transformer_metrics = json.load(handle)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_output = config.results_dir / "hierarchical_compare_metrics.json"
    predictions_output = config.results_dir / "hierarchical_compare_predictions.csv"

    report = {
        "stock_pool": config.experiment.stock_pool,
        "test_period": {
            "start": str(pd.Timestamp(config.splits.val_end) + pd.Timedelta(days=1)),
            "end": config.splits.test_end,
        },
        "hierarchical": {
            "available_layers": available_layers,
            "missing_layers": missing_layers,
            "test_size": int(len(combined_predictions)),
            **hierarchical_metrics,
        },
        "financial_transformer": transformer_metrics,
    }

    with metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if not combined_predictions.empty:
        combined_predictions.to_csv(predictions_output, index=False, encoding="utf-8-sig")

    return {
        "metrics_path": str(metrics_output),
        "predictions_path": str(predictions_output),
        "available_layers": available_layers,
        "missing_layers": missing_layers,
        **hierarchical_metrics,
    }