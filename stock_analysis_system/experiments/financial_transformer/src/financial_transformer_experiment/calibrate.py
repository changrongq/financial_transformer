from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import apply_probability_calibration
from .calibration import calibrated_test_predictions_path
from .calibration import fit_probability_calibration
from .calibration import save_calibration_summary
from .calibration import save_probability_calibration
from .config import ExperimentConfig
from .data import build_datasets, collate_sequence_batch
from .model import create_model
from .training import _evaluate


def _build_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequence_batch)


def run_probability_calibration(config: ExperimentConfig) -> dict:
    checkpoint_path = config.artifacts_dir / "financial_transformer.pt"
    if not checkpoint_path.exists():
        raise RuntimeError("未找到 Transformer 模型文件，请先运行 train.py。")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config", {})
    datasets = build_datasets(config)
    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    model = create_model(
        input_dim=datasets.input_dim,
        d_model=int(model_config.get("d_model", config.model.d_model)),
        nhead=int(model_config.get("nhead", config.model.nhead)),
        num_layers=int(model_config.get("num_layers", config.model.num_layers)),
        dim_feedforward=int(model_config.get("dim_feedforward", config.model.dim_feedforward)),
        dropout=float(model_config.get("dropout", config.model.dropout)),
        ffn_type=str(model_config.get("ffn_type", config.model.ffn_type)),
        position_encoding=str(model_config.get("position_encoding", config.model.position_encoding)),
        model_variant=str(model_config.get("model_variant", config.model.model_variant)),
        feature_cols=config.experiment.feature_cols,
        grouped_feature_groups=dict(model_config.get("grouped_feature_groups", config.model.grouped_feature_groups)),
        use_group_gates=bool(model_config.get("use_group_gates", config.model.use_group_gates)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_metrics, val_predictions = _evaluate(
        model,
        _build_loader(datasets.val, config.training.batch_size),
        device,
        exit_label_mode=config.multitask.exit_label_mode,
    )
    test_metrics, test_predictions = _evaluate(
        model,
        _build_loader(datasets.test, config.training.batch_size),
        device,
        exit_label_mode=config.multitask.exit_label_mode,
    )

    calibration = fit_probability_calibration(val_predictions)
    calibration_file = save_probability_calibration(config.results_dir, calibration)
    if not test_predictions.empty:
        test_predictions.to_csv(config.results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    calibrated_test = apply_probability_calibration(test_predictions, calibration)
    if not calibrated_test.empty:
        gain_term = calibrated_test["pred_max_gain"].clip(lower=0.0, upper=0.5)
        drawdown_penalty = calibrated_test["pred_max_drawdown"].abs().clip(lower=0.01, upper=0.3)
        calibrated_test["signal_score"] = (
            0.45 * calibrated_test["pred_short_up"]
            + 0.35 * calibrated_test["pred_take_profit"]
            + 0.20 * (1.0 - calibrated_test["pred_stop_loss"])
            + 0.10 * gain_term
            - 0.05 * drawdown_penalty
        )
        calibrated_test["prediction_prob"] = calibrated_test["pred_short_up"]
        calibrated_test.to_csv(calibrated_test_predictions_path(config.results_dir), index=False, encoding="utf-8-sig")

    summary = {
        "calibration_file": str(calibration_file),
        "val_size": len(val_predictions),
        "test_size": len(test_predictions),
        "val_metrics": {key: float(value) for key, value in val_metrics.items()},
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "tasks": calibration["tasks"],
    }
    summary_file = save_calibration_summary(config.results_dir, summary)
    return {
        **summary,
        "summary_file": str(summary_file),
        "test_predictions_file": str(config.results_dir / "test_predictions.csv"),
        "test_predictions_calibrated_file": str(calibrated_test_predictions_path(config.results_dir)),
    }