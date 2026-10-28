from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import ExperimentConfig


CALIBRATION_TASKS = ["short_up", "take_profit", "stop_loss"]


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_probability_calibration(predictions: pd.DataFrame) -> dict:
    tasks: dict[str, dict] = {}
    for task in CALIBRATION_TASKS:
        pred_col = f"pred_{task}"
        label_col = f"label_{task}"
        if pred_col not in predictions.columns or label_col not in predictions.columns:
            tasks[task] = {"enabled": False, "coef": 1.0, "intercept": 0.0}
            continue

        frame = predictions[[pred_col, label_col]].dropna().copy()
        if frame.empty or frame[label_col].nunique() < 2:
            tasks[task] = {"enabled": False, "coef": 1.0, "intercept": 0.0}
            continue

        x = _safe_logit(frame[pred_col].to_numpy()).reshape(-1, 1)
        y = frame[label_col].astype(int).to_numpy()
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        tasks[task] = {
            "enabled": True,
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        }

    return {
        "method": "platt_logit",
        "tasks": tasks,
    }


def apply_probability_calibration(predictions: pd.DataFrame, calibration: dict | None) -> pd.DataFrame:
    if predictions.empty or not calibration:
        return predictions.copy()

    result = predictions.copy()
    tasks = calibration.get("tasks", {})
    for task in CALIBRATION_TASKS:
        pred_col = f"pred_{task}"
        raw_col = f"raw_pred_{task}"
        if pred_col not in result.columns:
            continue
        params = tasks.get(task, {})
        if raw_col not in result.columns:
            result[raw_col] = result[pred_col]
        if not params.get("enabled", False):
            continue
        logits = _safe_logit(result[raw_col].to_numpy())
        calibrated_logits = params["coef"] * logits + params["intercept"]
        result[pred_col] = 1.0 / (1.0 + np.exp(-calibrated_logits))
    return result


def calibration_path(results_dir: Path) -> Path:
    return results_dir / "probability_calibration.json"


def save_probability_calibration(results_dir: Path, calibration: dict) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = calibration_path(results_dir)
    path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_probability_calibration(results_dir: Path) -> dict | None:
    path = calibration_path(results_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_summary_path(results_dir: Path) -> Path:
    return results_dir / "probability_calibration_summary.json"


def calibrated_test_predictions_path(results_dir: Path) -> Path:
    return results_dir / "test_predictions_calibrated.csv"


def save_calibration_summary(results_dir: Path, payload: dict) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = calibration_summary_path(results_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path