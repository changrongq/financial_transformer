from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import load_config
from .features import LAYER_FEATURES
from .logging_utils import setup_logger
from .multitask_models import MultiTaskForecastModel

logger = setup_logger("hierarchical.multitask_training")
MULTITASK_LAYERS = ["micro", "small", "mid", "large"]

TARGET_COLUMNS = {
    "short_up": "target_up_5d",
    "take_profit": "target_take_profit_45d",
    "stop_loss": "target_stop_loss_45d",
    "max_gain": "max_gain_45d",
    "max_drawdown": "max_drawdown_45d",
}


def _feature_columns(layer: str | None = None) -> list[str]:
    base_columns = sorted({column for columns in LAYER_FEATURES.values() for column in columns})
    if layer is None:
        return [*base_columns, "market_cap_yi", "layer_micro", "layer_small", "layer_mid", "layer_large"]
    return [*base_columns, "market_cap_yi"]


class MultiTaskSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, targets: dict[str, np.ndarray], meta: pd.DataFrame):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.targets = {name: torch.tensor(values, dtype=torch.float32) for name, values in targets.items()}
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        sample_targets = {name: values[idx] for name, values in self.targets.items()}
        return self.x[idx], sample_targets, self.meta.iloc[idx].to_dict()


def _load_multitask_features(features_db: str, split: str, layer: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(features_db)
    try:
        if layer is None:
            query = "SELECT * FROM multitask_features WHERE split = ? ORDER BY stock_code, date"
            params = [split]
        else:
            query = "SELECT * FROM multitask_features WHERE split = ? AND layer = ? ORDER BY stock_code, date"
            params = [split, layer]
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    finally:
        conn.close()
    if df.empty:
        return df
    if layer is None:
        for layer_name in MULTITASK_LAYERS:
            df[f"layer_{layer_name}"] = (df["layer"] == layer_name).astype(float)
    return df


def _build_multitask_sequences(df: pd.DataFrame, feature_cols: list[str], lookback: int):
    xs = []
    targets = {name: [] for name in TARGET_COLUMNS}
    metas = []
    for stock_code, group in df.groupby("stock_code"):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        target_arrays = {name: group[column].to_numpy(dtype=np.float32) for name, column in TARGET_COLUMNS.items()}
        for idx in range(lookback - 1, len(group)):
            window = values[idx - lookback + 1: idx + 1]
            if not np.isfinite(window).all():
                continue
            target_values = {name: float(array[idx]) for name, array in target_arrays.items()}
            if not all(np.isfinite(value) for value in target_values.values()):
                continue
            xs.append(window)
            for name, value in target_values.items():
                targets[name].append(value)
            metas.append({
                "stock_code": str(stock_code).zfill(6),
                "date": str(group.loc[idx, "date"].date()),
                "layer": group.loc[idx, "layer"],
            })
    if not xs:
        empty_x = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
        empty_targets = {name: np.empty((0,), dtype=np.float32) for name in TARGET_COLUMNS}
        return empty_x, empty_targets, pd.DataFrame(columns=["stock_code", "date", "layer"])
    return np.stack(xs), {name: np.array(values, dtype=np.float32) for name, values in targets.items()}, pd.DataFrame(metas)


def _make_loader(features_db: str, split: str, feature_cols: list[str], lookback: int, batch_size: int, shuffle: bool, layer: str | None = None):
    df = _load_multitask_features(features_db, split, layer=layer)
    x, targets, meta = _build_multitask_sequences(df, feature_cols, lookback)
    dataset = MultiTaskSequenceDataset(x, targets, meta)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), len(dataset), targets


def _artifact_prefix(layer: str | None) -> str:
    return "multitask" if layer is None else f"multitask_{layer}"


def _compute_pos_weight(labels: np.ndarray) -> float | None:
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if positives <= 0 or negatives <= 0:
        return None
    return negatives / positives


def _classification_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    outputs = {name: [] for name in TARGET_COLUMNS}
    labels = {name: [] for name in TARGET_COLUMNS}
    metas = []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            batch_outputs = model(x)
            outputs["short_up"].extend(torch.sigmoid(batch_outputs["short_up_logits"]).cpu().numpy().tolist())
            outputs["take_profit"].extend(torch.sigmoid(batch_outputs["take_profit_logits"]).cpu().numpy().tolist())
            outputs["stop_loss"].extend(torch.sigmoid(batch_outputs["stop_loss_logits"]).cpu().numpy().tolist())
            outputs["max_gain"].extend(batch_outputs["max_gain"].cpu().numpy().tolist())
            outputs["max_drawdown"].extend(batch_outputs["max_drawdown"].cpu().numpy().tolist())
            for name in TARGET_COLUMNS:
                labels[name].extend(y[name].cpu().numpy().tolist())
            metas.append(pd.DataFrame(meta))

    metrics = {
        "auc_short_up": _classification_auc(np.array(labels["short_up"]), np.array(outputs["short_up"])),
        "auc_take_profit": _classification_auc(np.array(labels["take_profit"]), np.array(outputs["take_profit"])),
        "auc_stop_loss": _classification_auc(np.array(labels["stop_loss"]), np.array(outputs["stop_loss"])),
        "mae_max_gain": float(mean_absolute_error(labels["max_gain"], outputs["max_gain"])) if labels["max_gain"] else 0.0,
        "mae_max_drawdown": float(mean_absolute_error(labels["max_drawdown"], outputs["max_drawdown"])) if labels["max_drawdown"] else 0.0,
    }
    metrics["composite"] = (
        metrics["auc_short_up"] + metrics["auc_take_profit"] + metrics["auc_stop_loss"]
    ) / 3.0 - metrics["mae_max_gain"] - metrics["mae_max_drawdown"]
    metrics["business_score"] = (
        0.45 * metrics["auc_short_up"]
        + 0.35 * metrics["auc_take_profit"]
        + 0.20 * metrics["auc_stop_loss"]
        - 0.20 * metrics["mae_max_gain"]
        - 0.10 * metrics["mae_max_drawdown"]
    )

    prediction_df = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    if not prediction_df.empty:
        prediction_df["pred_short_up"] = outputs["short_up"]
        prediction_df["pred_take_profit"] = outputs["take_profit"]
        prediction_df["pred_stop_loss"] = outputs["stop_loss"]
        prediction_df["pred_max_gain"] = outputs["max_gain"]
        prediction_df["pred_max_drawdown"] = outputs["max_drawdown"]
        prediction_df["label_short_up"] = labels["short_up"]
        prediction_df["label_take_profit"] = labels["take_profit"]
        prediction_df["label_stop_loss"] = labels["stop_loss"]
        prediction_df["label_max_gain"] = labels["max_gain"]
        prediction_df["label_max_drawdown"] = labels["max_drawdown"]
    return metrics, prediction_df


def train_multitask_model(
    config_path: str | Path,
    epochs_override: int | None = None,
    patience_override: int | None = None,
    batch_size_override: int | None = None,
    layer: str | None = None,
) -> dict:
    config = load_config(config_path)
    if layer is not None and layer not in MULTITASK_LAYERS:
        raise ValueError(f"未知多任务分层: {layer}")
    if epochs_override is not None:
        config.multitask.epochs = int(epochs_override)
    if patience_override is not None:
        config.multitask.early_stopping_patience = int(patience_override)
    if batch_size_override is not None:
        config.multitask.batch_size = int(batch_size_override)
    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    feature_cols = _feature_columns(layer=layer)
    lookback = config.training.lookback_window

    train_loader, train_size, train_targets = _make_loader(config.databases.features_db, "train", feature_cols, lookback, config.multitask.batch_size, True, layer=layer)
    val_loader, val_size, _ = _make_loader(config.databases.features_db, "val", feature_cols, lookback, config.multitask.batch_size, False, layer=layer)
    test_loader, test_size, _ = _make_loader(config.databases.features_db, "test", feature_cols, lookback, config.multitask.batch_size, False, layer=layer)
    if train_size == 0 or val_size == 0:
        raise RuntimeError(f"多任务样本不足，无法训练。 layer={layer or 'all'}")

    model = MultiTaskForecastModel(
        input_dim=len(feature_cols),
        hidden_dim=config.multitask.hidden_dim,
        dropout=config.multitask.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.multitask.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.multitask.epochs)

    pos_weights = {
        name: _compute_pos_weight(train_targets[name])
        for name in ["short_up", "take_profit", "stop_loss"]
    }
    cls_losses = {
        name: nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weights[name], dtype=torch.float32, device=device) if pos_weights[name] is not None else None
        )
        for name in ["short_up", "take_profit", "stop_loss"]
    }
    reg_loss = nn.SmoothL1Loss(beta=0.05)

    best_metric = -1e9
    patience = 0
    best_state = None
    history = []

    for epoch in range(1, config.multitask.epochs + 1):
        model.train()
        batch_losses = []
        for x, y, _ in train_loader:
            x = x.to(device)
            y = {name: tensor.to(device) for name, tensor in y.items()}
            outputs = model(x)
            loss = (
                config.multitask.task_weights["short_up"] * cls_losses["short_up"](outputs["short_up_logits"], y["short_up"])
                + config.multitask.task_weights["take_profit"] * cls_losses["take_profit"](outputs["take_profit_logits"], y["take_profit"])
                + config.multitask.task_weights["stop_loss"] * cls_losses["stop_loss"](outputs["stop_loss_logits"], y["stop_loss"])
                + config.multitask.task_weights["max_gain"] * reg_loss(outputs["max_gain"], y["max_gain"])
                + config.multitask.task_weights["max_drawdown"] * reg_loss(outputs["max_drawdown"], y["max_drawdown"])
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        scheduler.step()

        val_metrics, _ = _evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), **val_metrics})
        logger.info(
            "multitask layer=%s epoch=%s loss=%.4f composite=%.4f auc_short=%.4f auc_tp=%.4f auc_sl=%.4f",
            layer or "all",
            epoch,
            float(np.mean(batch_losses)),
            val_metrics["composite"],
            val_metrics["auc_short_up"],
            val_metrics["auc_take_profit"],
            val_metrics["auc_stop_loss"],
        )

        if val_metrics[config.multitask.validation_metric] > best_metric:
            best_metric = val_metrics[config.multitask.validation_metric]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.multitask.early_stopping_patience:
                logger.info("multitask layer=%s early stopping at epoch %s", layer or "all", epoch)
                break

    if best_state is None:
        raise RuntimeError("多任务训练未生成有效模型。")

    model.load_state_dict(best_state)
    test_metrics, test_predictions = _evaluate(model, test_loader, device)

    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = _artifact_prefix(layer)
    model_path = config.models_dir / f"{prefix}_model.pt"
    history_path = config.results_dir / f"{prefix}_history.json"
    metrics_path = config.results_dir / f"{prefix}_metrics.json"
    pred_path = config.results_dir / f"{prefix}_predictions.csv"

    torch.save(
        {
            "layer": layer,
            "feature_cols": feature_cols,
            "lookback": lookback,
            "hidden_dim": config.multitask.hidden_dim,
            "dropout": config.multitask.dropout,
            "model_state_dict": model.state_dict(),
            "task_weights": config.multitask.task_weights,
        },
        model_path,
    )
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"layer": layer or "all", "train_size": train_size, "val_size": val_size, "test_size": test_size, **test_metrics}, handle, ensure_ascii=False, indent=2)
    if not test_predictions.empty:
        test_predictions["model_layer"] = layer or "all"
        test_predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return {"layer": layer or "all", "model_path": str(model_path), "metrics_path": str(metrics_path), **test_metrics}