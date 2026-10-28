from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .config import load_config
from .features import LAYER_FEATURES
from .logging_utils import setup_logger
from .models import build_model

logger = setup_logger("hierarchical.training")


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.meta.iloc[idx].to_dict()


def _load_layer_features(features_db: str, layer: str, split: str) -> pd.DataFrame:
    conn = sqlite3.connect(features_db)
    try:
        query = "SELECT * FROM hierarchical_features WHERE layer = ? AND split = ? ORDER BY stock_code, date"
        return pd.read_sql_query(query, conn, params=[layer, split], parse_dates=["date"])
    finally:
        conn.close()


def _build_sequences(df: pd.DataFrame, feature_cols: list[str], lookback: int):
    xs, ys, metas = [], [], []
    for stock_code, group in df.groupby("stock_code"):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        labels = group["target_up"].to_numpy(dtype=np.float32)
        future_returns = group["future_return"].to_numpy(dtype=np.float32)
        dates = group["date"].astype(str).to_list()
        for idx in range(lookback - 1, len(group)):
            window = values[idx - lookback + 1: idx + 1]
            if not np.isfinite(window).all():
                continue
            xs.append(window)
            ys.append(labels[idx])
            metas.append({
                "stock_code": stock_code,
                "date": dates[idx],
                "future_return": float(future_returns[idx]),
            })
    if not xs:
        return np.empty((0, lookback, len(feature_cols))), np.empty((0,)), pd.DataFrame(columns=["stock_code", "date", "future_return"])
    return np.stack(xs), np.array(ys), pd.DataFrame(metas)


def _make_loader(features_db: str, layer: str, split: str, feature_cols: list[str], lookback: int, batch_size: int, shuffle: bool):
    df = _load_layer_features(features_db, layer, split)
    x, y, meta = _build_sequences(df, feature_cols, lookback)
    dataset = SequenceDataset(x, y, meta)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), len(dataset), y


def _compute_pos_weight(labels: np.ndarray) -> float | None:
    if labels.size == 0:
        return None
    positive = float(labels.sum())
    negative = float(labels.size - positive)
    if positive <= 0 or negative <= 0:
        return None
    return negative / positive


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, labels, metas = [], [], []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            preds.extend(prob.tolist())
            labels.extend(y.numpy().tolist())
            if isinstance(meta, dict):
                meta_df = pd.DataFrame(meta)
            else:
                meta_df = pd.DataFrame(list(meta))
            metas.append(meta_df)
    if not preds:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0}, pd.DataFrame()
    pred_label = [1 if p >= 0.5 else 0 for p in preds]
    metrics = {
        "accuracy": accuracy_score(labels, pred_label),
        "precision": precision_score(labels, pred_label, zero_division=0),
        "recall": recall_score(labels, pred_label, zero_division=0),
        "auc": roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.0,
    }
    meta_df = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    if not meta_df.empty:
        meta_df["prediction_prob"] = preds
        meta_df["label"] = labels
    return metrics, meta_df


def train_layer_model(config_path: str, layer: str) -> dict:
    config = load_config(config_path)
    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    feature_cols = LAYER_FEATURES[layer]
    lookback = config.training.lookback_window

    train_loader, train_size, train_labels = _make_loader(config.databases.features_db, layer, "train", feature_cols, lookback, config.training.batch_size, True)
    val_loader, val_size, _ = _make_loader(config.databases.features_db, layer, "val", feature_cols, lookback, config.training.batch_size, False)
    test_loader, test_size, _ = _make_loader(config.databases.features_db, layer, "test", feature_cols, lookback, config.training.batch_size, False)

    if train_size == 0 or val_size == 0:
        raise RuntimeError(f"{layer} 分层样本不足，无法训练。")

    model = build_model(layer, len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    pos_weight = _compute_pos_weight(train_labels)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight is not None else None
    )
    logger.info("layer=%s train_pos_rate=%.4f pos_weight=%s", layer, float(train_labels.mean()) if train_labels.size else 0.0, f"{pos_weight:.4f}" if pos_weight is not None else "none")

    best_metric = -1.0
    patience = 0
    history = []
    best_state = None

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        epoch_losses = []
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics, _ = _evaluate(model, val_loader, device)
        val_metric = val_metrics.get(config.training.validation_metric, val_metrics["auc"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), **val_metrics})
        logger.info("layer=%s epoch=%s loss=%.4f auc=%.4f acc=%.4f", layer, epoch, np.mean(epoch_losses), val_metrics["auc"], val_metrics["accuracy"])

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience:
                logger.info("layer=%s early stopping at epoch %s", layer, epoch)
                break

    if best_state is None:
        raise RuntimeError("训练未生成有效模型。")

    model.load_state_dict(best_state)
    test_metrics, test_predictions = _evaluate(model, test_loader, device)
    test_metrics = {k: float(v) for k, v in test_metrics.items()}

    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.models_dir / f"{layer}_model.pt"
    history_path = config.results_dir / f"{layer}_history.json"
    metrics_path = config.results_dir / f"{layer}_metrics.json"
    pred_path = config.results_dir / f"{layer}_predictions.csv"

    torch.save({
        "layer": layer,
        "feature_cols": feature_cols,
        "lookback": lookback,
        "model_state_dict": model.state_dict(),
    }, model_path)

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"layer": layer, "train_size": train_size, "val_size": val_size, "test_size": test_size, **test_metrics}, f, ensure_ascii=False, indent=2)
    if not test_predictions.empty:
        test_predictions["layer"] = layer
        test_predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return {"model_path": str(model_path), "metrics_path": str(metrics_path), **test_metrics}

