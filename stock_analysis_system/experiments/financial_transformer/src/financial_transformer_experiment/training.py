from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from hierarchical_system.logging_utils import setup_logger

from .calibration import apply_probability_calibration
from .calibration import calibrated_test_predictions_path
from .calibration import fit_probability_calibration
from .calibration import save_probability_calibration
from .config import ExperimentConfig
from .data import TARGET_COLUMNS, build_datasets, collate_sequence_batch
from .model import create_model
from .scoring import enrich_prediction_frame, strategy_score_frame

logger = setup_logger("financial_transformer.experiment")


@dataclass(frozen=True)
class TrainingStagePlan:
    name: str
    dataset_name: str
    epochs: int
    learning_rate: float
    patience_limit: int
    task_weights: dict[str, float] | None = None
    trainable_modules: list[str] | None = None


def _parameter_matches_module_prefix(parameter_name: str, module_prefix: str) -> bool:
    return parameter_name == module_prefix or parameter_name.startswith(f"{module_prefix}.")


def _configure_trainable_modules(model: nn.Module, trainable_modules: list[str] | None) -> list[nn.Parameter]:
    all_named_parameters = list(model.named_parameters())
    if not trainable_modules:
        for _, parameter in all_named_parameters:
            parameter.requires_grad = True
        return [parameter for _, parameter in all_named_parameters]

    requested_modules = [str(module_name).strip() for module_name in trainable_modules if str(module_name).strip()]
    if not requested_modules:
        raise ValueError("trainable_modules 不能为空列表。")

    trainable_parameters: list[nn.Parameter] = []
    matched_modules: set[str] = set()
    for parameter_name, parameter in all_named_parameters:
        is_trainable = any(
            _parameter_matches_module_prefix(parameter_name, module_prefix)
            for module_prefix in requested_modules
        )
        parameter.requires_grad = is_trainable
        if is_trainable:
            trainable_parameters.append(parameter)
            for module_prefix in requested_modules:
                if _parameter_matches_module_prefix(parameter_name, module_prefix):
                    matched_modules.add(module_prefix)

    unmatched_modules = [module_name for module_name in requested_modules if module_name not in matched_modules]
    if unmatched_modules:
        available_modules = sorted(name for name, _ in model.named_modules() if name)
        raise ValueError(
            "以下 trainable_modules 未匹配到模型参数: "
            f"{unmatched_modules}. 可用模块示例: {available_modules}"
        )
    if not trainable_parameters:
        raise ValueError("当前 phase 没有任何可训练参数，请检查 trainable_modules 配置。")
    return trainable_parameters


def _merge_task_weights(base_task_weights: dict[str, float], overrides: dict[str, float] | None) -> dict[str, float]:
    merged = {str(name): float(value) for name, value in base_task_weights.items()}
    if overrides:
        for name, value in overrides.items():
            merged[str(name)] = float(value)
    return merged


def _clone_config_with_task_weights(config: ExperimentConfig, task_weights: dict[str, float] | None) -> ExperimentConfig:
    if task_weights is None:
        return config
    return replace(
        config,
        multitask=replace(
            config.multitask,
            task_weights=_merge_task_weights(config.multitask.task_weights, task_weights),
        ),
    )


def _validate_dataset_diagnostics(diagnostics: dict[str, object] | None, config: ExperimentConfig) -> None:
    if not diagnostics:
        return

    failures: list[str] = []
    retained_ratio = diagnostics.get("retained_ratio")
    min_data_retained_ratio = config.training.min_data_retained_ratio
    if min_data_retained_ratio is not None and retained_ratio is not None:
        retained_ratio = float(retained_ratio)
        if retained_ratio < float(min_data_retained_ratio):
            failures.append(
                f"data retained ratio {retained_ratio:.4f} < training.min_data_retained_ratio {float(min_data_retained_ratio):.4f}"
            )

    raw_row_count = int(diagnostics.get("raw_row_count") or 0)
    max_missing_feature_ratio = config.training.max_missing_feature_ratio
    if max_missing_feature_ratio is not None and raw_row_count > 0:
        violating_columns: list[str] = []
        for item in diagnostics.get("top_missing_columns") or []:
            column_name = str(item.get("column"))
            missing_count = int(item.get("missing_count") or 0)
            missing_ratio = missing_count / raw_row_count
            if missing_ratio > float(max_missing_feature_ratio):
                violating_columns.append(f"{column_name}={missing_count}/{raw_row_count} ({missing_ratio:.4f})")
        if violating_columns:
            failures.append(
                "missing feature ratio exceeded training.max_missing_feature_ratio "
                f"{float(max_missing_feature_ratio):.4f}: {', '.join(violating_columns)}"
            )

    zero_turnover_fill_ratio = diagnostics.get("zero_turnover_fill_ratio")
    max_zero_turnover_fill_ratio = config.training.max_zero_turnover_fill_ratio
    if max_zero_turnover_fill_ratio is not None and zero_turnover_fill_ratio is not None:
        zero_turnover_fill_ratio = float(zero_turnover_fill_ratio)
        if zero_turnover_fill_ratio > float(max_zero_turnover_fill_ratio):
            failures.append(
                "zero turnover fill ratio exceeded training.max_zero_turnover_fill_ratio "
                f"{zero_turnover_fill_ratio:.4f} > {float(max_zero_turnover_fill_ratio):.4f}"
            )

    if failures:
        raise RuntimeError("训练前数据质量检查失败: " + " | ".join(failures))


class _IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices: list[int]):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        index_tensor = torch.tensor(self.indices, dtype=torch.long)
        self.targets = {name: values[index_tensor] for name, values in base_dataset.targets.items()}
        self.meta = base_dataset.meta.iloc[self.indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.indices[index]]


class _DateGroupedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, shuffle: bool):
        self.batch_size = batch_size
        self.shuffle = shuffle
        grouped = dataset.meta.groupby("date").indices if hasattr(dataset, "meta") else {}
        self.groups = [np.array(indices, dtype=np.int64) for _, indices in sorted(grouped.items()) if len(indices) > 0]

    def __iter__(self):
        group_order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.shuffle(group_order)
        for group_index in group_order:
            indices = self.groups[group_index].copy()
            if self.shuffle:
                np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return int(sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups))


def _build_loader(dataset, batch_size: int, shuffle: bool, group_by_date: bool = False) -> DataLoader:
    if group_by_date and hasattr(dataset, "meta") and not dataset.meta.empty:
        return DataLoader(
            dataset,
            batch_sampler=_DateGroupedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle),
            collate_fn=collate_sequence_batch,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sequence_batch)


def _classification_auc(labels: list[float], preds: list[float]) -> float:
    if not labels or len(set(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, preds))


def _safe_corr(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    usable = pd.DataFrame({"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")}).dropna()
    if len(usable) < 2:
        return 0.0
    value = usable["a"].corr(usable["b"], method=method)
    if pd.isna(value):
        return 0.0
    return float(value)


def _compute_pos_weight(labels: torch.Tensor) -> float | None:
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    if positives <= 0 or negatives <= 0:
        return None
    return negatives / positives


def _compute_time_decay_weights(meta: pd.DataFrame, half_life_days: float) -> pd.Series:
    if meta.empty:
        return pd.Series(dtype=np.float32)
    date_series = pd.to_datetime(meta["date"])
    latest_date = date_series.max()
    age_days = (latest_date - date_series).dt.days.clip(lower=0).astype(float)
    raw_weights = np.power(0.5, age_days / float(half_life_days))
    mean_weight = float(np.mean(raw_weights)) if len(raw_weights) else 1.0
    normalized = raw_weights / max(mean_weight, 1e-6)
    return pd.Series(normalized, index=meta.index, dtype=np.float32)


def _attach_time_decay_weights(dataset, half_life_days: float | None) -> dict[str, float] | None:
    if half_life_days is None or not hasattr(dataset, "meta") or dataset.meta.empty:
        return None
    sample_weights = _compute_time_decay_weights(dataset.meta, half_life_days)
    dataset.meta = dataset.meta.copy()
    dataset.meta["sample_weight"] = sample_weights.astype(float)
    return {
        "min": float(sample_weights.min()),
        "mean": float(sample_weights.mean()),
        "max": float(sample_weights.max()),
    }


def _weighted_loss_mean(loss_tensor: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if loss_tensor.ndim > 1:
        loss_tensor = loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)
    return (loss_tensor * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


def _build_loss_functions(train_dataset, config: ExperimentConfig, device: torch.device):
    pos_weights = {
        name: _compute_pos_weight(train_dataset.targets[name])
        for name in ["short_up", "take_profit", "stop_loss"]
    }
    cls_losses = {
        name: nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weights[name], dtype=torch.float32, device=device) if pos_weights[name] is not None else None,
            reduction="none",
        )
        for name in ["short_up", "take_profit", "stop_loss"]
    }
    reg_loss = nn.SmoothL1Loss(beta=config.multitask.regression_loss_beta, reduction="none")
    return cls_losses, reg_loss


def _build_recent_finetune_dataset(train_dataset, window_days: int | None):
    if window_days is None or not hasattr(train_dataset, "meta") or train_dataset.meta.empty:
        return None, None
    train_dates = pd.to_datetime(train_dataset.meta["date"])
    latest_date = train_dates.max()
    cutoff_date = latest_date - pd.Timedelta(days=int(window_days))
    recent_indices = np.flatnonzero((train_dates >= cutoff_date).to_numpy()).tolist()
    if not recent_indices:
        return None, None
    recent_dataset = _IndexedDataset(train_dataset, recent_indices)
    stats = {
        "window_days": int(window_days),
        "sample_count": int(len(recent_dataset)),
        "min_date": str(pd.to_datetime(recent_dataset.meta["date"]).min().date()),
        "max_date": str(pd.to_datetime(recent_dataset.meta["date"]).max().date()),
    }
    return recent_dataset, stats


def _run_training_stage(
    *,
    stage_name: str,
    model: nn.Module,
    train_dataset,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience_limit: int,
    starting_epoch: int,
    history: list[dict],
    best_metric: float,
    best_state,
    best_epoch: int,
    best_val_metrics: dict[str, float] | None,
    trainable_modules: list[str] | None = None,
):
    use_date_grouped_batches = (
        config.multitask.short_up_ranking_weight > 0.0 and config.multitask.short_up_ranking_same_date_only
    )
    train_loader = _build_loader(train_dataset, config.training.batch_size, True, group_by_date=use_date_grouped_batches)
    cls_losses, reg_loss = _build_loss_functions(train_dataset, config, device)
    trainable_parameters = _configure_trainable_modules(model, trainable_modules)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    patience = 0
    stage_best_metric = -1e9
    stage_best_state = {key: value.cpu() for key, value in model.state_dict().items()}
    stage_best_epoch = starting_epoch

    for stage_epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []

        for x, y, _, padding_mask, meta in train_loader:
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            y = {name: tensor.to(device) for name, tensor in y.items()}
            sample_weights = torch.tensor(
                [float(item.get("sample_weight", 1.0)) for item in meta],
                dtype=torch.float32,
                device=device,
            )
            outputs = model(x, padding_mask=padding_mask)
            loss = (
                config.multitask.task_weights["short_up"] * _weighted_loss_mean(cls_losses["short_up"](outputs["short_up_logits"], y["short_up"]), sample_weights)
                + config.multitask.task_weights["take_profit"] * _weighted_loss_mean(cls_losses["take_profit"](outputs["take_profit_logits"], y["take_profit"]), sample_weights)
                + config.multitask.task_weights["stop_loss"] * _weighted_loss_mean(cls_losses["stop_loss"](outputs["stop_loss_logits"], y["stop_loss"]), sample_weights)
                + config.multitask.task_weights["max_gain"] * _weighted_loss_mean(reg_loss(outputs["max_gain"], y["max_gain"]), sample_weights)
                + config.multitask.task_weights["max_drawdown"] * _weighted_loss_mean(reg_loss(outputs["max_drawdown"], y["max_drawdown"]), sample_weights)
            )
            if config.multitask.short_up_ranking_weight > 0.0:
                future_returns = torch.tensor(
                    [float(item["future_return"]) for item in meta],
                    dtype=torch.float32,
                    device=device,
                )
                ranking_loss = _pairwise_ranking_loss(
                    outputs["short_up_logits"],
                    future_returns=future_returns,
                    dates=[str(item["date"]) for item in meta],
                    min_return_gap=config.multitask.short_up_ranking_min_return_gap,
                    same_date_only=config.multitask.short_up_ranking_same_date_only,
                    max_pairs_per_batch=config.multitask.short_up_ranking_max_pairs_per_batch,
                )
                loss = loss + config.multitask.short_up_ranking_weight * ranking_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        scheduler.step()
        val_metrics, _ = _evaluate(model, val_loader, device, exit_label_mode=config.multitask.exit_label_mode)
        global_epoch = starting_epoch + stage_epoch
        epoch_row = {
            "epoch": global_epoch,
            "stage": stage_name,
            "stage_epoch": stage_epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            **{key: float(value) for key, value in val_metrics.items()},
        }
        history.append(epoch_row)
        validation_metric_name = config.multitask.validation_metric
        validation_metric = float(val_metrics.get(validation_metric_name, val_metrics["business_score"]))
        logger.info(
            "stage=%s epoch=%s loss=%.4f business_score=%.4f early_stop_metric[%s]=%.4f auc_short=%.4f auc_tp=%.4f auc_sl=%.4f",
            stage_name,
            global_epoch,
            epoch_row["train_loss"],
            epoch_row["business_score"],
            validation_metric_name,
            validation_metric,
            epoch_row["auc_short_up"],
            epoch_row["auc_take_profit"],
            epoch_row["auc_stop_loss"],
        )

        if validation_metric > best_metric:
            best_metric = validation_metric
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            best_epoch = global_epoch
            best_val_metrics = {key: float(value) for key, value in val_metrics.items()}
            patience = 0
            logger.info(
                "new best stage=%s epoch=%s early_stop_metric[%s]=%.4f",
                stage_name,
                global_epoch,
                validation_metric_name,
                validation_metric,
            )
        else:
            patience += 1
            if patience >= patience_limit:
                logger.info("early stopping stage=%s at epoch %s", stage_name, global_epoch)
                break

        if validation_metric > stage_best_metric:
            stage_best_metric = validation_metric
            stage_best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            stage_best_epoch = global_epoch

    return (
        best_metric,
        best_state,
        best_epoch,
        best_val_metrics,
        starting_epoch + stage_epoch,
        stage_best_metric,
        stage_best_state,
        stage_best_epoch,
    )


def _pairwise_ranking_loss(
    logits: torch.Tensor,
    future_returns: torch.Tensor,
    dates: list[str],
    min_return_gap: float,
    same_date_only: bool,
    max_pairs_per_batch: int,
) -> torch.Tensor:
    batch_size = int(logits.shape[0])
    if batch_size < 2:
        return logits.new_tensor(0.0)

    return_diff = future_returns.unsqueeze(1) - future_returns.unsqueeze(0)
    pair_mask = return_diff > float(min_return_gap)
    if same_date_only:
        date_codes, _ = pd.factorize(np.asarray(dates, dtype=object))
        date_codes_tensor = torch.tensor(date_codes, device=logits.device)
        pair_mask = pair_mask & (date_codes_tensor.unsqueeze(1) == date_codes_tensor.unsqueeze(0))

    pair_mask.fill_diagonal_(False)
    pair_indices = pair_mask.nonzero(as_tuple=False)
    if pair_indices.numel() == 0:
        return logits.new_tensor(0.0)

    if max_pairs_per_batch > 0 and pair_indices.shape[0] > max_pairs_per_batch:
        sampled = torch.randperm(pair_indices.shape[0], device=logits.device)[:max_pairs_per_batch]
        pair_indices = pair_indices[sampled]

    positive_idx = pair_indices[:, 0]
    negative_idx = pair_indices[:, 1]
    logit_diff = logits[positive_idx] - logits[negative_idx]
    pair_weights = return_diff[positive_idx, negative_idx].clamp_min(float(min_return_gap))
    pair_weights = pair_weights / pair_weights.mean().clamp_min(1e-6)
    return (torch.nn.functional.softplus(-logit_diff) * pair_weights).mean()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    exit_label_mode: str = "window_any",
):
    model.eval()
    outputs = {name: [] for name in TARGET_COLUMNS}
    labels = {name: [] for name in TARGET_COLUMNS}
    metas = []

    with torch.no_grad():
        for x, y, _, padding_mask, meta in loader:
            batch_outputs = model(x.to(device), padding_mask=padding_mask.to(device))
            outputs["short_up"].extend(torch.sigmoid(batch_outputs["short_up_logits"]).cpu().numpy().tolist())
            outputs["take_profit"].extend(torch.sigmoid(batch_outputs["take_profit_logits"]).cpu().numpy().tolist())
            outputs["stop_loss"].extend(torch.sigmoid(batch_outputs["stop_loss_logits"]).cpu().numpy().tolist())
            outputs["max_gain"].extend(batch_outputs["max_gain"].cpu().numpy().tolist())
            outputs["max_drawdown"].extend(batch_outputs["max_drawdown"].cpu().numpy().tolist())
            for name in TARGET_COLUMNS:
                labels[name].extend(y[name].cpu().numpy().tolist())
            metas.append(pd.DataFrame(meta))

    metrics = {
        "auc_short_up": _classification_auc(labels["short_up"], outputs["short_up"]),
        "auc_take_profit": _classification_auc(labels["take_profit"], outputs["take_profit"]),
        "auc_stop_loss": _classification_auc(labels["stop_loss"], outputs["stop_loss"]),
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
    metrics["auc"] = metrics["auc_short_up"]

    meta_df = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    if not meta_df.empty:
        meta_df["pred_short_up"] = outputs["short_up"]
        meta_df["pred_take_profit"] = outputs["take_profit"]
        meta_df["pred_stop_loss"] = outputs["stop_loss"]
        meta_df["pred_max_gain"] = outputs["max_gain"]
        meta_df["pred_max_drawdown"] = outputs["max_drawdown"]
        meta_df["label_short_up"] = labels["short_up"]
        meta_df["label_take_profit"] = labels["take_profit"]
        meta_df["label_stop_loss"] = labels["stop_loss"]
        meta_df["label_max_gain"] = labels["max_gain"]
        meta_df["label_max_drawdown"] = labels["max_drawdown"]
        meta_df["prediction_prob"] = meta_df["pred_short_up"]
        meta_df["label"] = meta_df["label_short_up"]
        meta_df = enrich_prediction_frame(meta_df, exit_label_mode=exit_label_mode)
        meta_df["signal_score"] = strategy_score_frame(meta_df, exit_label_mode=exit_label_mode)
        metrics["signal_future_pearson"] = _safe_corr(meta_df["signal_score"], meta_df["future_return"], "pearson")
        metrics["signal_future_spearman"] = _safe_corr(meta_df["signal_score"], meta_df["future_return"], "spearman")

        passed = meta_df[
            (pd.to_numeric(meta_df["pred_short_up"], errors="coerce") >= 0.5)
            & (pd.to_numeric(meta_df["pred_take_profit"], errors="coerce") >= 0.5)
            & (pd.to_numeric(meta_df["pred_stop_loss"], errors="coerce") < 0.5)
        ].copy()
        if not passed.empty:
            metrics["pass_signal_future_spearman"] = _safe_corr(passed["signal_score"], passed["future_return"], "spearman")
            metrics["pass_mean_future_return"] = float(pd.to_numeric(passed["future_return"], errors="coerce").mean())
        else:
            metrics["pass_signal_future_spearman"] = 0.0
            metrics["pass_mean_future_return"] = 0.0

        top1 = (
            meta_df.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
            .groupby("date", as_index=False)
            .head(1)
        )
        metrics["top1_mean_future_return"] = float(pd.to_numeric(top1["future_return"], errors="coerce").mean()) if not top1.empty else 0.0
    return metrics, meta_df


def _prepare_experiment_runtime(config: ExperimentConfig):
    datasets = build_datasets(config)
    _validate_dataset_diagnostics(datasets.diagnostics, config)
    if len(datasets.train) == 0 or len(datasets.val) == 0:
        raise RuntimeError("实验数据不足，无法完成训练。请检查股票池和时间切分。")

    recent_dataset, recent_finetune_stats = _build_recent_finetune_dataset(
        datasets.train,
        config.training.recent_finetune_window_days,
    )

    time_decay_stats = _attach_time_decay_weights(datasets.train, config.training.time_decay_half_life_days)
    if time_decay_stats is not None:
        logger.info(
            "enabled time decay half_life_days=%.1f sample_weight[min=%.4f mean=%.4f max=%.4f]",
            config.training.time_decay_half_life_days,
            time_decay_stats["min"],
            time_decay_stats["mean"],
            time_decay_stats["max"],
        )

    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    model = create_model(
        input_dim=datasets.input_dim,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        ffn_type=config.model.ffn_type,
        position_encoding=config.model.position_encoding,
        model_variant=config.model.model_variant,
        feature_cols=config.experiment.feature_cols,
        grouped_feature_groups=config.model.grouped_feature_groups,
        use_group_gates=config.model.use_group_gates,
    ).to(device)

    return {
        "datasets": datasets,
        "recent_dataset": recent_dataset,
        "recent_finetune_stats": recent_finetune_stats,
        "time_decay_stats": time_decay_stats,
        "device": device,
        "model": model,
        "val_loader": _build_loader(datasets.val, config.training.batch_size, False),
        "test_loader": _build_loader(datasets.test, config.training.batch_size, False),
    }


def _execute_stage_plans(
    *,
    base_config: ExperimentConfig,
    model: nn.Module,
    datasets,
    recent_dataset,
    val_loader: DataLoader,
    device: torch.device,
    stage_plans: list[TrainingStagePlan],
):
    dataset_lookup = {"train": datasets.train}
    if recent_dataset is not None:
        dataset_lookup["recent"] = recent_dataset

    best_metric = -1e9
    best_state = None
    best_epoch = 0
    best_val_metrics: dict[str, float] | None = None
    history: list[dict] = []
    last_epoch = 0
    current_stage_state = None

    for stage_plan in stage_plans:
        if int(stage_plan.epochs) <= 0:
            logger.info("skip stage=%s because epochs=%s", stage_plan.name, stage_plan.epochs)
            continue

        train_dataset = dataset_lookup.get(stage_plan.dataset_name)
        if train_dataset is None:
            raise ValueError(
                f"stage {stage_plan.name} 请求的数据集 {stage_plan.dataset_name!r} 不存在，可用数据集: {sorted(dataset_lookup)}"
            )

        if current_stage_state is not None:
            model.load_state_dict(current_stage_state)

        stage_config = _clone_config_with_task_weights(base_config, stage_plan.task_weights)
        logger.info(
            "start stage=%s dataset=%s epochs=%s lr=%.6f patience=%s trainable_modules=%s task_weights=%s",
            stage_plan.name,
            stage_plan.dataset_name,
            stage_plan.epochs,
            stage_plan.learning_rate,
            stage_plan.patience_limit,
            stage_plan.trainable_modules,
            stage_config.multitask.task_weights,
        )
        (
            best_metric,
            best_state,
            best_epoch,
            best_val_metrics,
            last_epoch,
            _stage_best_metric,
            stage_best_state,
            _stage_best_epoch,
        ) = _run_training_stage(
            stage_name=stage_plan.name,
            model=model,
            train_dataset=train_dataset,
            val_loader=val_loader,
            config=stage_config,
            device=device,
            epochs=int(stage_plan.epochs),
            learning_rate=float(stage_plan.learning_rate),
            patience_limit=int(stage_plan.patience_limit),
            starting_epoch=last_epoch,
            history=history,
            best_metric=best_metric,
            best_state=best_state,
            best_epoch=best_epoch,
            best_val_metrics=best_val_metrics,
            trainable_modules=stage_plan.trainable_modules,
        )
        current_stage_state = stage_best_state

    if best_state is None:
        raise RuntimeError("训练未能产生有效模型。")

    return {
        "best_metric": float(best_metric),
        "best_state": best_state,
        "best_epoch": int(best_epoch),
        "best_val_metrics": best_val_metrics or {},
        "history": history,
    }


def _persist_experiment_outputs(
    *,
    config: ExperimentConfig,
    model: nn.Module,
    datasets,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    best_state,
    best_epoch: int,
    best_metric: float,
    best_val_metrics: dict[str, float],
    history: list[dict],
    time_decay_stats: dict[str, float] | None,
    recent_finetune_stats: dict[str, object] | None,
    stage_plans: list[TrainingStagePlan],
    plan_name: str,
) -> dict:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_quality_summary_path = None
    data_quality_by_stock_path = None
    if datasets.diagnostics:
        data_quality_summary_path = config.results_dir / "data_quality_summary.json"
        data_quality_by_stock_path = config.results_dir / "data_quality_by_stock.csv"
        summary_payload = {key: value for key, value in datasets.diagnostics.items() if key != "per_stock"}
        with data_quality_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
        per_stock_rows = datasets.diagnostics.get("per_stock")
        if per_stock_rows:
            pd.DataFrame(per_stock_rows).to_csv(data_quality_by_stock_path, index=False, encoding="utf-8-sig")

    model.load_state_dict(best_state)
    _, val_predictions = _evaluate(model, val_loader, device, exit_label_mode=config.multitask.exit_label_mode)
    calibration = fit_probability_calibration(val_predictions)
    test_metrics, predictions = _evaluate(model, test_loader, device, exit_label_mode=config.multitask.exit_label_mode)
    test_metrics = {key: float(value) for key, value in test_metrics.items()}

    model_path = config.artifacts_dir / "financial_transformer.pt"
    history_path = config.results_dir / "history.json"
    metrics_path = config.results_dir / "metrics.json"
    predictions_path = config.results_dir / "test_predictions.csv"
    calibrated_predictions_path = calibrated_test_predictions_path(config.results_dir)
    calibration_file = save_probability_calibration(config.results_dir, calibration)
    stock_universe_path = config.results_dir / "selected_stock_universe.csv"

    if datasets.selected_stock_codes:
        pd.DataFrame({"stock_code": datasets.selected_stock_codes}).to_csv(stock_universe_path, index=False, encoding="utf-8-sig")

    torch.save(
        {
            "feature_cols": config.experiment.feature_cols,
            "selected_stock_codes": datasets.selected_stock_codes,
            "seq_len": config.model.seq_len,
            "min_seq_len": int(config.model.min_seq_len or config.model.seq_len),
            "multitask": True,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "d_model": config.model.d_model,
                "nhead": config.model.nhead,
                "num_layers": config.model.num_layers,
                "dim_feedforward": config.model.dim_feedforward,
                "dropout": config.model.dropout,
                "ffn_type": config.model.ffn_type,
                "position_encoding": config.model.position_encoding,
                "model_variant": config.model.model_variant,
                "grouped_feature_groups": config.model.grouped_feature_groups,
                "use_group_gates": config.model.use_group_gates,
                "stock_universe_path": str(stock_universe_path),
                "training_plan_name": plan_name,
                "training_stage_names": [stage.name for stage in stage_plans],
            },
        },
        model_path,
    )

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_size": len(datasets.train),
                "val_size": len(datasets.val),
                "test_size": len(datasets.test),
                "stock_pool": config.experiment.stock_pool,
                "layer_filter": config.experiment.layer_filter,
                "max_stocks": config.experiment.max_stocks,
                "seq_len": config.model.seq_len,
                "min_seq_len": int(config.model.min_seq_len or config.model.seq_len),
                "model_variant": config.model.model_variant,
                "use_group_gates": config.model.use_group_gates,
                "ffn_type": config.model.ffn_type,
                "position_encoding": config.model.position_encoding,
                "probability_calibration": str(calibration_file),
                "stock_universe_path": str(stock_universe_path),
                "stock_universe_size": len(datasets.selected_stock_codes),
                "data_quality_summary_path": str(data_quality_summary_path) if data_quality_summary_path is not None else None,
                "data_quality_by_stock_path": str(data_quality_by_stock_path) if data_quality_by_stock_path is not None else None,
                "data_quality_retained_ratio": float(datasets.diagnostics.get("retained_ratio", 0.0)) if datasets.diagnostics is not None else None,
                "sequence_filtering": datasets.diagnostics.get("sequence_filtering") if datasets.diagnostics is not None else None,
                "top_missing_columns": datasets.diagnostics.get("top_missing_columns") if datasets.diagnostics is not None else None,
                "short_up_ranking_weight": config.multitask.short_up_ranking_weight,
                "short_up_ranking_min_return_gap": config.multitask.short_up_ranking_min_return_gap,
                "short_up_ranking_same_date_only": config.multitask.short_up_ranking_same_date_only,
                "short_up_ranking_max_pairs_per_batch": config.multitask.short_up_ranking_max_pairs_per_batch,
                "multitask_hold_days": config.multitask.hold_days,
                "multitask_take_profit_threshold": config.multitask.take_profit_threshold,
                "multitask_stop_loss_threshold": config.multitask.stop_loss_threshold,
                "multitask_exit_label_mode": config.multitask.exit_label_mode,
                "time_decay_half_life_days": config.training.time_decay_half_life_days,
                "time_decay_weight_stats": time_decay_stats,
                "recent_finetune_window_days": config.training.recent_finetune_window_days,
                "recent_finetune_epochs": config.training.recent_finetune_epochs,
                "recent_finetune_learning_rate_factor": config.training.recent_finetune_learning_rate_factor,
                "recent_finetune_patience": config.training.recent_finetune_patience,
                "min_data_retained_ratio": config.training.min_data_retained_ratio,
                "max_missing_feature_ratio": config.training.max_missing_feature_ratio,
                "recent_finetune_stats": recent_finetune_stats,
                "training_plan_name": plan_name,
                "training_stages": [
                    {
                        "name": stage.name,
                        "dataset_name": stage.dataset_name,
                        "epochs": int(stage.epochs),
                        "learning_rate": float(stage.learning_rate),
                        "patience_limit": int(stage.patience_limit),
                        "task_weights": stage.task_weights,
                        "trainable_modules": stage.trainable_modules,
                    }
                    for stage in stage_plans
                ],
                **test_metrics,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    if not predictions.empty:
        predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        calibrated_predictions = apply_probability_calibration(predictions, calibration)
        calibrated_predictions = enrich_prediction_frame(calibrated_predictions, exit_label_mode=config.multitask.exit_label_mode)
        calibrated_predictions["signal_score"] = strategy_score_frame(calibrated_predictions, exit_label_mode=config.multitask.exit_label_mode)
        calibrated_predictions.to_csv(calibrated_predictions_path, index=False, encoding="utf-8-sig")

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "best_epoch": int(best_epoch),
        "best_validation_metric": float(best_metric),
        "best_val_metrics": best_val_metrics,
        "training_plan_name": plan_name,
        "training_stage_names": [stage.name for stage in stage_plans],
        **test_metrics,
    }


def run_training_plan(
    config: ExperimentConfig,
    stage_plans: list[TrainingStagePlan],
    *,
    plan_name: str = "standard",
) -> dict:
    runtime = _prepare_experiment_runtime(config)
    stage_results = _execute_stage_plans(
        base_config=config,
        model=runtime["model"],
        datasets=runtime["datasets"],
        recent_dataset=runtime["recent_dataset"],
        val_loader=runtime["val_loader"],
        device=runtime["device"],
        stage_plans=stage_plans,
    )
    return _persist_experiment_outputs(
        config=config,
        model=runtime["model"],
        datasets=runtime["datasets"],
        val_loader=runtime["val_loader"],
        test_loader=runtime["test_loader"],
        device=runtime["device"],
        best_state=stage_results["best_state"],
        best_epoch=stage_results["best_epoch"],
        best_metric=stage_results["best_metric"],
        best_val_metrics=stage_results["best_val_metrics"],
        history=stage_results["history"],
        time_decay_stats=runtime["time_decay_stats"],
        recent_finetune_stats=runtime["recent_finetune_stats"],
        stage_plans=stage_plans,
        plan_name=plan_name,
    )


def run_experiment(config: ExperimentConfig) -> dict:
    stage_plans = [
        TrainingStagePlan(
            name="base_train",
            dataset_name="train",
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            patience_limit=config.training.early_stopping_patience,
        )
    ]
    if config.training.recent_finetune_window_days is not None and config.training.recent_finetune_epochs > 0:
        stage_plans.append(
            TrainingStagePlan(
                name="recent_finetune",
                dataset_name="recent",
                epochs=config.training.recent_finetune_epochs,
                learning_rate=config.training.learning_rate * config.training.recent_finetune_learning_rate_factor,
                patience_limit=int(config.training.recent_finetune_patience or config.training.early_stopping_patience),
            )
        )
    return run_training_plan(config, stage_plans, plan_name="standard")