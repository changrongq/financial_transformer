from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass(frozen=True)
class DatabaseConfig:
    trade_db: str
    market_cap_db: str


@dataclass(frozen=True)
class ExperimentSection:
    feature_cols: list[str]
    stock_pool: list[str] = field(default_factory=list)
    layer_filter: str | None = None
    max_stocks: int | None = None
    role_vector_root_dir: str | None = None
    role_vector_glob: str = "single_stock_soft_switch_v1_*_recent/role_activity_vectors.csv"
    manual_concept_labels_path: str | None = None
    concept_label_source: str = "db_only"
    concept_sentiment_top_k: int | None = None
    concept_sentiment_weighting: str = "uniform"
    concept_exposure_lookback_days: int = 20
    stock_cv_total_folds: int | None = None
    train_stock_folds: list[int] = field(default_factory=list)
    val_stock_folds: list[int] = field(default_factory=list)
    test_stock_folds: list[int] = field(default_factory=list)
    prediction_horizon: int = 5
    target_return_threshold: float = 0.03


@dataclass(frozen=True)
class MultiTaskSection:
    short_horizon: int = 5
    short_threshold: float = 0.03
    hold_days: int = 45
    take_profit_threshold: float = 0.12
    stop_loss_threshold: float = -0.06
    exit_label_mode: str = "window_any"
    validation_metric: str = "business_score"
    regression_loss_beta: float = 0.05
    short_up_ranking_weight: float = 0.0
    short_up_ranking_min_return_gap: float = 0.02
    short_up_ranking_same_date_only: bool = True
    short_up_ranking_max_pairs_per_batch: int = 2048
    task_weights: dict[str, float] = field(default_factory=lambda: {
        "short_up": 1.75,
        "take_profit": 1.35,
        "stop_loss": 1.0,
        "max_gain": 0.2,
        "max_drawdown": 0.2,
    })


@dataclass(frozen=True)
class SplitSection:
    train_end: str
    val_end: str
    test_end: str


@dataclass(frozen=True)
class ModelSection:
    seq_len: int = 60
    min_seq_len: int | None = None
    model_variant: str = "flat"
    grouped_feature_groups: dict[str, list[str]] = field(default_factory=dict)
    use_group_gates: bool = False
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    ffn_type: str = "mlp"
    position_encoding: str = "sinusoidal"


@dataclass(frozen=True)
class TrainingSection:
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    label_smoothing: float = 0.05
    time_decay_half_life_days: float | None = None
    recent_finetune_window_days: int | None = None
    recent_finetune_epochs: int = 0
    recent_finetune_learning_rate_factor: float = 0.1
    recent_finetune_patience: int | None = None
    min_data_retained_ratio: float | None = 0.8
    max_missing_feature_ratio: float | None = 0.15
    max_zero_turnover_fill_ratio: float | None = None
    device: str = "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path
    databases: DatabaseConfig
    experiment: ExperimentSection
    multitask: MultiTaskSection
    splits: SplitSection
    model: ModelSection
    training: TrainingSection
    results_dir: Path
    artifacts_dir: Path


def load_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    root_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    def norm(path_value: str) -> str:
        path = Path(path_value)
        if not path.is_absolute():
            path = (root_dir / path).resolve()
        return str(path)

    experiment_raw = dict(raw.get("experiment", {}))
    role_vector_root_dir = experiment_raw.get("role_vector_root_dir")
    if role_vector_root_dir:
        experiment_raw["role_vector_root_dir"] = norm(str(role_vector_root_dir))
    manual_concept_labels_path = experiment_raw.get("manual_concept_labels_path")
    if manual_concept_labels_path:
        experiment_raw["manual_concept_labels_path"] = norm(str(manual_concept_labels_path))
    experiment = ExperimentSection(**experiment_raw)
    if not experiment.stock_pool and not experiment.layer_filter:
        raise ValueError("experiment.stock_pool 和 experiment.layer_filter 至少要提供一个。")
    if experiment.role_vector_root_dir and not Path(experiment.role_vector_root_dir).exists():
        raise ValueError(f"experiment.role_vector_root_dir 不存在: {experiment.role_vector_root_dir}")
    if experiment.concept_label_source not in {"db_only", "manual_first", "manual_only"}:
        raise ValueError("experiment.concept_label_source 只能是 db_only、manual_first 或 manual_only。")
    if experiment.concept_sentiment_top_k is not None and int(experiment.concept_sentiment_top_k) <= 0:
        raise ValueError("experiment.concept_sentiment_top_k 必须大于 0。")
    if experiment.concept_sentiment_weighting not in {"uniform", "strength", "exposure", "strength_exposure"}:
        raise ValueError("experiment.concept_sentiment_weighting 只能是 uniform、strength、exposure 或 strength_exposure。")
    if int(experiment.concept_exposure_lookback_days) < 3:
        raise ValueError("experiment.concept_exposure_lookback_days 必须大于等于 3。")
    if experiment.stock_cv_total_folds is not None:
        if int(experiment.stock_cv_total_folds) <= 1:
            raise ValueError("experiment.stock_cv_total_folds 必须大于 1。")
        valid_folds = set(range(int(experiment.stock_cv_total_folds)))
        for fold_list in [experiment.train_stock_folds, experiment.val_stock_folds, experiment.test_stock_folds]:
            if any(int(fold) not in valid_folds for fold in fold_list):
                raise ValueError("stock fold 索引超出 experiment.stock_cv_total_folds 范围。")

    model_raw = dict(raw.get("model", {}))
    seq_len = int(model_raw.get("seq_len", ModelSection.seq_len))
    min_seq_len = model_raw.get("min_seq_len")
    if min_seq_len is None:
        model_raw["min_seq_len"] = seq_len
    else:
        min_seq_len = int(min_seq_len)
        if min_seq_len <= 0 or min_seq_len > seq_len:
            raise ValueError("model.min_seq_len 必须在 1 到 model.seq_len 之间。")
        model_raw["min_seq_len"] = min_seq_len
    if model_raw.get("model_variant", ModelSection.model_variant) == "grouped" and not model_raw.get("grouped_feature_groups"):
        raise ValueError("grouped 模型必须提供 model.grouped_feature_groups。")

    training_raw = dict(raw.get("training", {}))
    time_decay_half_life_days = training_raw.get("time_decay_half_life_days")
    if time_decay_half_life_days is not None:
        time_decay_half_life_days = float(time_decay_half_life_days)
        if time_decay_half_life_days <= 0:
            raise ValueError("training.time_decay_half_life_days 必须大于 0。")
        training_raw["time_decay_half_life_days"] = time_decay_half_life_days
    recent_finetune_window_days = training_raw.get("recent_finetune_window_days")
    if recent_finetune_window_days is not None:
        recent_finetune_window_days = int(recent_finetune_window_days)
        if recent_finetune_window_days <= 0:
            raise ValueError("training.recent_finetune_window_days 必须大于 0。")
        training_raw["recent_finetune_window_days"] = recent_finetune_window_days
    recent_finetune_epochs = int(training_raw.get("recent_finetune_epochs", 0) or 0)
    if recent_finetune_epochs < 0:
        raise ValueError("training.recent_finetune_epochs 不能小于 0。")
    training_raw["recent_finetune_epochs"] = recent_finetune_epochs
    recent_finetune_learning_rate_factor = float(training_raw.get("recent_finetune_learning_rate_factor", 0.1))
    if recent_finetune_learning_rate_factor <= 0:
        raise ValueError("training.recent_finetune_learning_rate_factor 必须大于 0。")
    training_raw["recent_finetune_learning_rate_factor"] = recent_finetune_learning_rate_factor
    recent_finetune_patience = training_raw.get("recent_finetune_patience")
    if recent_finetune_patience is not None:
        recent_finetune_patience = int(recent_finetune_patience)
        if recent_finetune_patience <= 0:
            raise ValueError("training.recent_finetune_patience 必须大于 0。")
        training_raw["recent_finetune_patience"] = recent_finetune_patience
    min_data_retained_ratio = training_raw.get("min_data_retained_ratio")
    if min_data_retained_ratio is not None:
        min_data_retained_ratio = float(min_data_retained_ratio)
        if min_data_retained_ratio <= 0 or min_data_retained_ratio > 1:
            raise ValueError("training.min_data_retained_ratio 必须在 (0, 1] 范围内。")
        training_raw["min_data_retained_ratio"] = min_data_retained_ratio
    max_missing_feature_ratio = training_raw.get("max_missing_feature_ratio")
    if max_missing_feature_ratio is not None:
        max_missing_feature_ratio = float(max_missing_feature_ratio)
        if max_missing_feature_ratio <= 0 or max_missing_feature_ratio > 1:
            raise ValueError("training.max_missing_feature_ratio 必须在 (0, 1] 范围内。")
        training_raw["max_missing_feature_ratio"] = max_missing_feature_ratio
    max_zero_turnover_fill_ratio = training_raw.get("max_zero_turnover_fill_ratio")
    if max_zero_turnover_fill_ratio is not None:
        max_zero_turnover_fill_ratio = float(max_zero_turnover_fill_ratio)
        if max_zero_turnover_fill_ratio <= 0 or max_zero_turnover_fill_ratio > 1:
            raise ValueError("training.max_zero_turnover_fill_ratio 必须在 (0, 1] 范围内。")
        training_raw["max_zero_turnover_fill_ratio"] = max_zero_turnover_fill_ratio
    if (recent_finetune_window_days is None) != (recent_finetune_epochs == 0):
        raise ValueError("training.recent_finetune_window_days 和 training.recent_finetune_epochs 需要同时启用或同时关闭。")

    multitask_raw = dict(raw.get("multitask", {}))
    exit_label_mode = str(multitask_raw.get("exit_label_mode", "window_any"))
    if exit_label_mode not in {"window_any", "first_hit", "realized_profit"}:
        raise ValueError("multitask.exit_label_mode 只能是 window_any、first_hit 或 realized_profit。")
    multitask_raw["exit_label_mode"] = exit_label_mode

    results_dir_value = raw.get("results_dir", "results")
    artifacts_dir_value = raw.get("artifacts_dir", "artifacts")

    return ExperimentConfig(
        root_dir=root_dir,
        databases=DatabaseConfig(
            trade_db=norm(raw["databases"]["trade_db"]),
            market_cap_db=norm(raw["databases"]["market_cap_db"]),
        ),
        experiment=experiment,
        multitask=MultiTaskSection(**multitask_raw),
        splits=SplitSection(**raw.get("splits", {})),
        model=ModelSection(**model_raw),
        training=TrainingSection(**training_raw),
        results_dir=(root_dir / results_dir_value).resolve(),
        artifacts_dir=(root_dir / artifacts_dir_value).resolve(),
    )