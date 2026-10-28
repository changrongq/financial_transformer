from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


@dataclass
class DatabaseConfig:
    trade_db: str
    market_cap_db: str
    features_db: str


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    lookback_window: int = 60
    prediction_horizon: int = 5
    target_return_threshold: float = 0.03
    validation_metric: str = "auc"
    num_workers: int = 0
    device: str = "cpu"


@dataclass
class MultiTaskConfig:
    enabled: bool = True
    short_horizon: int = 5
    short_threshold: float = 0.03
    hold_days: int = 45
    take_profit_threshold: float = 0.12
    stop_loss_threshold: float = -0.06
    exit_label_mode: str = "window_any"
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    validation_metric: str = "composite"
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    hidden_dim: int = 128
    dropout: float = 0.15
    task_weights: dict[str, float] = field(default_factory=lambda: {
        "short_up": 1.5,
        "take_profit": 1.25,
        "stop_loss": 1.0,
        "max_gain": 0.25,
        "max_drawdown": 0.25,
    })


@dataclass
class EnsembleConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "micro": 0.15,
        "small": 0.25,
        "mid": 0.30,
        "large": 0.30,
    })


@dataclass
class SystemConfig:
    root_dir: Path
    tushare_token: str
    databases: DatabaseConfig
    training: TrainingConfig
    multitask: MultiTaskConfig
    ensemble: EnsembleConfig
    results_dir: Path
    models_dir: Path


DEFAULT_CONFIG = {
    "tushare": {"token": "${TUSHARE_TOKEN}"},
    "databases": {
        "trade_db": "../stock_history.db",
        "market_cap_db": "../market_cap_full.db",
        "features_db": "data/hierarchical_features.db",
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "early_stopping_patience": 10,
        "lookback_window": 60,
        "prediction_horizon": 5,
        "target_return_threshold": 0.03,
        "validation_metric": "auc",
        "num_workers": 0,
        "device": "cpu",
    },
    "multitask": {
        "enabled": True,
        "short_horizon": 5,
        "short_threshold": 0.03,
        "hold_days": 45,
        "take_profit_threshold": 0.12,
        "stop_loss_threshold": -0.06,
        "exit_label_mode": "window_any",
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.0001,
        "early_stopping_patience": 10,
        "validation_metric": "composite",
        "optimizer": "adamw",
        "scheduler": "cosine",
        "hidden_dim": 128,
        "dropout": 0.15,
        "task_weights": {
            "short_up": 1.5,
            "take_profit": 1.25,
            "stop_loss": 1.0,
            "max_gain": 0.25,
            "max_drawdown": 0.25,
        },
    },
    "ensemble": {
        "weights": {"micro": 0.15, "small": 0.25, "mid": 0.30, "large": 0.30}
    },
}


def _resolve_env(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, "")
    return value


def load_config(config_path: str | Path) -> SystemConfig:
    config_path = Path(config_path).resolve()
    root_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    merged = DEFAULT_CONFIG | raw
    db = merged["databases"]
    training = merged["training"]
    multitask = merged.get("multitask", DEFAULT_CONFIG["multitask"])
    ensemble = merged["ensemble"]

    def norm(path_value: str) -> str:
        path = Path(path_value)
        if not path.is_absolute():
            path = (root_dir / path).resolve()
        return str(path)

    return SystemConfig(
        root_dir=root_dir,
        tushare_token=_resolve_env(merged["tushare"]["token"]),
        databases=DatabaseConfig(
            trade_db=norm(db["trade_db"]),
            market_cap_db=norm(db["market_cap_db"]),
            features_db=norm(db["features_db"]),
        ),
        training=TrainingConfig(**training),
        multitask=MultiTaskConfig(**multitask),
        ensemble=EnsembleConfig(weights=ensemble["weights"]),
        results_dir=(root_dir / "results").resolve(),
        models_dir=(root_dir / "models").resolve(),
    )
