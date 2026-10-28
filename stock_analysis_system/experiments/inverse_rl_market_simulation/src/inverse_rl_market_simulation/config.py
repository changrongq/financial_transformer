from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from inverse_rl_market_simulation.agents import get_search_space_param_names


@dataclass(frozen=True)
class DatabaseConfig:
    trade_db: Path
    root_config_path: Path


@dataclass(frozen=True)
class ExperimentSection:
    stock_code: str
    start_date: str | None = None
    end_date: str | None = None
    window_days: int = 120
    stability_window_days: int = 60
    min_history_days: int = 80


@dataclass(frozen=True)
class OptimizationSection:
    random_seed: int = 7
    n_trials: int = 80
    search_space: str = "fixed_role"


@dataclass(frozen=True)
class ResultsSection:
    results_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path
    databases: DatabaseConfig
    experiment: ExperimentSection
    optimization: OptimizationSection
    results: ResultsSection


def _resolve_path(root_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (root_dir / path).resolve()
    return path


def _load_root_trade_db(root_config_path: Path) -> Path:
    with root_config_path.open("r", encoding="utf-8") as handle:
        root_raw = yaml.safe_load(handle) or {}
    trade_db = root_raw.get("databases", {}).get("trade_db")
    if not trade_db:
        raise ValueError("databases.trade_db 未提供，且 root config 中也没有该字段。")
    return Path(str(trade_db)).resolve()


def load_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    root_dir = config_path.parent
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    db_raw = dict(raw.get("databases", {}))
    root_config_path = _resolve_path(root_dir, db_raw.get("root_config_path") or "../../config.yaml")
    assert root_config_path is not None
    trade_db = _resolve_path(root_dir, db_raw.get("trade_db"))
    if trade_db is None:
        trade_db = _load_root_trade_db(root_config_path)
    if not trade_db.exists():
        raise ValueError(f"trade_db 不存在: {trade_db}")

    experiment = ExperimentSection(**dict(raw.get("experiment", {})))
    if not experiment.stock_code:
        raise ValueError("experiment.stock_code 必须提供。")
    if experiment.window_days < experiment.min_history_days:
        raise ValueError("experiment.window_days 不能小于 experiment.min_history_days。")
    if experiment.stability_window_days <= 0:
        raise ValueError("experiment.stability_window_days 必须大于 0。")

    optimization = OptimizationSection(**dict(raw.get("optimization", {})))
    if optimization.n_trials <= 0:
        raise ValueError("optimization.n_trials 必须大于 0。")
    get_search_space_param_names(optimization.search_space)

    results_value = raw.get("results", {}).get("results_dir", "results/single_stock_mvp")
    results_dir = _resolve_path(root_dir, str(results_value))
    assert results_dir is not None

    return ExperimentConfig(
        root_dir=root_dir,
        databases=DatabaseConfig(trade_db=trade_db, root_config_path=root_config_path),
        experiment=experiment,
        optimization=optimization,
        results=ResultsSection(results_dir=results_dir),
    )