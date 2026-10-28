from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.training import TrainingStagePlan, run_training_plan


def _resolve_output_dir(base_dir: Path, override_value: str | None, config_root: Path, suffix: str) -> Path:
    if override_value:
        override_path = Path(override_value)
        if not override_path.is_absolute():
            override_path = (config_root / override_path).resolve()
        return override_path
    return (base_dir / suffix).resolve()


def _load_curriculum_section(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    curriculum = raw.get("curriculum")
    if not isinstance(curriculum, dict):
        raise ValueError("配置缺少 curriculum 段，无法运行课程训练。")
    return curriculum


def _build_stage_plans(curriculum_raw: dict, base_config) -> list[TrainingStagePlan]:
    phases = curriculum_raw.get("phases") or []
    if not isinstance(phases, list) or not phases:
        raise ValueError("curriculum.phases 不能为空。")

    default_learning_rate = float(base_config.training.learning_rate)
    default_patience = int(base_config.training.early_stopping_patience)
    stage_plans: list[TrainingStagePlan] = []
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            raise ValueError(f"curriculum.phases[{index - 1}] 必须是字典。")
        name = str(phase.get("name") or f"phase_{index}")
        dataset_name = str(phase.get("dataset") or phase.get("dataset_name") or "train")
        epochs = int(phase.get("epochs", base_config.training.epochs))
        learning_rate = float(phase.get("learning_rate", default_learning_rate))
        patience_limit = int(phase.get("patience", phase.get("patience_limit", default_patience)))
        task_weights = phase.get("task_weights")
        trainable_modules = phase.get("trainable_modules")
        if task_weights is not None and not isinstance(task_weights, dict):
            raise ValueError(f"phase {name} 的 task_weights 必须是字典。")
        if trainable_modules is not None and not isinstance(trainable_modules, list):
            raise ValueError(f"phase {name} 的 trainable_modules 必须是列表。")
        if epochs <= 0:
            raise ValueError(f"phase {name} 的 epochs 必须大于 0。")
        if learning_rate <= 0:
            raise ValueError(f"phase {name} 的 learning_rate 必须大于 0。")
        if patience_limit <= 0:
            raise ValueError(f"phase {name} 的 patience 必须大于 0。")

        stage_plans.append(
            TrainingStagePlan(
                name=name,
                dataset_name=dataset_name,
                epochs=epochs,
                learning_rate=learning_rate,
                patience_limit=patience_limit,
                task_weights={str(key): float(value) for key, value in (task_weights or {}).items()} or None,
                trainable_modules=[str(module_name) for module_name in (trainable_modules or [])] or None,
            )
        )
    return stage_plans


def load_curriculum_plan(config_path: str | Path):
    resolved_config_path = Path(config_path).resolve()
    base_config = load_config(resolved_config_path)
    curriculum_raw = _load_curriculum_section(resolved_config_path)
    plan_name = str(curriculum_raw.get("plan_name") or resolved_config_path.stem)
    config_root = resolved_config_path.parent
    results_dir = _resolve_output_dir(
        base_config.results_dir,
        curriculum_raw.get("results_dir"),
        config_root,
        f"curriculum_{plan_name}",
    )
    artifacts_dir = _resolve_output_dir(
        base_config.artifacts_dir,
        curriculum_raw.get("artifacts_dir"),
        config_root,
        f"curriculum_{plan_name}",
    )
    curriculum_config = replace(base_config, results_dir=results_dir, artifacts_dir=artifacts_dir)
    stage_plans = _build_stage_plans(curriculum_raw, curriculum_config)
    return curriculum_config, plan_name, stage_plans


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train financial transformer with curriculum phases")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config.yaml"))
    args = parser.parse_args()

    config, plan_name, stage_plans = load_curriculum_plan(args.config)
    print(run_training_plan(config, stage_plans, plan_name=plan_name))