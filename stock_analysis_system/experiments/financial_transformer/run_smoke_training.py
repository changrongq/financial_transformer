from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import normalize_stock_code
from financial_transformer_experiment.training import run_experiment


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_stock_pool(stock_frame_path: Path) -> list[str]:
    frame = pd.read_csv(stock_frame_path, dtype={"stock_code": str})
    if "stock_code" not in frame.columns:
        raise RuntimeError(f"stock frame missing required column 'stock_code': {stock_frame_path}")
    stock_pool = sorted({normalize_stock_code(code) for code in frame["stock_code"].astype(str).tolist() if str(code).strip()})
    if not stock_pool:
        raise RuntimeError(f"no stock_code rows found in stock frame: {stock_frame_path}")
    return stock_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run financial transformer smoke training on a filtered stock frame")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stock-frame", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--validation-metric", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    stock_frame_path = Path(args.stock_frame).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not stock_frame_path.exists():
        raise RuntimeError(f"stock frame file not found: {stock_frame_path}")

    if args.seed is not None:
        _set_seed(int(args.seed))

    base_config = load_config(config_path)
    stock_pool = _load_stock_pool(stock_frame_path)

    training_override = base_config.training
    if args.epochs is not None:
        training_override = replace(training_override, epochs=int(args.epochs))
    if args.batch_size is not None:
        training_override = replace(training_override, batch_size=int(args.batch_size))
    if args.early_stopping_patience is not None:
        training_override = replace(training_override, early_stopping_patience=int(args.early_stopping_patience))
    if args.device is not None:
        training_override = replace(training_override, device=str(args.device))

    run_config = replace(
        base_config,
        experiment=replace(
            base_config.experiment,
            stock_pool=stock_pool,
            layer_filter=None,
            max_stocks=None,
        ),
        multitask=(
            replace(base_config.multitask, validation_metric=str(args.validation_metric))
            if args.validation_metric is not None
            else base_config.multitask
        ),
        training=training_override,
        results_dir=output_dir,
        artifacts_dir=output_dir / "artifacts",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_spec = {
        "config": str(config_path),
        "stock_frame": str(stock_frame_path),
        "output_dir": str(output_dir),
        "stock_count": len(stock_pool),
        "stock_pool_preview": stock_pool[:20],
        "epochs": int(run_config.training.epochs),
        "batch_size": int(run_config.training.batch_size),
        "early_stopping_patience": int(run_config.training.early_stopping_patience),
        "validation_metric": str(run_config.multitask.validation_metric),
        "seed": int(args.seed) if args.seed is not None else None,
        "device": str(run_config.training.device),
    }
    (output_dir / "smoke_run_spec.json").write_text(json.dumps(run_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    result = run_experiment(run_config)
    print(json.dumps({"run_spec": run_spec, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()