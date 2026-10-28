from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import build_datasets
from financial_transformer_experiment.model import create_model
from financial_transformer_experiment.training import _build_loader, _evaluate


def _freeze_to_checkpoint_universe(config):
    checkpoint_path = config.artifacts_dir / "financial_transformer.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stock_pool = checkpoint.get("selected_stock_codes") or []
    stock_pool = sorted({str(code).zfill(6) for code in stock_pool if str(code).strip()})
    if not stock_pool:
        return config
    return replace(
        config,
        experiment=replace(
            config.experiment,
            stock_pool=stock_pool,
            layer_filter=None,
            max_stocks=None,
        ),
    )


def _build_model(config, input_dim: int, device: torch.device, checkpoint: dict):
    model_config = checkpoint.get("model_config", {})
    feature_cols = checkpoint.get("feature_cols", config.experiment.feature_cols)
    return create_model(
        input_dim=input_dim,
        d_model=int(model_config.get("d_model", config.model.d_model)),
        nhead=int(model_config.get("nhead", config.model.nhead)),
        num_layers=int(model_config.get("num_layers", config.model.num_layers)),
        dim_feedforward=int(model_config.get("dim_feedforward", config.model.dim_feedforward)),
        dropout=float(model_config.get("dropout", config.model.dropout)),
        ffn_type=str(model_config.get("ffn_type", config.model.ffn_type)),
        position_encoding=str(model_config.get("position_encoding", config.model.position_encoding)),
        model_variant=str(model_config.get("model_variant", config.model.model_variant)),
        feature_cols=feature_cols,
        grouped_feature_groups=dict(model_config.get("grouped_feature_groups", config.model.grouped_feature_groups)),
        use_group_gates=bool(model_config.get("use_group_gates", config.model.use_group_gates)),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate val/test predictions from an existing transformer checkpoint")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--freeze-universe", action="store_true", help="Freeze stock universe to the checkpoint's selected_stock_codes")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = config.artifacts_dir / "financial_transformer.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint not found: {checkpoint_path}")

    if args.freeze_universe:
        config = _freeze_to_checkpoint_universe(config)

    datasets = build_datasets(config)
    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_model(config, input_dim=datasets.input_dim, device=device, checkpoint=checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loader = _build_loader(datasets.val, config.training.batch_size, False)
    test_loader = _build_loader(datasets.test, config.training.batch_size, False)
    val_metrics, val_predictions = _evaluate(model, val_loader, device, exit_label_mode=config.multitask.exit_label_mode)
    test_metrics, test_predictions = _evaluate(model, test_loader, device, exit_label_mode=config.multitask.exit_label_mode)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(config.results_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(config.results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    payload = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path),
        "freeze_universe": bool(args.freeze_universe),
        "val_rows": int(len(val_predictions)),
        "test_rows": int(len(test_predictions)),
        "val_dates": int(val_predictions["date"].nunique()) if not val_predictions.empty else 0,
        "test_dates": int(test_predictions["date"].nunique()) if not test_predictions.empty else 0,
        "val_metrics": {key: float(value) for key, value in val_metrics.items()},
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "val_output": str(config.results_dir / "val_predictions.csv"),
        "test_output": str(config.results_dir / "test_predictions.csv"),
    }
    with (config.results_dir / "split_prediction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()