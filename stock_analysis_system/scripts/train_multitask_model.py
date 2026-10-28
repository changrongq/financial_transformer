from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.multitask_training import train_multitask_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multitask stock forecasting model.")
    parser.add_argument("--epochs", type=int, default=None, help="Override multitask epochs for a shorter or longer run.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override multitask batch size.")
    parser.add_argument("--layer", choices=["micro", "small", "mid", "large"], default=None, help="Train an independent multitask model for a specific market-cap layer.")
    parser.add_argument("--all-layers", action="store_true", help="Train independent multitask models for all four market-cap layers.")
    args = parser.parse_args()

    if args.all_layers:
        for layer_name in ["micro", "small", "mid", "large"]:
            print(
                train_multitask_model(
                    ROOT / "config.yaml",
                    epochs_override=args.epochs,
                    patience_override=args.patience,
                    batch_size_override=args.batch_size,
                    layer=layer_name,
                )
            )
    else:
        print(
            train_multitask_model(
                ROOT / "config.yaml",
                epochs_override=args.epochs,
                patience_override=args.patience,
                batch_size_override=args.batch_size,
                layer=args.layer,
            )
        )