from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.calibrate import run_probability_calibration
from financial_transformer_experiment.config import load_config


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fit probability calibration for an existing financial transformer model")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(run_probability_calibration(config), ensure_ascii=False, indent=2))