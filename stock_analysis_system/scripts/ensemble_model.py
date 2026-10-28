from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.ensemble import weighted_ensemble


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    prediction_files = {
        layer: str(ROOT / "results" / f"{layer}_predictions.csv")
        for layer in ["micro", "small", "mid", "large"]
    }
    result = weighted_ensemble(
        prediction_files=prediction_files,
        weights=config.ensemble.weights,
        output_file=str(ROOT / "results" / "ensemble_predictions.csv"),
    )
    print(result.head(20))
