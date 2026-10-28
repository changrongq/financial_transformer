from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.training_review import review_mid_model


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    report_path = review_mid_model(ROOT / "results", config.results_dir / "mid_model_review.md")
    print({"report_path": str(report_path)})