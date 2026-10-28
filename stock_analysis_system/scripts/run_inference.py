from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.inference import ProductionInferenceEngine


if __name__ == "__main__":
    engine = ProductionInferenceEngine(ROOT / "config.yaml")
    result = engine.predict_latest(top_n=50)
    print(result.head(20))
