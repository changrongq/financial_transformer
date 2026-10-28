from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.multitask_inference import ProductionMultiTaskInferenceEngine


if __name__ == "__main__":
    engine = ProductionMultiTaskInferenceEngine(ROOT / "config.yaml")
    latest, strategy_a = engine.predict_latest(top_n=50)
    print("latest_multitask_top")
    print(latest.head(20).to_string(index=False))
    print("strategy_a_top")
    print(strategy_a.head(20).to_string(index=False))