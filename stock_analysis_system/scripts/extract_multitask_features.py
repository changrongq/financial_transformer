from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.multitask_data import extract_multitask_features, summarize_multitask_targets


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    feature_df = extract_multitask_features(config)
    print(feature_df.head())
    print(summarize_multitask_targets(feature_df).to_string(index=False))