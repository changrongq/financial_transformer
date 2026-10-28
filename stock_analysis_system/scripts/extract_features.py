from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.prepare import extract_hierarchical_features


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    feature_df = extract_hierarchical_features(config)
    print(feature_df.head())
    print(f"特征行数: {len(feature_df)}")
