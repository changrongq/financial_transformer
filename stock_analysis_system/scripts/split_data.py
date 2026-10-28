from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.splitter import build_stock_splits


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    split_df = build_stock_splits(
        market_cap_db=config.databases.market_cap_db,
        features_db=config.databases.features_db,
        output_csv=str(ROOT / "data" / "stock_splits.csv"),
    )
    print(split_df["split"].value_counts())
