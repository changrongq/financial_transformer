from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.data_refresh import refresh_latest_data_bundle


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    print(refresh_latest_data_bundle(config, refresh_market_caps=True, refresh_existing_market_caps=True))