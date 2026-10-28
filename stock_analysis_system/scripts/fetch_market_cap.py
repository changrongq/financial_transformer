from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.market_cap import fetch_and_store_market_caps


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    fetch_and_store_market_caps(config)
    print("市值抓取完成")
