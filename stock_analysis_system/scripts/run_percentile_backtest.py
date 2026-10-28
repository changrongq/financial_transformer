from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.percentile_backtest import run_percentile_backtest


if __name__ == "__main__":
    artifacts = run_percentile_backtest(ROOT / "config.yaml")
    print(artifacts.summary.to_string(index=False))