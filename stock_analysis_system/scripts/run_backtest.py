from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.backtest import Backtester, resolve_prediction_file


if __name__ == "__main__":
    backtester = Backtester(ROOT / "config.yaml")
    result = backtester.run(
        prediction_file=resolve_prediction_file(ROOT / "results"),
        hold_days=5,
        top_n=10,
        initial_capital=100000,
    )
    print(result.summary)
