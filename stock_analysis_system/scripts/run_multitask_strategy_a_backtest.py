from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.multitask_backtest import MultiTaskStrategyABacktester, write_multitask_strategy_a_backtest


if __name__ == "__main__":
    backtester = MultiTaskStrategyABacktester(ROOT / "config.yaml")
    result = backtester.run(
        start_date="2025-10-01",
        end_date="2026-04-13",
        hold_days=45,
        take_profit=0.12,
        stop_loss=-0.06,
        initial_capital=1_000_000.0,
        commission=0.001,
        slippage=0.001,
        min_lot=100,
    )
    output_paths = write_multitask_strategy_a_backtest(result, ROOT / "results")
    print(json.dumps({**result.summary, **output_paths}, ensure_ascii=False, indent=2))