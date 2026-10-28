from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.multitask_backtest import MultiTaskStrategyABacktester, write_multitask_strategy_a_backtest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run layered multitask Strategy A backtest")
    parser.add_argument("--layers", default="micro,small")
    parser.add_argument("--short-up-threshold", type=float, default=0.59)
    parser.add_argument("--take-profit-threshold", type=float, default=0.25)
    parser.add_argument("--stop-loss-threshold", type=float, default=0.30)
    parser.add_argument("--output-prefix", default="multitask_strategy_a_layered_micro_small")
    args = parser.parse_args()

    allowed_layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    layer_threshold_overrides = {
        layer: {
            "short_up_threshold": args.short_up_threshold,
            "take_profit_threshold": args.take_profit_threshold,
            "stop_loss_threshold": args.stop_loss_threshold,
        }
        for layer in allowed_layers
    }

    backtester = MultiTaskStrategyABacktester(ROOT / "config.yaml")
    result = backtester.run(
        start_date="2025-10-01",
        end_date="2026-04-13",
        short_up_threshold=args.short_up_threshold,
        take_profit_threshold=args.take_profit_threshold,
        stop_loss_threshold=args.stop_loss_threshold,
        allowed_layers=allowed_layers,
        layer_threshold_overrides=layer_threshold_overrides,
        hold_days=45,
        take_profit=0.12,
        stop_loss=-0.06,
        initial_capital=1_000_000.0,
        commission=0.001,
        slippage=0.001,
        min_lot=100,
    )
    output_paths = write_multitask_strategy_a_backtest(
        result,
        ROOT / "results",
        output_prefix=args.output_prefix,
    )
    print(json.dumps({**result.summary, **output_paths}, ensure_ascii=False, indent=2))