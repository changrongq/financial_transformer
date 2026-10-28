from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.realistic_backtest import RealisticPortfolioBacktester


if __name__ == "__main__":
    backtester = RealisticPortfolioBacktester(ROOT / "config.yaml")
    result = backtester.run(
        prediction_file=ROOT / "results" / "ensemble_predictions.csv",
        percentile=0.05,
        max_positions=20,
        hold_days=5,
        buy_cost=0.001,
        sell_cost=0.001,
        initial_capital=100000.0,
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_path = results_dir / "realistic_ensemble_top5pct_trades.csv"
    equity_path = results_dir / "realistic_ensemble_top5pct_equity.csv"
    summary_path = results_dir / "realistic_ensemble_top5pct_summary.json"
    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    result.equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2)

    print(result.summary)