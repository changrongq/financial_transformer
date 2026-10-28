from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.multitask_backtest import MultiTaskStrategyABacktester, write_multitask_strategy_a_backtest


def _parse_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("阈值网格不能为空")
    return values


def _ranking_score(row: dict, min_trades: int, min_signals: int) -> float:
    trade_penalty = 0.0 if row["trade_count"] >= min_trades else (min_trades - row["trade_count"]) * 0.02
    signal_penalty = 0.0 if row["signal_count"] >= min_signals else (min_signals - row["signal_count"]) * 0.005
    return (
        row["total_return"] * 1.0
        + row["sharpe_ratio"] * 0.15
        + row["win_rate"] * 0.05
        + row["profit_factor"] * 0.03
        + row["max_drawdown"] * 0.5
        - trade_penalty
        - signal_penalty
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search Strategy A thresholds for multitask backtesting")
    parser.add_argument("--start-date", default="2025-10-01")
    parser.add_argument("--end-date", default="2026-04-13")
    parser.add_argument("--short-up-grid", default="0.55,0.60,0.65,0.70")
    parser.add_argument("--take-profit-grid", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--stop-loss-grid", default="0.15,0.20,0.25")
    parser.add_argument("--allowed-layers", default="")
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--min-signals", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    backtester = MultiTaskStrategyABacktester(ROOT / "config.yaml")
    short_up_grid = _parse_grid(args.short_up_grid)
    take_profit_grid = _parse_grid(args.take_profit_grid)
    stop_loss_grid = _parse_grid(args.stop_loss_grid)
    allowed_layers = [layer.strip() for layer in args.allowed_layers.split(",") if layer.strip()]

    search_rows = []
    best_result = None
    best_summary = None
    best_feasible_result = None
    best_feasible_summary = None

    for short_up_threshold, take_profit_threshold, stop_loss_threshold in product(short_up_grid, take_profit_grid, stop_loss_grid):
        result = backtester.run(
            start_date=args.start_date,
            end_date=args.end_date,
            short_up_threshold=short_up_threshold,
            take_profit_threshold=take_profit_threshold,
            stop_loss_threshold=stop_loss_threshold,
            allowed_layers=allowed_layers or None,
            hold_days=45,
            take_profit=0.12,
            stop_loss=-0.06,
            initial_capital=1_000_000.0,
            commission=0.001,
            slippage=0.001,
            min_lot=100,
        )
        row = dict(result.summary)
        row["meets_min_trades"] = row["trade_count"] >= args.min_trades
        row["meets_min_signals"] = row["signal_count"] >= args.min_signals
        row["meets_constraints"] = row["meets_min_trades"] and row["meets_min_signals"]
        row["ranking_score"] = _ranking_score(row, min_trades=args.min_trades, min_signals=args.min_signals)
        search_rows.append(row)
        if best_summary is None or row["ranking_score"] > best_summary["ranking_score"]:
            best_summary = row
            best_result = result
        if row["meets_constraints"] and (best_feasible_summary is None or row["ranking_score"] > best_feasible_summary["ranking_score"]):
            best_feasible_summary = row
            best_feasible_result = result

    if best_summary is None or best_result is None:
        raise RuntimeError("阈值搜索未产生任何结果")

    selected_summary = best_feasible_summary or best_summary
    selected_result = best_feasible_result or best_result

    search_df = pd.DataFrame(search_rows).sort_values(
        ["meets_constraints", "ranking_score", "total_return", "sharpe_ratio", "max_drawdown", "trade_count"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    results_dir = ROOT / "results"
    output_tag = f"_{args.output_tag}" if args.output_tag else ""
    search_csv_path = results_dir / f"multitask_strategy_a_threshold_search{output_tag}.csv"
    top_csv_path = results_dir / f"multitask_strategy_a_threshold_search_top{output_tag}.csv"
    best_json_path = results_dir / f"multitask_strategy_a_threshold_best{output_tag}.json"

    search_df.to_csv(search_csv_path, index=False, encoding="utf-8-sig")
    search_df.head(args.top_k).to_csv(top_csv_path, index=False, encoding="utf-8-sig")

    suffix = f"multitask_strategy_a_search_mt{args.min_trades}_ms{args.min_signals}{output_tag}"
    detail_paths = write_multitask_strategy_a_backtest(selected_result, results_dir, output_prefix=suffix)
    best_payload = {
        "grid_size": len(search_rows),
        "constraints": {
            "min_trades": args.min_trades,
            "min_signals": args.min_signals,
            "allowed_layers": allowed_layers,
        },
        "best": selected_summary,
        "best_overall": best_summary,
        "best_feasible": best_feasible_summary,
        "detail_outputs": detail_paths,
        "search_results": {
            "all": str(search_csv_path),
            "top": str(top_csv_path),
        },
    }
    with best_json_path.open("w", encoding="utf-8") as handle:
        json.dump(best_payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps({**best_payload, "top_rows": search_df.head(args.top_k).to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()