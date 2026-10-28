from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.backtest import TransformerStrategyBacktester, write_backtest_outputs
from financial_transformer_experiment.config import load_config


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


def _parse_top_k_grid(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in text.split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        if stripped in {"none", "null", "all"}:
            values.append(None)
        else:
            values.append(int(stripped))
    if not values:
        raise ValueError("top_k 网格不能为空")
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
    parser = argparse.ArgumentParser(description="Grid search multitask transformer strategy thresholds")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--short-up-grid", default="0.53,0.56,0.59,0.62,0.65")
    parser.add_argument("--take-profit-grid", default="0.15,0.20,0.25,0.30,0.35")
    parser.add_argument("--stop-loss-grid", default="0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--top-k-grid", default="none")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-signals", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--hold-days", type=int, default=45)
    parser.add_argument("--take-profit-exit", type=float, default=0.12)
    parser.add_argument("--stop-loss-exit", type=float, default=-0.06)
    args = parser.parse_args()

    config = load_config(args.config)
    backtester = TransformerStrategyBacktester(config)
    short_up_grid = _parse_grid(args.short_up_grid)
    take_profit_grid = _parse_grid(args.take_profit_grid)
    stop_loss_grid = _parse_grid(args.stop_loss_grid)
    top_k_grid = _parse_top_k_grid(args.top_k_grid)

    search_rows = []
    best_summary = None
    best_result = None
    best_feasible_summary = None
    best_feasible_result = None

    for short_up_threshold, take_profit_threshold, stop_loss_threshold, top_k_per_day in product(
        short_up_grid,
        take_profit_grid,
        stop_loss_grid,
        top_k_grid,
    ):
        try:
            result = backtester.run(
                start_date=args.start_date,
                end_date=args.end_date,
                short_up_threshold=short_up_threshold,
                take_profit_threshold=take_profit_threshold,
                stop_loss_threshold=stop_loss_threshold,
                top_k_per_day=top_k_per_day,
                use_calibration=args.use_calibration,
                hold_days=args.hold_days,
                take_profit=args.take_profit_exit,
                stop_loss=args.stop_loss_exit,
                initial_capital=1_000_000.0,
                commission=0.001,
                slippage=0.001,
                min_lot=100,
            )
        except RuntimeError:
            continue

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
        raise RuntimeError("阈值搜索没有产生任何有效回测结果")

    selected_summary = best_feasible_summary or best_summary
    selected_result = best_feasible_result or best_result
    search_df = pd.DataFrame(search_rows).sort_values(
        ["meets_constraints", "ranking_score", "total_return", "sharpe_ratio", "max_drawdown", "trade_count"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    output_tag = f"_{args.output_tag}" if args.output_tag else ""
    results_dir = config.results_dir
    search_csv_path = results_dir / f"transformer_multitask_threshold_search{output_tag}.csv"
    top_csv_path = results_dir / f"transformer_multitask_threshold_search_top{output_tag}.csv"
    best_json_path = results_dir / f"transformer_multitask_threshold_best{output_tag}.json"

    search_df.to_csv(search_csv_path, index=False, encoding="utf-8-sig")
    search_df.head(args.top_n).to_csv(top_csv_path, index=False, encoding="utf-8-sig")

    prefix = f"transformer_multitask_search_mt{args.min_trades}_ms{args.min_signals}{output_tag}"
    detail_paths = write_backtest_outputs(selected_result, results_dir, prefix)
    payload = {
        "grid_size": len(search_rows),
        "constraints": {
            "min_trades": args.min_trades,
            "min_signals": args.min_signals,
        },
        "execution": {
            "hold_days": args.hold_days,
            "take_profit_exit": args.take_profit_exit,
            "stop_loss_exit": args.stop_loss_exit,
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
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps({**payload, "top_rows": search_df.head(args.top_n).to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()