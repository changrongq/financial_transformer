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

for path in [PROJECT_SRC, SRC, ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_lightweight_rerank_probe import StaticRerankBacktester  # noqa: E402
from financial_transformer_experiment.backtest import write_backtest_outputs  # noqa: E402
from financial_transformer_experiment.config import load_config  # noqa: E402


def _parse_float_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("grid cannot be empty")
    return sorted(dict.fromkeys(values))


def _parse_int_grid(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("grid cannot be empty")
    return sorted(dict.fromkeys(values))


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
    parser = argparse.ArgumentParser(description="Grid search execution parameters on a static rerank prediction pool")
    parser.add_argument("--config", default="config_moneyflow_v1_recent_oos_v1.yaml")
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--score-col", default="rerank_score")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--short-up-threshold", type=float, required=True)
    parser.add_argument("--take-profit-threshold", type=float, required=True)
    parser.add_argument("--stop-loss-threshold", type=float, required=True)
    parser.add_argument("--top-k-grid", default="1")
    parser.add_argument("--take-profit-exit-grid", default="0.14,0.16,0.18")
    parser.add_argument("--stop-loss-exit-grid", default="-0.06,-0.05,-0.04")
    parser.add_argument("--hold-days-grid", default="30,45")
    parser.add_argument("--min-trades", type=int, default=2)
    parser.add_argument("--min-signals", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-tag", default="static_rerank_execution_search")
    args = parser.parse_args()

    config = load_config(str((ROOT / args.config).resolve()))
    predictions_path = Path(args.predictions_file)
    if not predictions_path.is_absolute():
        predictions_path = (ROOT / predictions_path).resolve()
    predictions = pd.read_csv(predictions_path, dtype={"stock_code": str})
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions = predictions.dropna(subset=["date", args.score_col]).copy()

    backtester = StaticRerankBacktester(config, predictions=predictions, score_col=args.score_col)
    top_k_grid = _parse_int_grid(args.top_k_grid)
    take_profit_exit_grid = _parse_float_grid(args.take_profit_exit_grid)
    stop_loss_exit_grid = _parse_float_grid(args.stop_loss_exit_grid)
    hold_days_grid = _parse_int_grid(args.hold_days_grid)

    rows = []
    best_summary = None
    best_result = None

    for top_k_per_day, take_profit_exit, stop_loss_exit, hold_days in product(
        top_k_grid,
        take_profit_exit_grid,
        stop_loss_exit_grid,
        hold_days_grid,
    ):
        try:
            result = backtester.run(
                start_date=args.start_date,
                end_date=args.end_date,
                short_up_threshold=args.short_up_threshold,
                take_profit_threshold=args.take_profit_threshold,
                stop_loss_threshold=args.stop_loss_threshold,
                top_k_per_day=top_k_per_day,
                hold_days=hold_days,
                take_profit=take_profit_exit,
                stop_loss=stop_loss_exit,
                initial_capital=1_000_000.0,
                commission=0.001,
                slippage=0.001,
                min_lot=100,
                min_position_weight=0.08,
                max_position_weight=0.20,
                max_total_position_fraction=1.0,
                cooldown_trigger_losses=0,
                cooldown_days=0,
                min_secondary_primary_ratio=0.0,
                rank_weight_power=0.0,
                min_score_ratio_to_leader=0.0,
            )
        except RuntimeError:
            continue

        row = dict(result.summary)
        row["top_k_per_day"] = top_k_per_day
        row["take_profit_exit"] = take_profit_exit
        row["stop_loss_exit"] = stop_loss_exit
        row["hold_days_exit"] = hold_days
        row["meets_min_trades"] = row["trade_count"] >= args.min_trades
        row["meets_min_signals"] = row["signal_count"] >= args.min_signals
        row["meets_constraints"] = row["meets_min_trades"] and row["meets_min_signals"]
        row["ranking_score"] = _ranking_score(row, min_trades=args.min_trades, min_signals=args.min_signals)
        rows.append(row)

        if best_summary is None or (
            row["meets_constraints"], row["ranking_score"]
        ) > (
            best_summary["meets_constraints"], best_summary["ranking_score"]
        ):
            best_summary = row
            best_result = result

    if best_summary is None or best_result is None:
        raise RuntimeError("static rerank execution search produced no valid backtest results")

    search_df = pd.DataFrame(rows).sort_values(
        ["meets_constraints", "ranking_score", "total_return", "sharpe_ratio", "max_drawdown", "trade_count"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    output_dir = predictions_path.parent
    search_csv_path = output_dir / f"execution_search_{args.output_tag}.csv"
    top_csv_path = output_dir / f"execution_search_top_{args.output_tag}.csv"
    best_json_path = output_dir / f"execution_search_best_{args.output_tag}.json"
    search_df.to_csv(search_csv_path, index=False, encoding="utf-8-sig")
    search_df.head(args.top_n).to_csv(top_csv_path, index=False, encoding="utf-8-sig")

    detail_paths = write_backtest_outputs(best_result, output_dir, f"execution_search_{args.output_tag}")
    payload = {
        "config": str((ROOT / args.config).resolve()),
        "predictions_file": str(predictions_path),
        "score_col": args.score_col,
        "fixed_signal_thresholds": {
            "short_up": args.short_up_threshold,
            "take_profit": args.take_profit_threshold,
            "stop_loss": args.stop_loss_threshold,
        },
        "grid_size": int(len(rows)),
        "constraints": {
            "min_trades": args.min_trades,
            "min_signals": args.min_signals,
        },
        "best": best_summary,
        "detail_outputs": detail_paths,
        "search_results": {
            "all": str(search_csv_path),
            "top": str(top_csv_path),
        },
    }
    best_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "top_rows": search_df.head(args.top_n).to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()