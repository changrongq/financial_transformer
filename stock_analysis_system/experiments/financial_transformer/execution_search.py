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


def _parse_float_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("grid cannot be empty")
    return values


def _parse_int_grid(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("grid cannot be empty")
    return values


def _parse_float_grid_allow_empty(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return sorted(dict.fromkeys(values))


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
        raise ValueError("top_k grid cannot be empty")
    return values


def _parse_window_size_grid(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    return sorted(dict.fromkeys(values))


def _load_trade_dates(config_path: str, use_calibration: bool) -> list[pd.Timestamp]:
    config = load_config(config_path)
    predictions_file = config.results_dir / ("test_predictions_calibrated.csv" if use_calibration else "test_predictions.csv")
    if not predictions_file.exists():
        raise FileNotFoundError(f"prediction file not found: {predictions_file}")
    df = pd.read_csv(predictions_file, usecols=["date"])
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().drop_duplicates().sort_values().tolist()
    if not dates:
        raise RuntimeError("no prediction dates found for execution search rolling windows")
    return [pd.Timestamp(date) for date in dates]


def _window_ranges(dates: list[pd.Timestamp], window_size: int, step_size: int) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if len(dates) < window_size:
        return []
    windows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for start_idx in range(0, len(dates) - window_size + 1, step_size):
        end_idx = start_idx + window_size - 1
        windows.append((dates[start_idx], dates[end_idx], start_idx))
    return windows


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


def _rolling_summary(
    backtester: TransformerStrategyBacktester,
    windows_by_size: dict[int, list[tuple[pd.Timestamp, pd.Timestamp, int]]],
    *,
    short_up_threshold: float,
    take_profit_threshold: float,
    stop_loss_threshold: float,
    top_k_per_day: int | None,
    use_calibration: bool,
    hold_days: int,
    take_profit: float,
    stop_loss: float,
    min_position_weight: float,
    max_position_weight: float,
    max_total_position_fraction: float,
    cooldown_trigger_losses: int,
    cooldown_days: int,
    min_secondary_primary_ratio: float,
    rank_weight_power: float,
    min_score_ratio_to_leader: float,
) -> dict:
    if not windows_by_size:
        return {
            "rolling_window_count": 0,
            "rolling_positive_windows": 0,
            "rolling_nonzero_trade_windows": 0,
            "rolling_positive_window_ratio": 0.0,
            "rolling_avg_total_return": 0.0,
            "rolling_median_total_return": 0.0,
            "rolling_avg_sharpe_ratio": 0.0,
            "rolling_avg_max_drawdown": 0.0,
            "rolling_min_total_return": 0.0,
            "rolling_summaries": [],
        }

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for window_size, windows in windows_by_size.items():
        size_rows: list[dict] = []
        for start_date, end_date, _ in windows:
            try:
                result = backtester.run(
                    start_date=str(start_date.date()),
                    end_date=str(end_date.date()),
                    short_up_threshold=short_up_threshold,
                    take_profit_threshold=take_profit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    top_k_per_day=top_k_per_day,
                    use_calibration=use_calibration,
                    hold_days=hold_days,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    initial_capital=1_000_000.0,
                    commission=0.001,
                    slippage=0.001,
                    min_lot=100,
                    min_position_weight=min_position_weight,
                    max_position_weight=max_position_weight,
                    max_total_position_fraction=max_total_position_fraction,
                    cooldown_trigger_losses=cooldown_trigger_losses,
                    cooldown_days=cooldown_days,
                    min_secondary_primary_ratio=min_secondary_primary_ratio,
                    rank_weight_power=rank_weight_power,
                    min_score_ratio_to_leader=min_score_ratio_to_leader,
                )
                row = dict(result.summary)
            except RuntimeError as exc:
                row = {
                    "requested_start_date": str(start_date.date()),
                    "requested_end_date": str(end_date.date()),
                    "signal_count": 0,
                    "trade_count": 0,
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "avg_hold_days": 0.0,
                    "error": str(exc),
                }
            row["window_size"] = window_size
            size_rows.append(row)
            all_rows.append(row)

        size_df = pd.DataFrame(size_rows)
        summary_rows.append(
            {
                "window_size": window_size,
                "window_count": int(len(size_df)),
                "positive_windows": int((size_df["total_return"] > 0).sum()),
                "nonzero_trade_windows": int((size_df["trade_count"] > 0).sum()),
                "avg_total_return": float(size_df["total_return"].mean()),
                "median_total_return": float(size_df["total_return"].median()),
                "avg_sharpe_ratio": float(size_df["sharpe_ratio"].mean()),
                "avg_max_drawdown": float(size_df["max_drawdown"].mean()),
                "min_total_return": float(size_df["total_return"].min()),
            }
        )

    combined_df = pd.DataFrame(all_rows)
    positive_windows = int((combined_df["total_return"] > 0).sum())
    window_count = int(len(combined_df))
    return {
        "rolling_window_count": window_count,
        "rolling_positive_windows": positive_windows,
        "rolling_nonzero_trade_windows": int((combined_df["trade_count"] > 0).sum()),
        "rolling_positive_window_ratio": float(positive_windows / window_count) if window_count else 0.0,
        "rolling_avg_total_return": float(combined_df["total_return"].mean()) if window_count else 0.0,
        "rolling_median_total_return": float(combined_df["total_return"].median()) if window_count else 0.0,
        "rolling_avg_sharpe_ratio": float(combined_df["sharpe_ratio"].mean()) if window_count else 0.0,
        "rolling_avg_max_drawdown": float(combined_df["max_drawdown"].mean()) if window_count else 0.0,
        "rolling_min_total_return": float(combined_df["total_return"].min()) if window_count else 0.0,
        "rolling_summaries": summary_rows,
    }


def _combined_ranking_score(row: dict, min_trades: int, min_signals: int, rolling_weight: float) -> float:
    base_score = _ranking_score(row, min_trades=min_trades, min_signals=min_signals)
    if row.get("rolling_window_count", 0) <= 0:
        return base_score
    rolling_score = (
        row["rolling_avg_total_return"] * 1.2
        + row["rolling_avg_sharpe_ratio"] * 0.12
        + row["rolling_avg_max_drawdown"] * 0.5
        + row["rolling_positive_window_ratio"] * 0.08
        + row["rolling_min_total_return"] * 0.4
    )
    return base_score + rolling_weight * rolling_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search execution parameters for a fixed transformer signal strategy")
    parser.add_argument("--config", default=str(ROOT / "config_two_stage_rerank_best_v1.yaml"))
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--short-up-threshold", type=float, required=True)
    parser.add_argument("--take-profit-threshold", type=float, required=True)
    parser.add_argument("--stop-loss-threshold", type=float, required=True)
    parser.add_argument("--top-k-grid", default="2")
    parser.add_argument("--take-profit-exit-grid", default="0.08,0.10,0.12,0.15")
    parser.add_argument("--stop-loss-exit-grid", default="-0.03,-0.04,-0.05,-0.06,-0.08")
    parser.add_argument("--hold-days-grid", default="15,20,25,30,35,45")
    parser.add_argument("--min-position-weight-grid", default="0.08")
    parser.add_argument("--max-position-weight-grid", default="0.20")
    parser.add_argument("--max-total-position-fraction-grid", default="1.0")
    parser.add_argument("--cooldown-trigger-losses-grid", default="0")
    parser.add_argument("--cooldown-days-grid", default="0")
    parser.add_argument("--min-secondary-primary-ratio-grid", default="0.0")
    parser.add_argument("--rank-weight-power-grid", default="0.0")
    parser.add_argument("--min-score-ratio-to-leader-grid", default="0.0")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-signals", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--rolling-window-sizes", default="")
    parser.add_argument("--rolling-step-size", type=int, default=5)
    parser.add_argument("--rolling-weight", type=float, default=1.0)
    parser.add_argument("--min-positive-window-ratio", type=float, default=0.0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--use-calibration", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    backtester = TransformerStrategyBacktester(config)
    top_k_grid = _parse_top_k_grid(args.top_k_grid)
    take_profit_exit_grid = _parse_float_grid(args.take_profit_exit_grid)
    stop_loss_exit_grid = _parse_float_grid(args.stop_loss_exit_grid)
    hold_days_grid = _parse_int_grid(args.hold_days_grid)
    min_position_weight_grid = _parse_float_grid_allow_empty(args.min_position_weight_grid)
    max_position_weight_grid = _parse_float_grid_allow_empty(args.max_position_weight_grid)
    max_total_position_fraction_grid = _parse_float_grid_allow_empty(args.max_total_position_fraction_grid)
    cooldown_trigger_losses_grid = _parse_int_grid(args.cooldown_trigger_losses_grid)
    cooldown_days_grid = _parse_int_grid(args.cooldown_days_grid)
    min_secondary_primary_ratio_grid = _parse_float_grid_allow_empty(args.min_secondary_primary_ratio_grid)
    rank_weight_power_grid = _parse_float_grid_allow_empty(args.rank_weight_power_grid)
    min_score_ratio_to_leader_grid = _parse_float_grid_allow_empty(args.min_score_ratio_to_leader_grid)
    rolling_window_sizes = _parse_window_size_grid(args.rolling_window_sizes)

    rolling_windows_by_size: dict[int, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
    if rolling_window_sizes:
        dates = _load_trade_dates(args.config, use_calibration=args.use_calibration)
        if args.start_date:
            dates = [date for date in dates if date >= pd.Timestamp(args.start_date)]
        if args.end_date:
            dates = [date for date in dates if date <= pd.Timestamp(args.end_date)]
        for window_size in rolling_window_sizes:
            windows = _window_ranges(dates, window_size=window_size, step_size=args.rolling_step_size)
            if windows:
                rolling_windows_by_size[window_size] = windows

    rows = []
    best_summary = None
    best_result = None
    best_feasible_summary = None
    best_feasible_result = None

    for top_k_per_day, take_profit_exit, stop_loss_exit, hold_days, min_position_weight, max_position_weight, max_total_position_fraction, cooldown_trigger_losses, cooldown_days, min_secondary_primary_ratio, rank_weight_power, min_score_ratio_to_leader in product(
        top_k_grid,
        take_profit_exit_grid,
        stop_loss_exit_grid,
        hold_days_grid,
        min_position_weight_grid,
        max_position_weight_grid,
        max_total_position_fraction_grid,
        cooldown_trigger_losses_grid,
        cooldown_days_grid,
        min_secondary_primary_ratio_grid,
        rank_weight_power_grid,
        min_score_ratio_to_leader_grid,
    ):
        if max_position_weight < min_position_weight:
            continue
        if max_total_position_fraction < max_position_weight:
            continue
        if cooldown_trigger_losses <= 0 and cooldown_days > 0:
            continue
        if cooldown_trigger_losses > 0 and cooldown_days <= 0:
            continue
        try:
            result = backtester.run(
                start_date=args.start_date,
                end_date=args.end_date,
                short_up_threshold=args.short_up_threshold,
                take_profit_threshold=args.take_profit_threshold,
                stop_loss_threshold=args.stop_loss_threshold,
                top_k_per_day=top_k_per_day,
                use_calibration=args.use_calibration,
                hold_days=hold_days,
                take_profit=take_profit_exit,
                stop_loss=stop_loss_exit,
                initial_capital=1_000_000.0,
                commission=0.001,
                slippage=0.001,
                min_lot=100,
                min_position_weight=min_position_weight,
                max_position_weight=max_position_weight,
                max_total_position_fraction=max_total_position_fraction,
                cooldown_trigger_losses=cooldown_trigger_losses,
                cooldown_days=cooldown_days,
                min_secondary_primary_ratio=min_secondary_primary_ratio,
                rank_weight_power=rank_weight_power,
                min_score_ratio_to_leader=min_score_ratio_to_leader,
            )
        except RuntimeError:
            continue

        row = dict(result.summary)
        row["take_profit_exit"] = take_profit_exit
        row["stop_loss_exit"] = stop_loss_exit
        row["hold_days_exit"] = hold_days
        row["min_position_weight"] = min_position_weight
        row["max_position_weight"] = max_position_weight
        row["max_total_position_fraction"] = max_total_position_fraction
        row["cooldown_trigger_losses"] = cooldown_trigger_losses
        row["cooldown_days"] = cooldown_days
        row["min_secondary_primary_ratio"] = min_secondary_primary_ratio
        row["rank_weight_power"] = rank_weight_power
        row["min_score_ratio_to_leader"] = min_score_ratio_to_leader
        row.update(
            _rolling_summary(
                backtester,
                rolling_windows_by_size,
                short_up_threshold=args.short_up_threshold,
                take_profit_threshold=args.take_profit_threshold,
                stop_loss_threshold=args.stop_loss_threshold,
                top_k_per_day=top_k_per_day,
                use_calibration=args.use_calibration,
                hold_days=hold_days,
                take_profit=take_profit_exit,
                stop_loss=stop_loss_exit,
                min_position_weight=min_position_weight,
                max_position_weight=max_position_weight,
                max_total_position_fraction=max_total_position_fraction,
                cooldown_trigger_losses=cooldown_trigger_losses,
                cooldown_days=cooldown_days,
                min_secondary_primary_ratio=min_secondary_primary_ratio,
                rank_weight_power=rank_weight_power,
                min_score_ratio_to_leader=min_score_ratio_to_leader,
            )
        )
        row["meets_min_trades"] = row["trade_count"] >= args.min_trades
        row["meets_min_signals"] = row["signal_count"] >= args.min_signals
        row["meets_min_positive_window_ratio"] = row.get("rolling_positive_window_ratio", 0.0) >= args.min_positive_window_ratio
        row["meets_constraints"] = row["meets_min_trades"] and row["meets_min_signals"] and row["meets_min_positive_window_ratio"]
        row["ranking_score"] = _ranking_score(row, min_trades=args.min_trades, min_signals=args.min_signals)
        row["combined_ranking_score"] = _combined_ranking_score(
            row,
            min_trades=args.min_trades,
            min_signals=args.min_signals,
            rolling_weight=args.rolling_weight,
        )
        rows.append(row)

        if best_summary is None or row["combined_ranking_score"] > best_summary["combined_ranking_score"]:
            best_summary = row
            best_result = result
        if row["meets_constraints"] and (
            best_feasible_summary is None or row["combined_ranking_score"] > best_feasible_summary["combined_ranking_score"]
        ):
            best_feasible_summary = row
            best_feasible_result = result

    if best_summary is None or best_result is None:
        raise RuntimeError("execution search produced no valid backtest results")

    selected_summary = best_feasible_summary or best_summary
    selected_result = best_feasible_result or best_result
    search_df = pd.DataFrame(rows).sort_values(
        [
            "meets_constraints",
            "combined_ranking_score",
            "ranking_score",
            "rolling_positive_window_ratio",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "trade_count",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    ).reset_index(drop=True)

    output_tag = f"_{args.output_tag}" if args.output_tag else ""
    search_csv_path = config.results_dir / f"execution_search{output_tag}.csv"
    top_csv_path = config.results_dir / f"execution_search_top{output_tag}.csv"
    best_json_path = config.results_dir / f"execution_search_best{output_tag}.json"
    search_df.to_csv(search_csv_path, index=False, encoding="utf-8-sig")
    search_df.head(args.top_n).to_csv(top_csv_path, index=False, encoding="utf-8-sig")

    prefix = f"execution_search_mt{args.min_trades}_ms{args.min_signals}{output_tag}"
    detail_paths = write_backtest_outputs(selected_result, config.results_dir, prefix)
    payload = {
        "grid_size": int(len(rows)),
        "fixed_signal_thresholds": {
            "short_up": args.short_up_threshold,
            "take_profit": args.take_profit_threshold,
            "stop_loss": args.stop_loss_threshold,
        },
        "execution_grids": {
            "min_position_weight": min_position_weight_grid,
            "max_position_weight": max_position_weight_grid,
            "max_total_position_fraction": max_total_position_fraction_grid,
            "cooldown_trigger_losses": cooldown_trigger_losses_grid,
            "cooldown_days": cooldown_days_grid,
            "min_secondary_primary_ratio": min_secondary_primary_ratio_grid,
            "rank_weight_power": rank_weight_power_grid,
            "min_score_ratio_to_leader": min_score_ratio_to_leader_grid,
        },
        "constraints": {
            "min_trades": args.min_trades,
            "min_signals": args.min_signals,
            "min_positive_window_ratio": args.min_positive_window_ratio,
        },
        "rolling_validation": {
            "window_sizes": rolling_window_sizes,
            "step_size": args.rolling_step_size,
            "weight": args.rolling_weight,
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