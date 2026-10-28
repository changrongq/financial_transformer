from __future__ import annotations

import argparse
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

from financial_transformer_experiment.backtest import TransformerStrategyBacktester
from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.two_stage_execution import build_test_two_stage_predictions
from financial_transformer_experiment.two_stage_execution import resolve_two_stage_execution_spec
from financial_transformer_experiment.two_stage_execution import resolve_single_stage_execution_defaults


DEFAULT_EXECUTION_HOLD_DAYS = 45
DEFAULT_EXECUTION_TAKE_PROFIT = 0.14
DEFAULT_EXECUTION_STOP_LOSS = -0.06
DEFAULT_MIN_POSITION_WEIGHT = 0.08
DEFAULT_MAX_POSITION_WEIGHT = 0.20
DEFAULT_COOLDOWN_TRIGGER_LOSSES = 0
DEFAULT_COOLDOWN_DAYS = 0
DEFAULT_LEADER_MAX_DAY_RANK = None
DEFAULT_LEADER_TAKE_PROFIT = None
DEFAULT_LEADER_STOP_LOSS = None
DEFAULT_LEADER_HOLD_DAYS = None


def _load_trade_dates(config_path: str, use_calibration: bool) -> list[pd.Timestamp]:
    config = load_config(config_path)
    spec = resolve_two_stage_execution_spec(config_path)
    predictions_file = config.results_dir / ("test_predictions_calibrated.csv" if use_calibration else "test_predictions.csv")
    if spec is not None and not predictions_file.exists():
        build_test_two_stage_predictions(config_path, use_calibration=use_calibration)
    if not predictions_file.exists():
        raise FileNotFoundError(f"prediction file not found: {predictions_file}")
    df = pd.read_csv(predictions_file, usecols=["date"])
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().drop_duplicates().sort_values().tolist()
    if not dates:
        raise RuntimeError("no prediction dates found for rolling backtest")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-window backtests for a fixed transformer strategy configuration")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--short-up-threshold", type=float)
    parser.add_argument("--take-profit-threshold", type=float)
    parser.add_argument("--stop-loss-threshold", type=float)
    parser.add_argument("--top-k-per-day")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--hold-days", type=int)
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--min-position-weight", type=float, default=DEFAULT_MIN_POSITION_WEIGHT)
    parser.add_argument("--max-position-weight", type=float, default=DEFAULT_MAX_POSITION_WEIGHT)
    parser.add_argument("--cooldown-trigger-losses", type=int, default=DEFAULT_COOLDOWN_TRIGGER_LOSSES)
    parser.add_argument("--cooldown-days", type=int, default=DEFAULT_COOLDOWN_DAYS)
    parser.add_argument("--min-score-ratio-to-leader", type=float)
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--leader-max-day-rank", type=int, default=DEFAULT_LEADER_MAX_DAY_RANK)
    parser.add_argument("--leader-take-profit", type=float, default=DEFAULT_LEADER_TAKE_PROFIT)
    parser.add_argument("--leader-stop-loss", type=float, default=DEFAULT_LEADER_STOP_LOSS)
    parser.add_argument("--leader-hold-days", type=int, default=DEFAULT_LEADER_HOLD_DAYS)
    args = parser.parse_args()

    config = load_config(args.config)
    spec = resolve_two_stage_execution_spec(args.config)
    single_stage_defaults = resolve_single_stage_execution_defaults(args.config)
    execution_defaults = spec or single_stage_defaults
    backtester = TransformerStrategyBacktester(config)
    short_up_threshold = args.short_up_threshold
    if short_up_threshold is None:
        calibrated_short_up_threshold = getattr(execution_defaults, "default_calibrated_short_up_threshold", None) if execution_defaults else None
        if args.use_calibration and calibrated_short_up_threshold is not None:
            short_up_threshold = calibrated_short_up_threshold
        else:
            short_up_threshold = execution_defaults.default_short_up_threshold if execution_defaults else 0.254
    take_profit_threshold = args.take_profit_threshold if args.take_profit_threshold is not None else (execution_defaults.default_take_profit_threshold if execution_defaults else 0.572)
    stop_loss_threshold = args.stop_loss_threshold if args.stop_loss_threshold is not None else (execution_defaults.default_stop_loss_threshold if execution_defaults else 0.50)
    dates = _load_trade_dates(args.config, use_calibration=args.use_calibration)
    if args.start_date:
        dates = [date for date in dates if date >= pd.Timestamp(args.start_date)]
    if args.end_date:
        dates = [date for date in dates if date <= pd.Timestamp(args.end_date)]
    windows = _window_ranges(dates, window_size=args.window_size, step_size=args.step_size)
    if not windows:
        raise RuntimeError("not enough prediction dates for rolling backtest windows")

    raw_top_k = args.top_k_per_day if args.top_k_per_day is not None else str(execution_defaults.default_top_k_per_day if execution_defaults else 2)
    top_k_per_day = None if str(raw_top_k).lower() in {"none", "null", "all"} else int(raw_top_k)
    min_score_ratio_to_leader = (
        args.min_score_ratio_to_leader
        if args.min_score_ratio_to_leader is not None
        else (execution_defaults.default_min_score_ratio_to_leader if execution_defaults else 0.0)
    )
    hold_days = args.hold_days if args.hold_days is not None else (execution_defaults.default_execution_hold_days if execution_defaults else DEFAULT_EXECUTION_HOLD_DAYS)
    take_profit = args.take_profit if args.take_profit is not None else (execution_defaults.default_execution_take_profit if execution_defaults else DEFAULT_EXECUTION_TAKE_PROFIT)
    stop_loss = args.stop_loss if args.stop_loss is not None else (execution_defaults.default_execution_stop_loss if execution_defaults else DEFAULT_EXECUTION_STOP_LOSS)

    rows = []
    for start_date, end_date, offset in windows:
        try:
            result = backtester.run(
                start_date=str(start_date.date()),
                end_date=str(end_date.date()),
                short_up_threshold=short_up_threshold,
                take_profit_threshold=take_profit_threshold,
                stop_loss_threshold=stop_loss_threshold,
                top_k_per_day=top_k_per_day,
                use_calibration=args.use_calibration,
                hold_days=hold_days,
                take_profit=take_profit,
                stop_loss=stop_loss,
                initial_capital=1_000_000.0,
                commission=0.001,
                slippage=0.001,
                min_lot=100,
                min_position_weight=args.min_position_weight,
                max_position_weight=args.max_position_weight,
                cooldown_trigger_losses=args.cooldown_trigger_losses,
                cooldown_days=args.cooldown_days,
                min_score_ratio_to_leader=min_score_ratio_to_leader,
                leader_max_day_rank=args.leader_max_day_rank,
                leader_take_profit=args.leader_take_profit,
                leader_stop_loss=args.leader_stop_loss,
                leader_hold_days=args.leader_hold_days,
            )
            row = dict(result.summary)
            row["window_index"] = len(rows)
            row["window_offset"] = offset
            rows.append(row)
        except RuntimeError as exc:
            rows.append(
                {
                    "requested_start_date": str(start_date.date()),
                    "requested_end_date": str(end_date.date()),
                    "short_up_threshold": short_up_threshold,
                    "take_profit_threshold": take_profit_threshold,
                    "stop_loss_threshold": stop_loss_threshold,
                    "top_k_per_day": top_k_per_day,
                    "signal_count": 0,
                    "trade_count": 0,
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "avg_hold_days": 0.0,
                    "window_index": len(rows),
                    "window_offset": offset,
                    "error": str(exc),
                    "leader_max_day_rank": args.leader_max_day_rank,
                    "leader_take_profit": args.leader_take_profit,
                    "leader_stop_loss": args.leader_stop_loss,
                    "leader_hold_days": args.leader_hold_days,
                }
            )

    summary_df = pd.DataFrame(rows)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    csv_path = config.results_dir / f"rolling_backtest{tag}.csv"
    json_path = config.results_dir / f"rolling_backtest{tag}.json"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "config": args.config,
        "results_dir": str(config.results_dir),
        "window_size": args.window_size,
        "step_size": args.step_size,
        "window_count": int(len(summary_df)),
        "thresholds": {
            "short_up": short_up_threshold,
            "take_profit": take_profit_threshold,
            "stop_loss": stop_loss_threshold,
            "top_k_per_day": top_k_per_day,
        },
        "execution_policy": {
            "hold_days": hold_days,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "min_position_weight": args.min_position_weight,
            "max_position_weight": args.max_position_weight,
            "cooldown_trigger_losses": args.cooldown_trigger_losses,
            "cooldown_days": args.cooldown_days,
            "min_score_ratio_to_leader": min_score_ratio_to_leader,
            "leader_max_day_rank": args.leader_max_day_rank,
            "leader_take_profit": args.leader_take_profit,
            "leader_stop_loss": args.leader_stop_loss,
            "leader_hold_days": args.leader_hold_days,
        },
        "aggregate": {
            "avg_total_return": float(summary_df["total_return"].mean()),
            "median_total_return": float(summary_df["total_return"].median()),
            "avg_sharpe_ratio": float(summary_df["sharpe_ratio"].mean()),
            "avg_max_drawdown": float(summary_df["max_drawdown"].mean()),
            "positive_windows": int((summary_df["total_return"] > 0).sum()),
            "nonzero_trade_windows": int((summary_df["trade_count"] > 0).sum()),
        },
        "output_csv": str(csv_path),
        "windows": summary_df.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()