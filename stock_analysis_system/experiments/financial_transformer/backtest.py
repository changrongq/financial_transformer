from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from financial_transformer_experiment.backtest import TransformerStrategyBacktester, write_backtest_outputs
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest financial transformer strategy")
    parser.add_argument("--config", default=str(ROOT / "config_two_stage_rerank_best_v1.yaml"))
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--short-up-threshold", type=float)
    parser.add_argument("--take-profit-threshold", type=float)
    parser.add_argument("--stop-loss-threshold", type=float)
    parser.add_argument("--hold-days", type=int)
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--top-k-per-day", type=int)
    parser.add_argument("--min-position-weight", type=float, default=DEFAULT_MIN_POSITION_WEIGHT)
    parser.add_argument("--max-position-weight", type=float, default=DEFAULT_MAX_POSITION_WEIGHT)
    parser.add_argument("--cooldown-trigger-losses", type=int, default=DEFAULT_COOLDOWN_TRIGGER_LOSSES)
    parser.add_argument("--cooldown-days", type=int, default=DEFAULT_COOLDOWN_DAYS)
    parser.add_argument("--stock-cooldown-days", type=int, default=0)
    parser.add_argument("--min-score-ratio-to-leader", type=float)
    parser.add_argument("--leader-max-day-rank", type=int)
    parser.add_argument("--leader-take-profit", type=float)
    parser.add_argument("--leader-stop-loss", type=float)
    parser.add_argument("--leader-hold-days", type=int)
    parser.add_argument("--state-profile", type=str)
    parser.add_argument("--output-prefix", type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    spec = resolve_two_stage_execution_spec(args.config)
    single_stage_defaults = resolve_single_stage_execution_defaults(args.config)
    predictions_suffix = "_calibrated" if args.use_calibration else ""
    predictions_path = config.results_dir / f"test_predictions{predictions_suffix}.csv"
    if spec is not None and not predictions_path.exists():
        build_test_two_stage_predictions(args.config, use_calibration=args.use_calibration)
    execution_defaults = spec or single_stage_defaults
    top_k_per_day = args.top_k_per_day if args.top_k_per_day is not None else (execution_defaults.default_top_k_per_day if execution_defaults else 2)
    hold_days = args.hold_days if args.hold_days is not None else (execution_defaults.default_execution_hold_days if execution_defaults else DEFAULT_EXECUTION_HOLD_DAYS)
    take_profit = args.take_profit if args.take_profit is not None else (execution_defaults.default_execution_take_profit if execution_defaults else DEFAULT_EXECUTION_TAKE_PROFIT)
    stop_loss = args.stop_loss if args.stop_loss is not None else (execution_defaults.default_execution_stop_loss if execution_defaults else DEFAULT_EXECUTION_STOP_LOSS)
    min_score_ratio_to_leader = (
        args.min_score_ratio_to_leader
        if args.min_score_ratio_to_leader is not None
        else (execution_defaults.default_min_score_ratio_to_leader if execution_defaults else 0.0)
    )
    leader_max_day_rank = args.leader_max_day_rank if args.leader_max_day_rank is not None else (execution_defaults.default_leader_max_day_rank if execution_defaults else None)
    leader_take_profit = args.leader_take_profit if args.leader_take_profit is not None else (execution_defaults.default_leader_take_profit if execution_defaults else None)
    leader_stop_loss = args.leader_stop_loss if args.leader_stop_loss is not None else (execution_defaults.default_leader_stop_loss if execution_defaults else None)
    leader_hold_days = args.leader_hold_days if args.leader_hold_days is not None else (execution_defaults.default_leader_hold_days if execution_defaults else None)
    short_up_threshold = args.short_up_threshold
    if short_up_threshold is None:
        calibrated_short_up_threshold = getattr(execution_defaults, "default_calibrated_short_up_threshold", None) if execution_defaults else None
        if args.use_calibration and calibrated_short_up_threshold is not None:
            short_up_threshold = calibrated_short_up_threshold
        else:
            short_up_threshold = execution_defaults.default_short_up_threshold if execution_defaults else 0.254
    take_profit_threshold = args.take_profit_threshold if args.take_profit_threshold is not None else (execution_defaults.default_take_profit_threshold if execution_defaults else 0.572)
    stop_loss_threshold = args.stop_loss_threshold if args.stop_loss_threshold is not None else (execution_defaults.default_stop_loss_threshold if execution_defaults else 0.50)
    backtester = TransformerStrategyBacktester(config)
    result = backtester.run(
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
        stock_cooldown_days=args.stock_cooldown_days,
        min_score_ratio_to_leader=min_score_ratio_to_leader,
        leader_max_day_rank=leader_max_day_rank,
        leader_take_profit=leader_take_profit,
        leader_stop_loss=leader_stop_loss,
        leader_hold_days=leader_hold_days,
        state_profile=args.state_profile,
    )
    output_prefix = args.output_prefix or "transformer_micro_multitask_backtest"
    outputs = write_backtest_outputs(result, config.results_dir, output_prefix)
    print(json.dumps({**result.summary, **outputs}, ensure_ascii=False, indent=2))