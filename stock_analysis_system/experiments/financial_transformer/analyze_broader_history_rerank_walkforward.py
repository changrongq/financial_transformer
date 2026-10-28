from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC, ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_lightweight_rerank_probe import (  # noqa: E402
    KNOWN_FAIR_PARAMS,
    RERANK_FEATURES,
    StaticRerankBacktester,
    _build_training_frame,
    _freeze_to_checkpoint_universe,
    _load_or_generate_split_predictions,
    _prepare_candidates,
    _resolve_fair_params,
)
from financial_transformer_experiment.config import load_config  # noqa: E402


def _load_config(config_name: str):
    return _freeze_to_checkpoint_universe(load_config(str((ROOT / config_name).resolve())))


def _run_window_backtest(config, predictions: pd.DataFrame, score_col: str, fair: dict, start_date: str, end_date: str) -> dict:
    backtester = StaticRerankBacktester(config, predictions=predictions, score_col=score_col)
    result = backtester.run(
        start_date=start_date,
        end_date=end_date,
        short_up_threshold=fair["short_up_threshold"],
        take_profit_threshold=fair["take_profit_threshold"],
        stop_loss_threshold=fair["stop_loss_threshold"],
        top_k_per_day=fair["top_k_per_day"],
        hold_days=fair["hold_days"],
        take_profit=fair["take_profit"],
        stop_loss=fair["stop_loss"],
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
    return result.summary


def _window_ranges(dates: list[pd.Timestamp], window_size: int, step_size: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(dates) < window_size:
        return []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start_idx in range(0, len(dates) - window_size + 1, step_size):
        end_idx = start_idx + window_size - 1
        ranges.append((dates[start_idx], dates[end_idx]))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for broader-history rerank on recent baseline windows")
    parser.add_argument("--history-config", default="config_moneyflow_v1.yaml")
    parser.add_argument("--eval-config", default="config_moneyflow_v1_recent_oos_v1.yaml")
    parser.add_argument("--history-split", default="both", choices=["val", "test", "both"])
    parser.add_argument("--candidate-pool", type=int, default=5)
    parser.add_argument("--label-mode", default="best_in_pool", choices=["best_in_pool", "positive_return"])
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--include-prior-recent-history", action="store_true")
    parser.add_argument("--output-prefix", default="broader_history_rerank_walkforward_report")
    parser.add_argument("--short-up-threshold", type=float, default=None)
    parser.add_argument("--take-profit-threshold", type=float, default=None)
    parser.add_argument("--stop-loss-threshold", type=float, default=None)
    parser.add_argument("--top-k-per-day", type=int, default=None)
    parser.add_argument("--hold-days", type=int, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    args = parser.parse_args()

    history_config = _load_config(args.history_config)
    eval_config = _load_config(args.eval_config)
    fair = _resolve_fair_params((ROOT / args.eval_config).stem, args)

    history_val, history_test = _load_or_generate_split_predictions(history_config)
    eval_val, eval_test = _load_or_generate_split_predictions(eval_config)

    history_frames = [history_val] if args.history_split == "val" else [history_test] if args.history_split == "test" else [history_val, history_test]
    history_candidates = []
    for frame in history_frames:
        candidates = _prepare_candidates(frame, history_config, fair, args.candidate_pool)
        if not candidates.empty:
            history_candidates.append(candidates)
    if not history_candidates:
        raise RuntimeError("history candidates are empty")
    fixed_history = pd.concat(history_candidates, ignore_index=True)

    eval_candidates = _prepare_candidates(eval_test, eval_config, fair, args.candidate_pool)
    if eval_candidates.empty:
        raise RuntimeError("eval candidates are empty")

    recent_dates = sorted(pd.to_datetime(eval_candidates["date"]).drop_duplicates().tolist())
    windows = _window_ranges(recent_dates, window_size=args.window_size, step_size=args.step_size)
    if not windows:
        raise RuntimeError("no walk-forward windows generated")

    rows: list[dict] = []
    for window_start, window_end in windows:
        train_parts = [fixed_history]
        if args.include_prior_recent_history:
            prior_recent = eval_candidates[eval_candidates["date"] < pd.Timestamp(window_start)].copy()
            if not prior_recent.empty:
                train_parts.append(prior_recent)
        train_candidates = pd.concat(train_parts, ignore_index=True)
        train_frame = _build_training_frame(train_candidates, args.label_mode)
        if train_frame["rerank_label"].nunique() < 2:
            continue

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(train_frame[RERANK_FEATURES], train_frame["rerank_label"].astype(int))

        window_candidates = eval_candidates[
            (eval_candidates["date"] >= pd.Timestamp(window_start))
            & (eval_candidates["date"] <= pd.Timestamp(window_end))
        ].copy()
        if window_candidates.empty:
            continue
        window_candidates["rerank_score"] = model.predict_proba(window_candidates[RERANK_FEATURES])[:, 1]

        base_summary = _run_window_backtest(
            eval_config,
            window_candidates,
            score_col="signal_score",
            fair=fair,
            start_date=str(pd.Timestamp(window_start).date()),
            end_date=str(pd.Timestamp(eval_config.splits.test_end).date()),
        )
        rerank_summary = _run_window_backtest(
            eval_config,
            window_candidates,
            score_col="rerank_score",
            fair=fair,
            start_date=str(pd.Timestamp(window_start).date()),
            end_date=str(pd.Timestamp(eval_config.splits.test_end).date()),
        )
        rows.append(
            {
                "window_start": str(pd.Timestamp(window_start).date()),
                "window_end": str(pd.Timestamp(window_end).date()),
                "train_dates": int(train_frame["date"].nunique()),
                "train_rows": int(len(train_frame)),
                "base_total_return": float(base_summary["total_return"]),
                "base_sharpe_ratio": float(base_summary["sharpe_ratio"]),
                "base_trade_count": int(base_summary["trade_count"]),
                "rerank_total_return": float(rerank_summary["total_return"]),
                "rerank_sharpe_ratio": float(rerank_summary["sharpe_ratio"]),
                "rerank_trade_count": int(rerank_summary["trade_count"]),
                "delta_total_return": float(rerank_summary["total_return"] - base_summary["total_return"]),
                "delta_sharpe_ratio": float(rerank_summary["sharpe_ratio"] - base_summary["sharpe_ratio"]),
            }
        )

    if not rows:
        raise RuntimeError("no valid walk-forward rows produced")

    report_df = pd.DataFrame(rows)
    summary = {
        "history_config": args.history_config,
        "eval_config": args.eval_config,
        "history_split": args.history_split,
        "candidate_pool": args.candidate_pool,
        "label_mode": args.label_mode,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "include_prior_recent_history": bool(args.include_prior_recent_history),
        "fair": fair,
        "window_count": int(len(report_df)),
        "improved_return_windows": int((report_df["delta_total_return"] > 0).sum()),
        "improved_sharpe_windows": int((report_df["delta_sharpe_ratio"] > 0).sum()),
        "avg_delta_total_return": float(report_df["delta_total_return"].mean()),
        "avg_delta_sharpe_ratio": float(report_df["delta_sharpe_ratio"].mean()),
        "rows": report_df.to_dict(orient="records"),
    }

    output_json = ROOT / "results" / f"{args.output_prefix}.json"
    output_md = ROOT / "results" / f"{args.output_prefix}.md"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Broader History Rerank Walkforward Report", ""]
    lines.append(f"- History config: {args.history_config}")
    lines.append(f"- Eval config: {args.eval_config}")
    lines.append(f"- History split: {args.history_split}")
    lines.append(f"- Candidate pool: {args.candidate_pool}")
    lines.append(f"- Label mode: {args.label_mode}")
    lines.append(f"- Window size / step: {args.window_size} / {args.step_size}")
    lines.append(f"- Include prior recent history: {args.include_prior_recent_history}")
    lines.append(
        f"- Fair: su={fair['short_up_threshold']}, tp={fair['take_profit_threshold']}, sl={fair['stop_loss_threshold']}, top_k={fair['top_k_per_day']}, hold={fair['hold_days']}, tp_exit={fair['take_profit']}, sl_exit={fair['stop_loss']}"
    )
    lines.append(f"- Improved return windows: {summary['improved_return_windows']} / {summary['window_count']}")
    lines.append(f"- Improved sharpe windows: {summary['improved_sharpe_windows']} / {summary['window_count']}")
    lines.append(f"- Avg delta total return: {summary['avg_delta_total_return'] * 100:.2f}%")
    lines.append(f"- Avg delta sharpe: {summary['avg_delta_sharpe_ratio']:.4f}")
    lines.append("")
    lines.append("## Windows")
    lines.append("")
    for row in summary["rows"]:
        lines.append(
            f"- {row['window_start']} -> {row['window_end']}: base_ret={row['base_total_return'] * 100:.2f}%, rerank_ret={row['rerank_total_return'] * 100:.2f}%, delta_ret={row['delta_total_return'] * 100:.2f}%, base_sharpe={row['base_sharpe_ratio']:.4f}, rerank_sharpe={row['rerank_sharpe_ratio']:.4f}, delta_sharpe={row['delta_sharpe_ratio']:.4f}"
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "window_count": summary["window_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()