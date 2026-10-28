from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.backtest import TransformerStrategyBacktester, write_backtest_outputs
from financial_transformer_experiment.calibration import apply_probability_calibration, load_probability_calibration
from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import build_datasets
from financial_transformer_experiment.model import create_model
from financial_transformer_experiment.training import _build_loader, _evaluate


FILTER_FEATURES = [
    "pred_short_up",
    "pred_take_profit",
    "pred_stop_loss",
    "pred_max_gain",
    "pred_max_drawdown",
    "signal_score",
]


def _strategy_score_frame(df: pd.DataFrame) -> pd.Series:
    gain_term = df["pred_max_gain"].clip(lower=0.0, upper=0.5)
    drawdown_penalty = df["pred_max_drawdown"].abs().clip(lower=0.01, upper=0.3)
    return (
        0.45 * df["pred_short_up"]
        + 0.35 * df["pred_take_profit"]
        + 0.20 * (1.0 - df["pred_stop_loss"])
        + 0.10 * gain_term
        - 0.05 * drawdown_penalty
    )


def _static_trade_quality_label(df: pd.DataFrame) -> pd.Series:
    good_trade = (df["label_take_profit"] >= 0.5) & (df["label_stop_loss"] < 0.5)
    return good_trade.astype(int)


def _ranking_score(row: dict, min_trades: int, min_signals: int) -> float:
    trade_penalty = 0.0 if row["trade_count"] >= min_trades else (min_trades - row["trade_count"]) * 0.02
    signal_penalty = 0.0 if row["signal_count"] >= min_signals else (min_signals - row["signal_count"]) * 0.005
    return (
        row["total_return"]
        + 0.15 * row["sharpe_ratio"]
        + 0.05 * row["win_rate"]
        + 0.03 * row["profit_factor"]
        + 0.5 * row["max_drawdown"]
        - trade_penalty
        - signal_penalty
    )


def _build_model(config, input_dim: int, device: torch.device) -> FinancialTransformer:
    return create_model(
        input_dim=input_dim,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        ffn_type=config.model.ffn_type,
        position_encoding=config.model.position_encoding,
        model_variant=config.model.model_variant,
        feature_cols=config.experiment.feature_cols,
        grouped_feature_groups=config.model.grouped_feature_groups,
        use_group_gates=config.model.use_group_gates,
    ).to(device)


def _known_stock_pool(config) -> list[str]:
    universe_path = config.results_dir / "selected_stock_universe.csv"
    if universe_path.exists():
        universe = pd.read_csv(universe_path, dtype={"stock_code": str})
        return sorted(universe["stock_code"].astype(str).str.zfill(6).dropna().unique().tolist())

    predictions_path = config.results_dir / "test_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path, dtype={"stock_code": str}, usecols=["stock_code"])
        return sorted(predictions["stock_code"].astype(str).str.zfill(6).dropna().unique().tolist())
    return []


def _freeze_to_known_universe(config):
    stock_pool = _known_stock_pool(config)
    if not stock_pool:
        return config
    return replace(
        config,
        experiment=replace(
            config.experiment,
            stock_pool=stock_pool,
            layer_filter=None,
            max_stocks=None,
        ),
    )


def _load_model_split_predictions(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _freeze_to_known_universe(config)
    datasets = build_datasets(config)
    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    model = _build_model(config, input_dim=datasets.input_dim, device=device)
    checkpoint = torch.load(config.artifacts_dir / "financial_transformer.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loader = _build_loader(datasets.val, config.training.batch_size, False)
    test_loader = _build_loader(datasets.test, config.training.batch_size, False)
    _, val_predictions = _evaluate(model, val_loader, device, exit_label_mode=config.multitask.exit_label_mode)
    _, test_predictions = _evaluate(model, test_loader, device, exit_label_mode=config.multitask.exit_label_mode)
    calibration = load_probability_calibration(config.results_dir)
    val_predictions = apply_probability_calibration(val_predictions, calibration)
    test_predictions = apply_probability_calibration(test_predictions, calibration)
    for frame in [val_predictions, test_predictions]:
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["signal_score"] = _strategy_score_frame(frame)
        frame["static_good_label"] = _static_trade_quality_label(frame)
    return val_predictions, test_predictions


def _prefixed_strategy_score_frame(df: pd.DataFrame, prefix: str) -> pd.Series:
    gain_term = df[f"{prefix}pred_max_gain"].clip(lower=0.0, upper=0.5)
    drawdown_penalty = df[f"{prefix}pred_max_drawdown"].abs().clip(lower=0.01, upper=0.3)
    return (
        0.45 * df[f"{prefix}pred_short_up"]
        + 0.35 * df[f"{prefix}pred_take_profit"]
        + 0.20 * (1.0 - df[f"{prefix}pred_stop_loss"])
        + 0.10 * gain_term
        - 0.05 * drawdown_penalty
    )


def _merge_two_stage_predictions(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    candidate_pool: int,
    primary_weight: float,
    secondary_weight: float,
) -> pd.DataFrame:
    merge_cols = [
        "stock_code",
        "date",
        "layer",
        "future_return",
        "label_short_up",
        "label_take_profit",
        "label_stop_loss",
        "label_max_gain",
        "label_max_drawdown",
        "label",
    ]
    pred_cols = [
        "pred_short_up",
        "pred_take_profit",
        "pred_stop_loss",
        "pred_max_gain",
        "pred_max_drawdown",
        "prediction_prob",
        "signal_score",
        "static_good_label",
    ]

    primary_frame = primary[merge_cols + pred_cols].rename(columns={col: f"primary_{col}" for col in pred_cols if col != "static_good_label"})
    primary_frame["static_good_label"] = primary["static_good_label"].astype(int)
    secondary_frame = secondary[["stock_code", "date", *[col for col in pred_cols if col != "static_good_label"]]].rename(
        columns={col: f"secondary_{col}" for col in pred_cols if col != "static_good_label"}
    )
    merged = primary_frame.merge(secondary_frame, on=["stock_code", "date"], how="inner")
    merged["primary_signal_score"] = _prefixed_strategy_score_frame(merged, "primary_")
    merged["secondary_signal_score"] = _prefixed_strategy_score_frame(merged, "secondary_")
    merged = merged.sort_values(["date", "primary_signal_score", "primary_pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)
    merged["primary_rank"] = merged.groupby("date").cumcount() + 1
    merged = merged[merged["primary_rank"] <= int(candidate_pool)].copy()

    total_weight = float(primary_weight) + float(secondary_weight)
    normalized_primary = float(primary_weight) / total_weight
    normalized_secondary = float(secondary_weight) / total_weight

    output = merged[
        [
            "stock_code",
            "date",
            "layer",
            "future_return",
            "label_short_up",
            "label_take_profit",
            "label_stop_loss",
            "label_max_gain",
            "label_max_drawdown",
            "label",
            "static_good_label",
        ]
    ].copy()
    for col in ["pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown", "prediction_prob"]:
        output[col] = normalized_primary * merged[f"primary_{col}"] + normalized_secondary * merged[f"secondary_{col}"]
    output["signal_score"] = _strategy_score_frame(output)
    output["rerank_primary_rank"] = merged["primary_rank"]
    output["primary_signal_score"] = merged["primary_signal_score"]
    output["secondary_signal_score"] = merged["secondary_signal_score"]
    output["output_mode"] = "blend"
    output["effective_primary_weight"] = normalized_primary
    output["effective_secondary_weight"] = normalized_secondary
    return output.reset_index(drop=True)


def _make_execution_kwargs(args) -> dict:
    return {
        "short_up_threshold": args.short_up_threshold,
        "take_profit_threshold": args.take_profit_threshold,
        "stop_loss_threshold": args.stop_loss_threshold,
        "top_k_per_day": args.top_k,
        "use_calibration": False,
        "hold_days": args.hold_days,
        "take_profit": args.take_profit_exit,
        "stop_loss": args.stop_loss_exit,
        "initial_capital": 1_000_000.0,
        "commission": 0.001,
        "slippage": 0.001,
        "min_lot": 100,
        "min_position_weight": args.min_position_weight,
        "max_position_weight": args.max_position_weight,
        "max_total_position_fraction": args.max_total_position_fraction,
        "cooldown_trigger_losses": args.cooldown_trigger_losses,
        "cooldown_days": args.cooldown_days,
        "min_score_ratio_to_leader": args.min_score_ratio_to_leader,
        "leader_max_day_rank": args.leader_max_day_rank,
        "leader_take_profit": args.leader_take_profit,
        "leader_stop_loss": args.leader_stop_loss,
        "leader_hold_days": args.leader_hold_days,
    }


def _build_execution_aware_training_frame(
    predictions: pd.DataFrame,
    output_config,
    output_dir: Path,
    args,
) -> tuple[pd.DataFrame, dict]:
    label_dir = output_dir / "_validation_execution_labeling"
    label_dir.mkdir(parents=True, exist_ok=True)
    labeling_predictions = predictions.copy()
    labeling_predictions.to_csv(label_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    labeling_config = replace(output_config, results_dir=label_dir)
    backtester = TransformerStrategyBacktester(labeling_config)
    result = backtester.run(
        start_date=str(labeling_predictions["date"].min().date()),
        end_date=str(labeling_predictions["date"].max().date()),
        **_make_execution_kwargs(args),
    )
    write_backtest_outputs(result, label_dir, "validation_execution_labeling")

    signals = result.signals.copy()
    trades = result.trades.copy()
    if signals.empty or trades.empty:
        raise RuntimeError("验证期默认执行线没有形成足够的信号或成交，无法生成 execution-aware 标签。")

    signals["stock_code"] = signals["stock_code"].astype(str).str.zfill(6)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    trades["stock_code"] = trades["stock_code"].astype(str).str.zfill(6)
    trades["signal_date"] = pd.to_datetime(trades["signal_date"])

    join_cols = ["stock_code", "signal_date"]
    labeled = signals.merge(
        trades[join_cols + ["buy_date", "sell_date", "exit_reason", "profit", "profit_pct", "hold_days"]],
        on=join_cols,
        how="left",
    )
    labeled = labeled[labeled["profit"].notna()].copy()
    if labeled.empty:
        raise RuntimeError("验证期选中的信号没有可回填的真实成交，无法训练 execution-aware 过滤器。")

    labeled["trade_quality_label"] = (pd.to_numeric(labeled["profit"], errors="coerce") > 0.0).astype(int)
    labeled["static_good_label"] = pd.to_numeric(labeled["static_good_label"], errors="coerce").fillna(0).astype(int)
    labeled["static_vs_realized_match"] = (labeled["static_good_label"] == labeled["trade_quality_label"]).astype(int)
    labeled["date"] = pd.to_datetime(labeled["date"])
    labeled.to_csv(output_dir / "execution_aware_validation_trades.csv", index=False, encoding="utf-8-sig")

    summary = {
        "validation_signal_count": int(len(signals)),
        "validation_trade_count": int(len(trades)),
        "training_row_count": int(len(labeled)),
        "training_positive_rate": float(labeled["trade_quality_label"].mean()),
        "static_label_match_rate": float(labeled["static_vs_realized_match"].mean()),
        "static_false_bad_realized_good": int(((labeled["static_good_label"] == 0) & (labeled["trade_quality_label"] == 1)).sum()),
        "static_false_good_realized_bad": int(((labeled["static_good_label"] == 1) & (labeled["trade_quality_label"] == 0)).sum()),
        "label_dir": str(label_dir),
    }
    return labeled, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate an execution-aware trade-quality filter on top of the current two-stage leader-tiered default")
    parser.add_argument("--primary-config", default=str(ROOT / "config_moneyflow_v1.yaml"))
    parser.add_argument("--secondary-config", default=str(ROOT / "config_moneyflow_selective_v1.yaml"))
    parser.add_argument("--output-config", default=str(ROOT / "config_two_stage_rerank_leader_tiered_v1.yaml"))
    parser.add_argument("--output-tag", default="trade_filter_v2_execution_aware")
    parser.add_argument("--candidate-pool", type=int, default=8)
    parser.add_argument("--primary-weight", type=float, default=0.55)
    parser.add_argument("--secondary-weight", type=float, default=0.45)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--short-up-threshold", type=float, default=0.254)
    parser.add_argument("--take-profit-threshold", type=float, default=0.572)
    parser.add_argument("--stop-loss-threshold", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--quality-threshold-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-signals", type=int, default=10)
    parser.add_argument("--hold-days", type=int, default=35)
    parser.add_argument("--take-profit-exit", type=float, default=0.14)
    parser.add_argument("--stop-loss-exit", type=float, default=-0.06)
    parser.add_argument("--min-score-ratio-to-leader", type=float, default=0.95)
    parser.add_argument("--leader-max-day-rank", type=int, default=1)
    parser.add_argument("--leader-take-profit", type=float, default=0.165)
    parser.add_argument("--leader-stop-loss", type=float, default=-0.035)
    parser.add_argument("--leader-hold-days", type=int, default=20)
    parser.add_argument("--min-position-weight", type=float, default=0.08)
    parser.add_argument("--max-position-weight", type=float, default=0.20)
    parser.add_argument("--max-total-position-fraction", type=float, default=1.0)
    parser.add_argument("--cooldown-trigger-losses", type=int, default=0)
    parser.add_argument("--cooldown-days", type=int, default=0)
    args = parser.parse_args()

    primary_config = load_config(args.primary_config)
    secondary_config = load_config(args.secondary_config)
    output_config = load_config(args.output_config)
    output_name = f"{Path(args.output_config).stem.replace('config_', '')}_{args.output_tag}"
    output_dir = output_config.root_dir / "results" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_val, primary_test = _load_model_split_predictions(primary_config)
    secondary_val, secondary_test = _load_model_split_predictions(secondary_config)
    val_predictions = _merge_two_stage_predictions(primary_val, secondary_val, args.candidate_pool, args.primary_weight, args.secondary_weight)
    test_predictions = _merge_two_stage_predictions(primary_test, secondary_test, args.candidate_pool, args.primary_weight, args.secondary_weight)

    train_df, label_summary = _build_execution_aware_training_frame(val_predictions, output_config, output_dir, args)
    if train_df["trade_quality_label"].nunique() < 2:
        raise RuntimeError("execution-aware 验证成交标签单一，无法训练过滤器。")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(train_df[FILTER_FEATURES], train_df["trade_quality_label"].astype(int))

    for frame in [val_predictions, test_predictions]:
        frame["pred_trade_quality"] = model.predict_proba(frame[FILTER_FEATURES])[:, 1]

    val_predictions.to_csv(output_dir / "val_predictions_calibrated.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(output_dir / "test_predictions_calibrated.csv", index=False, encoding="utf-8-sig")

    model_payload = {
        "label_source": "execution_aware_realized_trade_profit",
        "primary_config": str(Path(args.primary_config).resolve()),
        "secondary_config": str(Path(args.secondary_config).resolve()),
        "output_config": str(Path(args.output_config).resolve()),
        "candidate_pool": int(args.candidate_pool),
        "primary_weight": float(args.primary_weight),
        "secondary_weight": float(args.secondary_weight),
        "features": FILTER_FEATURES,
        "intercept": float(model.intercept_[0]),
        "coef": {feature: float(coef) for feature, coef in zip(FILTER_FEATURES, model.coef_[0])},
        "label_summary": label_summary,
    }
    (output_dir / "trade_quality_model.json").write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    backtest_config = replace(output_config, results_dir=output_dir)
    backtester = TransformerStrategyBacktester(backtest_config)
    execution_kwargs = _make_execution_kwargs(args)
    baseline_result = backtester.run(start_date=args.start_date, end_date=args.end_date, **execution_kwargs)
    baseline_paths = write_backtest_outputs(baseline_result, output_dir, "baseline_no_filter")

    quality_grid = [float(item.strip()) for item in args.quality_threshold_grid.split(",") if item.strip()]
    rows = []
    best_summary = None
    best_result = None
    for quality_threshold in quality_grid:
        try:
            result = backtester.run(
                start_date=args.start_date,
                end_date=args.end_date,
                min_filter_column="pred_trade_quality",
                min_filter_value=quality_threshold,
                **execution_kwargs,
            )
        except RuntimeError:
            continue
        row = dict(result.summary)
        row["quality_threshold"] = quality_threshold
        row["ranking_score"] = _ranking_score(row, min_trades=args.min_trades, min_signals=args.min_signals)
        rows.append(row)
        if best_summary is None or row["ranking_score"] > best_summary["ranking_score"]:
            best_summary = row
            best_result = result

    if best_summary is None or best_result is None:
        raise RuntimeError("execution-aware 过滤器搜索没有得到有效回测结果。")

    search_df = pd.DataFrame(rows).sort_values(
        ["ranking_score", "total_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    search_df.to_csv(output_dir / "trade_quality_filter_search.csv", index=False, encoding="utf-8-sig")
    detail_paths = write_backtest_outputs(best_result, output_dir, "trade_quality_filter_best")

    payload = {
        "best": best_summary,
        "baseline": baseline_result.summary,
        "baseline_outputs": baseline_paths,
        "detail_outputs": detail_paths,
        "search_results": str(output_dir / "trade_quality_filter_search.csv"),
        "model": model_payload,
    }
    (output_dir / "trade_quality_filter_best.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()