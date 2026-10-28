from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
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


def _candidate_gate(
    df: pd.DataFrame,
    short_up_threshold: float,
    take_profit_threshold: float,
    stop_loss_threshold: float,
) -> pd.DataFrame:
    return df[
        (df["pred_short_up"] > short_up_threshold)
        & (df["pred_take_profit"] > take_profit_threshold)
        & (df["pred_stop_loss"] < stop_loss_threshold)
    ].copy()


def _training_candidates(
    df: pd.DataFrame,
    short_up_threshold: float,
    take_profit_threshold: float,
    stop_loss_threshold: float,
) -> tuple[pd.DataFrame, str]:
    strict = _candidate_gate(df, short_up_threshold, take_profit_threshold, stop_loss_threshold)
    if not strict.empty:
        return strict, "strict"

    relaxed = _candidate_gate(
        df,
        short_up_threshold=max(short_up_threshold - 0.05, 0.0),
        take_profit_threshold=max(take_profit_threshold - 0.05, 0.0),
        stop_loss_threshold=min(stop_loss_threshold + 0.05, 1.0),
    )
    if not relaxed.empty:
        return relaxed, "relaxed"

    return df.copy(), "all_validation"


def _trade_quality_label(df: pd.DataFrame) -> pd.Series:
    good_trade = (df["label_take_profit"] >= 0.5) & (df["label_stop_loss"] < 0.5)
    return good_trade.astype(int)


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


def _freeze_to_saved_universe(config):
    universe_path = config.results_dir / "selected_stock_universe.csv"
    if not universe_path.exists():
        return config
    universe = pd.read_csv(universe_path, dtype={"stock_code": str})
    stock_pool = sorted(universe["stock_code"].astype(str).str.zfill(6).dropna().unique().tolist())
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


def _load_split_predictions(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    val_calibrated = apply_probability_calibration(val_predictions, calibration)
    test_calibrated = apply_probability_calibration(test_predictions, calibration)
    for frame in [val_predictions, test_predictions, val_calibrated, test_calibrated]:
        frame["signal_score"] = _strategy_score_frame(frame)
    return val_predictions, test_predictions, val_calibrated, test_calibrated


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate a trade-quality filter on top of an existing transformer run")
    parser.add_argument("--config", default=str(ROOT / "config_moneyflow_v1.yaml"))
    parser.add_argument("--output-tag", default="trade_filter_v1")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--short-up-threshold", type=float, default=0.50)
    parser.add_argument("--take-profit-threshold", type=float, default=0.46)
    parser.add_argument("--stop-loss-threshold", type=float, default=0.58)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--quality-threshold-grid", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-signals", type=int, default=10)
    args = parser.parse_args()

    config = _freeze_to_saved_universe(load_config(args.config))
    output_dir = config.root_dir / "results" / f"{config.results_dir.name}_{args.output_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    val_predictions, test_predictions, val_calibrated, test_calibrated = _load_split_predictions(config)
    val_candidates, candidate_mode = _training_candidates(
        val_predictions,
        short_up_threshold=args.short_up_threshold,
        take_profit_threshold=args.take_profit_threshold,
        stop_loss_threshold=args.stop_loss_threshold,
    )

    val_candidates["trade_quality_label"] = _trade_quality_label(val_candidates)
    if val_candidates["trade_quality_label"].nunique() < 2:
        raise RuntimeError("验证集候选标签单一，无法训练交易过滤器。")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(val_candidates[FILTER_FEATURES], val_candidates["trade_quality_label"].astype(int))

    for frame in [val_predictions, test_predictions, val_calibrated, test_calibrated]:
        frame["pred_trade_quality"] = model.predict_proba(frame[FILTER_FEATURES])[:, 1]
        frame["trade_quality_label"] = _trade_quality_label(frame)

    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    val_calibrated.to_csv(output_dir / "val_predictions_calibrated.csv", index=False, encoding="utf-8-sig")
    test_calibrated.to_csv(output_dir / "test_predictions_calibrated.csv", index=False, encoding="utf-8-sig")

    model_payload = {
        "features": FILTER_FEATURES,
        "candidate_mode": candidate_mode,
        "intercept": float(model.intercept_[0]),
        "coef": {feature: float(coef) for feature, coef in zip(FILTER_FEATURES, model.coef_[0])},
        "validation_candidate_count": int(len(val_candidates)),
        "validation_positive_rate": float(val_candidates["trade_quality_label"].mean()),
    }
    (output_dir / "trade_quality_model.json").write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    backtest_config = replace(config, results_dir=output_dir)
    backtester = TransformerStrategyBacktester(backtest_config)
    quality_grid = [float(item.strip()) for item in args.quality_threshold_grid.split(",") if item.strip()]

    rows = []
    best_summary = None
    best_result = None
    for quality_threshold in quality_grid:
        try:
            result = backtester.run(
                start_date=args.start_date,
                end_date=args.end_date,
                short_up_threshold=args.short_up_threshold,
                take_profit_threshold=args.take_profit_threshold,
                stop_loss_threshold=args.stop_loss_threshold,
                top_k_per_day=args.top_k,
                use_calibration=False,
                hold_days=45,
                take_profit=0.12,
                stop_loss=-0.06,
                initial_capital=1_000_000.0,
                commission=0.001,
                slippage=0.001,
                min_lot=100,
                min_filter_column="pred_trade_quality",
                min_filter_value=quality_threshold,
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
        raise RuntimeError("过滤器搜索没有得到有效回测结果。")

    search_df = pd.DataFrame(rows).sort_values(
        ["ranking_score", "total_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    search_df.to_csv(output_dir / "trade_quality_filter_search.csv", index=False, encoding="utf-8-sig")
    detail_paths = write_backtest_outputs(best_result, output_dir, "trade_quality_filter_best")
    payload = {
        "config": str(Path(args.config).resolve()),
        "base_thresholds": {
            "short_up_threshold": args.short_up_threshold,
            "take_profit_threshold": args.take_profit_threshold,
            "stop_loss_threshold": args.stop_loss_threshold,
            "top_k": args.top_k,
        },
        "best": best_summary,
        "detail_outputs": detail_paths,
        "search_results": str(output_dir / "trade_quality_filter_search.csv"),
        "model": model_payload,
    }
    (output_dir / "trade_quality_filter_best.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()