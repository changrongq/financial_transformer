from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"
PROJECT_SRC = EXPERIMENT_ROOT.parents[1] / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config


def _strategy_score_frame(df: pd.DataFrame, prefix: str) -> pd.Series:
    gain_term = df[f"{prefix}pred_max_gain"].clip(lower=0.0, upper=0.5)
    drawdown_penalty = df[f"{prefix}pred_max_drawdown"].abs().clip(lower=0.01, upper=0.3)
    return (
        0.45 * df[f"{prefix}pred_short_up"]
        + 0.35 * df[f"{prefix}pred_take_profit"]
        + 0.20 * (1.0 - df[f"{prefix}pred_stop_loss"])
        + 0.10 * gain_term
        - 0.05 * drawdown_penalty
    )


def _load_prediction_frame(results_dir: Path, use_calibrated: bool) -> pd.DataFrame:
    file_name = "test_predictions_calibrated.csv" if use_calibrated else "test_predictions.csv"
    path = results_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"prediction file not found: {path}")
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _resolve_output_predictions(merged: pd.DataFrame, output_mode: str, primary_weight: float, secondary_weight: float) -> pd.DataFrame:
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
        ]
    ].copy()

    prediction_cols = ["pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown", "prediction_prob"]
    if output_mode == "secondary_only":
        for col in prediction_cols:
            output[col] = merged[f"secondary_{col}"]
    elif output_mode == "blend":
        total_weight = primary_weight + secondary_weight
        normalized_primary = primary_weight / total_weight
        normalized_secondary = secondary_weight / total_weight
        for col in prediction_cols:
            output[col] = normalized_primary * merged[f"primary_{col}"] + normalized_secondary * merged[f"secondary_{col}"]
        primary_weight = normalized_primary
        secondary_weight = normalized_secondary
    else:
        raise ValueError(f"unsupported output mode: {output_mode}")

    output["signal_score"] = _strategy_score_frame(output, "")
    output["rerank_primary_rank"] = merged["primary_rank"]
    output["primary_signal_score"] = merged["primary_signal_score"]
    output["secondary_signal_score"] = merged["secondary_signal_score"]
    output["output_mode"] = output_mode
    output["effective_primary_weight"] = primary_weight
    output["effective_secondary_weight"] = secondary_weight
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build two-stage rerank predictions from two transformer runs")
    parser.add_argument("--primary-config", default=str(EXPERIMENT_ROOT / "config_moneyflow_v1.yaml"))
    parser.add_argument("--secondary-config", default=str(EXPERIMENT_ROOT / "config_moneyflow_selective_v1.yaml"))
    parser.add_argument("--output-config", default=str(EXPERIMENT_ROOT / "config_two_stage_rerank_v1.yaml"))
    parser.add_argument("--candidate-pool", type=int, default=10)
    parser.add_argument("--primary-weight", type=float, default=0.65)
    parser.add_argument("--secondary-weight", type=float, default=0.35)
    parser.add_argument("--output-mode", choices=["blend", "secondary_only"], default="blend")
    parser.add_argument("--use-calibrated", action="store_true")
    args = parser.parse_args()

    primary_config = load_config(args.primary_config)
    secondary_config = load_config(args.secondary_config)
    output_config = load_config(args.output_config)

    primary = _load_prediction_frame(primary_config.results_dir, use_calibrated=args.use_calibrated)
    secondary = _load_prediction_frame(secondary_config.results_dir, use_calibrated=args.use_calibrated)

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
    ]

    primary = primary[merge_cols + pred_cols].rename(columns={col: f"primary_{col}" for col in pred_cols})
    secondary = secondary[["stock_code", "date", *pred_cols]].rename(columns={col: f"secondary_{col}" for col in pred_cols})
    merged = primary.merge(secondary, on=["stock_code", "date"], how="inner")
    merged["primary_signal_score"] = _strategy_score_frame(merged, "primary_")
    merged["secondary_signal_score"] = _strategy_score_frame(merged, "secondary_")
    merged = merged.sort_values(["date", "primary_signal_score", "primary_pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)
    merged["primary_rank"] = merged.groupby("date").cumcount() + 1
    merged = merged[merged["primary_rank"] <= args.candidate_pool].copy()

    primary_weight = float(args.primary_weight)
    secondary_weight = float(args.secondary_weight)
    output = _resolve_output_predictions(
        merged,
        output_mode=args.output_mode,
        primary_weight=primary_weight,
        secondary_weight=secondary_weight,
    )

    output_config.results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_config.results_dir / "test_predictions.csv"
    summary_path = output_config.results_dir / "two_stage_rerank_summary.json"
    output.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    payload = {
        "primary_results_dir": str(primary_config.results_dir),
        "secondary_results_dir": str(secondary_config.results_dir),
        "output_results_dir": str(output_config.results_dir),
        "candidate_pool": int(args.candidate_pool),
        "output_mode": args.output_mode,
        "primary_weight": primary_weight,
        "secondary_weight": secondary_weight,
        "effective_primary_weight": float(output["effective_primary_weight"].iloc[0]) if not output.empty else primary_weight,
        "effective_secondary_weight": float(output["effective_secondary_weight"].iloc[0]) if not output.empty else secondary_weight,
        "use_calibrated": bool(args.use_calibrated),
        "rows": int(len(output)),
        "dates": int(output["date"].nunique()) if not output.empty else 0,
        "stocks": int(output["stock_code"].nunique()) if not output.empty else 0,
        "predictions_file": str(predictions_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))