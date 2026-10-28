from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, load_config
from .predict import predict_latest
from .scoring import enrich_prediction_frame, strategy_score_frame


@dataclass(frozen=True)
class TwoStageExecutionSpec:
    primary_config: Path
    secondary_config: Path
    output_config: Path
    candidate_pool: int
    primary_weight: float
    secondary_weight: float
    output_mode: str
    default_short_up_threshold: float
    default_take_profit_threshold: float
    default_stop_loss_threshold: float
    default_top_k_per_day: int
    default_min_score_ratio_to_leader: float = 0.0
    default_execution_hold_days: int = 45
    default_execution_take_profit: float = 0.12
    default_execution_stop_loss: float = -0.06
    default_leader_max_day_rank: int | None = None
    default_leader_take_profit: float | None = None
    default_leader_stop_loss: float | None = None
    default_leader_hold_days: int | None = None
    default_output_tag: str | None = None

@dataclass(frozen=True)
class SingleStageExecutionDefaults:
    default_short_up_threshold: float
    default_take_profit_threshold: float
    default_stop_loss_threshold: float
    default_top_k_per_day: int
    default_calibrated_short_up_threshold: float | None = None
    default_min_score_ratio_to_leader: float = 0.0
    default_execution_hold_days: int = 45
    default_execution_take_profit: float = 0.14
    default_execution_stop_loss: float = -0.06
    default_leader_max_day_rank: int | None = None
    default_leader_take_profit: float | None = None
    default_leader_stop_loss: float | None = None
    default_leader_hold_days: int | None = None


def resolve_single_stage_execution_defaults(config_path: str | Path) -> SingleStageExecutionDefaults | None:
    config_path = Path(config_path).resolve()
    stem = config_path.stem
    if stem == "config_moneyflow_v1_concept_sentiment_manual_micro_rank_mild_smoke_v1":
        return SingleStageExecutionDefaults(
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=30,
            default_execution_take_profit=0.16,
            default_execution_stop_loss=-0.06,
        )
    if stem == "config_moneyflow_v1_concept_sentiment_manual_micro_smoke_v1":
        return SingleStageExecutionDefaults(
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=30,
            default_execution_take_profit=0.16,
            default_execution_stop_loss=-0.06,
        )
    if stem == "config_moneyflow_v1_concept_sentiment_manual_micro_exposure_smoke_recent_oos_v1":
        return SingleStageExecutionDefaults(
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=45,
            default_execution_take_profit=0.16,
            default_execution_stop_loss=-0.06,
        )
    if stem == "config_moneyflow_v1_concept_sentiment_manual_micro_recent_oos_v1":
        return SingleStageExecutionDefaults(
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=30,
            default_execution_take_profit=0.16,
            default_execution_stop_loss=-0.06,
        )
    if stem == "config_moneyflow_v1_concept_sentiment_manual_micro_exposure_smoke_v1":
        return SingleStageExecutionDefaults(
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=45,
            default_execution_take_profit=0.16,
            default_execution_stop_loss=-0.06,
        )
    if stem == "config_moneyflow_v1_concept_sentiment_manual_micro_v1":
        return SingleStageExecutionDefaults(
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_calibrated_short_up_threshold=0.32,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=30,
            default_execution_take_profit=0.16,
            default_execution_stop_loss=-0.06,
        )
    if stem == "config_moneyflow_v1_large_v1":
        return SingleStageExecutionDefaults(
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=45,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
        )
    return None


def resolve_two_stage_execution_spec(config_path: str | Path) -> TwoStageExecutionSpec | None:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    stem = config_path.stem
    if stem == "config_two_stage_rerank_large_recent_oos_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1_large_recent_oos_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_large_recent_oos_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=45,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_output_tag="two_stage_rerank_large_recent_oos_v1",
        )
    if stem == "config_two_stage_rerank_large_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1_large_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_large_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.50,
            default_take_profit_threshold=0.46,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=45,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_output_tag="two_stage_rerank_large_v1",
        )
    if stem == "config_two_stage_rerank_conceptsentv2_selective_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1_concept_sentiment_v2.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_v1_concept_sentiment_selective_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_output_tag="two_stage_rerank_conceptsentv2_selective_v1",
        )
    if stem == "config_two_stage_rerank_conceptsentv2_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1_concept_sentiment_v2.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_output_tag="two_stage_rerank_conceptsentv2_v1",
        )
    if stem == "config_two_stage_rerank_best_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_output_tag="two_stage_rerank_best",
        )
    if stem == "config_two_stage_rerank_leader_tiered_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.95,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_leader_max_day_rank=1,
            default_leader_take_profit=0.165,
            default_leader_stop_loss=-0.035,
            default_leader_hold_days=20,
            default_output_tag="two_stage_rerank_leader_tiered_v1",
        )
    if stem == "config_two_stage_rerank_leader_tiered_latest_incremental_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1_latest_incremental_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_latest_incremental_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.95,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_leader_max_day_rank=1,
            default_leader_take_profit=0.165,
            default_leader_stop_loss=-0.035,
            default_leader_hold_days=20,
            default_output_tag="two_stage_rerank_leader_tiered_latest_incremental_v1",
        )
    if stem == "config_two_stage_rerank_leader_tiered_latest_incremental_v2":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1_latest_incremental_v2.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_latest_incremental_v2.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.95,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_leader_max_day_rank=1,
            default_leader_take_profit=0.165,
            default_leader_stop_loss=-0.035,
            default_leader_hold_days=20,
            default_output_tag="two_stage_rerank_leader_tiered_latest_incremental_v2",
        )
    if stem == "config_two_stage_rerank_leader_tiered_latest_incremental_v3":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1_latest_incremental_v3.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_latest_incremental_v3.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.95,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_leader_max_day_rank=1,
            default_leader_take_profit=0.165,
            default_leader_stop_loss=-0.035,
            default_leader_hold_days=20,
            default_output_tag="two_stage_rerank_leader_tiered_latest_incremental_v3",
        )
    if stem == "config_two_stage_rerank_conservative_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=8,
            primary_weight=0.55,
            secondary_weight=0.45,
            output_mode="blend",
            default_short_up_threshold=0.254,
            default_take_profit_threshold=0.572,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=3,
            default_min_score_ratio_to_leader=0.97,
            default_execution_hold_days=35,
            default_execution_take_profit=0.14,
            default_execution_stop_loss=-0.06,
            default_output_tag="leader_ratio97_topk3_v1",
        )
    if stem == "config_two_stage_rerank_v1":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=10,
            primary_weight=0.65,
            secondary_weight=0.35,
            output_mode="blend",
            default_short_up_threshold=0.262,
            default_take_profit_threshold=0.574,
            default_stop_loss_threshold=0.50,
            default_top_k_per_day=2,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=45,
            default_execution_take_profit=0.12,
            default_execution_stop_loss=-0.06,
            default_output_tag="two_stage_rerank_v1",
        )
    if stem == "config_two_stage_rerank_v2":
        return TwoStageExecutionSpec(
            primary_config=(root / "config_moneyflow_v1.yaml").resolve(),
            secondary_config=(root / "config_moneyflow_selective_v1.yaml").resolve(),
            output_config=config_path,
            candidate_pool=10,
            primary_weight=0.65,
            secondary_weight=0.35,
            output_mode="secondary_only",
            default_short_up_threshold=0.256,
            default_take_profit_threshold=0.492,
            default_stop_loss_threshold=0.58,
            default_top_k_per_day=1,
            default_min_score_ratio_to_leader=0.0,
            default_execution_hold_days=45,
            default_execution_take_profit=0.12,
            default_execution_stop_loss=-0.06,
            default_output_tag="two_stage_rerank_v2",
        )
    return None


def _strategy_score_frame(df: pd.DataFrame, prefix: str) -> pd.Series:
    return strategy_score_frame(df, prefix=prefix)


def _load_latest_frame(results_dir: Path, use_calibration: bool) -> pd.DataFrame:
    file_name = "latest_predictions_calibrated.csv" if use_calibration else "latest_predictions.csv"
    path = results_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"latest prediction file not found: {path}")
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    if "prediction_prob" not in df.columns:
        df["prediction_prob"] = df["pred_short_up"]
    return df


def _load_test_frame(results_dir: Path, use_calibration: bool) -> pd.DataFrame:
    file_name = "test_predictions_calibrated.csv" if use_calibration else "test_predictions.csv"
    path = results_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"test prediction file not found: {path}")
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    if "prediction_prob" not in df.columns:
        df["prediction_prob"] = df["pred_short_up"]
    return df


def _resolve_output_predictions(merged: pd.DataFrame, output_mode: str, primary_weight: float, secondary_weight: float) -> pd.DataFrame:
    output = merged[["stock_code", "date", "layer", "close", "market_cap_yi", "future_return_reference"]].copy()
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

    output = enrich_prediction_frame(output, prefix="")
    output["signal_score"] = _strategy_score_frame(output, "")
    output["rerank_primary_rank"] = merged["primary_rank"]
    output["primary_signal_score"] = merged["primary_signal_score"]
    output["secondary_signal_score"] = merged["secondary_signal_score"]
    output["output_mode"] = output_mode
    output["effective_primary_weight"] = primary_weight
    output["effective_secondary_weight"] = secondary_weight
    return output.sort_values(["signal_score", "pred_take_profit", "pred_short_up"], ascending=[False, False, False]).reset_index(drop=True)


def build_latest_two_stage_predictions(config_path: str | Path, use_calibration: bool = False, top_n: int | None = None) -> dict:
    spec = resolve_two_stage_execution_spec(config_path)
    if spec is None:
        raise RuntimeError(f"config is not registered as a two-stage execution spec: {config_path}")

    primary_config = load_config(spec.primary_config)
    secondary_config = load_config(spec.secondary_config)
    output_config = load_config(spec.output_config)

    predict_latest(primary_config, use_calibration=use_calibration)
    predict_latest(secondary_config, use_calibration=use_calibration)

    primary = _load_latest_frame(primary_config.results_dir, use_calibration=use_calibration)
    secondary = _load_latest_frame(secondary_config.results_dir, use_calibration=use_calibration)

    base_cols = ["stock_code", "date", "layer", "close", "market_cap_yi", "future_return_reference"]
    pred_cols = [
        "pred_short_up",
        "pred_take_profit",
        "pred_stop_loss",
        "pred_max_gain",
        "pred_max_drawdown",
        "prediction_prob",
        "signal_score",
    ]

    primary = primary[base_cols + pred_cols].rename(columns={col: f"primary_{col}" for col in pred_cols})
    secondary = secondary[["stock_code", "date", *pred_cols]].rename(columns={col: f"secondary_{col}" for col in pred_cols})
    primary = enrich_prediction_frame(primary, prefix="primary_")
    secondary = enrich_prediction_frame(secondary, prefix="secondary_")
    merged = primary.merge(secondary, on=["stock_code", "date"], how="inner")
    merged["primary_signal_score"] = _strategy_score_frame(merged, "primary_")
    merged["secondary_signal_score"] = _strategy_score_frame(merged, "secondary_")
    merged = merged.sort_values(["date", "primary_signal_score", "primary_pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)
    merged["primary_rank"] = merged.groupby("date").cumcount() + 1
    merged = merged[merged["primary_rank"] <= spec.candidate_pool].copy()

    output = _resolve_output_predictions(
        merged,
        output_mode=spec.output_mode,
        primary_weight=spec.primary_weight,
        secondary_weight=spec.secondary_weight,
    )
    if top_n:
        output = output.head(top_n).reset_index(drop=True)

    output_config.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_calibrated" if use_calibration else ""
    output_path = output_config.results_dir / f"latest_predictions{suffix}.csv"
    summary_path = output_config.results_dir / f"latest_predictions{suffix}_summary.json"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    payload = {
        "primary_results_dir": str(primary_config.results_dir),
        "secondary_results_dir": str(secondary_config.results_dir),
        "output_results_dir": str(output_config.results_dir),
        "candidate_pool": spec.candidate_pool,
        "output_mode": spec.output_mode,
        "primary_weight": spec.primary_weight,
        "secondary_weight": spec.secondary_weight,
        "effective_primary_weight": float(output["effective_primary_weight"].iloc[0]) if not output.empty else spec.primary_weight,
        "effective_secondary_weight": float(output["effective_secondary_weight"].iloc[0]) if not output.empty else spec.secondary_weight,
        "use_calibration": bool(use_calibration),
        "prediction_count": int(len(output)),
        "output_file": str(output_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return {
        "output_file": str(output_path),
        "top_stock": output.iloc[0]["stock_code"] if not output.empty else None,
        "top_signal_score": float(output.iloc[0]["signal_score"]) if not output.empty else None,
        "prediction_count": int(len(output)),
    }


def build_test_two_stage_predictions(config_path: str | Path, use_calibration: bool = False) -> dict:
    spec = resolve_two_stage_execution_spec(config_path)
    if spec is None:
        raise RuntimeError(f"config is not registered as a two-stage execution spec: {config_path}")

    primary_config = load_config(spec.primary_config)
    secondary_config = load_config(spec.secondary_config)
    output_config = load_config(spec.output_config)

    primary = _load_test_frame(primary_config.results_dir, use_calibration=use_calibration)
    secondary = _load_test_frame(secondary_config.results_dir, use_calibration=use_calibration)

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
    merged = merged[merged["primary_rank"] <= spec.candidate_pool].copy()

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
    prediction_cols = [
        "pred_short_up",
        "pred_take_profit",
        "pred_stop_loss",
        "pred_max_gain",
        "pred_max_drawdown",
        "prediction_prob",
    ]
    if spec.output_mode == "secondary_only":
        for col in prediction_cols:
            output[col] = merged[f"secondary_{col}"]
        effective_primary_weight = 0.0
        effective_secondary_weight = 1.0
    elif spec.output_mode == "blend":
        total_weight = spec.primary_weight + spec.secondary_weight
        effective_primary_weight = spec.primary_weight / total_weight
        effective_secondary_weight = spec.secondary_weight / total_weight
        for col in prediction_cols:
            output[col] = (
                effective_primary_weight * merged[f"primary_{col}"]
                + effective_secondary_weight * merged[f"secondary_{col}"]
            )
    else:
        raise ValueError(f"unsupported output mode: {spec.output_mode}")

    output = enrich_prediction_frame(output, prefix="")
    output["signal_score"] = _strategy_score_frame(output, "")
    output["rerank_primary_rank"] = merged["primary_rank"]
    output["primary_signal_score"] = merged["primary_signal_score"]
    output["secondary_signal_score"] = merged["secondary_signal_score"]
    output["output_mode"] = spec.output_mode
    output["effective_primary_weight"] = effective_primary_weight
    output["effective_secondary_weight"] = effective_secondary_weight

    output_config.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_calibrated" if use_calibration else ""
    predictions_path = output_config.results_dir / f"test_predictions{suffix}.csv"
    summary_path = output_config.results_dir / f"two_stage_rerank_summary{suffix}.json"
    output.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    payload = {
        "primary_results_dir": str(primary_config.results_dir),
        "secondary_results_dir": str(secondary_config.results_dir),
        "output_results_dir": str(output_config.results_dir),
        "candidate_pool": spec.candidate_pool,
        "output_mode": spec.output_mode,
        "primary_weight": spec.primary_weight,
        "secondary_weight": spec.secondary_weight,
        "effective_primary_weight": float(output["effective_primary_weight"].iloc[0]) if not output.empty else spec.primary_weight,
        "effective_secondary_weight": float(output["effective_secondary_weight"].iloc[0]) if not output.empty else spec.secondary_weight,
        "use_calibration": bool(use_calibration),
        "rows": int(len(output)),
        "dates": int(output["date"].nunique()) if not output.empty else 0,
        "stocks": int(output["stock_code"].nunique()) if not output.empty else 0,
        "predictions_file": str(predictions_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def predict_latest_for_execution(config_path: str | Path, top_n: int | None = None, use_calibration: bool = False) -> dict:
    spec = resolve_two_stage_execution_spec(config_path)
    if spec is None:
        config = load_config(config_path)
        return predict_latest(config, top_n=top_n, use_calibration=use_calibration)
    return build_latest_two_stage_predictions(config_path, use_calibration=use_calibration, top_n=top_n)