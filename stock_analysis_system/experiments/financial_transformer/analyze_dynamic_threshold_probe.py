from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_lightweight_rerank_probe import _run_backtest  # noqa: E402
from financial_transformer_experiment.backtest import write_backtest_outputs  # noqa: E402
from financial_transformer_experiment.config import load_config  # noqa: E402
from financial_transformer_experiment.two_stage_execution import (  # noqa: E402
    build_test_two_stage_predictions,
    resolve_single_stage_execution_defaults,
    resolve_two_stage_execution_spec,
)


def _load_predictions(config_path: Path, use_calibration: bool) -> tuple[object, pd.DataFrame, Path]:
    config = load_config(str(config_path))
    suffix = "_calibrated" if use_calibration else ""
    predictions_path = config.results_dir / f"test_predictions{suffix}.csv"
    if not predictions_path.exists():
        spec = resolve_two_stage_execution_spec(str(config_path))
        if spec is None:
            raise RuntimeError(f"missing predictions file: {predictions_path}")
        build_test_two_stage_predictions(str(config_path), use_calibration=use_calibration)
    if not predictions_path.exists():
        raise RuntimeError(f"failed to materialize predictions file: {predictions_path}")

    df = pd.read_csv(predictions_path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_columns = [
        "pred_short_up",
        "pred_take_profit",
        "pred_stop_loss",
        "pred_max_gain",
        "pred_max_drawdown",
        "signal_score",
        "primary_signal_score",
        "secondary_signal_score",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    required = ["date", "pred_short_up", "pred_take_profit", "pred_stop_loss", "signal_score"]
    df = df.dropna(subset=required).copy()
    return config, df, predictions_path


def _resolve_execution_defaults(config_path: Path, use_calibration: bool) -> dict:
    execution_defaults = resolve_two_stage_execution_spec(str(config_path)) or resolve_single_stage_execution_defaults(str(config_path))
    if execution_defaults is None:
        raise RuntimeError(f"no execution defaults for {config_path.name}")

    calibrated_short_up_threshold = getattr(execution_defaults, "default_calibrated_short_up_threshold", None)
    if use_calibration and calibrated_short_up_threshold is not None:
        short_up_threshold = float(calibrated_short_up_threshold)
    else:
        short_up_threshold = float(execution_defaults.default_short_up_threshold)

    return {
        "short_up_threshold": short_up_threshold,
        "take_profit_threshold": float(execution_defaults.default_take_profit_threshold),
        "stop_loss_threshold": float(execution_defaults.default_stop_loss_threshold),
        "top_k_per_day": int(execution_defaults.default_top_k_per_day),
        "hold_days": int(execution_defaults.default_execution_hold_days),
        "take_profit": float(execution_defaults.default_execution_take_profit),
        "stop_loss": float(execution_defaults.default_execution_stop_loss),
        "min_score_ratio_to_leader": float(getattr(execution_defaults, "default_min_score_ratio_to_leader", 0.0) or 0.0),
        "leader_max_day_rank": getattr(execution_defaults, "default_leader_max_day_rank", None),
        "leader_take_profit": getattr(execution_defaults, "default_leader_take_profit", None),
        "leader_stop_loss": getattr(execution_defaults, "default_leader_stop_loss", None),
        "leader_hold_days": getattr(execution_defaults, "default_leader_hold_days", None),
    }


def _anchor_quantile(series: pd.Series, static_threshold: float) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return 0.5
    quantile = float((valid <= float(static_threshold)).mean())
    return min(max(quantile, 0.05), 0.95)


def _clamp_quantile(value: float) -> float:
    return min(max(float(value), 0.05), 0.95)


def _parse_offset_grid(raw_offsets: str) -> list[float]:
    offsets = [float(item.strip()) for item in raw_offsets.split(",") if item.strip()]
    if not offsets:
        raise RuntimeError("quantile offset grid is empty")
    return sorted(set(offsets))


def _build_quantile_candidates(anchor_quantiles: dict, raw_offsets: str) -> dict[str, list[float]]:
    offsets = _parse_offset_grid(raw_offsets)
    return {
        key: sorted(set(_clamp_quantile(anchor_quantiles[key] + offset) for offset in offsets))
        for key in ["short_up_quantile", "take_profit_quantile", "stop_loss_quantile"]
    }


def _build_threshold_plan(
    eval_predictions: pd.DataFrame,
    defaults: dict,
    rolling_window_dates: int,
    min_history_dates: int,
    quantiles: dict,
    threshold_source: str,
) -> pd.DataFrame:
    unique_dates = sorted(pd.Timestamp(date) for date in eval_predictions["date"].dropna().unique())
    rows: list[dict] = []
    for index, date in enumerate(unique_dates):
        if rolling_window_dates > 0:
            history_dates = unique_dates[max(0, index - rolling_window_dates):index]
        else:
            history_dates = unique_dates[:index]
        history = eval_predictions[eval_predictions["date"].isin(history_dates)].copy()
        if len(history_dates) < min_history_dates or history.empty:
            rows.append(
                {
                    "date": date,
                    "history_dates": len(history_dates),
                    "threshold_source": "static_fallback",
                    "short_up_threshold": defaults["short_up_threshold"],
                    "take_profit_threshold": defaults["take_profit_threshold"],
                    "stop_loss_threshold": defaults["stop_loss_threshold"],
                }
            )
            continue
        rows.append(
            {
                "date": date,
                "history_dates": len(history_dates),
                    "threshold_source": threshold_source,
                    "short_up_threshold": float(history["pred_short_up"].quantile(quantiles["short_up_quantile"])),
                    "take_profit_threshold": float(history["pred_take_profit"].quantile(quantiles["take_profit_quantile"])),
                    "stop_loss_threshold": float(history["pred_stop_loss"].quantile(quantiles["stop_loss_quantile"])),
            }
        )
    plan = pd.DataFrame(rows)
    return plan


def _apply_thresholds(predictions: pd.DataFrame, threshold_plan: pd.DataFrame) -> pd.DataFrame:
    working = predictions.merge(threshold_plan, on="date", how="left")
    passed = working[
        (working["pred_short_up"] > working["short_up_threshold"])
        & (working["pred_take_profit"] > working["take_profit_threshold"])
        & (working["pred_stop_loss"] < working["stop_loss_threshold"])
    ].copy()
    return passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)


def _static_threshold_plan(predictions: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    unique_dates = sorted(pd.Timestamp(date) for date in predictions["date"].dropna().unique())
    return pd.DataFrame(
        {
            "date": unique_dates,
            "history_dates": 0,
            "threshold_source": "static",
            "short_up_threshold": defaults["short_up_threshold"],
            "take_profit_threshold": defaults["take_profit_threshold"],
            "stop_loss_threshold": defaults["stop_loss_threshold"],
        }
    )


def _run_probe(
    config,
    predictions: pd.DataFrame,
    defaults: dict,
    state_profile: str | None,
) -> object:
    fair = {
        "short_up_threshold": defaults["short_up_threshold"],
        "take_profit_threshold": defaults["take_profit_threshold"],
        "stop_loss_threshold": defaults["stop_loss_threshold"],
        "top_k_per_day": defaults["top_k_per_day"],
        "hold_days": defaults["hold_days"],
        "take_profit": defaults["take_profit"],
        "stop_loss": defaults["stop_loss"],
    }
    return _run_backtest(
        config,
        predictions,
        score_col="signal_score",
        fair=fair,
        state_profile=state_profile,
        min_score_ratio_to_leader=defaults["min_score_ratio_to_leader"],
        leader_max_day_rank=defaults["leader_max_day_rank"],
        leader_take_profit=defaults["leader_take_profit"],
        leader_stop_loss=defaults["leader_stop_loss"],
        leader_hold_days=defaults["leader_hold_days"],
    )


def _to_serializable_daily(plan: pd.DataFrame) -> list[dict]:
    records = plan.copy()
    records["date"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
    return records.to_dict(orient="records")


def _result_sort_key(summary: dict) -> tuple[float, float, int, int]:
    return (
        float(summary["total_return"]),
        float(summary["sharpe_ratio"]),
        int(summary["trade_count"]),
        int(summary["signal_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe dynamic percentile thresholds on existing mainline predictions")
    parser.add_argument("--config", default="config_two_stage_rerank_leader_tiered_latest_incremental_v3.yaml")
    parser.add_argument("--reference-config", default="config_two_stage_rerank_leader_tiered_v1.yaml")
    parser.add_argument("--rolling-window-dates", type=int, default=20)
    parser.add_argument("--min-history-dates", type=int, default=1)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--quantile-offsets", default="-0.05,0.0,0.05")
    parser.add_argument("--max-grid-results", type=int, default=10)
    parser.add_argument("--output-tag", default="dynamic_threshold_probe")
    parser.add_argument("--state-profile", default=None)
    parser.add_argument("--use-calibration", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    reference_config_path = (ROOT / args.reference_config).resolve()

    config, eval_predictions, eval_predictions_path = _load_predictions(config_path, args.use_calibration)
    _, reference_predictions, reference_predictions_path = _load_predictions(reference_config_path, args.use_calibration)
    defaults = _resolve_execution_defaults(config_path, args.use_calibration)
    anchor_quantiles = {
        "short_up_quantile": _anchor_quantile(reference_predictions["pred_short_up"], defaults["short_up_threshold"]),
        "take_profit_quantile": _anchor_quantile(reference_predictions["pred_take_profit"], defaults["take_profit_threshold"]),
        "stop_loss_quantile": _anchor_quantile(reference_predictions["pred_stop_loss"], defaults["stop_loss_threshold"]),
    }

    static_plan = _static_threshold_plan(eval_predictions, defaults)
    anchor_plan = _build_threshold_plan(
        eval_predictions,
        defaults,
        rolling_window_dates=args.rolling_window_dates,
        min_history_dates=args.min_history_dates,
        quantiles=anchor_quantiles,
        threshold_source="rolling_quantile_anchor",
    )

    static_predictions = _apply_thresholds(eval_predictions, static_plan)
    anchor_predictions = _apply_thresholds(eval_predictions, anchor_plan)

    base_result = _run_probe(config, static_predictions, defaults, args.state_profile)
    anchor_result = _run_probe(config, anchor_predictions, defaults, args.state_profile)

    best_quantiles = dict(anchor_quantiles)
    best_plan = anchor_plan
    best_result = anchor_result
    grid_results: list[dict] = []
    if args.grid_search:
        quantile_grid = _build_quantile_candidates(anchor_quantiles, args.quantile_offsets)
        for short_up_quantile, take_profit_quantile, stop_loss_quantile in product(
            quantile_grid["short_up_quantile"],
            quantile_grid["take_profit_quantile"],
            quantile_grid["stop_loss_quantile"],
        ):
            quantiles = {
                "short_up_quantile": short_up_quantile,
                "take_profit_quantile": take_profit_quantile,
                "stop_loss_quantile": stop_loss_quantile,
            }
            plan = _build_threshold_plan(
                eval_predictions,
                defaults,
                rolling_window_dates=args.rolling_window_dates,
                min_history_dates=args.min_history_dates,
                quantiles=quantiles,
                threshold_source="rolling_quantile_grid",
            )
            predictions = _apply_thresholds(eval_predictions, plan)
            result = _run_probe(config, predictions, defaults, args.state_profile)
            candidate = {
                "quantiles": quantiles,
                "summary": result.summary,
                "delta": {
                    "total_return": float(result.summary["total_return"] - base_result.summary["total_return"]),
                    "sharpe_ratio": float(result.summary["sharpe_ratio"] - base_result.summary["sharpe_ratio"]),
                    "trade_count": int(result.summary["trade_count"] - base_result.summary["trade_count"]),
                    "signal_count": int(result.summary["signal_count"] - base_result.summary["signal_count"]),
                },
            }
            grid_results.append(candidate)
            if _result_sort_key(result.summary) > _result_sort_key(best_result.summary):
                best_quantiles = quantiles
                best_plan = plan
                best_result = result

        grid_results.sort(key=lambda item: _result_sort_key(item["summary"]), reverse=True)
    else:
        quantile_grid = {key: [value] for key, value in anchor_quantiles.items()}

    output_prefix = args.output_tag
    outputs = write_backtest_outputs(best_result, config.results_dir, output_prefix)
    best_plan_path = config.results_dir / f"{output_prefix}_daily_thresholds.csv"
    best_plan_to_write = best_plan.copy()
    best_plan_to_write["date"] = pd.to_datetime(best_plan_to_write["date"]).dt.strftime("%Y-%m-%d")
    best_plan_to_write.to_csv(best_plan_path, index=False, encoding="utf-8-sig")

    report = {
        "config": config_path.name,
        "reference_config": reference_config_path.name,
        "predictions_path": str(eval_predictions_path),
        "reference_predictions_path": str(reference_predictions_path),
        "rolling_window_dates": int(args.rolling_window_dates),
        "min_history_dates": int(args.min_history_dates),
        "grid_search": bool(args.grid_search),
        "quantile_offsets": args.quantile_offsets,
        "use_calibration": bool(args.use_calibration),
        "anchor_quantiles": anchor_quantiles,
        "searched_quantiles": quantile_grid,
        "base_summary": base_result.summary,
        "anchor_summary": anchor_result.summary,
        "best_quantiles": best_quantiles,
        "dynamic_summary": best_result.summary,
        "delta": {
            "total_return": float(best_result.summary["total_return"] - base_result.summary["total_return"]),
            "sharpe_ratio": float(best_result.summary["sharpe_ratio"] - base_result.summary["sharpe_ratio"]),
            "trade_count": int(best_result.summary["trade_count"] - base_result.summary["trade_count"]),
            "signal_count": int(best_result.summary["signal_count"] - base_result.summary["signal_count"]),
        },
        "top_grid_results": grid_results[: max(int(args.max_grid_results), 0)],
        "daily_thresholds": _to_serializable_daily(best_plan),
        "artifacts": {
            **outputs,
            "daily_thresholds": str(best_plan_path),
        },
    }
    report_path = config.results_dir / f"{output_prefix}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()