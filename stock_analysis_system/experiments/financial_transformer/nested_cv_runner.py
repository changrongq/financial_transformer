from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import ExperimentConfig, load_config
from financial_transformer_experiment.training import run_experiment


def _resolve_config_path(plan_path: Path, config_text: str) -> Path:
    candidate = Path(config_text)
    if candidate.is_absolute():
        return candidate.resolve()
    return (plan_path.parent / candidate).resolve()


def _load_plan(plan_path: Path) -> dict:
    with plan_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _candidate_slug(config_path: Path) -> str:
    return config_path.stem


def _build_variant_config(
    base_config: ExperimentConfig,
    *,
    results_dir: Path,
    artifacts_dir: Path,
    train_end: str,
    val_end: str,
    test_end: str,
    stock_cv_total_folds: int | None = None,
    train_stock_folds: list[int] | None = None,
    val_stock_folds: list[int] | None = None,
    test_stock_folds: list[int] | None = None,
) -> ExperimentConfig:
    experiment = replace(
        base_config.experiment,
        stock_cv_total_folds=stock_cv_total_folds,
        train_stock_folds=train_stock_folds or [],
        val_stock_folds=val_stock_folds or [],
        test_stock_folds=test_stock_folds or [],
    )
    splits = replace(base_config.splits, train_end=train_end, val_end=val_end, test_end=test_end)
    return replace(
        base_config,
        experiment=experiment,
        splits=splits,
        results_dir=results_dir.resolve(),
        artifacts_dir=artifacts_dir.resolve(),
    )


def _run_inner_stock_rotation(
    base_config: ExperimentConfig,
    *,
    outer_window: dict,
    n_folds: int,
    candidate_root: Path,
) -> list[dict]:
    rows: list[dict] = []
    for fold_index in range(n_folds):
        train_folds = [fold for fold in range(n_folds) if fold != fold_index]
        val_folds = [fold_index]
        fold_root = candidate_root / outer_window["name"] / f"inner_fold_{fold_index}"
        config = _build_variant_config(
            base_config,
            results_dir=fold_root / "results",
            artifacts_dir=fold_root / "artifacts",
            train_end=outer_window["train_end"],
            val_end=outer_window["val_end"],
            test_end=outer_window["val_end"],
            stock_cv_total_folds=n_folds,
            train_stock_folds=train_folds,
            val_stock_folds=val_folds,
            test_stock_folds=[],
        )
        result = run_experiment(config)
        rows.append(
            {
                "window_name": outer_window["name"],
                "fold_index": int(fold_index),
                "train_stock_folds": train_folds,
                "val_stock_folds": val_folds,
                "best_epoch": int(result.get("best_epoch", 0)),
                "best_validation_metric": float(result.get("best_validation_metric", 0.0)),
                **{f"val_{key}": float(value) for key, value in (result.get("best_val_metrics") or {}).items()},
            }
        )
    return rows


def _run_outer_window(
    base_config: ExperimentConfig,
    *,
    outer_window: dict,
    candidate_root: Path,
) -> dict:
    outer_root = candidate_root / outer_window["name"] / "outer_full"
    config = _build_variant_config(
        base_config,
        results_dir=outer_root / "results",
        artifacts_dir=outer_root / "artifacts",
        train_end=outer_window["train_end"],
        val_end=outer_window["val_end"],
        test_end=outer_window["test_end"],
    )
    result = run_experiment(config)
    return {
        "window_name": outer_window["name"],
        "train_end": outer_window["train_end"],
        "val_end": outer_window["val_end"],
        "test_end": outer_window["test_end"],
        "best_epoch": int(result.get("best_epoch", 0)),
        "best_validation_metric": float(result.get("best_validation_metric", 0.0)),
        **{f"test_{key}": float(value) for key, value in result.items() if isinstance(value, (int, float)) and key not in {"best_epoch", "best_validation_metric"}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested CV structure evaluation for financial transformer experiments")
    parser.add_argument("--plan", default=str(ROOT / "config_nested_cv_plan_v1.yaml"))
    parser.add_argument("--candidate")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--inner-folds", type=int)
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = _load_plan(plan_path)
    candidate_items = [args.candidate] if args.candidate else list(plan.get("candidate_structures", []))
    if args.max_candidates:
        candidate_items = candidate_items[: args.max_candidates]
    if not candidate_items:
        raise RuntimeError("nested cv plan did not provide any candidate structures")

    outer_windows = list((plan.get("outer_time_rolling") or {}).get("windows", []))
    if args.max_windows:
        outer_windows = outer_windows[: args.max_windows]
    if not outer_windows:
        raise RuntimeError("nested cv plan did not provide any outer windows")

    n_folds = int(args.inner_folds or ((plan.get("inner_stock_rotation") or {}).get("n_folds") or 5))
    results_root = ROOT / "results" / "nested_cv" / (plan_path.stem + (f"_{args.output_tag}" if args.output_tag else ""))
    results_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    payload_candidates: list[dict] = []

    for candidate_text in candidate_items:
        config_path = _resolve_config_path(plan_path, candidate_text)
        base_config = load_config(config_path)
        candidate_slug = _candidate_slug(config_path)
        candidate_root = results_root / candidate_slug
        candidate_root.mkdir(parents=True, exist_ok=True)

        inner_rows: list[dict] = []
        outer_rows: list[dict] = []
        for outer_window in outer_windows:
            inner_rows.extend(
                _run_inner_stock_rotation(
                    base_config,
                    outer_window=outer_window,
                    n_folds=n_folds,
                    candidate_root=candidate_root,
                )
            )
            outer_rows.append(
                _run_outer_window(
                    base_config,
                    outer_window=outer_window,
                    candidate_root=candidate_root,
                )
            )

        inner_df = pd.DataFrame(inner_rows)
        outer_df = pd.DataFrame(outer_rows)
        inner_csv = candidate_root / "inner_stock_rotation_summary.csv"
        outer_csv = candidate_root / "outer_window_summary.csv"
        inner_df.to_csv(inner_csv, index=False, encoding="utf-8-sig")
        outer_df.to_csv(outer_csv, index=False, encoding="utf-8-sig")

        summary_row = {
            "candidate": candidate_slug,
            "config_path": str(config_path),
            "inner_mean_best_validation_metric": float(inner_df["best_validation_metric"].mean()) if not inner_df.empty else 0.0,
            "inner_std_best_validation_metric": float(inner_df["best_validation_metric"].std(ddof=0)) if len(inner_df) > 1 else 0.0,
            "outer_mean_test_business_score": float(outer_df["test_business_score"].mean()) if "test_business_score" in outer_df.columns else 0.0,
            "outer_mean_test_auc_short_up": float(outer_df["test_auc_short_up"].mean()) if "test_auc_short_up" in outer_df.columns else 0.0,
            "outer_positive_business_score_windows": int((outer_df["test_business_score"] > 0).sum()) if "test_business_score" in outer_df.columns else 0,
            "outer_window_count": int(len(outer_df)),
            "inner_fold_count": int(len(inner_df)),
            "inner_summary_csv": str(inner_csv),
            "outer_summary_csv": str(outer_csv),
        }
        summary_rows.append(summary_row)
        payload_candidates.append(
            {
                **summary_row,
                "inner_rows": inner_rows,
                "outer_rows": outer_rows,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["outer_mean_test_business_score", "inner_mean_best_validation_metric"],
        ascending=[False, False],
    ).reset_index(drop=True)
    summary_csv = results_root / "nested_cv_summary.csv"
    summary_json = results_root / "nested_cv_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    payload = {
        "plan": str(plan_path),
        "results_root": str(results_root),
        "n_folds": n_folds,
        "outer_window_count": len(outer_windows),
        "candidates": payload_candidates,
        "summary_csv": str(summary_csv),
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()