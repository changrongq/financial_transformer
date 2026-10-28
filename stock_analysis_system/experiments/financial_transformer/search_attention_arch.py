from __future__ import annotations

import argparse
from dataclasses import replace
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

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.training import run_experiment


def _parse_grid(raw: str, cast_fn):
    return [cast_fn(item.strip()) for item in raw.split(",") if item.strip()]


def _build_config(base_config, *, nhead: int, d_model: int):
    return replace(
        base_config,
        model=replace(
            base_config.model,
            nhead=nhead,
            d_model=d_model,
            dim_feedforward=max(d_model * 4, 128),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Transformer attention architecture candidates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="2,4,8")
    parser.add_argument("--d-models", default="64,128,192")
    parser.add_argument("--limit", type=int, help="Optional limit on number of combinations")
    args = parser.parse_args()

    base_config = load_config(args.config)
    heads = _parse_grid(args.heads, int)
    d_models = _parse_grid(args.d_models, int)
    combinations = [(nhead, d_model) for nhead in heads for d_model in d_models]
    if args.limit is not None:
        combinations = combinations[: max(int(args.limit), 0)]

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for nhead, d_model in combinations:
        if d_model % nhead != 0:
            rows.append({
                "nhead": int(nhead),
                "d_model": int(d_model),
                "status": "skipped",
                "reason": "d_model_not_divisible_by_nhead",
            })
            continue
        candidate_root = output_path.parent / f"nhead_{nhead}_dmodel_{d_model}"
        config = _build_config(
            base_config,
            nhead=int(nhead),
            d_model=int(d_model),
        )
        config = replace(
            config,
            results_dir=(candidate_root / "results").resolve(),
            artifacts_dir=(candidate_root / "artifacts").resolve(),
        )
        try:
            result = run_experiment(config)
            metrics_path = config.results_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            rows.append({
                "nhead": int(nhead),
                "d_model": int(d_model),
                "status": "completed",
                "best_epoch": int(result.get("best_epoch", 0)),
                "best_validation_metric": float(result.get("best_validation_metric", 0.0)),
                "test_business_score": float(metrics.get("business_score", 0.0)),
                "test_signal_future_spearman": float(metrics.get("signal_future_spearman", 0.0)),
                "test_pass_signal_future_spearman": float(metrics.get("pass_signal_future_spearman", 0.0)),
                "test_top1_mean_future_return": float(metrics.get("top1_mean_future_return", 0.0)),
                "data_quality_retained_ratio": float(metrics.get("data_quality_retained_ratio", 0.0) or 0.0),
                "results_dir": str(config.results_dir),
                "artifacts_dir": str(config.artifacts_dir),
            })
        except Exception as exc:
            rows.append({
                "nhead": int(nhead),
                "d_model": int(d_model),
                "status": "failed",
                "reason": str(exc),
                "results_dir": str(config.results_dir),
                "artifacts_dir": str(config.artifacts_dir),
            })

    result_df = pd.DataFrame(rows)
    if not result_df.empty and "best_validation_metric" in result_df.columns:
        result_df = result_df.sort_values(["status", "best_validation_metric"], ascending=[True, False]).reset_index(drop=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": str(summary_path), "count": int(len(rows))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()