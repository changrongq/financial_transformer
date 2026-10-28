from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import ExperimentConfig, load_config
from financial_transformer_experiment.data import build_datasets
from financial_transformer_experiment.training import run_experiment


def _parse_candidates(text: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            seq_text, min_text = item.split(":", 1)
            seq_len = int(seq_text.strip())
            min_seq_len = int(min_text.strip())
        else:
            seq_len = int(item)
            min_seq_len = seq_len
        if seq_len <= 0:
            raise ValueError("候选窗口长度必须大于 0。")
        if min_seq_len <= 0 or min_seq_len > seq_len:
            raise ValueError("候选 min_seq_len 必须在 1 到 seq_len 之间。")
        candidates.append((seq_len, min_seq_len))
    if not candidates:
        raise ValueError("至少需要提供一个候选窗口。")
    return candidates


def _candidate_name(seq_len: int, min_seq_len: int) -> str:
    if seq_len == min_seq_len:
        return f"fixed_{seq_len}"
    return f"dynamic_{seq_len}_{min_seq_len}"


def _candidate_config(base: ExperimentConfig, seq_len: int, min_seq_len: int, tag: str) -> ExperimentConfig:
    candidate_name = _candidate_name(seq_len, min_seq_len)
    model = replace(base.model, seq_len=seq_len, min_seq_len=min_seq_len)
    results_dir = (EXPERIMENT_ROOT / "results" / tag / candidate_name).resolve()
    artifacts_dir = (EXPERIMENT_ROOT / "artifacts" / tag / candidate_name).resolve()
    return replace(base, model=model, results_dir=results_dir, artifacts_dir=artifacts_dir)


def _load_existing_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_sample_sizes(config: ExperimentConfig) -> dict[str, int]:
    datasets = build_datasets(config)
    return {
        "train_size": len(datasets.train),
        "val_size": len(datasets.val),
        "test_size": len(datasets.test),
    }


def _summarize_metrics(config: ExperimentConfig, metrics: dict, source: str) -> dict:
    return {
        "candidate": _candidate_name(config.model.seq_len, int(config.model.min_seq_len or config.model.seq_len)),
        "seq_len": int(config.model.seq_len),
        "min_seq_len": int(config.model.min_seq_len or config.model.seq_len),
        "mode": "fixed" if int(config.model.seq_len) == int(config.model.min_seq_len or config.model.seq_len) else "dynamic",
        "train_size": int(metrics.get("train_size", 0)),
        "val_size": int(metrics.get("val_size", 0)),
        "test_size": int(metrics.get("test_size", 0)),
        "business_score": float(metrics.get("business_score", 0.0)) if "business_score" in metrics else None,
        "auc_short_up": float(metrics.get("auc_short_up", 0.0)) if "auc_short_up" in metrics else None,
        "auc_take_profit": float(metrics.get("auc_take_profit", 0.0)) if "auc_take_profit" in metrics else None,
        "auc_stop_loss": float(metrics.get("auc_stop_loss", 0.0)) if "auc_stop_loss" in metrics else None,
        "mae_max_gain": float(metrics.get("mae_max_gain", 0.0)) if "mae_max_gain" in metrics else None,
        "mae_max_drawdown": float(metrics.get("mae_max_drawdown", 0.0)) if "mae_max_drawdown" in metrics else None,
        "results_dir": str(config.results_dir),
        "artifacts_dir": str(config.artifacts_dir),
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep financial transformer window candidates")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config_moneyflow_v1.yaml"))
    parser.add_argument("--candidates", default="45,60,75,90,105,120")
    parser.add_argument("--tag", default="window_search_v1")
    parser.add_argument("--sample-size-only", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    candidates = _parse_candidates(args.candidates)
    summary_rows: list[dict] = []

    for seq_len, min_seq_len in candidates:
        config = _candidate_config(base, seq_len, min_seq_len, args.tag)
        metrics_path = config.results_dir / "metrics.json"

        if args.sample_size_only:
            sample_sizes = _collect_sample_sizes(config)
            summary_rows.append(_summarize_metrics(config, sample_sizes, source="sample_size_only"))
            print(json.dumps(summary_rows[-1], ensure_ascii=False))
            continue

        existing_metrics = None if args.force_retrain else _load_existing_metrics(metrics_path)
        if existing_metrics is not None:
            summary_rows.append(_summarize_metrics(config, existing_metrics, source="existing_metrics"))
            print(json.dumps(summary_rows[-1], ensure_ascii=False))
            continue

        result = run_experiment(config)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = _summarize_metrics(config, metrics, source="fresh_train")
        row["model_path"] = result.get("model_path")
        row["metrics_path"] = result.get("metrics_path")
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if not summary_rows:
        raise RuntimeError("窗口搜索没有产生任何结果。")

    output_dir = (EXPERIMENT_ROOT / "results" / args.tag).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["business_score", "train_size", "seq_len"],
        ascending=[False, False, True],
        na_position="last",
    )
    summary_csv = output_dir / "window_search_summary.csv"
    summary_json = output_dir / "window_search_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "rows": len(summary_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()