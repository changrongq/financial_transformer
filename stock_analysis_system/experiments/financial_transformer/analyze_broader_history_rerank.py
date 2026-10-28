from __future__ import annotations

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
    _build_training_frame,
    _freeze_to_checkpoint_universe,
    _load_or_generate_split_predictions,
    _prepare_candidates,
    _run_backtest,
)
from financial_transformer_experiment.config import load_config  # noqa: E402


def _load_config(config_name: str):
    return _freeze_to_checkpoint_universe(load_config(str((ROOT / config_name).resolve())))


def _training_frames(config_name: str, train_split: str) -> list[pd.DataFrame]:
    config = _load_config(config_name)
    val_predictions, test_predictions = _load_or_generate_split_predictions(config)
    if train_split == "val":
        return [val_predictions]
    if train_split == "test":
        return [test_predictions]
    if train_split == "both":
        return [val_predictions, test_predictions]
    raise RuntimeError(f"unsupported train split: {train_split}")


def _fit_and_eval(
    *,
    train_configs: list[str],
    train_split: str,
    eval_config_name: str,
    candidate_pool: int,
    label_mode: str,
) -> dict:
    eval_config = _load_config(eval_config_name)
    fair = KNOWN_FAIR_PARAMS[(ROOT / eval_config_name).stem]

    pooled_candidates: list[pd.DataFrame] = []
    for train_config_name in train_configs:
        train_config = _load_config(train_config_name)
        for frame in _training_frames(train_config_name, train_split):
            candidates = _prepare_candidates(frame, train_config, fair, candidate_pool)
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["source_config"] = train_config_name
            pooled_candidates.append(candidates)
    if not pooled_candidates:
        raise RuntimeError("no training candidates collected")

    train_candidates = pd.concat(pooled_candidates, ignore_index=True)
    train_frame = _build_training_frame(train_candidates, label_mode)
    if train_frame["rerank_label"].nunique() < 2:
        raise RuntimeError("training labels are single-class")

    _, eval_test_predictions = _load_or_generate_split_predictions(eval_config)
    eval_candidates = _prepare_candidates(eval_test_predictions, eval_config, fair, candidate_pool)
    if eval_candidates.empty:
        raise RuntimeError("evaluation candidates are empty")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(train_frame[RERANK_FEATURES], train_frame["rerank_label"].astype(int))
    eval_candidates = eval_candidates.copy()
    eval_candidates["rerank_score"] = model.predict_proba(eval_candidates[RERANK_FEATURES])[:, 1]

    base_result = _run_backtest(eval_config, eval_candidates, score_col="signal_score", fair=fair)
    rerank_result = _run_backtest(eval_config, eval_candidates, score_col="rerank_score", fair=fair)
    return {
        "train_configs": train_configs,
        "train_split": train_split,
        "eval_config": eval_config_name,
        "candidate_pool": candidate_pool,
        "label_mode": label_mode,
        "training_rows": int(len(train_frame)),
        "training_dates": int(train_frame["date"].nunique()),
        "training_positive_rate": float(train_frame["rerank_label"].mean()),
        "base_summary": base_result.summary,
        "rerank_summary": rerank_result.summary,
        "delta": {
            "total_return": float(rerank_result.summary["total_return"] - base_result.summary["total_return"]),
            "sharpe_ratio": float(rerank_result.summary["sharpe_ratio"] - base_result.summary["sharpe_ratio"]),
            "trade_count": int(rerank_result.summary["trade_count"] - base_result.summary["trade_count"]),
            "signal_count": int(rerank_result.summary["signal_count"] - base_result.summary["signal_count"]),
        },
        "coef": {feature: float(coef) for feature, coef in zip(RERANK_FEATURES, model.coef_[0])},
    }


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    scenarios = [
        {
            "name": "recent_local_val_to_recent",
            "train_configs": ["config_moneyflow_v1_recent_oos_v1.yaml"],
            "train_split": "val",
            "eval_config": "config_moneyflow_v1_recent_oos_v1.yaml",
        },
        {
            "name": "old_val_to_recent",
            "train_configs": ["config_moneyflow_v1.yaml"],
            "train_split": "val",
            "eval_config": "config_moneyflow_v1_recent_oos_v1.yaml",
        },
        {
            "name": "old_both_to_recent",
            "train_configs": ["config_moneyflow_v1.yaml"],
            "train_split": "both",
            "eval_config": "config_moneyflow_v1_recent_oos_v1.yaml",
        },
        {
            "name": "old_both_to_old",
            "train_configs": ["config_moneyflow_v1.yaml"],
            "train_split": "both",
            "eval_config": "config_moneyflow_v1.yaml",
        },
        {
            "name": "old_recent_val_to_recent",
            "train_configs": ["config_moneyflow_v1.yaml", "config_moneyflow_v1_recent_oos_v1.yaml"],
            "train_split": "val",
            "eval_config": "config_moneyflow_v1_recent_oos_v1.yaml",
        },
    ]

    report = {
        "candidate_pool": 5,
        "label_mode": "best_in_pool",
        "scenarios": [],
    }
    for scenario in scenarios:
        result = _fit_and_eval(
            train_configs=scenario["train_configs"],
            train_split=scenario["train_split"],
            eval_config_name=scenario["eval_config"],
            candidate_pool=5,
            label_mode="best_in_pool",
        )
        report["scenarios"].append({"name": scenario["name"], **result})

    output_json = ROOT / "results" / "broader_history_rerank_report.json"
    output_md = ROOT / "results" / "broader_history_rerank_report.md"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Broader History Rerank Report", ""]
    lines.append("Candidate pool: 5")
    lines.append("Label mode: best_in_pool")
    lines.append("")
    for item in report["scenarios"]:
        base = item["base_summary"]
        rerank = item["rerank_summary"]
        delta = item["delta"]
        lines.append(f"## {item['name']}")
        lines.append("")
        lines.append(f"- Train configs: {', '.join(item['train_configs'])}")
        lines.append(f"- Train split: {item['train_split']}")
        lines.append(f"- Eval config: {item['eval_config']}")
        lines.append(f"- Training rows/dates: {item['training_rows']} / {item['training_dates']}")
        lines.append(
            f"- Base: total_return={_format_pct(base['total_return'])}, sharpe={_format_float(base['sharpe_ratio'])}, trades={base['trade_count']}, signals={base['signal_count']}"
        )
        lines.append(
            f"- Rerank: total_return={_format_pct(rerank['total_return'])}, sharpe={_format_float(rerank['sharpe_ratio'])}, trades={rerank['trade_count']}, signals={rerank['signal_count']}"
        )
        lines.append(
            f"- Delta: total_return={_format_pct(delta['total_return'])}, sharpe={_format_float(delta['sharpe_ratio'])}, trade_count={delta['trade_count']}, signal_count={delta['signal_count']}"
        )
        lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "scenario_count": len(report["scenarios"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()