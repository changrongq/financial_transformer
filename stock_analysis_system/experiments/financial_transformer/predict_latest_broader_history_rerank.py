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
    RERANK_FEATURES,
    _build_training_frame,
    _freeze_to_checkpoint_universe,
    _load_or_generate_split_predictions,
    _prepare_candidates,
)
from financial_transformer_experiment.config import load_config  # noqa: E402
from financial_transformer_experiment.predict import predict_latest  # noqa: E402


VARIANT_RULES = {
    "default_exec": {
        "short_up_threshold": 0.5,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
    "transfer_exec": {
        "short_up_threshold": 0.5,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 30,
        "take_profit": 0.14,
        "stop_loss": -0.04,
    },
}


def _load_latest_frame(results_dir: Path, use_calibration: bool) -> pd.DataFrame:
    suffix = "_calibrated" if use_calibration else ""
    path = results_dir / f"latest_predictions{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"latest prediction file not found: {path}")
    frame = pd.read_csv(path, dtype={"stock_code": str})
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def run_latest_broader_history_rerank(
    *,
    variant: str = "default_exec",
    train_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_split: str = "both",
    candidate_pool: int = 5,
    label_mode: str = "best_in_pool",
    use_calibration: bool = False,
    top_n: int | None = None,
    output_dir: str | Path | None = None,
    refresh_latest: bool = False,
    allow_stale: bool = False,
) -> dict:
    if variant not in VARIANT_RULES:
        raise ValueError(f"unsupported variant: {variant}")

    fair = dict(VARIANT_RULES[variant])
    train_config_name = str(train_config_path or "config_moneyflow_v1.yaml")
    eval_config_name = str(eval_config_path or "config_moneyflow_v1.yaml")
    train_config = _freeze_to_checkpoint_universe(load_config(str((ROOT / train_config_name).resolve())))
    eval_config = load_config(str((ROOT / eval_config_name).resolve()))

    try:
        latest_frame = _load_latest_frame(eval_config.results_dir, use_calibration=use_calibration)
        if refresh_latest:
            raise FileNotFoundError("refresh requested")
    except FileNotFoundError:
        predict_latest(eval_config, top_n=top_n, use_calibration=use_calibration)
        latest_frame = _load_latest_frame(eval_config.results_dir, use_calibration=use_calibration)

    latest_available_date = pd.to_datetime(latest_frame["date"], errors="coerce").max()
    if pd.isna(latest_available_date):
        raise RuntimeError("latest prediction frame has no valid dates")
    if not allow_stale:
        latest_frame = latest_frame[latest_frame["date"] == latest_available_date].copy()

    history_val, history_test = _load_or_generate_split_predictions(
        train_config,
        require_val=train_split in {"val", "both"},
        require_test=train_split in {"test", "both"},
    )
    history_frames = []
    if train_split in {"val", "both"} and history_val is not None:
        history_frames.append(history_val)
    if train_split in {"test", "both"} and history_test is not None:
        history_frames.append(history_test)
    if not history_frames:
        raise RuntimeError("no historical training frames available")

    history_candidates = []
    for frame in history_frames:
        candidates = _prepare_candidates(frame, train_config, fair, candidate_pool)
        if not candidates.empty:
            history_candidates.append(candidates)
    if not history_candidates:
        raise RuntimeError("historical candidate pool is empty")

    train_candidates = pd.concat(history_candidates, ignore_index=True)
    train_frame = _build_training_frame(train_candidates, label_mode)
    if train_frame["rerank_label"].nunique() < 2:
        raise RuntimeError("historical rerank labels are single-class")

    latest_candidates = _prepare_candidates(latest_frame, eval_config, fair, candidate_pool)
    if latest_candidates.empty:
        raise RuntimeError("latest candidate pool is empty under the selected fair rule")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(train_frame[RERANK_FEATURES], train_frame["rerank_label"].astype(int))
    latest_candidates = latest_candidates.copy()
    latest_candidates["rerank_score"] = model.predict_proba(latest_candidates[RERANK_FEATURES])[:, 1]
    latest_candidates = latest_candidates.sort_values(["date", "rerank_score", "pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)
    latest_candidates["rerank_rank"] = latest_candidates.groupby("date").cumcount() + 1
    selected = latest_candidates[latest_candidates["rerank_rank"] <= int(fair["top_k_per_day"])].copy()

    resolved_output_dir = Path(output_dir).resolve() if output_dir else (ROOT / "results" / "broader_history_rerank_variants_v1" / variant / "latest")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    latest_candidates.to_csv(resolved_output_dir / "latest_candidate_pool.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(resolved_output_dir / "latest_selected.csv", index=False, encoding="utf-8-sig")

    payload = {
        "variant": variant,
        "train_config": str((ROOT / train_config_name).resolve()),
        "eval_config": str((ROOT / eval_config_name).resolve()),
        "train_split": train_split,
        "candidate_pool": candidate_pool,
        "label_mode": label_mode,
        "use_calibration": bool(use_calibration),
        "fair": fair,
        "latest_available_date": str(pd.Timestamp(latest_available_date).date()),
        "allow_stale": bool(allow_stale),
        "latest_date_count": int(latest_candidates["date"].nunique()),
        "latest_candidate_count": int(len(latest_candidates)),
        "selected_count": int(len(selected)),
        "top_stock": str(selected.iloc[0]["stock_code"]) if not selected.empty else None,
        "top_rerank_score": float(selected.iloc[0]["rerank_score"]) if not selected.empty else None,
        "training_rows": int(len(train_frame)),
        "training_dates": int(train_frame["date"].nunique()),
        "training_positive_rate": float(train_frame["rerank_label"].mean()),
        "model": {
            "intercept": float(model.intercept_[0]),
            "coef": {feature: float(coef) for feature, coef in zip(RERANK_FEATURES, model.coef_[0])},
        },
        "outputs": {
            "latest_candidate_pool": str(resolved_output_dir / "latest_candidate_pool.csv"),
            "latest_selected": str(resolved_output_dir / "latest_selected.csv"),
        },
    }
    (resolved_output_dir / "latest_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply broader-history rerank to the latest single-model prediction pool")
    parser.add_argument("--variant", default="default_exec", choices=sorted(VARIANT_RULES))
    parser.add_argument("--train-config", default="config_moneyflow_v1.yaml")
    parser.add_argument("--eval-config", default="config_moneyflow_v1.yaml")
    parser.add_argument("--train-split", default="both", choices=["val", "test", "both"])
    parser.add_argument("--candidate-pool", type=int, default=5)
    parser.add_argument("--label-mode", default="best_in_pool", choices=["best_in_pool", "positive_return"])
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refresh-latest", action="store_true")
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()
    payload = run_latest_broader_history_rerank(
        variant=args.variant,
        train_config_path=args.train_config,
        eval_config_path=args.eval_config,
        train_split=args.train_split,
        candidate_pool=args.candidate_pool,
        label_mode=args.label_mode,
        use_calibration=args.use_calibration,
        top_n=args.top_n,
        output_dir=args.output_dir,
        refresh_latest=args.refresh_latest,
        allow_stale=args.allow_stale,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()