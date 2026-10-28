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
from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import build_datasets
from financial_transformer_experiment.model import create_model
from financial_transformer_experiment.scoring import enrich_prediction_frame, strategy_score_frame
from financial_transformer_experiment.training import _build_loader, _evaluate


KNOWN_FAIR_PARAMS = {
    "config_moneyflow_v1_recent_oos_v1": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
    "config_moneyflow_v1": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
    "config_moneyflow_v1_latest_incremental_v1": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
    "config_moneyflow_v1_latest_incremental_v2": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
}


def _resolve_fair_params(config_stem: str, args) -> dict:
    fair = dict(KNOWN_FAIR_PARAMS.get(config_stem, {}))

    override_map = {
        "short_up_threshold": args.short_up_threshold,
        "take_profit_threshold": args.take_profit_threshold,
        "stop_loss_threshold": args.stop_loss_threshold,
        "top_k_per_day": args.top_k_per_day,
        "hold_days": args.hold_days,
        "take_profit": args.take_profit,
        "stop_loss": args.stop_loss,
    }
    for key, value in override_map.items():
        if value is not None:
            fair[key] = value

    required_keys = [
        "short_up_threshold",
        "take_profit_threshold",
        "stop_loss_threshold",
        "top_k_per_day",
        "hold_days",
        "take_profit",
        "stop_loss",
    ]
    missing_keys = [key for key in required_keys if key not in fair]
    if missing_keys:
        raise RuntimeError(
            f"missing fair params for {config_stem}: {', '.join(missing_keys)}; provide them via CLI overrides"
        )
    return fair


RERANK_FEATURES = [
    "pred_short_up",
    "pred_take_profit",
    "pred_stop_loss",
    "pred_max_gain",
    "pred_max_drawdown",
    "signal_score",
    "first_hit_edge",
    "first_hit_confidence",
    "day_rank",
    "score_ratio_to_leader",
    "score_gap_to_leader",
]


def _resolve_rerank_features(raw_features: str | None) -> list[str]:
    if raw_features is None or not raw_features.strip():
        return list(RERANK_FEATURES)

    features = [item.strip() for item in raw_features.split(",") if item.strip()]
    if not features:
        raise RuntimeError("resolved rerank feature set is empty")

    invalid = [feature for feature in features if feature not in RERANK_FEATURES]
    if invalid:
        raise RuntimeError(
            f"unsupported rerank features: {', '.join(invalid)}; allowed: {', '.join(RERANK_FEATURES)}"
        )
    return features


def _freeze_to_checkpoint_universe(config):
    checkpoint_path = config.artifacts_dir / "financial_transformer.pt"
    if not checkpoint_path.exists():
        return config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stock_pool = checkpoint.get("selected_stock_codes") or []
    stock_pool = sorted({str(code).zfill(6) for code in stock_pool if str(code).strip()})
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


def _build_model(config, input_dim: int, device: torch.device, checkpoint: dict):
    model_config = checkpoint.get("model_config", {})
    feature_cols = checkpoint.get("feature_cols", config.experiment.feature_cols)
    return create_model(
        input_dim=input_dim,
        d_model=int(model_config.get("d_model", config.model.d_model)),
        nhead=int(model_config.get("nhead", config.model.nhead)),
        num_layers=int(model_config.get("num_layers", config.model.num_layers)),
        dim_feedforward=int(model_config.get("dim_feedforward", config.model.dim_feedforward)),
        dropout=float(model_config.get("dropout", config.model.dropout)),
        ffn_type=str(model_config.get("ffn_type", config.model.ffn_type)),
        position_encoding=str(model_config.get("position_encoding", config.model.position_encoding)),
        model_variant=str(model_config.get("model_variant", config.model.model_variant)),
        feature_cols=feature_cols,
        grouped_feature_groups=dict(model_config.get("grouped_feature_groups", config.model.grouped_feature_groups)),
        use_group_gates=bool(model_config.get("use_group_gates", config.model.use_group_gates)),
    ).to(device)


def _load_or_generate_split_predictions(config, *, require_val: bool = True, require_test: bool = True) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    val_path = config.results_dir / "val_predictions.csv"
    test_path = config.results_dir / "test_predictions.csv"
    val_predictions = pd.read_csv(val_path, dtype={"stock_code": str}) if require_val and val_path.exists() else None
    test_predictions = pd.read_csv(test_path, dtype={"stock_code": str}) if require_test and test_path.exists() else None
    if ((not require_val) or val_predictions is not None) and ((not require_test) or test_predictions is not None):
        return val_predictions, test_predictions

    checkpoint_path = config.artifacts_dir / "financial_transformer.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    datasets = build_datasets(config)
    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    model = _build_model(config, input_dim=datasets.input_dim, device=device, checkpoint=checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    config.results_dir.mkdir(parents=True, exist_ok=True)
    if require_val and val_predictions is None:
        val_loader = _build_loader(datasets.val, config.training.batch_size, False)
        _, val_predictions = _evaluate(model, val_loader, device, exit_label_mode=config.multitask.exit_label_mode)
        val_predictions.to_csv(val_path, index=False, encoding="utf-8-sig")
    if require_test and test_predictions is None:
        test_loader = _build_loader(datasets.test, config.training.batch_size, False)
        _, test_predictions = _evaluate(model, test_loader, device, exit_label_mode=config.multitask.exit_label_mode)
        test_predictions.to_csv(test_path, index=False, encoding="utf-8-sig")
    return val_predictions, test_predictions


def _select_training_frames(
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    train_split: str,
) -> list[pd.DataFrame]:
    if train_split == "val":
        return [val_predictions]
    if train_split == "test":
        return [test_predictions]
    if train_split == "both":
        return [val_predictions, test_predictions]
    raise RuntimeError(f"unsupported train split: {train_split}")


def _prepare_candidates(df: pd.DataFrame, config, fair: dict, candidate_pool: int) -> pd.DataFrame:
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["stock_code"] = working["stock_code"].astype(str).str.zfill(6)
    numeric_columns = [
        "pred_short_up",
        "pred_take_profit",
        "pred_stop_loss",
        "pred_max_gain",
        "pred_max_drawdown",
        "future_return",
        "label_take_profit",
        "label_stop_loss",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["date", "pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown"]).copy()
    working = enrich_prediction_frame(working, exit_label_mode=config.multitask.exit_label_mode)
    working["signal_score"] = strategy_score_frame(working, exit_label_mode=config.multitask.exit_label_mode)
    working = working[
        (working["pred_short_up"] > fair["short_up_threshold"])
        & (working["pred_take_profit"] > fair["take_profit_threshold"])
        & (working["pred_stop_loss"] < fair["stop_loss_threshold"])
    ].copy()
    if working.empty:
        return working
    working = working.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).copy()
    working["day_rank"] = working.groupby("date").cumcount() + 1
    working = working[working["day_rank"] <= candidate_pool].copy()
    leader_score = working.groupby("date")["signal_score"].transform("max")
    working["score_ratio_to_leader"] = working["signal_score"].div(leader_score.where(leader_score.abs() > 1e-9))
    working["score_gap_to_leader"] = leader_score - working["signal_score"]
    return working.reset_index(drop=True)


def _build_training_frame(candidates: pd.DataFrame, label_mode: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    frame = candidates.copy()
    if label_mode == "best_in_pool":
        best_idx = frame.groupby("date")["future_return"].idxmax()
        frame["rerank_label"] = 0
        frame.loc[best_idx, "rerank_label"] = 1
    elif label_mode == "positive_return":
        frame["rerank_label"] = (frame["future_return"] > 0).astype(int)
    else:
        raise RuntimeError(f"unsupported label mode: {label_mode}")
    return frame


class StaticRerankBacktester(TransformerStrategyBacktester):
    def __init__(self, config, predictions: pd.DataFrame, score_col: str):
        super().__init__(config)
        self.predictions = predictions.copy()
        self.score_col = score_col

    def _load_predictions(self, *args, **kwargs):
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        top_k_per_day = kwargs.get("top_k_per_day")
        min_score_ratio_to_leader = kwargs.get("min_score_ratio_to_leader", 0.0)
        df = self.predictions.copy()
        if start_date is not None:
            df = df[df["date"] >= pd.Timestamp(start_date)].copy()
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)].copy()
        if df.empty:
            raise RuntimeError("指定区间内没有可用于回测的 Transformer 多任务预测。")
        df["signal_score"] = pd.to_numeric(df[self.score_col], errors="coerce")
        if min_score_ratio_to_leader is not None and float(min_score_ratio_to_leader) > 0.0 and not df.empty:
            leader_score = df.groupby("date")["signal_score"].transform("max")
            df["score_ratio_to_leader"] = df["signal_score"].div(leader_score.where(leader_score.abs() > 1e-9))
            df = df[df["score_ratio_to_leader"] >= float(min_score_ratio_to_leader)].copy()
        df = df.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).copy()
        if top_k_per_day is not None and int(top_k_per_day) > 0:
            df = df.groupby("date", group_keys=False).head(int(top_k_per_day)).reset_index(drop=True)
        return df.reset_index(drop=True)


def _run_backtest(
    config,
    predictions: pd.DataFrame,
    score_col: str,
    fair: dict,
    state_profile: str | None = None,
    min_score_ratio_to_leader: float = 0.0,
    leader_max_day_rank: int | None = None,
    leader_take_profit: float | None = None,
    leader_stop_loss: float | None = None,
    leader_hold_days: int | None = None,
):
    backtester = StaticRerankBacktester(config, predictions=predictions, score_col=score_col)
    return backtester.run(
        short_up_threshold=fair["short_up_threshold"],
        take_profit_threshold=fair["take_profit_threshold"],
        stop_loss_threshold=fair["stop_loss_threshold"],
        top_k_per_day=fair["top_k_per_day"],
        hold_days=fair["hold_days"],
        take_profit=fair["take_profit"],
        stop_loss=fair["stop_loss"],
        initial_capital=1_000_000.0,
        commission=0.001,
        slippage=0.001,
        min_lot=100,
        min_position_weight=0.08,
        max_position_weight=0.20,
        max_total_position_fraction=1.0,
        cooldown_trigger_losses=0,
        cooldown_days=0,
        min_secondary_primary_ratio=0.0,
        rank_weight_power=0.0,
        min_score_ratio_to_leader=min_score_ratio_to_leader,
        leader_max_day_rank=leader_max_day_rank,
        leader_take_profit=leader_take_profit,
        leader_stop_loss=leader_stop_loss,
        leader_hold_days=leader_hold_days,
        state_profile=state_profile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight linear reranker on validation candidate pools and test it on the current run")
    parser.add_argument("--config", default="config_moneyflow_v1_recent_oos_v1.yaml")
    parser.add_argument("--train-config", default=None, help="Optional config to source validation candidates from; defaults to --config")
    parser.add_argument("--train-configs", default=None, help="Optional comma-separated configs to pool validation candidates from; overrides --train-config")
    parser.add_argument("--eval-config", default=None, help="Optional config to source test candidates from; defaults to --config")
    parser.add_argument("--train-split", default="val", choices=["val", "test", "both"])
    parser.add_argument("--candidate-pool", type=int, default=5)
    parser.add_argument("--label-mode", default="best_in_pool", choices=["best_in_pool", "positive_return"])
    parser.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated rerank features; defaults to the full built-in feature set",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=1.0,
        help="Blend weight for rerank_score against signal_score; 1.0 means pure rerank, 0.0 means base signal only",
    )
    parser.add_argument("--output-tag", default="lightweight_rerank_probe")
    parser.add_argument("--short-up-threshold", type=float, default=None)
    parser.add_argument("--take-profit-threshold", type=float, default=None)
    parser.add_argument("--stop-loss-threshold", type=float, default=None)
    parser.add_argument("--top-k-per-day", type=int, default=None)
    parser.add_argument("--hold-days", type=int, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--state-profile", type=str, default=None)
    parser.add_argument("--min-score-ratio-to-leader", type=float, default=0.0)
    parser.add_argument("--leader-max-day-rank", type=int, default=None)
    parser.add_argument("--leader-take-profit", type=float, default=None)
    parser.add_argument("--leader-stop-loss", type=float, default=None)
    parser.add_argument("--leader-hold-days", type=int, default=None)
    args = parser.parse_args()

    default_config_path = (ROOT / args.config).resolve()
    eval_config_path = (ROOT / (args.eval_config or args.config)).resolve()

    if args.train_configs:
        train_config_paths = [
            (ROOT / item.strip()).resolve()
            for item in args.train_configs.split(",")
            if item.strip()
        ]
    else:
        train_config_paths = [(ROOT / (args.train_config or args.config)).resolve()]

    train_configs = [_freeze_to_checkpoint_universe(load_config(str(path))) for path in train_config_paths]
    eval_config = _freeze_to_checkpoint_universe(load_config(str(eval_config_path)))
    fair = _resolve_fair_params(eval_config_path.stem, args)
    rerank_features = _resolve_rerank_features(args.features)
    if not 0.0 <= float(args.blend_weight) <= 1.0:
        raise RuntimeError("--blend-weight must be between 0.0 and 1.0")

    output_dir = eval_config.root_dir / "results" / f"{eval_config.results_dir.name}_{args.output_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, eval_test_predictions = _load_or_generate_split_predictions(eval_config, require_val=False, require_test=True)
    pooled_val_candidates: list[pd.DataFrame] = []
    for train_config in train_configs:
        require_val = args.train_split in {"val", "both"}
        require_test = args.train_split in {"test", "both"}
        train_val_predictions, train_test_predictions = _load_or_generate_split_predictions(
            train_config,
            require_val=require_val,
            require_test=require_test,
        )
        for train_frame in _select_training_frames(train_val_predictions, train_test_predictions, args.train_split):
            train_candidates = _prepare_candidates(train_frame, train_config, fair, args.candidate_pool)
            if train_candidates.empty:
                continue
            train_candidates = train_candidates.copy()
            train_candidates["source_config"] = train_config.results_dir.name
            pooled_val_candidates.append(train_candidates)
    if not pooled_val_candidates:
        raise RuntimeError("validation candidate pool is empty")

    val_candidates = pd.concat(pooled_val_candidates, ignore_index=True)
    test_candidates = _prepare_candidates(eval_test_predictions, eval_config, fair, args.candidate_pool)
    val_train = _build_training_frame(val_candidates, args.label_mode)
    if val_train["rerank_label"].nunique() < 2:
        raise RuntimeError("validation rerank labels are single-class")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(val_train[rerank_features], val_train["rerank_label"].astype(int))

    val_candidates = val_candidates.copy()
    test_candidates = test_candidates.copy()
    val_candidates["rerank_score"] = model.predict_proba(val_candidates[rerank_features])[:, 1]
    test_candidates["rerank_score"] = model.predict_proba(test_candidates[rerank_features])[:, 1]
    val_candidates["blended_rerank_score"] = (
        (1.0 - float(args.blend_weight)) * val_candidates["signal_score"]
        + float(args.blend_weight) * val_candidates["rerank_score"]
    )
    test_candidates["blended_rerank_score"] = (
        (1.0 - float(args.blend_weight)) * test_candidates["signal_score"]
        + float(args.blend_weight) * test_candidates["rerank_score"]
    )

    base_result = _run_backtest(
        eval_config,
        test_candidates,
        score_col="signal_score",
        fair=fair,
        state_profile=args.state_profile,
        min_score_ratio_to_leader=args.min_score_ratio_to_leader,
        leader_max_day_rank=args.leader_max_day_rank,
        leader_take_profit=args.leader_take_profit,
        leader_stop_loss=args.leader_stop_loss,
        leader_hold_days=args.leader_hold_days,
    )
    rerank_score_col = "rerank_score" if float(args.blend_weight) >= 0.999999 else "blended_rerank_score"
    rerank_result = _run_backtest(
        eval_config,
        test_candidates,
        score_col=rerank_score_col,
        fair=fair,
        state_profile=args.state_profile,
        min_score_ratio_to_leader=args.min_score_ratio_to_leader,
        leader_max_day_rank=args.leader_max_day_rank,
        leader_take_profit=args.leader_take_profit,
        leader_stop_loss=args.leader_stop_loss,
        leader_hold_days=args.leader_hold_days,
    )

    selected_compare = (
        test_candidates.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
        .groupby("date", group_keys=False)
        .head(1)[["date", "stock_code", "day_rank", "future_return", "signal_score"]]
        .rename(columns={"stock_code": "base_stock_code", "day_rank": "base_rank", "future_return": "base_future_return", "signal_score": "base_score"})
        .merge(
            test_candidates.sort_values(["date", "rerank_score", "pred_take_profit"], ascending=[True, False, False])
            .groupby("date", group_keys=False)
            .head(1)[["date", "stock_code", "day_rank", "future_return", "rerank_score"]]
            .rename(columns={"stock_code": "rerank_stock_code", "day_rank": "rerank_rank", "future_return": "rerank_future_return"}),
            on="date",
            how="outer",
        )
        .sort_values("date")
    )
    selected_compare["date"] = pd.to_datetime(selected_compare["date"]).dt.strftime("%Y-%m-%d")
    selected_compare.to_csv(output_dir / "selected_compare.csv", index=False, encoding="utf-8-sig")

    model_payload = {
        "features": rerank_features,
        "candidate_pool": int(args.candidate_pool),
        "label_mode": args.label_mode,
        "blend_weight": float(args.blend_weight),
        "intercept": float(model.intercept_[0]),
        "coef": {feature: float(coef) for feature, coef in zip(rerank_features, model.coef_[0])},
        "validation_rows": int(len(val_train)),
        "validation_dates": int(val_train["date"].nunique()),
        "validation_positive_rate": float(val_train["rerank_label"].mean()),
    }
    (output_dir / "rerank_model.json").write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    test_candidates.to_csv(output_dir / "test_candidate_pool.csv", index=False, encoding="utf-8-sig")
    val_candidates.to_csv(output_dir / "val_candidate_pool.csv", index=False, encoding="utf-8-sig")
    detail_paths = write_backtest_outputs(rerank_result, output_dir, "lightweight_rerank_best")

    payload = {
        "config": str(default_config_path),
        "train_config": str(train_config_paths[0]) if len(train_config_paths) == 1 else None,
        "train_configs": [str(path) for path in train_config_paths],
        "train_split": args.train_split,
        "eval_config": str(eval_config_path),
        "state_profile": args.state_profile,
        "min_score_ratio_to_leader": float(args.min_score_ratio_to_leader),
        "leader_max_day_rank": int(args.leader_max_day_rank) if args.leader_max_day_rank is not None else None,
        "leader_take_profit": float(args.leader_take_profit) if args.leader_take_profit is not None else None,
        "leader_stop_loss": float(args.leader_stop_loss) if args.leader_stop_loss is not None else None,
        "leader_hold_days": int(args.leader_hold_days) if args.leader_hold_days is not None else None,
        "fair": fair,
        "base_summary": base_result.summary,
        "rerank_summary": rerank_result.summary,
        "selected_compare": str(output_dir / "selected_compare.csv"),
        "model": model_payload,
        "detail_outputs": detail_paths,
    }
    (output_dir / "lightweight_rerank_probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()