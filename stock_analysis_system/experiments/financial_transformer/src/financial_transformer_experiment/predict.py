from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .calibration import apply_probability_calibration
from .calibration import load_probability_calibration
from .config import ExperimentConfig
from .data import build_feature_frame
from .model import create_model
from .scoring import enrich_prediction_frame, strategy_score_frame


def predict_latest(config: ExperimentConfig, top_n: int | None = None, use_calibration: bool = False) -> dict:
    checkpoint_path = config.artifacts_dir / "financial_transformer.pt"
    if not checkpoint_path.exists():
        raise RuntimeError("未找到 Transformer 模型文件，请先运行 train.py。")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_cols = checkpoint["feature_cols"]
    seq_len = int(checkpoint.get("seq_len", config.model.seq_len))
    min_seq_len = int(checkpoint.get("min_seq_len", config.model.min_seq_len or seq_len))
    model_config = checkpoint.get("model_config", {})
    calibration = load_probability_calibration(config.results_dir) if use_calibration else None

    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
    model = create_model(
        input_dim=len(feature_cols),
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_df = build_feature_frame(config)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    rows: list[dict] = []
    with torch.no_grad():
        for stock_code, group in feature_df.groupby("stock_code"):
            group = group.sort_values("date").reset_index(drop=True)
            usable = group.dropna(subset=feature_cols).reset_index(drop=True)
            if len(usable) < min_seq_len:
                continue
            latest_window = usable[feature_cols].tail(seq_len).to_numpy(dtype=np.float32)
            if not np.isfinite(latest_window).all():
                continue
            x = torch.tensor(latest_window[None, :, :], dtype=torch.float32, device=device)
            outputs = model(x)
            pred_short_up = float(torch.sigmoid(outputs["short_up_logits"]).item())
            pred_take_profit = float(torch.sigmoid(outputs["take_profit_logits"]).item())
            pred_stop_loss = float(torch.sigmoid(outputs["stop_loss_logits"]).item())
            pred_max_gain = float(outputs["max_gain"].item())
            pred_max_drawdown = float(outputs["max_drawdown"].item())
            latest = usable.iloc[-1]
            rows.append(
                {
                    "stock_code": stock_code,
                    "date": str(pd.Timestamp(latest["date"]).date()),
                    "layer": latest.get("layer", config.experiment.layer_filter or "unknown"),
                    "raw_pred_short_up": pred_short_up,
                    "raw_pred_take_profit": pred_take_profit,
                    "raw_pred_stop_loss": pred_stop_loss,
                    "pred_short_up": pred_short_up,
                    "pred_take_profit": pred_take_profit,
                    "pred_stop_loss": pred_stop_loss,
                    "pred_max_gain": pred_max_gain,
                    "pred_max_drawdown": pred_max_drawdown,
                    "risk_reward_ratio": float(max(pred_max_gain, 0.0) / max(abs(pred_max_drawdown), 0.01)),
                    "close": float(latest.get("close", 0.0)),
                    "market_cap_yi": float(latest.get("market_cap_yi", 0.0)),
                    "future_return_reference": float(latest.get("future_return_5d", 0.0)) if pd.notna(latest.get("future_return_5d", np.nan)) else 0.0,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("没有生成任何预测结果，请检查股票池、模型和历史数据是否足够。")

    result = apply_probability_calibration(result, calibration)
    result = enrich_prediction_frame(result, exit_label_mode=config.multitask.exit_label_mode)
    result["signal_score"] = strategy_score_frame(result, exit_label_mode=config.multitask.exit_label_mode)
    result = result.sort_values(["signal_score", "pred_take_profit", "pred_short_up"], ascending=[False, False, False]).reset_index(drop=True)
    if top_n:
        result = result.head(top_n)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = "_calibrated" if use_calibration else ""
    output_path = config.results_dir / f"latest_predictions{file_suffix}.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_path = config.results_dir / f"latest_predictions{file_suffix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "top_n": int(top_n or len(result)),
                "stock_pool_size": len(config.experiment.stock_pool),
                "layer_filter": config.experiment.layer_filter,
                "prediction_count": int(len(result)),
                "probability_calibration_applied": bool(calibration),
                "output_file": str(output_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "output_file": str(output_path),
        "top_stock": result.iloc[0]["stock_code"],
        "top_signal_score": float(result.iloc[0]["signal_score"]),
        "prediction_count": int(len(result)),
    }