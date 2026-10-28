from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import build_feature_frame
from financial_transformer_experiment.model import create_model


def _entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
    if probabilities.size == 0:
        return 0.0
    probabilities = probabilities / probabilities.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-9)
    q = q / max(q.sum(), 1e-9)
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask]))) if mask.any() else 0.0
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _attention_feature_focus(attention: np.ndarray, feature_frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    sequence_len = attention.shape[-1]
    if sequence_len == 0:
        return {"top_features": [], "feature_scores": {}}
    weights = attention.mean(axis=(0, 1, 2)) if attention.ndim == 4 else attention.mean(axis=(0, 1))
    weights = np.asarray(weights, dtype=float)
    weights = weights[-sequence_len:]
    focus_scores: dict[str, float] = {}
    for feature_name in feature_cols:
        values = pd.to_numeric(feature_frame[feature_name], errors="coerce").tail(sequence_len).fillna(0.0).to_numpy(dtype=float)
        if values.size != weights.size:
            continue
        focus_scores[feature_name] = float(np.abs(np.corrcoef(weights, values)[0, 1])) if values.std() > 0 and weights.std() > 0 else 0.0
    top_features = sorted(focus_scores.items(), key=lambda item: item[1], reverse=True)[:5]
    return {"top_features": top_features, "feature_scores": focus_scores}


def _load_sample_frame(config, stock_code: str | None, max_rows: int) -> tuple[pd.DataFrame, str]:
    feature_frame = build_feature_frame(config)
    if stock_code:
        normalized = str(stock_code).strip().zfill(6)
        feature_frame = feature_frame[feature_frame["stock_code"].astype(str).str.zfill(6) == normalized].copy()
    if feature_frame.empty:
        raise RuntimeError("没有可用样本用于注意力诊断。")
    feature_frame = feature_frame.sort_values(["stock_code", "date"]).reset_index(drop=True)
    stock_code_resolved = str(feature_frame["stock_code"].iloc[0]).zfill(6)
    return feature_frame.tail(max_rows).reset_index(drop=True), stock_code_resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Transformer attention heads on a trained checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stock-code", default=None)
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--max-seconds", type=int, default=90)
    parser.add_argument("--sample-stocks", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else config.artifacts_dir / "financial_transformer.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"未找到模型文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    sample_stocks = max(int(args.sample_stocks), 1)
    if not args.stock_code:
        config = replace(
            config,
            experiment=replace(
                config.experiment,
                max_stocks=min(int(config.experiment.max_stocks or sample_stocks), sample_stocks),
            ),
        )
    model_config = checkpoint.get("model_config", {})
    feature_cols = checkpoint.get("feature_cols", config.experiment.feature_cols)
    sample_frame_all = build_feature_frame(config)
    if args.stock_code:
        normalized = str(args.stock_code).strip().zfill(6)
        sample_frame_all = sample_frame_all[sample_frame_all["stock_code"].astype(str).str.zfill(6) == normalized].copy()
    if sample_frame_all.empty:
        raise RuntimeError("没有可用样本用于注意力诊断。")

    stock_codes = list(sample_frame_all["stock_code"].astype(str).dropna().unique())[:sample_stocks]

    device = torch.device("cpu")
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

    start_ts = time.time()
    attentions_accum: list[list[np.ndarray]] = []
    used_stock_codes: list[str] = []
    sample_rows_total = 0
    for stock_code in stock_codes:
        stock_frame = sample_frame_all[sample_frame_all["stock_code"].astype(str) == str(stock_code)].sort_values("date")
        input_matrix = stock_frame[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().tail(int(args.max_rows)).to_numpy(dtype=np.float32)
        if input_matrix.size == 0:
            continue
        sample_rows_total += int(input_matrix.shape[0])
        sample_tensor = torch.tensor(input_matrix[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            _outputs, attentions = model(sample_tensor, return_attention=True)
        if attentions:
            attentions_accum.append([layer.detach().cpu().numpy() for layer in attentions])
            used_stock_codes.append(str(stock_code).zfill(6))
        if (time.time() - start_ts) > float(args.max_seconds):
            break

    if not attentions_accum:
        raise RuntimeError("未采集到有效注意力矩阵。")

    attentions: list[torch.Tensor] = []
    layer_count = len(attentions_accum[0])
    min_seq_len = min(sample[0].shape[-1] for sample in attentions_accum)
    for layer_index in range(layer_count):
        cropped = [sample[layer_index][:, :, :min_seq_len, :min_seq_len] for sample in attentions_accum]
        merged = np.concatenate(cropped, axis=0)
        attentions.append(torch.tensor(merged, dtype=torch.float32))

    head_summaries: list[dict[str, object]] = []
    head_entropies: list[float] = []
    normalized_attentions: list[np.ndarray] = []
    for layer_index, layer_attention in enumerate(attentions):
        layer_attention_np = layer_attention.detach().cpu().numpy()
        normalized = _normalize_rows(layer_attention_np)
        normalized_attentions.append(normalized)
        per_head_entropy = [float(np.mean([_entropy(step) for step in head_matrix.reshape(-1, head_matrix.shape[-1])])) for head_matrix in normalized[0]]
        layer_mean = normalized.mean(axis=0)
        head_summary = {
            "layer_index": int(layer_index),
            "head_count": int(layer_mean.shape[0]),
            "head_entropy": per_head_entropy,
            "head_entropy_mean": float(np.mean(per_head_entropy)) if per_head_entropy else 0.0,
        }
        head_summaries.append(head_summary)
        head_entropies.extend(per_head_entropy)

    redundancy_rows: list[dict[str, object]] = []
    if normalized_attentions:
        base_layer = normalized_attentions[0][0]
        for head_i in range(base_layer.shape[0]):
            for head_j in range(head_i + 1, base_layer.shape[0]):
                head_i_matrix = base_layer[head_i].reshape(-1, base_layer.shape[-1]).mean(axis=0)
                head_j_matrix = base_layer[head_j].reshape(-1, base_layer.shape[-1]).mean(axis=0)
                redundancy_rows.append({
                    "layer_index": 0,
                    "head_i": int(head_i),
                    "head_j": int(head_j),
                    "js_divergence": float(_js_divergence(head_i_matrix, head_j_matrix)),
                })

    sample_frame = sample_frame_all[sample_frame_all["stock_code"].astype(str).isin(used_stock_codes)].copy()
    focus = _attention_feature_focus(normalized_attentions[0] if normalized_attentions else np.array([]), sample_frame, feature_cols)
    mean_entropy = float(np.mean(head_entropies)) if head_entropies else 0.0
    mean_js = float(np.mean([row["js_divergence"] for row in redundancy_rows])) if redundancy_rows else 0.0
    diagnosis = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path),
        "stock_codes": used_stock_codes,
        "sample_rows": int(sample_rows_total),
        "layer_count": int(len(attentions)),
        "head_summaries": head_summaries,
        "redundancy": redundancy_rows,
        "feature_focus": focus,
        "overall": {
            "mean_head_entropy": mean_entropy,
            "mean_js_divergence": mean_js,
            "specialization_hint": "good" if mean_entropy > 0.5 and mean_js < 0.5 else "weak" if mean_entropy < 0.2 and mean_js > 0.7 else "mixed",
            "runtime_seconds": float(time.time() - start_ts),
            "aligned_sequence_length": int(min_seq_len),
        },
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "mean_head_entropy": mean_entropy, "mean_js_divergence": mean_js}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()