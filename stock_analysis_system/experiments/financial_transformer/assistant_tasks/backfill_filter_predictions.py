"""对 Filter 模型执行 2026-03-05 ~ 2026-06-04 的历史区间预测。

每到一个交易日，只使用截至当天的特征来做预测（模拟实盘逐日预测），
然后拼接已有的 test_predictions.csv（1月5日-3月4日），形成完整的预测序列。

输出：backfill_predictions.csv (完整序列)
"""

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

ROOT = Path("d:/qcr/Stock/stock_analysis_system")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/financial_transformer/src"))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import build_feature_frame
from financial_transformer_experiment.model import create_model

CONFIG_PATH = ROOT / "experiments/financial_transformer/config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml"
OUTPUT_DIR = ROOT / "experiments/financial_transformer/results/formal_filtered_v5_e80_bsstop_seed33"

START_DATE = "2026-03-05"
END_DATE = "2026-06-04"

config = load_config(str(CONFIG_PATH))

# 加载 checkpoint
checkpoint_path = config.artifacts_dir / "financial_transformer.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
feature_cols = checkpoint["feature_cols"]
seq_len = int(checkpoint.get("seq_len", config.model.seq_len))
min_seq_len = int(checkpoint.get("min_seq_len", config.model.min_seq_len or seq_len))
model_config = checkpoint.get("model_config", {})

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

# 构建全量特征
print("Building feature frame...")
feature_df = build_feature_frame(config)
feature_df = feature_df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
print(f"Feature frame: {len(feature_df)} rows, {feature_df['stock_code'].nunique()} stocks")
print(f"Date range: {feature_df['date'].min()} ~ {feature_df['date'].max()}")

# 生成交易日序列
all_dates = sorted(feature_df["date"].dropna().unique())
target_dates = [d for d in all_dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]
print(f"Target dates: {len(target_dates)} trading days ({target_dates[0]} ~ {target_dates[-1]})")

rows = []
skipped = 0

with torch.no_grad():
    for i, target_date in enumerate(target_dates):
        # 只使用截至 target_date 的特征
        cut_df = feature_df[feature_df["date"] <= target_date].copy()

        for stock_code, group in cut_df.groupby("stock_code"):
            group = group.sort_values("date").reset_index(drop=True)
            usable = group.dropna(subset=feature_cols)
            if len(usable) < min_seq_len:
                skipped += 1
                continue
            latest_window = usable[feature_cols].tail(seq_len).to_numpy(dtype=np.float32)
            if not np.isfinite(latest_window).all():
                skipped += 1
                continue

            x = torch.tensor(latest_window[None, :, :], dtype=torch.float32, device=device)
            outputs = model(x)

            pred_short_up = float(torch.sigmoid(outputs["short_up_logits"]).item())
            pred_take_profit = float(torch.sigmoid(outputs["take_profit_logits"]).item())
            pred_stop_loss = float(torch.sigmoid(outputs["stop_loss_logits"]).item())
            pred_max_gain = float(outputs["max_gain"].item())
            pred_max_drawdown = float(outputs["max_drawdown"].item())

            rows.append({
                "stock_code": stock_code,
                "date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
                "pred_short_up": pred_short_up,
                "pred_take_profit": pred_take_profit,
                "pred_stop_loss": pred_stop_loss,
                "pred_max_gain": pred_max_gain,
                "pred_max_drawdown": pred_max_drawdown,
            })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(target_dates)}] {target_date} - predicted {len(rows)} rows so far")

backfill_df = pd.DataFrame(rows)
print(f"\nBackfill complete: {len(backfill_df)} predictions, {skipped} skipped")

# 读取已有的 test_predictions（1月5日-3月4日）
test_path = OUTPUT_DIR / "test_predictions.csv"
test_df = pd.read_csv(test_path, dtype={"stock_code": str})
test_df["date"] = pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")
test_cols = ["stock_code", "date", "pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown"]
test_df = test_df[test_cols]

# 拼接：test (1/5-3/4) + 先前的预测 (3/5~6/4)
combined = pd.concat([test_df, backfill_df], ignore_index=True)
combined = combined.sort_values(["date", "stock_code"]).reset_index(drop=True)

out_path = OUTPUT_DIR / "backfill_predictions.csv"
combined.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nCombined predictions saved to: {out_path}")
print(f"  Total rows: {len(combined)}")
print(f"  Date range: {combined['date'].min()} ~ {combined['date'].max()}")
print(f"  Unique dates: {combined['date'].nunique()}")
print(f"  Unique stocks: {combined['stock_code'].nunique()}")
