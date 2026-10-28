from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .features import LAYER_FEATURES, build_layer_features, layer_from_market_cap_yi
from .logging_utils import setup_logger
from .models import build_model

logger = setup_logger("hierarchical.inference")


@dataclass
class LayerArtifact:
    layer: str
    model: torch.nn.Module
    feature_cols: list[str]
    lookback: int


class ProductionInferenceEngine:
    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)
        self.device = torch.device(
            self.config.training.device if torch.cuda.is_available() or self.config.training.device == "cpu" else "cpu"
        )
        self.artifacts = self._load_artifacts()

    def _load_artifacts(self) -> dict[str, LayerArtifact]:
        artifacts: dict[str, LayerArtifact] = {}
        for layer in ["micro", "small", "mid", "large"]:
            model_path = self.config.models_dir / f"{layer}_model.pt"
            if not model_path.exists():
                logger.warning("未找到模型文件: %s", model_path)
                continue
            checkpoint = torch.load(model_path, map_location=self.device)
            feature_cols = checkpoint.get("feature_cols", LAYER_FEATURES[layer])
            lookback = int(checkpoint.get("lookback", self.config.training.lookback_window))
            model = build_model(layer, len(feature_cols)).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            artifacts[layer] = LayerArtifact(layer=layer, model=model, feature_cols=feature_cols, lookback=lookback)
        if not artifacts:
            raise RuntimeError("未找到任何已训练模型，请先运行分层训练脚本。")
        return artifacts

    def _load_latest_market_caps(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.config.databases.market_cap_db)
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()]
            date_col = "trade_date" if "trade_date" in cols else "update_date" if "update_date" in cols else None
            select_cols = ["stock_code", "total_mv"]
            if date_col:
                select_cols.append(f"{date_col} AS trade_date")
            df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM market_caps", conn)
        finally:
            conn.close()
        sort_cols = ["stock_code"] + (["trade_date"] if "trade_date" in df.columns else [])
        df = df.sort_values(sort_cols).drop_duplicates("stock_code", keep="last")
        df["market_cap_yi"] = df["total_mv"].astype(float) / 10000.0
        return df[["stock_code", "market_cap_yi"]]

    def _load_trade_data(self, stock_code: str, min_rows: int) -> pd.DataFrame:
        conn = sqlite3.connect(self.config.databases.trade_db)
        try:
            query = "SELECT * FROM stock_data WHERE stock_code = ? ORDER BY 日期"
            df = pd.read_sql_query(query, conn, params=[stock_code])
            # 真实换手率（修正股权结构污染，见 free_float.py）
            try:
                rto_df = pd.read_sql_query(
                    "SELECT stock_code, date, real_turnover_rate "
                    "FROM real_turnover_rate WHERE stock_code = ?",
                    conn,
                    params=[stock_code],
                )
            except Exception:
                rto_df = pd.DataFrame()
        finally:
            conn.close()

        if df.empty:
            return df

        rename_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
        }
        df = df.rename(columns=rename_map)
        keep = [c for c in ["stock_code", "date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"] if c in df.columns]
        df = df[keep].copy()
        for col in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_turnover_mask = pd.Series(False, index=df.index)
        if "volume" in df.columns:
            missing_turnover_mask = df["volume"].isna()
            if "amount" in df.columns:
                missing_turnover_mask = missing_turnover_mask & df["amount"].isna()
            df.loc[missing_turnover_mask, "volume"] = 0.0
        if "amount" in df.columns:
            df["amount"] = df["amount"].fillna(df["close"] * df["volume"])
        else:
            df["amount"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
        if "turnover_rate" not in df.columns:
            df["turnover_rate"] = np.nan
        df.loc[df["turnover_rate"].isna() & missing_turnover_mask, "turnover_rate"] = 0.0
        df["date"] = pd.to_datetime(df["date"])
        # 真实换手率：用修正后的真实换手率替代名义换手率（缺失回退名义）
        if not rto_df.empty:
            rto_df = rto_df.rename(columns={"date": "_rto_date"})
            rto_df["_rto_date"] = pd.to_datetime(rto_df["_rto_date"], errors="coerce")
            rto_df["real_turnover_rate"] = pd.to_numeric(rto_df["real_turnover_rate"], errors="coerce")
            df = df.merge(
                rto_df[["stock_code", "_rto_date", "real_turnover_rate"]],
                left_on=["stock_code", "date"],
                right_on=["stock_code", "_rto_date"],
                how="left",
            ).drop(columns=["_rto_date"])
            df["turnover_rate"] = df["real_turnover_rate"].fillna(df["turnover_rate"])
        return df.tail(min_rows).reset_index(drop=True)

    def predict_latest(self, top_n: int = 50, min_history: int | None = None) -> pd.DataFrame:
        caps = self._load_latest_market_caps()
        rows: list[dict[str, Any]] = []
        min_history = min_history or max(a.lookback for a in self.artifacts.values()) + 80

        for _, cap_row in caps.iterrows():
            stock_code = str(cap_row["stock_code"])
            market_cap_yi = float(cap_row["market_cap_yi"])
            layer = layer_from_market_cap_yi(market_cap_yi)
            artifact = self.artifacts.get(layer)
            if artifact is None:
                continue
            try:
                price_df = self._load_trade_data(stock_code, min_history)
                if price_df.empty or len(price_df) < artifact.lookback:
                    continue
                feat_df = build_layer_features(price_df, market_cap_yi).replace([np.inf, -np.inf], np.nan)
                feat_df = feat_df.dropna().reset_index(drop=True)
                if len(feat_df) < artifact.lookback:
                    continue
                latest_window = feat_df[artifact.feature_cols].tail(artifact.lookback)
                if latest_window.isna().any().any() or not np.isfinite(latest_window.to_numpy(dtype="float32")).all():
                    continue
                x = torch.tensor(latest_window.to_numpy(dtype="float32")[None, :, :], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    prob = torch.sigmoid(artifact.model(x)).item()
                latest = feat_df.iloc[-1]
                rows.append({
                    "stock_code": stock_code,
                    "date": str(pd.to_datetime(latest["date"]).date()),
                    "layer": layer,
                    "market_cap_yi": market_cap_yi,
                    "prediction_prob": float(prob),
                    "close": float(latest["close"]),
                    "turnover_rate": float(latest.get("turnover_rate", latest.get("turnover_rate_est", 0.0)) or 0.0),
                    "amount": float(latest.get("amount", 0.0) or 0.0),
                    "trend_strength": float(latest.get("trend_strength", 0.0) or 0.0),
                })
            except Exception as exc:
                logger.warning("最新推理失败 %s: %s", stock_code, exc)

        result = pd.DataFrame(rows)
        if result.empty:
            raise RuntimeError("没有生成任何最新预测结果。")

        layer_weight = pd.Series(self.config.ensemble.weights)
        result["ensemble_score"] = result["prediction_prob"] * result["layer"].map(layer_weight).fillna(0.0)
        result = result.sort_values(["ensemble_score", "prediction_prob", "amount"], ascending=False).reset_index(drop=True)
        output_file = self.config.results_dir / "latest_picks.csv"
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False, encoding="utf-8-sig")
        return result.head(top_n)
