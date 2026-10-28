from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .features import build_layer_features, layer_from_market_cap_yi
from .logging_utils import setup_logger
from .multitask_models import MultiTaskForecastModel
from .multitask_training import _feature_columns, MULTITASK_LAYERS

logger = setup_logger("hierarchical.multitask_inference")


@dataclass
class MultiTaskLayerArtifact:
    layer: str
    model: MultiTaskForecastModel
    feature_cols: list[str]
    lookback: int


class ProductionMultiTaskInferenceEngine:
    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)
        self.device = torch.device(
            self.config.training.device if torch.cuda.is_available() or self.config.training.device == "cpu" else "cpu"
        )
        self.artifacts = self._load_artifacts()

    def _load_artifacts(self) -> dict[str, MultiTaskLayerArtifact]:
        artifacts: dict[str, MultiTaskLayerArtifact] = {}
        for layer in MULTITASK_LAYERS:
            model_path = self.config.models_dir / f"multitask_{layer}_model.pt"
            if not model_path.exists():
                logger.warning("未找到多任务模型文件: %s", model_path)
                continue
            checkpoint = torch.load(model_path, map_location=self.device)
            feature_cols = checkpoint.get("feature_cols", _feature_columns(layer=layer))
            lookback = int(checkpoint.get("lookback", self.config.training.lookback_window))
            hidden_dim = int(checkpoint.get("hidden_dim", self.config.multitask.hidden_dim))
            dropout = float(checkpoint.get("dropout", self.config.multitask.dropout))
            model = MultiTaskForecastModel(input_dim=len(feature_cols), hidden_dim=hidden_dim, dropout=dropout).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            artifacts[layer] = MultiTaskLayerArtifact(layer=layer, model=model, feature_cols=feature_cols, lookback=lookback)
        if not artifacts:
            raise RuntimeError("未找到任何已训练的分层多任务模型，请先运行 scripts/train_multitask_model.py --all-layers。")
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
            df = pd.read_sql_query("SELECT * FROM stock_data WHERE stock_code = ? ORDER BY 日期", conn, params=[stock_code])
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
        for column in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
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
        return df.tail(min_rows).reset_index(drop=True)

    @staticmethod
    def _strategy_score(short_up: float, take_profit: float, stop_loss: float, max_gain: float, max_drawdown: float) -> float:
        gain_term = max(min(max_gain, 0.5), 0.0)
        drawdown_penalty = min(abs(max_drawdown), 0.3)
        return 0.45 * short_up + 0.35 * take_profit + 0.20 * (1.0 - stop_loss) + 0.10 * gain_term - 0.05 * drawdown_penalty

    def predict_latest(self, top_n: int = 50, min_history: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        caps = self._load_latest_market_caps()
        rows: list[dict[str, Any]] = []
        min_history = min_history or max(artifact.lookback for artifact in self.artifacts.values()) + 80

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
                    outputs = artifact.model(x)

                pred_short_up = float(torch.sigmoid(outputs["short_up_logits"]).item())
                pred_take_profit = float(torch.sigmoid(outputs["take_profit_logits"]).item())
                pred_stop_loss = float(torch.sigmoid(outputs["stop_loss_logits"]).item())
                pred_max_gain = float(outputs["max_gain"].item())
                pred_max_drawdown = float(outputs["max_drawdown"].item())
                latest = feat_df.iloc[-1]
                strategy_a_pass = pred_short_up >= 0.60 and pred_take_profit >= 0.40 and pred_stop_loss <= 0.20
                strategy_score = self._strategy_score(pred_short_up, pred_take_profit, pred_stop_loss, pred_max_gain, pred_max_drawdown)
                risk_reward_ratio = float(max(pred_max_gain, 0.0) / max(abs(pred_max_drawdown), 0.01))

                rows.append(
                    {
                        "stock_code": stock_code,
                        "date": str(pd.to_datetime(latest["date"]).date()),
                        "layer": layer,
                        "market_cap_yi": market_cap_yi,
                        "pred_short_up": pred_short_up,
                        "pred_take_profit": pred_take_profit,
                        "pred_stop_loss": pred_stop_loss,
                        "pred_max_gain": pred_max_gain,
                        "pred_max_drawdown": pred_max_drawdown,
                        "strategy_score": strategy_score,
                        "strategy_a_pass": bool(strategy_a_pass),
                        "risk_reward_ratio": risk_reward_ratio,
                        "close": float(latest["close"]),
                        "turnover_rate": float(latest.get("turnover_rate", 0.0) or 0.0),
                        "amount": float(latest.get("amount", 0.0) or 0.0),
                        "trend_strength": float(latest.get("trend_strength", 0.0) or 0.0),
                    }
                )
            except Exception as exc:
                logger.warning("多任务最新推理失败 %s: %s", stock_code, exc)

        result = pd.DataFrame(rows)
        if result.empty:
            raise RuntimeError("没有生成任何多任务最新预测结果。")

        result = result.sort_values(
            ["strategy_a_pass", "strategy_score", "pred_take_profit", "risk_reward_ratio", "amount"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        strategy_view = result[result["strategy_a_pass"]].reset_index(drop=True)

        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.config.results_dir / "latest_multitask_picks.csv", index=False, encoding="utf-8-sig")
        strategy_view.to_csv(self.config.results_dir / "latest_multitask_strategy_a.csv", index=False, encoding="utf-8-sig")
        return result.head(top_n), strategy_view.head(top_n)