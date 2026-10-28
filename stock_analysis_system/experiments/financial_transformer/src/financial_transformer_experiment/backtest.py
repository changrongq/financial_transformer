from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .calibration import apply_probability_calibration
from .calibration import calibrated_test_predictions_path
from .calibration import load_probability_calibration
from .config import ExperimentConfig
from .scoring import enrich_prediction_frame, strategy_score_frame


AGENT_PROXY_DAILY_PATH = Path(__file__).resolve().parents[2] / "results" / "agent_proxy_market_state_report" / "agent_proxy_daily.csv"


@dataclass
class BacktestResult:
    summary: dict
    trades: pd.DataFrame
    equity: pd.DataFrame
    signals: pd.DataFrame
    state_plan: pd.DataFrame | None = None


@dataclass(frozen=True)
class StatefulExecutionPlan:
    date: pd.Timestamp
    state_name: str
    state_reason: str
    use_soft_context_score: bool
    allow_entries: bool
    top_k_per_day: int
    min_score_ratio_to_leader: float
    execution_hold_days: int
    execution_take_profit: float
    execution_stop_loss: float
    leader_max_day_rank: int | None
    leader_take_profit: float | None
    leader_stop_loss: float | None
    leader_hold_days: int | None
    institution_activity_apr_z: float
    margin_trend_activity_apr_z: float
    crowding_activity_apr_z: float
    soft_context_defensive_pressure: float
    soft_context_supportive_pressure: float


class TransformerStrategyBacktester:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._agent_proxy_daily_cache: pd.DataFrame | None = None

    @staticmethod
    def _strategy_score_frame(df: pd.DataFrame) -> pd.Series:
        return strategy_score_frame(df)

    def _load_predictions(
        self,
        start_date: str | None,
        end_date: str | None,
        short_up_threshold: float,
        take_profit_threshold: float,
        stop_loss_threshold: float,
        top_k_per_day: int | None = None,
        use_calibration: bool = False,
        min_secondary_primary_ratio: float = 0.0,
        min_score_ratio_to_leader: float = 0.0,
        min_filter_column: str | None = None,
        min_filter_value: float | None = None,
    ) -> pd.DataFrame:
        predictions_path = calibrated_test_predictions_path(self.config.results_dir) if use_calibration else self.config.results_dir / "test_predictions.csv"
        if not predictions_path.exists():
            raise RuntimeError("未找到 test_predictions.csv，请先运行 Transformer 训练。")

        df = pd.read_csv(predictions_path, dtype={"stock_code": str})
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"])
        required_cols = [
            "pred_short_up",
            "pred_take_profit",
            "pred_stop_loss",
            "pred_max_gain",
            "pred_max_drawdown",
        ]
        for column in required_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["date", *required_cols]).copy()
        if use_calibration and "raw_pred_short_up" not in df.columns:
            df = apply_probability_calibration(df, load_probability_calibration(self.config.results_dir))
        df = enrich_prediction_frame(df, exit_label_mode=self.config.multitask.exit_label_mode)
        if start_date is not None:
            df = df[df["date"] >= pd.Timestamp(start_date)].copy()
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)].copy()
        if df.empty:
            raise RuntimeError("指定区间内没有可用于回测的 Transformer 多任务预测。")

        if "layer" not in df.columns:
            df["layer"] = self.config.experiment.layer_filter or "unknown"
        df["signal_score"] = strategy_score_frame(df, exit_label_mode=self.config.multitask.exit_label_mode)
        df["strategy_a_pass"] = (
            (df["pred_short_up"] > short_up_threshold)
            & (df["pred_take_profit"] > take_profit_threshold)
            & (df["pred_stop_loss"] < stop_loss_threshold)
        )
        if min_filter_column and min_filter_value is not None:
            if min_filter_column not in df.columns:
                raise RuntimeError(f"预测文件缺少过滤列: {min_filter_column}")
            df[min_filter_column] = pd.to_numeric(df[min_filter_column], errors="coerce")
            df["strategy_a_pass"] = df["strategy_a_pass"] & (df[min_filter_column] >= float(min_filter_value))
        if min_secondary_primary_ratio > 0.0:
            if {"primary_signal_score", "secondary_signal_score"}.issubset(df.columns):
                primary_score = pd.to_numeric(df["primary_signal_score"], errors="coerce")
                secondary_score = pd.to_numeric(df["secondary_signal_score"], errors="coerce")
                ratio = secondary_score.div(primary_score.where(primary_score.abs() > 1e-9))
                df["secondary_primary_ratio"] = ratio
                df["strategy_a_pass"] = df["strategy_a_pass"] & (ratio >= min_secondary_primary_ratio)
            else:
                df["secondary_primary_ratio"] = np.nan
        df = df[df["strategy_a_pass"]].copy()
        if min_score_ratio_to_leader > 0.0 and not df.empty:
            leader_score = df.groupby("date")["signal_score"].transform("max")
            df["score_ratio_to_leader"] = df["signal_score"].div(leader_score.where(leader_score.abs() > 1e-9))
            df = df[df["score_ratio_to_leader"] >= min_score_ratio_to_leader].copy()
        if top_k_per_day is not None and top_k_per_day > 0 and not df.empty:
            df = (
                df.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
                .groupby("date", group_keys=False)
                .head(top_k_per_day)
                .reset_index(drop=True)
            )
        return df.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)

    def _load_price_history(self, stock_code: str) -> pd.DataFrame:
        stock_code = str(stock_code).zfill(6)
        if stock_code in self._price_cache:
            return self._price_cache[stock_code]
        conn = sqlite3.connect(self.config.databases.trade_db)
        try:
            df = pd.read_sql_query(
                "SELECT stock_code, 日期, 开盘, 最高, 最低, 收盘 FROM stock_data WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[stock_code],
            )
        finally:
            conn.close()
        if df.empty:
            result = pd.DataFrame(columns=["stock_code", "date", "open", "high", "low", "close"])
        else:
            result = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"})
            result["date"] = pd.to_datetime(result["date"])
            for column in ["open", "high", "low", "close"]:
                result[column] = pd.to_numeric(result[column], errors="coerce")
            result = result.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
        self._price_cache[stock_code] = result
        return result

    def _daily_bar(self, stock_code: str, current_date: pd.Timestamp) -> pd.Series | None:
        prices = self._load_price_history(stock_code)
        bar = prices[prices["date"] == current_date]
        if bar.empty:
            return None
        return bar.iloc[0]

    def _next_trade_date(self, stock_code: str, signal_date: pd.Timestamp) -> pd.Timestamp | None:
        prices = self._load_price_history(stock_code)
        future = prices.loc[prices["date"] > signal_date, "date"]
        if future.empty:
            return None
        return pd.Timestamp(future.iloc[0])

    def _load_market_state_daily_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame(columns=["date"])
        stock_codes = sorted({str(code).zfill(6) for code in predictions["stock_code"].dropna().astype(str)})
        if not stock_codes:
            return pd.DataFrame(columns=["date"])
        start_date = (pd.Timestamp(predictions["date"].min()) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(predictions["date"].max()).strftime("%Y-%m-%d")
        frames: list[pd.DataFrame] = []
        conn = sqlite3.connect(self.config.databases.trade_db)
        try:
            for start_index in range(0, len(stock_codes), 400):
                batch_codes = stock_codes[start_index:start_index + 400]
                placeholders = ",".join(["?"] * len(batch_codes))
                query = (
                    "SELECT stock_code, 日期, 收盘, 换手率 "
                    f"FROM stock_data WHERE stock_code IN ({placeholders}) AND 日期 >= ? AND 日期 <= ? ORDER BY stock_code, 日期"
                )
                frame = pd.read_sql_query(query, conn, params=[*batch_codes, start_date, end_date])
                if not frame.empty:
                    frames.append(frame)
        finally:
            conn.close()
        if not frames:
            return pd.DataFrame(columns=["date"])
        market = pd.concat(frames, ignore_index=True)
        market["date"] = pd.to_datetime(market["日期"], errors="coerce")
        market["close"] = pd.to_numeric(market["收盘"], errors="coerce")
        market["turnover_rate"] = pd.to_numeric(market["换手率"], errors="coerce")
        market = market.dropna(subset=["date", "close"]).copy()
        market["stock_code"] = market["stock_code"].astype(str).str.zfill(6)
        market = market.sort_values(["stock_code", "date"]).reset_index(drop=True)
        # 真实换手率：用修正后的真实换手率替代名义换手率（regime 截面均值同样受股权结构污染）
        try:
            rto = pd.read_sql_query(
                "SELECT stock_code, date, real_turnover_rate FROM real_turnover_rate",
                sqlite3.connect(self.config.databases.trade_db),
            )
            if not rto.empty:
                rto["stock_code"] = rto["stock_code"].astype(str).str.zfill(6)
                rto["date"] = pd.to_datetime(rto["date"], errors="coerce")
                rto["real_turnover_rate"] = pd.to_numeric(rto["real_turnover_rate"], errors="coerce")
                market = market.merge(
                    rto[["stock_code", "date", "real_turnover_rate"]],
                    on=["stock_code", "date"], how="left",
                )
                market["turnover_rate"] = market["real_turnover_rate"].fillna(market["turnover_rate"])
        except Exception:
            pass
        market["return_1d"] = market.groupby("stock_code")["close"].pct_change()
        daily = market.groupby("date", as_index=False).agg(
            regime_cross_return_mean=("return_1d", "mean"),
            regime_cross_return_vol=("return_1d", "std"),
            regime_cross_up_ratio=("return_1d", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            regime_cross_turnover_mean=("turnover_rate", "mean"),
        )
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["regime_cross_return_mean"] = daily["regime_cross_return_mean"].fillna(0.0)
        daily["regime_cross_return_vol"] = daily["regime_cross_return_vol"].fillna(0.0)
        daily["regime_cross_up_ratio"] = daily["regime_cross_up_ratio"].fillna(0.0)
        daily["regime_cross_turnover_mean"] = daily["regime_cross_turnover_mean"].ffill().fillna(0.0)
        daily["regime_breadth_5d"] = daily["regime_cross_up_ratio"].rolling(5, min_periods=3).mean()
        daily["regime_return_mean_5d"] = daily["regime_cross_return_mean"].rolling(5, min_periods=3).mean()
        daily["regime_return_vol_5d"] = daily["regime_cross_return_vol"].rolling(5, min_periods=3).mean()
        turnover_mean_5d = daily["regime_cross_turnover_mean"].rolling(5, min_periods=3).mean()
        turnover_mean_20d = daily["regime_cross_turnover_mean"].rolling(20, min_periods=5).mean()
        daily["regime_turnover_ratio_5_20"] = turnover_mean_5d.div(turnover_mean_20d.replace(0.0, np.nan))
        return daily[[
            "date",
            "regime_breadth_5d",
            "regime_return_mean_5d",
            "regime_return_vol_5d",
            "regime_turnover_ratio_5_20",
        ]].copy()

    def _load_agent_proxy_daily_metrics(self) -> pd.DataFrame:
        if self._agent_proxy_daily_cache is not None:
            return self._agent_proxy_daily_cache.copy()
        if not AGENT_PROXY_DAILY_PATH.exists():
            empty = pd.DataFrame(columns=[
                "date",
                "institution_activity",
                "margin_trend_activity",
                "crowding_activity",
                "institution_activity_apr_z",
                "margin_trend_activity_apr_z",
                "crowding_activity_apr_z",
            ])
            self._agent_proxy_daily_cache = empty
            return empty.copy()

        proxy = pd.read_csv(AGENT_PROXY_DAILY_PATH)
        keep_cols = ["date", "institution_activity", "margin_trend_activity", "crowding_activity"]
        proxy = proxy[keep_cols].copy()
        proxy["date"] = pd.to_datetime(proxy["date"], errors="coerce")
        for column in keep_cols[1:]:
            proxy[column] = pd.to_numeric(proxy[column], errors="coerce")
        proxy = proxy.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        baseline = proxy[(proxy["date"] >= pd.Timestamp("2026-04-01")) & (proxy["date"] <= pd.Timestamp("2026-04-30"))].copy()
        if baseline.empty:
            baseline = proxy.copy()
        for column in ["institution_activity", "margin_trend_activity", "crowding_activity"]:
            baseline_mean = float(baseline[column].mean()) if column in baseline else 0.0
            baseline_std = float(baseline[column].std(ddof=0)) if column in baseline else 0.0
            if baseline_std <= 1e-9 or np.isnan(baseline_std):
                proxy[f"{column}_apr_z"] = 0.0
            else:
                proxy[f"{column}_apr_z"] = (proxy[column] - baseline_mean) / baseline_std

        self._agent_proxy_daily_cache = proxy
        return proxy.copy()

    @staticmethod
    def _soft_context_components(
        institution_activity_apr_z: float,
        margin_trend_activity_apr_z: float,
        crowding_activity_apr_z: float,
    ) -> tuple[float, float, float]:
        institution_weakness = float(np.clip(-float(institution_activity_apr_z or 0.0), 0.0, 3.0) / 3.0)
        margin_support = float(np.clip(float(margin_trend_activity_apr_z or 0.0), 0.0, 3.0) / 3.0)
        crowding_pressure = float(np.clip(float(crowding_activity_apr_z or 0.0), 0.0, 3.0) / 3.0)
        return institution_weakness, margin_support, crowding_pressure

    def _position_weight(
        self,
        row: pd.Series,
        max_signal_score: float,
        min_position_weight: float,
        max_position_weight: float,
    ) -> float:
        edge = max(float(row["pred_max_gain"]), 0.02) * max(float(row["pred_take_profit"]), 0.01) * max(float(row["pred_short_up"]), 0.01)
        risk = max(abs(float(row["pred_max_drawdown"])), 0.02) * max(float(row["pred_stop_loss"]), 0.05)
        normalized = min((edge / risk) / max(max_signal_score, 1e-6), 1.0)
        if max_position_weight <= 0:
            return 0.0
        floor_weight = max(min_position_weight, 0.0)
        cap_weight = max(max_position_weight, floor_weight)
        return min(cap_weight, floor_weight + max(cap_weight - floor_weight, 0.0) * normalized)

    def _prepare_signals(
        self,
        predictions: pd.DataFrame,
        initial_capital: float,
        min_position_weight: float,
        max_position_weight: float,
        rank_weight_power: float,
    ) -> pd.DataFrame:
        if predictions.empty:
            return predictions
        prepared = []
        for signal_date, group in predictions.groupby("date", sort=True):
            ordered_group = group.sort_values(["signal_score", "pred_take_profit"], ascending=[False, False]).reset_index(drop=True)
            max_signal_score = float(ordered_group["signal_score"].max())
            day_rows: list[dict] = []
            for rank_index, (_, row) in enumerate(ordered_group.iterrows(), start=1):
                base_target_weight = self._position_weight(
                    row,
                    max_signal_score=max_signal_score,
                    min_position_weight=min_position_weight,
                    max_position_weight=max_position_weight,
                )
                rank_weight_multiplier = 1.0 / (float(rank_index) ** max(rank_weight_power, 0.0))
                day_rows.append(
                    {
                        **row.to_dict(),
                        "signal_date": pd.Timestamp(signal_date),
                        "day_rank": rank_index,
                        "base_target_weight": base_target_weight,
                        "rank_weight_multiplier": rank_weight_multiplier,
                    }
                )

            base_weight_sum = float(sum(item["base_target_weight"] for item in day_rows))
            tilted_weight_sum = float(sum(item["base_target_weight"] * item["rank_weight_multiplier"] for item in day_rows))
            normalization = base_weight_sum / tilted_weight_sum if tilted_weight_sum > 0 else 1.0

            for item in day_rows:
                target_weight = item["base_target_weight"] * item["rank_weight_multiplier"] * normalization
                prepared.append(
                    {
                        **item,
                        "target_weight": target_weight,
                        "target_amount": initial_capital * target_weight,
                    }
                )
        return pd.DataFrame(prepared).sort_values(["signal_date", "signal_score"], ascending=[True, False]).reset_index(drop=True)

    def _build_stateful_execution_plan(
        self,
        predictions: pd.DataFrame,
        state_profile: str,
        fallback_top_k_per_day: int | None,
        fallback_min_score_ratio_to_leader: float,
        fallback_hold_days: int,
        fallback_take_profit: float,
        fallback_stop_loss: float,
        fallback_leader_max_day_rank: int | None,
        fallback_leader_take_profit: float | None,
        fallback_leader_stop_loss: float | None,
        fallback_leader_hold_days: int | None,
    ) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame()
        if state_profile not in {"market_state_gate_v1", "market_state_gate_v2", "market_state_soft_context_v1"}:
            raise RuntimeError(f"未知 state_profile: {state_profile}")

        metrics_rows = []
        for signal_date, group in predictions.groupby("date", sort=True):
            ordered = group.sort_values(["signal_score", "pred_take_profit"], ascending=[False, False]).reset_index(drop=True)
            top_scores = ordered["signal_score"].tolist()
            metrics_rows.append(
                {
                    "date": pd.Timestamp(signal_date),
                    "pass_count": int(len(ordered)),
                    "top1_score": float(top_scores[0]),
                    "top1_pred_take_profit": float(ordered.iloc[0]["pred_take_profit"]),
                    "top1_pred_stop_loss": float(ordered.iloc[0]["pred_stop_loss"]),
                    "score_gap_top1_top2": float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else float(top_scores[0]),
                }
            )
        metrics = pd.DataFrame(metrics_rows)
        market_daily = self._load_market_state_daily_metrics(predictions)
        if not market_daily.empty:
            metrics = metrics.merge(market_daily, on="date", how="left")
        else:
            metrics["regime_breadth_5d"] = np.nan
            metrics["regime_return_mean_5d"] = np.nan
            metrics["regime_return_vol_5d"] = np.nan
            metrics["regime_turnover_ratio_5_20"] = np.nan
        proxy_daily = self._load_agent_proxy_daily_metrics()
        if not proxy_daily.empty:
            metrics = metrics.merge(proxy_daily, on="date", how="left")
        else:
            metrics["institution_activity_apr_z"] = np.nan
            metrics["margin_trend_activity_apr_z"] = np.nan
            metrics["crowding_activity_apr_z"] = np.nan

        q_pass_low = float(metrics["pass_count"].quantile(0.25))
        q_pass_high = float(metrics["pass_count"].quantile(0.75))
        q_score_low = float(metrics["top1_score"].quantile(0.35))
        q_score_high = float(metrics["top1_score"].quantile(0.70))
        q_gap_low = float(metrics["score_gap_top1_top2"].quantile(0.35))
        q_gap_high = float(metrics["score_gap_top1_top2"].quantile(0.65))
        q_breadth_low = float(metrics["regime_breadth_5d"].dropna().quantile(0.35)) if metrics["regime_breadth_5d"].notna().any() else 0.45
        q_breadth_high = float(metrics["regime_breadth_5d"].dropna().quantile(0.70)) if metrics["regime_breadth_5d"].notna().any() else 0.55
        q_return_low = float(metrics["regime_return_mean_5d"].dropna().quantile(0.35)) if metrics["regime_return_mean_5d"].notna().any() else 0.0
        q_return_high = float(metrics["regime_return_mean_5d"].dropna().quantile(0.70)) if metrics["regime_return_mean_5d"].notna().any() else 0.0
        q_turnover_high = float(metrics["regime_turnover_ratio_5_20"].dropna().quantile(0.70)) if metrics["regime_turnover_ratio_5_20"].notna().any() else 1.0
        default_top_k = int(top_k_per_day) if (top_k_per_day := fallback_top_k_per_day) is not None else 1
        default_leader_rank = int(fallback_leader_max_day_rank) if fallback_leader_max_day_rank is not None else None

        plans: list[StatefulExecutionPlan] = []
        for row in metrics.itertuples(index=False):
            state_name = "neutral"
            state_reason = "default"
            allow_entries = True
            top_k_per_day = default_top_k
            min_score_ratio_to_leader = float(fallback_min_score_ratio_to_leader)
            execution_hold_days = int(fallback_hold_days)
            execution_take_profit = float(fallback_take_profit)
            execution_stop_loss = float(fallback_stop_loss)
            leader_max_day_rank = default_leader_rank
            leader_take_profit = fallback_leader_take_profit
            leader_stop_loss = fallback_leader_stop_loss
            leader_hold_days = fallback_leader_hold_days
            use_soft_context_score = False

            institution_activity_apr_z = float(getattr(row, "institution_activity_apr_z", 0.0) or 0.0)
            margin_trend_activity_apr_z = float(getattr(row, "margin_trend_activity_apr_z", 0.0) or 0.0)
            crowding_activity_apr_z = float(getattr(row, "crowding_activity_apr_z", 0.0) or 0.0)
            institution_weakness, margin_support, crowding_pressure = self._soft_context_components(
                institution_activity_apr_z=institution_activity_apr_z,
                margin_trend_activity_apr_z=margin_trend_activity_apr_z,
                crowding_activity_apr_z=crowding_activity_apr_z,
            )

            market_is_weak = (
                (pd.notna(row.regime_breadth_5d) and row.regime_breadth_5d <= q_breadth_low)
                or (pd.notna(row.regime_return_mean_5d) and row.regime_return_mean_5d <= q_return_low)
            )
            market_is_strong = (
                (pd.notna(row.regime_breadth_5d) and row.regime_breadth_5d >= q_breadth_high)
                and (pd.notna(row.regime_return_mean_5d) and row.regime_return_mean_5d >= q_return_high)
            )
            turnover_is_hot = pd.notna(row.regime_turnover_ratio_5_20) and row.regime_turnover_ratio_5_20 >= q_turnover_high
            defensive_trigger = False
            rotation_trigger = False
            acceleration_trigger = False

            if state_profile == "market_state_soft_context_v1":
                use_soft_context_score = True
                if default_top_k > 1 or float(fallback_min_score_ratio_to_leader) > 0.0:
                    min_score_ratio_to_leader = float(np.clip(
                        float(fallback_min_score_ratio_to_leader)
                        + 0.040 * institution_weakness
                        + 0.030 * crowding_pressure
                        - 0.020 * margin_support,
                        0.0,
                        0.98,
                    ))
                else:
                    min_score_ratio_to_leader = float(fallback_min_score_ratio_to_leader)
                execution_hold_days = max(
                    15,
                    int(round(
                        int(fallback_hold_days)
                        * (1.0 - 0.18 * institution_weakness - 0.15 * crowding_pressure + 0.08 * margin_support)
                    )),
                )
                execution_take_profit = float(max(
                    0.08,
                    float(fallback_take_profit) * (1.0 - 0.08 * crowding_pressure + 0.05 * margin_support),
                ))
                execution_stop_loss = float(min(
                    -0.02,
                    float(fallback_stop_loss) + 0.012 * institution_weakness + 0.010 * crowding_pressure - 0.006 * margin_support,
                ))
                if leader_max_day_rank is not None and leader_max_day_rank > 0:
                    leader_take_profit = float(max(
                        0.10,
                        float(fallback_leader_take_profit or fallback_take_profit) * (1.0 - 0.06 * crowding_pressure + 0.05 * margin_support),
                    ))
                    leader_stop_loss = float(min(
                        -0.02,
                        float(fallback_leader_stop_loss or fallback_stop_loss) + 0.010 * institution_weakness + 0.008 * crowding_pressure - 0.005 * margin_support,
                    ))
                    leader_hold_days = max(
                        12,
                        int(round(
                            int(fallback_leader_hold_days or fallback_hold_days)
                            * (1.0 - 0.16 * institution_weakness - 0.12 * crowding_pressure + 0.06 * margin_support)
                        )),
                    )
                dominant_pressure = max(institution_weakness, crowding_pressure, margin_support)
                if dominant_pressure < 0.20:
                    state_name = "neutral"
                    state_reason = "proxy_context_near_april_baseline"
                elif institution_weakness >= max(crowding_pressure, margin_support):
                    state_name = "soft_defensive_bias"
                    state_reason = "weak_institution_proxy_bias"
                elif crowding_pressure >= max(institution_weakness, margin_support):
                    state_name = "soft_crowding_bias"
                    state_reason = "crowding_proxy_bias"
                else:
                    state_name = "soft_supportive_bias"
                    state_reason = "margin_support_proxy_bias"
            elif state_profile == "market_state_gate_v2":
                defensive_trigger = (row.top1_score <= q_score_low and row.pass_count <= q_pass_low and market_is_weak)
                rotation_trigger = (row.pass_count >= q_pass_high and row.score_gap_top1_top2 <= q_gap_low and turnover_is_hot)
                acceleration_trigger = (row.top1_score >= q_score_high and row.score_gap_top1_top2 >= q_gap_high and row.pass_count >= default_top_k and market_is_strong)
            else:
                defensive_trigger = ((row.top1_score <= q_score_low or row.pass_count <= q_pass_low) and market_is_weak)
                rotation_trigger = (row.pass_count >= q_pass_high and row.score_gap_top1_top2 <= q_gap_low and turnover_is_hot)
                acceleration_trigger = (row.top1_score >= q_score_high and row.score_gap_top1_top2 >= q_gap_high and row.pass_count >= default_top_k and market_is_strong)

            if state_profile in {"market_state_gate_v1", "market_state_gate_v2"} and defensive_trigger:
                state_name = "defensive"
                state_reason = "weak_leader_with_weak_market_breadth"
                allow_entries = False
                top_k_per_day = 0
                min_score_ratio_to_leader = 0.0
                execution_hold_days = min(int(fallback_hold_days), 20)
                execution_take_profit = min(float(fallback_take_profit), 0.10)
                execution_stop_loss = max(float(fallback_stop_loss), -0.04)
                leader_max_day_rank = 1
                leader_take_profit = 0.12
                leader_stop_loss = -0.03
                leader_hold_days = 15
            elif rotation_trigger:
                state_name = "rotation"
                state_reason = "crowded_candidates_in_hot_rotating_tape"
                top_k_per_day = 1
                min_score_ratio_to_leader = 0.0
                execution_hold_days = min(int(fallback_hold_days), 25)
                execution_take_profit = min(float(fallback_take_profit), 0.12)
                execution_stop_loss = max(float(fallback_stop_loss), -0.05)
                leader_max_day_rank = 1
                leader_take_profit = fallback_leader_take_profit if fallback_leader_take_profit is not None else 0.14
                leader_stop_loss = fallback_leader_stop_loss if fallback_leader_stop_loss is not None else -0.04
                leader_hold_days = min(int(fallback_leader_hold_days or execution_hold_days), 20)
            elif acceleration_trigger:
                state_name = "acceleration"
                state_reason = "strong_leader_in_broad_supportive_market"
                top_k_per_day = max(default_top_k, 3)
                min_score_ratio_to_leader = 0.90 if fallback_min_score_ratio_to_leader <= 0 else min(float(fallback_min_score_ratio_to_leader), 0.90)
                execution_hold_days = max(int(fallback_hold_days), 35)
                execution_take_profit = max(float(fallback_take_profit), 0.14)
                execution_stop_loss = float(fallback_stop_loss)
                leader_max_day_rank = 1
                leader_take_profit = max(float(fallback_leader_take_profit or fallback_take_profit), 0.18)
                leader_stop_loss = min(float(fallback_leader_stop_loss or fallback_stop_loss), -0.035)
                leader_hold_days = max(int(fallback_leader_hold_days or fallback_hold_days), 25)

            plans.append(
                StatefulExecutionPlan(
                    date=pd.Timestamp(row.date),
                    state_name=state_name,
                    state_reason=state_reason,
                    use_soft_context_score=use_soft_context_score,
                    allow_entries=allow_entries,
                    top_k_per_day=int(top_k_per_day),
                    min_score_ratio_to_leader=float(min_score_ratio_to_leader),
                    execution_hold_days=int(execution_hold_days),
                    execution_take_profit=float(execution_take_profit),
                    execution_stop_loss=float(execution_stop_loss),
                    leader_max_day_rank=int(leader_max_day_rank) if leader_max_day_rank is not None else None,
                    leader_take_profit=float(leader_take_profit) if leader_take_profit is not None else None,
                    leader_stop_loss=float(leader_stop_loss) if leader_stop_loss is not None else None,
                    leader_hold_days=int(leader_hold_days) if leader_hold_days is not None else None,
                    institution_activity_apr_z=institution_activity_apr_z,
                    margin_trend_activity_apr_z=margin_trend_activity_apr_z,
                    crowding_activity_apr_z=crowding_activity_apr_z,
                    soft_context_defensive_pressure=float(max(institution_weakness, crowding_pressure)),
                    soft_context_supportive_pressure=float(margin_support),
                )
            )
        return pd.DataFrame([plan.__dict__ for plan in plans])

    def _apply_stateful_execution_plan(
        self,
        predictions: pd.DataFrame,
        state_plan: pd.DataFrame,
    ) -> pd.DataFrame:
        if predictions.empty or state_plan.empty:
            return predictions
        working = predictions.copy()
        working = working.merge(state_plan, on="date", how="left")
        base_leader_score = working.groupby("date")["signal_score"].transform("max")
        working["base_score_ratio_to_leader"] = working["signal_score"].div(base_leader_score.where(base_leader_score.abs() > 1e-9))

        ranking_score_col = "signal_score"
        if "use_soft_context_score" in working.columns and working["use_soft_context_score"].fillna(False).any():
            crowding_pressure = working["crowding_activity_apr_z"].clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
            institution_weakness = (-working["institution_activity_apr_z"]).clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
            margin_support = working["margin_trend_activity_apr_z"].clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
            first_hit_confidence = pd.to_numeric(working.get("first_hit_confidence"), errors="coerce")
            first_hit_confidence = first_hit_confidence.fillna(working["pred_take_profit"] * (1.0 - working["pred_stop_loss"]))
            vulnerability = (
                0.45 * working["pred_stop_loss"].clip(lower=0.0, upper=1.0)
                + 0.35 * working["pred_max_drawdown"].abs().clip(lower=0.01, upper=0.30)
                + 0.20 * (1.0 - working["pred_take_profit"].clip(lower=0.0, upper=1.0))
            )
            resilience = (
                0.35 * working["base_score_ratio_to_leader"].clip(lower=0.0, upper=1.5)
                + 0.30 * first_hit_confidence.clip(lower=0.0, upper=1.0)
                + 0.20 * working["pred_take_profit"].clip(lower=0.0, upper=1.0)
                + 0.15 * working["pred_short_up"].clip(lower=0.0, upper=1.0)
            )
            weak_institution_penalty = 0.65 * (1.0 - working["base_score_ratio_to_leader"].clip(lower=0.0, upper=1.0)) + 0.35 * working["pred_stop_loss"].clip(lower=0.0, upper=1.0)
            working["soft_context_score_delta"] = 0.04 * (
                0.55 * margin_support * resilience
                - 0.60 * crowding_pressure * vulnerability
                - 0.75 * institution_weakness * weak_institution_penalty
            )
            working["soft_context_score_delta"] = working["soft_context_score_delta"].where(working["use_soft_context_score"].fillna(False), 0.0)
            working["state_rerank_score"] = working["signal_score"] + working["soft_context_score_delta"]
            ranking_score_col = "state_rerank_score"
        else:
            working["soft_context_score_delta"] = 0.0
            working["state_rerank_score"] = working["signal_score"]

        leader_score = working.groupby("date")[ranking_score_col].transform("max")
        working["score_ratio_to_leader"] = working[ranking_score_col].div(leader_score.where(leader_score.abs() > 1e-9))
        working = working[working["allow_entries"].fillna(True)].copy()
        working = working[
            working["score_ratio_to_leader"] >= working["min_score_ratio_to_leader"].fillna(0.0)
        ].copy()
        if working.empty:
            return working
        working = (
            working.sort_values(["date", ranking_score_col, "pred_take_profit"], ascending=[True, False, False])
            .groupby("date", group_keys=False)
            .head(working["top_k_per_day"].fillna(1).astype(int).max())
            .copy()
        )
        working["state_rank_within_day"] = working.groupby("date").cumcount() + 1
        working = working[
            working["state_rank_within_day"] <= working["top_k_per_day"].fillna(1).astype(int)
        ].copy()
        return working.sort_values(["date", ranking_score_col, "pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)

    def _invert_in_down_regime(
        self,
        predictions: pd.DataFrame,
        regime_return_lookback: int = 5,
        regime_boundary: float = 0.0,
    ) -> pd.DataFrame:
        """下跌 regime 自动反转 signal_score。

        核心洞察：模型存在"滞后追涨"特性——在持续下跌市里，模型给高分的
        强势股补跌最凶（追在顶部）。因此当大盘近 N 日均收益为负（下跌市）时，
        将 signal_score 取反（1 - score），使模型 bottom 变成 top，
        实现"下跌市反向选股"。

        参数：
          regime_return_lookback: 判断 regime 用的大盘均收益回看天数
          regime_boundary: 下跌市判定边界（大盘 N 日均收益 < 此值即反转）
        """
        if predictions.empty:
            return predictions
        market_daily = self._load_market_state_daily_metrics(predictions)
        if market_daily.empty or "regime_return_mean_5d" not in market_daily.columns:
            return predictions

        regime_col = f"regime_return_mean_{regime_return_lookback}d"
        if regime_col not in market_daily.columns:
            regime_col = "regime_return_mean_5d"

        market_daily = market_daily[["date", regime_col]].rename(columns={regime_col: "_regime_mean"})
        market_daily["date"] = pd.to_datetime(market_daily["date"], errors="coerce")

        working = predictions.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working.merge(market_daily, on="date", how="left")

        is_down = working["_regime_mean"].fillna(0.0) < regime_boundary
        working["regime_inverted"] = is_down.astype(int)
        # 下跌市：signal_score 取反（1 - score，保持 [0,1] 且完全反向排序）
        # 同时把完整排序纳入 _prepare_signals 所用的 signal_score
        working["signal_score_raw"] = working["signal_score"].copy()
        working.loc[is_down, "signal_score"] = 1.0 - working.loc[is_down, "signal_score"].clip(0.0, 1.0)

        return working.drop(columns=["_regime_mean"]).reset_index(drop=True)

    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        short_up_threshold: float = 0.59,
        take_profit_threshold: float = 0.25,
        stop_loss_threshold: float = 0.30,
        top_k_per_day: int | None = None,
        use_calibration: bool = False,
        hold_days: int = 45,
        take_profit: float = 0.12,
        stop_loss: float = -0.06,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        min_lot: int = 100,
        min_position_weight: float = 0.08,
        max_position_weight: float = 0.20,
        max_total_position_fraction: float = 1.0,
        cooldown_trigger_losses: int = 0,
        cooldown_days: int = 0,
        stock_cooldown_days: int = 0,
        min_secondary_primary_ratio: float = 0.0,
        rank_weight_power: float = 0.0,
        min_score_ratio_to_leader: float = 0.0,
        min_filter_column: str | None = None,
        min_filter_value: float | None = None,
        leader_max_day_rank: int | None = None,
        leader_take_profit: float | None = None,
        leader_stop_loss: float | None = None,
        leader_hold_days: int | None = None,
        state_profile: str | None = None,
        invert_down_regime: bool = False,
        regime_return_lookback: int = 5,
        regime_boundary: float = 0.0,
    ) -> BacktestResult:
        start_date = start_date or str((pd.Timestamp(self.config.splits.val_end) + pd.Timedelta(days=1)).date())
        end_date = end_date or self.config.splits.test_end
        predictions = self._load_predictions(
            start_date=start_date,
            end_date=end_date,
            short_up_threshold=short_up_threshold,
            take_profit_threshold=take_profit_threshold,
            stop_loss_threshold=stop_loss_threshold,
            top_k_per_day=None if (stock_cooldown_days > 0 or state_profile is not None) else top_k_per_day,
            use_calibration=use_calibration,
            min_secondary_primary_ratio=min_secondary_primary_ratio,
            min_score_ratio_to_leader=0.0 if state_profile is not None else min_score_ratio_to_leader,
            min_filter_column=min_filter_column,
            min_filter_value=min_filter_value,
        )
        if invert_down_regime:
            predictions = self._invert_in_down_regime(
                predictions,
                regime_return_lookback=regime_return_lookback,
                regime_boundary=regime_boundary,
            )
        state_plan = pd.DataFrame()
        if state_profile is not None:
            state_plan = self._build_stateful_execution_plan(
                predictions,
                state_profile=state_profile,
                fallback_top_k_per_day=top_k_per_day,
                fallback_min_score_ratio_to_leader=min_score_ratio_to_leader,
                fallback_hold_days=hold_days,
                fallback_take_profit=take_profit,
                fallback_stop_loss=stop_loss,
                fallback_leader_max_day_rank=leader_max_day_rank,
                fallback_leader_take_profit=leader_take_profit,
                fallback_leader_stop_loss=leader_stop_loss,
                fallback_leader_hold_days=leader_hold_days,
            )
            predictions = self._apply_stateful_execution_plan(predictions, state_plan)
        signals = self._prepare_signals(
            predictions,
            initial_capital=initial_capital,
            min_position_weight=min_position_weight,
            max_position_weight=max_position_weight,
            rank_weight_power=rank_weight_power,
        )
        if signals.empty:
            raise RuntimeError("当前阈值下没有任何可回测信号。")
        if state_profile is not None and not state_plan.empty:
            if "execution_take_profit" not in signals.columns:
                per_date_plan = state_plan.rename(columns={"date": "signal_date"})
                signals = signals.merge(per_date_plan, on="signal_date", how="left")
            for numeric_col in [
                "execution_take_profit",
                "execution_stop_loss",
                "leader_max_day_rank",
                "leader_take_profit",
                "leader_stop_loss",
                "leader_hold_days",
            ]:
                if numeric_col in signals.columns:
                    signals[numeric_col] = pd.to_numeric(signals[numeric_col], errors="coerce")
            signals["exit_take_profit"] = signals["execution_take_profit"].fillna(float(take_profit))
            signals["exit_stop_loss"] = signals["execution_stop_loss"].fillna(float(stop_loss))
            signals["exit_hold_days"] = signals["execution_hold_days"].fillna(int(hold_days)).astype(int)
            leader_rank_limit = signals["leader_max_day_rank"].fillna(0).astype(int)
            leader_mask = leader_rank_limit.gt(0) & (signals["day_rank"] <= leader_rank_limit)
            signals.loc[leader_mask & signals["leader_take_profit"].notna(), "exit_take_profit"] = signals.loc[
                leader_mask & signals["leader_take_profit"].notna(), "leader_take_profit"
            ]
            signals.loc[leader_mask & signals["leader_stop_loss"].notna(), "exit_stop_loss"] = signals.loc[
                leader_mask & signals["leader_stop_loss"].notna(), "leader_stop_loss"
            ]
            signals.loc[leader_mask & signals["leader_hold_days"].notna(), "exit_hold_days"] = signals.loc[
                leader_mask & signals["leader_hold_days"].notna(), "leader_hold_days"
            ].astype(int)
        else:
            signals["exit_take_profit"] = float(take_profit)
            signals["exit_stop_loss"] = float(stop_loss)
            signals["exit_hold_days"] = int(hold_days)
            if leader_max_day_rank is not None and leader_max_day_rank > 0:
                leader_mask = signals["day_rank"] <= int(leader_max_day_rank)
                if leader_take_profit is not None:
                    signals.loc[leader_mask, "exit_take_profit"] = float(leader_take_profit)
                if leader_stop_loss is not None:
                    signals.loc[leader_mask, "exit_stop_loss"] = float(leader_stop_loss)
                if leader_hold_days is not None:
                    signals.loc[leader_mask, "exit_hold_days"] = int(leader_hold_days)

        signal_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)
        trade_dates = set(predictions["date"].drop_duplicates().tolist())
        for stock_code in predictions["stock_code"].drop_duplicates():
            prices = self._load_price_history(stock_code)
            if not prices.empty:
                trade_dates.update(prices[(prices["date"] >= signal_start) & (prices["date"] <= requested_end)]["date"].tolist())
        trade_calendar = sorted(date for date in trade_dates if signal_start <= pd.Timestamp(date) <= requested_end)
        trade_calendar_index = {pd.Timestamp(date): index for index, date in enumerate(trade_calendar)}

        cash = initial_capital
        pending_orders: list[dict] = []
        open_positions: list[dict] = []
        trade_rows: list[dict] = []
        equity_rows: list[dict] = []
        scheduled_signal_rows: list[dict] = []
        previous_total_assets = initial_capital
        consecutive_stop_losses = 0
        cooldown_until: pd.Timestamp | None = None
        cooldown_activations = 0
        stock_cooldown_until: dict[str, int] = {}

        for current_index, current_date in enumerate(trade_calendar):
            current_date = pd.Timestamp(current_date)

            today_orders = [order for order in pending_orders if order["buy_date"] == current_date]
            pending_orders = [order for order in pending_orders if order["buy_date"] != current_date]
            for order in today_orders:
                if any(position["stock_code"] == order["stock_code"] for position in open_positions):
                    continue
                bar = self._daily_bar(order["stock_code"], current_date)
                if bar is None:
                    continue
                buy_price = float(bar["open"]) * (1.0 + slippage)
                current_exposure_amount = sum(float(position["buy_amount"]) for position in open_positions)
                exposure_capacity = max(initial_capital * max_total_position_fraction - current_exposure_amount, 0.0)
                max_target_amount = min(order["target_amount"], initial_capital * max_position_weight, exposure_capacity, cash)
                quantity = math.floor(max_target_amount / (buy_price * (1.0 + commission)) / min_lot) * min_lot
                if quantity < min_lot:
                    continue
                buy_amount = quantity * buy_price
                total_cost = buy_amount * (1.0 + commission)
                if total_cost > cash:
                    continue
                cash -= total_cost
                open_positions.append(
                    {
                        "stock_code": order["stock_code"],
                        "layer": order.get("layer"),
                        "signal_date": order["signal_date"],
                        "buy_date": current_date,
                        "buy_price": buy_price,
                        "buy_quantity": quantity,
                        "buy_amount": buy_amount,
                        "pred_short_up": order["pred_short_up"],
                        "pred_take_profit": order["pred_take_profit"],
                        "pred_stop_loss": order["pred_stop_loss"],
                        "pred_max_gain": order["pred_max_gain"],
                        "pred_max_drawdown": order["pred_max_drawdown"],
                        "signal_score": order["signal_score"],
                        "target_weight": order["target_weight"],
                        "exit_take_profit": float(order.get("exit_take_profit", take_profit)),
                        "exit_stop_loss": float(order.get("exit_stop_loss", stop_loss)),
                        "exit_hold_days": int(order.get("exit_hold_days", hold_days)),
                        "max_gain": 0.0,
                        "max_drawdown": 0.0,
                    }
                )

            still_open: list[dict] = []
            for position in open_positions:
                bar = self._daily_bar(position["stock_code"], current_date)
                if bar is None:
                    still_open.append(position)
                    continue

                high_return = float(bar["high"] / position["buy_price"] - 1.0)
                low_return = float(bar["low"] / position["buy_price"] - 1.0)
                position["max_gain"] = max(position["max_gain"], high_return)
                position["max_drawdown"] = min(position["max_drawdown"], low_return)
                hold_days_count = int(sum(1 for date in trade_calendar if position["buy_date"] <= pd.Timestamp(date) <= current_date))
                position_take_profit = float(position.get("exit_take_profit", take_profit))
                position_stop_loss = float(position.get("exit_stop_loss", stop_loss))
                position_hold_days = int(position.get("exit_hold_days", hold_days))

                exit_reason = None
                raw_exit_price = None
                if low_return <= position_stop_loss:
                    exit_reason = "止损"
                    raw_exit_price = min(float(bar["open"]), position["buy_price"] * (1.0 + position_stop_loss))
                elif high_return >= position_take_profit:
                    exit_reason = "止盈"
                    raw_exit_price = max(float(bar["open"]), position["buy_price"] * (1.0 + position_take_profit))
                elif hold_days_count >= position_hold_days:
                    exit_reason = "时间止损"
                    raw_exit_price = float(bar["close"])

                if exit_reason is None:
                    still_open.append(position)
                    continue

                sell_price = raw_exit_price * (1.0 - slippage)
                sell_amount = position["buy_quantity"] * sell_price
                net_sell_amount = sell_amount * (1.0 - commission)
                cash += net_sell_amount
                profit = net_sell_amount - position["buy_amount"] * (1.0 + commission)
                trade_rows.append(
                    {
                        "stock_code": position["stock_code"],
                        "layer": position.get("layer"),
                        "signal_date": str(pd.Timestamp(position["signal_date"]).date()),
                        "buy_date": str(position["buy_date"].date()),
                        "buy_price": position["buy_price"],
                        "buy_quantity": position["buy_quantity"],
                        "buy_amount": position["buy_amount"],
                        "sell_date": str(current_date.date()),
                        "sell_price": sell_price,
                        "sell_amount": net_sell_amount,
                        "hold_days": hold_days_count,
                        "max_gain": position["max_gain"],
                        "max_drawdown": position["max_drawdown"],
                        "exit_reason": exit_reason,
                        "profit": profit,
                        "profit_pct": profit / max(position["buy_amount"], 1e-6),
                        "pred_short_up": position["pred_short_up"],
                        "pred_take_profit": position["pred_take_profit"],
                        "pred_stop_loss": position["pred_stop_loss"],
                        "pred_max_gain": position["pred_max_gain"],
                        "pred_max_drawdown": position["pred_max_drawdown"],
                        "signal_score": position["signal_score"],
                        "exit_take_profit": position_take_profit,
                        "exit_stop_loss": position_stop_loss,
                        "exit_hold_days": position_hold_days,
                    }
                )
                if exit_reason == "止损":
                    consecutive_stop_losses += 1
                    if cooldown_trigger_losses > 0 and cooldown_days > 0 and consecutive_stop_losses >= cooldown_trigger_losses:
                        cooldown_end_index = min(current_index + cooldown_days, len(trade_calendar) - 1)
                        cooldown_until = pd.Timestamp(trade_calendar[cooldown_end_index])
                        cooldown_activations += 1
                        consecutive_stop_losses = 0
                else:
                    consecutive_stop_losses = 0
            open_positions = still_open

            today_signals = signals[signals["signal_date"] == current_date]
            held_or_pending = {position["stock_code"] for position in open_positions} | {order["stock_code"] for order in pending_orders}
            is_cooldown_active = cooldown_until is not None and current_date <= cooldown_until
            if not is_cooldown_active:
                scheduled_today = 0
                for _, row in today_signals.iterrows():
                    if top_k_per_day is not None and scheduled_today >= top_k_per_day:
                        break
                    stock_code = str(row["stock_code"]).zfill(6)
                    if stock_code in held_or_pending:
                        continue
                    if stock_cooldown_days > 0:
                        current_trade_index = trade_calendar_index.get(current_date)
                        cooldown_end_index = stock_cooldown_until.get(stock_code)
                        if current_trade_index is not None and cooldown_end_index is not None and current_trade_index <= cooldown_end_index:
                            continue
                    buy_date = self._next_trade_date(stock_code, current_date)
                    if buy_date is None or buy_date > requested_end:
                        continue
                    pending_orders.append({**row.to_dict(), "stock_code": stock_code, "buy_date": buy_date})
                    scheduled_signal_rows.append({**row.to_dict(), "stock_code": stock_code, "buy_date": str(pd.Timestamp(buy_date).date())})
                    scheduled_today += 1
                    if stock_cooldown_days > 0:
                        current_trade_index = trade_calendar_index.get(current_date)
                        if current_trade_index is not None:
                            stock_cooldown_until[stock_code] = current_trade_index + stock_cooldown_days

            stock_value = 0.0
            for position in open_positions:
                bar = self._daily_bar(position["stock_code"], current_date)
                if bar is None:
                    continue
                stock_value += position["buy_quantity"] * float(bar["close"])
            total_assets = cash + stock_value
            equity_rows.append(
                {
                    "date": str(current_date.date()),
                    "total_assets": total_assets,
                    "stock_value": stock_value,
                    "cash": cash,
                    "position_percent": stock_value / total_assets if total_assets > 0 else 0.0,
                    "daily_pnl": total_assets - previous_total_assets,
                    "cumulative_pnl": total_assets - initial_capital,
                }
            )
            previous_total_assets = total_assets

        trades_df = pd.DataFrame(trade_rows).sort_values(["buy_date", "stock_code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
        scheduled_signals_df = (
            pd.DataFrame(scheduled_signal_rows).sort_values(["signal_date", "signal_score"], ascending=[True, False]).reset_index(drop=True)
            if scheduled_signal_rows
            else pd.DataFrame()
        )
        equity_df = pd.DataFrame(equity_rows)
        if equity_df.empty:
            equity_df = pd.DataFrame([
                {
                    "date": start_date,
                    "total_assets": initial_capital,
                    "stock_value": 0.0,
                    "cash": initial_capital,
                    "position_percent": 0.0,
                    "daily_pnl": 0.0,
                    "cumulative_pnl": 0.0,
                }
            ])
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df["daily_return"] = equity_df["total_assets"].pct_change().fillna(0.0)
        drawdown = equity_df["total_assets"] / equity_df["total_assets"].cummax() - 1.0
        profitable = trades_df[trades_df["profit"] > 0] if not trades_df.empty else pd.DataFrame()
        losing = trades_df[trades_df["profit"] < 0] if not trades_df.empty else pd.DataFrame()
        days_count = max(len(equity_df), 1)

        summary = {
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "short_up_threshold": short_up_threshold,
            "take_profit_threshold": take_profit_threshold,
            "stop_loss_threshold": stop_loss_threshold,
            "top_k_per_day": top_k_per_day,
            "state_profile": state_profile,
            "signal_count": int(len(scheduled_signals_df)),
            "trade_count": int(len(trades_df)),
            "total_return": float(equity_df.iloc[-1]["total_assets"] / initial_capital - 1.0),
            "annualized_return": float((equity_df.iloc[-1]["total_assets"] / initial_capital) ** (252 / days_count) - 1.0) if days_count > 1 else 0.0,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "sharpe_ratio": float(equity_df["daily_return"].mean() / equity_df["daily_return"].std() * np.sqrt(252)) if len(equity_df) > 1 and float(equity_df["daily_return"].std()) > 0 else 0.0,
            "win_rate": float((trades_df["profit"] > 0).mean()) if not trades_df.empty else 0.0,
            "profit_factor": float(profitable["profit"].sum() / abs(losing["profit"].sum())) if not losing.empty else 0.0,
            "avg_hold_days": float(trades_df["hold_days"].mean()) if not trades_df.empty else 0.0,
            "min_position_weight": float(min_position_weight),
            "max_position_weight": float(max_position_weight),
            "max_total_position_fraction": float(max_total_position_fraction),
            "cooldown_trigger_losses": int(cooldown_trigger_losses),
            "cooldown_days": int(cooldown_days),
            "stock_cooldown_days": int(stock_cooldown_days),
            "cooldown_activations": int(cooldown_activations),
            "min_secondary_primary_ratio": float(min_secondary_primary_ratio),
            "rank_weight_power": float(rank_weight_power),
            "min_score_ratio_to_leader": float(min_score_ratio_to_leader),
            "min_filter_column": min_filter_column,
            "min_filter_value": float(min_filter_value) if min_filter_value is not None else None,
            "leader_max_day_rank": int(leader_max_day_rank) if leader_max_day_rank is not None else None,
            "leader_take_profit": float(leader_take_profit) if leader_take_profit is not None else None,
            "leader_stop_loss": float(leader_stop_loss) if leader_stop_loss is not None else None,
            "leader_hold_days": int(leader_hold_days) if leader_hold_days is not None else None,
        }
        if not state_plan.empty:
            summary["state_date_counts"] = {str(key): int(value) for key, value in state_plan["state_name"].value_counts().items()}
        if not scheduled_signals_df.empty and "state_name" in scheduled_signals_df.columns:
            summary["state_signal_counts"] = {str(key): int(value) for key, value in scheduled_signals_df["state_name"].value_counts().items()}
        state_plan_df = state_plan.sort_values(["date"]).reset_index(drop=True) if not state_plan.empty else None
        return BacktestResult(
            summary=summary,
            trades=trades_df,
            equity=equity_df.drop(columns=["daily_return"]),
            signals=scheduled_signals_df,
            state_plan=state_plan_df,
        )


def write_backtest_outputs(result: BacktestResult, results_dir: Path, prefix: str) -> dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_path = results_dir / f"{prefix}_trades.csv"
    equity_path = results_dir / f"{prefix}_equity.csv"
    signals_path = results_dir / f"{prefix}_signals.csv"
    summary_path = results_dir / f"{prefix}_summary.json"
    state_plan_path = results_dir / f"{prefix}_state_plan.csv"
    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    result.equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    result.signals.to_csv(signals_path, index=False, encoding="utf-8-sig")
    if result.state_plan is not None:
        state_plan_to_write = result.state_plan.copy()
        if "date" in state_plan_to_write.columns:
            state_plan_to_write["date"] = pd.to_datetime(state_plan_to_write["date"]).dt.strftime("%Y-%m-%d")
        state_plan_to_write.to_csv(state_plan_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2)
    outputs = {
        "trades": str(trades_path),
        "equity": str(equity_path),
        "signals": str(signals_path),
        "summary": str(summary_path),
    }
    if result.state_plan is not None:
        outputs["state_plan"] = str(state_plan_path)
    return outputs