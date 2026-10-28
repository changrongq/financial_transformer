from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sqlite3

import numpy as np
import pandas as pd

from .config import load_config


@dataclass
class MultiTaskBacktestResult:
    summary: dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_holdings: pd.DataFrame
    signals: pd.DataFrame


class MultiTaskStrategyABacktester:
    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._predictions_cache: pd.DataFrame | None = None

    def _load_all_layer_predictions(self) -> pd.DataFrame:
        if self._predictions_cache is not None:
            return self._predictions_cache

        frames = []
        for layer in ["micro", "small", "mid", "large"]:
            path = self.config.results_dir / f"multitask_{layer}_predictions.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            frames.append(df)
        if not frames:
            raise RuntimeError("未找到任何分层多任务预测文件，请先完成多任务训练。")

        combined = pd.concat(frames, ignore_index=True)
        combined["stock_code"] = combined["stock_code"].astype(str).str.zfill(6)
        combined["date"] = pd.to_datetime(combined["date"])
        for column in [
            "pred_short_up",
            "pred_take_profit",
            "pred_stop_loss",
            "pred_max_gain",
            "pred_max_drawdown",
        ]:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
        combined = combined.dropna(subset=["date", "pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown"])
        combined["signal_score"] = self._strategy_score_series(combined)
        self._predictions_cache = combined.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)
        return self._predictions_cache

    def _load_layer_predictions(
        self,
        start_date: str,
        end_date: str,
        short_up_threshold: float,
        take_profit_threshold: float,
        stop_loss_threshold: float,
        allowed_layers: list[str] | None = None,
        layer_threshold_overrides: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        combined = self._load_all_layer_predictions()
        combined = combined[(combined["date"] >= pd.Timestamp(start_date)) & (combined["date"] <= pd.Timestamp(end_date))].copy()
        if combined.empty:
            raise RuntimeError("指定区间内没有可用于回测的多任务预测数据。")

        if allowed_layers is not None:
            allowed_layer_set = {layer.strip() for layer in allowed_layers if layer and layer.strip()}
            combined = combined[combined["layer"].isin(allowed_layer_set)].copy()
            if combined.empty:
                raise RuntimeError("指定层内没有可用于回测的多任务预测数据。")

        combined["effective_short_up_threshold"] = short_up_threshold
        combined["effective_take_profit_threshold"] = take_profit_threshold
        combined["effective_stop_loss_threshold"] = stop_loss_threshold

        if layer_threshold_overrides:
            for layer_name, overrides in layer_threshold_overrides.items():
                layer_mask = combined["layer"] == layer_name
                if not layer_mask.any():
                    continue
                if "short_up_threshold" in overrides:
                    combined.loc[layer_mask, "effective_short_up_threshold"] = float(overrides["short_up_threshold"])
                if "take_profit_threshold" in overrides:
                    combined.loc[layer_mask, "effective_take_profit_threshold"] = float(overrides["take_profit_threshold"])
                if "stop_loss_threshold" in overrides:
                    combined.loc[layer_mask, "effective_stop_loss_threshold"] = float(overrides["stop_loss_threshold"])

        combined["strategy_a_pass"] = (
            (combined["pred_short_up"] > combined["effective_short_up_threshold"])
            & (combined["pred_take_profit"] > combined["effective_take_profit_threshold"])
            & (combined["pred_stop_loss"] < combined["effective_stop_loss_threshold"])
        )
        combined = combined.sort_values(["date", "strategy_a_pass", "signal_score", "pred_take_profit"], ascending=[True, False, False, False]).reset_index(drop=True)
        return combined

    @staticmethod
    def _strategy_score_series(df: pd.DataFrame) -> pd.Series:
        gain_term = df["pred_max_gain"].clip(lower=0.0, upper=0.5)
        drawdown_penalty = df["pred_max_drawdown"].abs().clip(lower=0.01, upper=0.3)
        return 0.45 * df["pred_short_up"] + 0.35 * df["pred_take_profit"] + 0.20 * (1.0 - df["pred_stop_loss"]) + 0.10 * gain_term - 0.05 * drawdown_penalty

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

    def _next_trade_date(self, stock_code: str, signal_date: pd.Timestamp) -> pd.Timestamp | None:
        prices = self._load_price_history(stock_code)
        future = prices.loc[prices["date"] > signal_date, "date"]
        if future.empty:
            return None
        return pd.Timestamp(future.iloc[0])

    def _daily_bar(self, stock_code: str, current_date: pd.Timestamp) -> pd.Series | None:
        prices = self._load_price_history(stock_code)
        bar = prices[prices["date"] == current_date]
        if bar.empty:
            return None
        return bar.iloc[0]

    def _position_weight(self, row: pd.Series, max_signal_score: float) -> float:
        edge = max(float(row["pred_max_gain"]), 0.02) * max(float(row["pred_take_profit"]), 0.01) * max(float(row["pred_short_up"]), 0.01)
        risk = max(abs(float(row["pred_max_drawdown"])), 0.02) * max(float(row["pred_stop_loss"]), 0.05)
        normalized = min((edge / risk) / max(max_signal_score, 1e-6), 1.0)
        return min(0.20, 0.08 + 0.12 * normalized)

    def _prepare_signals(self, predictions: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        signals = predictions[predictions["strategy_a_pass"]].copy()
        if signals.empty:
            return signals
        prepared = []
        for signal_date, group in signals.groupby("date", sort=True):
            max_signal_score = float(group["signal_score"].max())
            for _, row in group.iterrows():
                target_weight = self._position_weight(row, max_signal_score=max_signal_score)
                prepared.append(
                    {
                        **row.to_dict(),
                        "signal_date": pd.Timestamp(signal_date),
                        "target_weight": target_weight,
                        "target_amount": initial_capital * target_weight,
                    }
                )
        return pd.DataFrame(prepared).sort_values(["signal_date", "signal_score"], ascending=[True, False]).reset_index(drop=True)

    @staticmethod
    def _compute_sortino(daily_returns: pd.Series) -> float:
        downside = daily_returns[daily_returns < 0]
        if downside.empty:
            return 0.0
        downside_std = float(downside.std())
        if downside_std <= 0:
            return 0.0
        return float(daily_returns.mean() / downside_std * np.sqrt(252))

    def run(
        self,
        start_date: str = "2025-10-01",
        end_date: str = "2026-04-13",
        short_up_threshold: float = 0.60,
        take_profit_threshold: float = 0.40,
        stop_loss_threshold: float = 0.20,
        allowed_layers: list[str] | None = None,
        layer_threshold_overrides: dict[str, dict[str, float]] | None = None,
        hold_days: int = 45,
        take_profit: float = 0.12,
        stop_loss: float = -0.06,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        min_lot: int = 100,
    ) -> MultiTaskBacktestResult:
        predictions = self._load_layer_predictions(
            start_date=start_date,
            end_date=end_date,
            short_up_threshold=short_up_threshold,
            take_profit_threshold=take_profit_threshold,
            stop_loss_threshold=stop_loss_threshold,
            allowed_layers=allowed_layers,
            layer_threshold_overrides=layer_threshold_overrides,
        )
        signal_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)
        effective_signal_end = pd.Timestamp(predictions["date"].max())
        signals = self._prepare_signals(predictions, initial_capital=initial_capital)

        trade_dates = set(predictions["date"].drop_duplicates().tolist())
        for stock_code in predictions["stock_code"].drop_duplicates():
            prices = self._load_price_history(stock_code)
            if not prices.empty:
                trade_dates.update(prices[(prices["date"] >= signal_start) & (prices["date"] <= requested_end)]["date"].tolist())
        trade_calendar = sorted(date for date in trade_dates if signal_start <= pd.Timestamp(date) <= requested_end)

        cash = initial_capital
        pending_orders: list[dict] = []
        open_positions: list[dict] = []
        trade_rows: list[dict] = []
        daily_rows: list[dict] = []
        holding_rows: list[dict] = []
        previous_total_assets = initial_capital

        for current_date in trade_calendar:
            current_date = pd.Timestamp(current_date)

            today_orders = [order for order in pending_orders if order["buy_date"] == current_date]
            pending_orders = [order for order in pending_orders if order["buy_date"] != current_date]
            for order in today_orders:
                if any(position["stock_code"] == order["stock_code"] for position in open_positions):
                    continue
                bar = self._daily_bar(order["stock_code"], current_date)
                if bar is None:
                    continue
                raw_buy_price = float(bar["open"])
                executed_buy_price = raw_buy_price * (1.0 + slippage)
                max_target_amount = min(order["target_amount"], initial_capital * 0.20, cash)
                quantity = math.floor(max_target_amount / (executed_buy_price * (1.0 + commission)) / min_lot) * min_lot
                if quantity < min_lot:
                    continue
                buy_amount = quantity * executed_buy_price
                total_cost = buy_amount * (1.0 + commission)
                if total_cost > cash:
                    continue
                cash -= total_cost
                open_positions.append(
                    {
                        "stock_code": order["stock_code"],
                        "layer": order.get("layer", order.get("model_layer")),
                        "signal_date": order["signal_date"],
                        "buy_date": current_date,
                        "buy_price": executed_buy_price,
                        "buy_quantity": quantity,
                        "buy_amount": buy_amount,
                        "pred_short_up": order["pred_short_up"],
                        "pred_take_profit": order["pred_take_profit"],
                        "pred_stop_loss": order["pred_stop_loss"],
                        "pred_max_gain": order["pred_max_gain"],
                        "pred_max_drawdown": order["pred_max_drawdown"],
                        "signal_score": order["signal_score"],
                        "target_weight": order["target_weight"],
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

                exit_reason = None
                raw_exit_price = None
                if low_return <= stop_loss:
                    exit_reason = "止损"
                    raw_exit_price = min(float(bar["open"]), position["buy_price"] * (1.0 + stop_loss))
                elif high_return >= take_profit:
                    exit_reason = "止盈"
                    raw_exit_price = max(float(bar["open"]), position["buy_price"] * (1.0 + take_profit))
                elif hold_days_count >= hold_days:
                    exit_reason = "时间止损"
                    raw_exit_price = float(bar["close"])

                if exit_reason is None:
                    still_open.append(position)
                    continue

                executed_sell_price = raw_exit_price * (1.0 - slippage)
                sell_amount = position["buy_quantity"] * executed_sell_price
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
                        "sell_price": executed_sell_price,
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
                    }
                )
            open_positions = still_open

            today_signals = signals[signals["signal_date"] == current_date] if not signals.empty else pd.DataFrame()
            held_or_pending = {position["stock_code"] for position in open_positions} | {order["stock_code"] for order in pending_orders}
            for _, row in today_signals.iterrows():
                stock_code = str(row["stock_code"]).zfill(6)
                if stock_code in held_or_pending:
                    continue
                buy_date = self._next_trade_date(stock_code, current_date)
                if buy_date is None or buy_date > requested_end:
                    continue
                pending_orders.append({**row.to_dict(), "stock_code": stock_code, "buy_date": buy_date})

            stock_value = 0.0
            for position in open_positions:
                bar = self._daily_bar(position["stock_code"], current_date)
                if bar is None:
                    continue
                market_value = position["buy_quantity"] * float(bar["close"])
                stock_value += market_value
                unrealized_profit = market_value * (1.0 - commission - slippage) - position["buy_amount"] * (1.0 + commission)
                holding_rows.append(
                    {
                        "date": str(current_date.date()),
                        "stock_code": position["stock_code"],
                        "layer": position.get("layer"),
                        "signal_date": str(pd.Timestamp(position["signal_date"]).date()),
                        "buy_date": str(position["buy_date"].date()),
                        "buy_quantity": position["buy_quantity"],
                        "buy_price": position["buy_price"],
                        "close_price": float(bar["close"]),
                        "market_value": market_value,
                        "unrealized_profit": unrealized_profit,
                        "unrealized_profit_pct": unrealized_profit / max(position["buy_amount"], 1e-6),
                        "pred_short_up": position["pred_short_up"],
                        "pred_take_profit": position["pred_take_profit"],
                        "pred_stop_loss": position["pred_stop_loss"],
                        "pred_max_gain": position["pred_max_gain"],
                        "pred_max_drawdown": position["pred_max_drawdown"],
                        "signal_score": position["signal_score"],
                    }
                )
            total_assets = cash + stock_value
            daily_pnl = total_assets - previous_total_assets
            cumulative_pnl = total_assets - initial_capital
            daily_rows.append(
                {
                    "date": str(current_date.date()),
                    "total_assets": total_assets,
                    "stock_value": stock_value,
                    "cash": cash,
                    "position_percent": stock_value / total_assets if total_assets > 0 else 0.0,
                    "daily_pnl": daily_pnl,
                    "cumulative_pnl": cumulative_pnl,
                }
            )
            previous_total_assets = total_assets

        for position in open_positions:
            last_date = requested_end
            bar = self._daily_bar(position["stock_code"], last_date)
            mark_price = float(bar["close"]) if bar is not None else position["buy_price"]
            mark_value = position["buy_quantity"] * mark_price * (1.0 - commission - slippage)
            hold_days_count = int(sum(1 for date in trade_calendar if position["buy_date"] <= pd.Timestamp(date) <= requested_end))
            profit = mark_value - position["buy_amount"] * (1.0 + commission)
            trade_rows.append(
                {
                    "stock_code": position["stock_code"],
                    "layer": position.get("layer"),
                    "signal_date": str(pd.Timestamp(position["signal_date"]).date()),
                    "buy_date": str(position["buy_date"].date()),
                    "buy_price": position["buy_price"],
                    "buy_quantity": position["buy_quantity"],
                    "buy_amount": position["buy_amount"],
                    "sell_date": None,
                    "sell_price": None,
                    "sell_amount": None,
                    "hold_days": hold_days_count,
                    "max_gain": position["max_gain"],
                    "max_drawdown": position["max_drawdown"],
                    "exit_reason": "持有中",
                    "profit": profit,
                    "profit_pct": profit / max(position["buy_amount"], 1e-6),
                    "pred_short_up": position["pred_short_up"],
                    "pred_take_profit": position["pred_take_profit"],
                    "pred_stop_loss": position["pred_stop_loss"],
                    "pred_max_gain": position["pred_max_gain"],
                    "pred_max_drawdown": position["pred_max_drawdown"],
                    "signal_score": position["signal_score"],
                }
            )

        trades_df = pd.DataFrame(trade_rows).sort_values(["buy_date", "stock_code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
        equity_df = pd.DataFrame(daily_rows)
        if equity_df.empty:
            equity_df = pd.DataFrame([{"date": start_date, "total_assets": initial_capital, "stock_value": 0.0, "cash": initial_capital, "position_percent": 0.0, "daily_pnl": 0.0, "cumulative_pnl": 0.0}])
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df["daily_return"] = equity_df["total_assets"].pct_change().fillna(0.0)
        peak = equity_df["total_assets"].cummax()
        drawdown = equity_df["total_assets"] / peak - 1.0
        holdings_df = pd.DataFrame(holding_rows).sort_values(["date", "stock_code"]).reset_index(drop=True) if holding_rows else pd.DataFrame(columns=["date", "stock_code", "layer", "signal_date", "buy_date", "buy_quantity", "buy_price", "close_price", "market_value", "unrealized_profit", "unrealized_profit_pct", "pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown", "signal_score"])

        closed_trades = trades_df[trades_df["exit_reason"] != "持有中"] if not trades_df.empty else pd.DataFrame()
        profitable = trades_df[trades_df["profit"] > 0] if not trades_df.empty else pd.DataFrame()
        losing = trades_df[trades_df["profit"] < 0] if not trades_df.empty else pd.DataFrame()
        days_count = max(len(equity_df), 1)
        annualized_return = float((equity_df.iloc[-1]["total_assets"] / initial_capital) ** (252 / days_count) - 1.0) if days_count > 1 else 0.0
        summary = {
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "effective_signal_start_date": str(predictions["date"].min().date()),
            "effective_signal_end_date": str(effective_signal_end.date()),
            "initial_capital": initial_capital,
            "commission": commission,
            "slippage": slippage,
            "min_lot": min_lot,
            "short_up_threshold": short_up_threshold,
            "take_profit_threshold": take_profit_threshold,
            "stop_loss_threshold": stop_loss_threshold,
            "allowed_layers": allowed_layers or ["micro", "small", "mid", "large"],
            "layer_threshold_overrides": layer_threshold_overrides or {},
            "signal_count": int(len(signals)),
            "trade_count": int(len(trades_df)),
            "closed_trade_count": int(len(closed_trades)),
            "open_trade_count": int(len(trades_df[trades_df["exit_reason"] == "持有中"])) if not trades_df.empty else 0,
            "total_return": float(equity_df.iloc[-1]["total_assets"] / initial_capital - 1.0),
            "annualized_return": annualized_return,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "sharpe_ratio": float(equity_df["daily_return"].mean() / equity_df["daily_return"].std() * np.sqrt(252)) if len(equity_df) > 1 and float(equity_df["daily_return"].std()) > 0 else 0.0,
            "sortino_ratio": self._compute_sortino(equity_df["daily_return"]),
            "win_rate": float((trades_df["profit"] > 0).mean()) if not trades_df.empty else 0.0,
            "avg_profit": float(profitable["profit"].mean()) if not profitable.empty else 0.0,
            "avg_loss": float(losing["profit"].mean()) if not losing.empty else 0.0,
            "profit_factor": float(profitable["profit"].sum() / abs(losing["profit"].sum())) if not losing.empty else 0.0,
            "avg_hold_days": float(trades_df["hold_days"].mean()) if not trades_df.empty else 0.0,
        }
        return MultiTaskBacktestResult(summary=summary, trades=trades_df, equity_curve=equity_df.drop(columns=["daily_return"]), daily_holdings=holdings_df, signals=signals)


def write_multitask_strategy_a_backtest(
    result: MultiTaskBacktestResult,
    results_dir: str | Path,
    output_prefix: str = "multitask_strategy_a",
) -> dict[str, str]:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / f"{output_prefix}_trades.csv"
    equity_path = output_dir / f"{output_prefix}_equity.csv"
    holdings_path = output_dir / f"{output_prefix}_daily_holdings.csv"
    summary_path = output_dir / f"{output_prefix}_summary.json"
    signals_path = output_dir / f"{output_prefix}_signals.csv"

    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    result.equity_curve.to_csv(equity_path, index=False, encoding="utf-8-sig")
    result.daily_holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    result.signals.to_csv(signals_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2)

    return {
        "trades": str(trades_path),
        "equity": str(equity_path),
        "daily_holdings": str(holdings_path),
        "summary": str(summary_path),
        "signals": str(signals_path),
    }