from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sqlite3

import pandas as pd

from .config import load_config


@dataclass
class RealisticBacktestResult:
    summary: dict
    trades: pd.DataFrame
    equity: pd.DataFrame


class RealisticPortfolioBacktester:
    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)
        self._price_cache: dict[str, pd.DataFrame] = {}

    def _prepare_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"stock_code", "date", "ensemble_score"}
        if not required.issubset(df.columns):
            raise ValueError("预测文件至少需要包含 stock_code、date、ensemble_score 列。")
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"])
        df["ensemble_score"] = pd.to_numeric(df["ensemble_score"], errors="coerce")
        return df.dropna(subset=["ensemble_score", "date"]).sort_values(["date", "ensemble_score"], ascending=[True, False]).reset_index(drop=True)

    def _load_predictions(self, prediction_file: str | Path) -> pd.DataFrame:
        path = Path(prediction_file)
        if not path.exists():
            raise FileNotFoundError(f"预测文件不存在: {path}")
        df = pd.read_csv(path)
        return self._prepare_predictions(df)

    def _load_close_prices(self, stock_code: str) -> pd.DataFrame:
        stock_code = str(stock_code).zfill(6)
        if stock_code in self._price_cache:
            return self._price_cache[stock_code]
        conn = sqlite3.connect(self.config.databases.trade_db)
        try:
            df = pd.read_sql_query(
                "SELECT 日期, 收盘 FROM stock_data WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[stock_code],
            )
        finally:
            conn.close()
        if df.empty:
            result = pd.DataFrame(columns=["date", "close"])
        else:
            result = df.rename(columns={"日期": "date", "收盘": "close"})
            result["date"] = pd.to_datetime(result["date"])
            result["close"] = pd.to_numeric(result["close"], errors="coerce")
            result = result.dropna(subset=["close"]).reset_index(drop=True)
        self._price_cache[stock_code] = result
        return result

    def _resolve_trade_window(self, stock_code: str, signal_date: pd.Timestamp, hold_days: int) -> dict | None:
        prices = self._load_close_prices(stock_code)
        if prices.empty:
            return None
        future = prices[prices["date"] > signal_date].reset_index(drop=True)
        if future.empty:
            return None
        buy_row = future.iloc[0]
        sell_idx = min(hold_days - 1, len(future) - 1)
        sell_row = future.iloc[sell_idx]
        if float(buy_row["close"]) <= 0:
            return None
        return {
            "buy_date": pd.Timestamp(buy_row["date"]),
            "buy_price": float(buy_row["close"]),
            "sell_date": pd.Timestamp(sell_row["date"]),
            "sell_price": float(sell_row["close"]),
        }

    def run(
        self,
        prediction_file: str | Path,
        percentile: float = 0.05,
        max_positions: int = 20,
        hold_days: int = 5,
        buy_cost: float = 0.001,
        sell_cost: float = 0.001,
        initial_capital: float = 100000.0,
    ) -> RealisticBacktestResult:
        predictions = self._load_predictions(prediction_file)
        return self.run_frame(
            predictions=predictions,
            percentile=percentile,
            max_positions=max_positions,
            hold_days=hold_days,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            initial_capital=initial_capital,
        )

    def run_frame(
        self,
        predictions: pd.DataFrame,
        percentile: float = 0.05,
        max_positions: int = 20,
        hold_days: int = 5,
        buy_cost: float = 0.001,
        sell_cost: float = 0.001,
        initial_capital: float = 100000.0,
    ) -> RealisticBacktestResult:
        predictions = self._prepare_predictions(predictions.copy())
        cash = initial_capital
        open_positions: list[dict] = []
        trade_rows: list[dict] = []
        equity_rows: list[dict] = []

        for signal_date, group in predictions.groupby("date", sort=True):
            signal_date = pd.Timestamp(signal_date)
            closing_positions = [position for position in open_positions if position["sell_date"] <= signal_date]
            remaining_positions = [position for position in open_positions if position["sell_date"] > signal_date]
            for position in closing_positions:
                proceeds = position["shares"] * position["sell_price"] * (1.0 - sell_cost)
                cash += proceeds
                trade_rows.append(
                    {
                        "signal_date": str(position["signal_date"].date()),
                        "buy_date": str(position["buy_date"].date()),
                        "sell_date": str(position["sell_date"].date()),
                        "stock_code": position["stock_code"],
                        "score": position["score"],
                        "shares": position["shares"],
                        "buy_price": position["buy_price"],
                        "sell_price": position["sell_price"],
                        "gross_return": (position["sell_price"] - position["buy_price"]) / position["buy_price"],
                        "net_profit": proceeds - position["cost_basis"],
                    }
                )
            open_positions = remaining_positions

            held_codes = {position["stock_code"] for position in open_positions}
            pick_count = max(1, int(math.ceil(len(group) * percentile)))
            candidates = group.head(pick_count).copy()
            candidates["stock_code"] = candidates["stock_code"].astype(str).str.zfill(6)
            candidates = candidates[~candidates["stock_code"].isin(held_codes)].reset_index(drop=True)

            available_slots = max_positions - len(open_positions)
            if available_slots > 0 and not candidates.empty and cash > 0:
                candidates = candidates.head(available_slots)
                allocation = cash / len(candidates)
                new_positions: list[dict] = []
                for _, row in candidates.iterrows():
                    trade_window = self._resolve_trade_window(str(row["stock_code"]), signal_date, hold_days)
                    if trade_window is None:
                        continue
                    buy_price = trade_window["buy_price"]
                    shares = math.floor(allocation / (buy_price * (1.0 + buy_cost)))
                    if shares <= 0:
                        continue
                    cost_basis = shares * buy_price * (1.0 + buy_cost)
                    if cost_basis > cash:
                        continue
                    cash -= cost_basis
                    new_positions.append(
                        {
                            "signal_date": signal_date,
                            "stock_code": str(row["stock_code"]),
                            "score": float(row["ensemble_score"]),
                            "buy_date": trade_window["buy_date"],
                            "buy_price": buy_price,
                            "sell_date": trade_window["sell_date"],
                            "sell_price": trade_window["sell_price"],
                            "shares": shares,
                            "cost_basis": cost_basis,
                        }
                    )
                open_positions.extend(new_positions)

            mark_to_market = cash
            for position in open_positions:
                latest_price = position["buy_price"]
                prices = self._load_close_prices(position["stock_code"])
                visible = prices[prices["date"] <= signal_date]
                if not visible.empty:
                    latest_price = float(visible.iloc[-1]["close"])
                mark_to_market += position["shares"] * latest_price
            equity_rows.append({"date": str(signal_date.date()), "cash": cash, "equity": mark_to_market, "open_positions": len(open_positions)})

        final_signal_date = pd.to_datetime(equity_rows[-1]["date"]) if equity_rows else pd.Timestamp.today().normalize()
        for position in sorted(open_positions, key=lambda item: item["sell_date"]):
            proceeds = position["shares"] * position["sell_price"] * (1.0 - sell_cost)
            cash += proceeds
            trade_rows.append(
                {
                    "signal_date": str(position["signal_date"].date()),
                    "buy_date": str(position["buy_date"].date()),
                    "sell_date": str(position["sell_date"].date()),
                    "stock_code": position["stock_code"],
                    "score": position["score"],
                    "shares": position["shares"],
                    "buy_price": position["buy_price"],
                    "sell_price": position["sell_price"],
                    "gross_return": (position["sell_price"] - position["buy_price"]) / position["buy_price"],
                    "net_profit": proceeds - position["cost_basis"],
                }
            )
            equity_rows.append({"date": str(max(final_signal_date, position["sell_date"]).date()), "cash": cash, "equity": cash, "open_positions": 0})

        trades_df = pd.DataFrame(trade_rows).sort_values(["buy_date", "stock_code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
        equity_df = pd.DataFrame(equity_rows)
        if equity_df.empty:
            equity_df = pd.DataFrame([{"date": str(pd.Timestamp.today().date()), "cash": initial_capital, "equity": initial_capital, "open_positions": 0}])
        equity_df = equity_df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0.0)

        peak = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] / peak - 1.0
        losing_profit = float(trades_df.loc[trades_df["net_profit"] < 0, "net_profit"].sum()) if not trades_df.empty else 0.0
        summary = {
            "percentile": percentile,
            "max_positions": max_positions,
            "hold_days": hold_days,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "initial_capital": initial_capital,
            "final_equity": float(equity_df.iloc[-1]["equity"]),
            "total_return": float(equity_df.iloc[-1]["equity"] / initial_capital - 1.0),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "sharpe_ratio": float(equity_df["daily_return"].mean() / equity_df["daily_return"].std() * (252 ** 0.5)) if len(equity_df) > 1 and float(equity_df["daily_return"].std()) > 0 else 0.0,
            "trade_count": int(len(trades_df)),
            "win_rate": float((trades_df["net_profit"] > 0).mean()) if not trades_df.empty else 0.0,
            "profit_factor": float(trades_df.loc[trades_df["net_profit"] > 0, "net_profit"].sum() / abs(losing_profit)) if losing_profit < 0 else 0.0,
        }
        return RealisticBacktestResult(summary=summary, trades=trades_df, equity=equity_df)