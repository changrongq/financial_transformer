from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import math

import numpy as np
import pandas as pd

from .config import load_config
from .logging_utils import setup_logger

logger = setup_logger("hierarchical.backtest")


def resolve_prediction_file(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    candidates = [
        results_dir / "ensemble_predictions.csv",
        results_dir / "micro_predictions.csv",
        results_dir / "latest_picks.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到可用预测文件，检查目录: {results_dir}")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict


class Backtester:
    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)

    def _load_predictions(self, prediction_file: str | Path) -> pd.DataFrame:
        path = Path(prediction_file)
        if not path.exists():
            raise FileNotFoundError(f"预测文件不存在: {path}")
        df = pd.read_csv(path)
        if "date" not in df.columns or "stock_code" not in df.columns:
            raise ValueError("预测文件至少需要包含 date 和 stock_code 列。")
        score_col = "ensemble_score" if "ensemble_score" in df.columns else "prediction_prob"
        df["score"] = pd.to_numeric(df[score_col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna(subset=["score"]).sort_values(["date", "score"], ascending=[True, False])

    def _load_close_prices(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.config.databases.trade_db)
        try:
            df = pd.read_sql_query(
                "SELECT stock_code, 日期, 收盘 FROM stock_data WHERE stock_code = ? AND 日期 BETWEEN ? AND ? ORDER BY 日期",
                conn,
                params=[stock_code, start_date, end_date],
            )
        finally:
            conn.close()
        if df.empty:
            return df
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.dropna(subset=["close"])

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        peak = equity_curve.cummax()
        drawdown = equity_curve / peak - 1
        return float(drawdown.min()) if not drawdown.empty else 0.0

    def run(
        self,
        prediction_file: str | Path,
        hold_days: int = 5,
        top_n: int = 10,
        initial_capital: float = 100000.0,
    ) -> BacktestResult:
        pred_df = self._load_predictions(prediction_file)
        if pred_df.empty:
            raise RuntimeError("预测文件为空。")

        grouped = pred_df.groupby(pred_df["date"].dt.normalize())
        cash = initial_capital
        equity_points = []
        trades = []

        for trade_date, group in grouped:
            picks = group.head(top_n).copy()
            if picks.empty:
                equity_points.append({"date": trade_date, "equity": cash})
                continue
            allocation = cash / max(len(picks), 1)
            day_profit = 0.0
            for _, row in picks.iterrows():
                start_date = trade_date.strftime("%Y-%m-%d")
                end_date = (trade_date + pd.Timedelta(days=hold_days + 10)).strftime("%Y-%m-%d")
                price_df = self._load_close_prices(str(row["stock_code"]), start_date, end_date)
                if len(price_df) < 2:
                    continue
                buy_idx = 1 if len(price_df) > 1 else 0
                sell_idx = min(buy_idx + hold_days - 1, len(price_df) - 1)
                buy_price = float(price_df.iloc[buy_idx]["close"])
                sell_price = float(price_df.iloc[sell_idx]["close"])
                if buy_price <= 0:
                    continue
                shares = math.floor(allocation / buy_price)
                if shares <= 0:
                    continue
                profit = (sell_price - buy_price) * shares
                ret = (sell_price - buy_price) / buy_price
                day_profit += profit
                trades.append({
                    "signal_date": str(trade_date.date()),
                    "buy_date": str(price_df.iloc[buy_idx]["date"].date()),
                    "sell_date": str(price_df.iloc[sell_idx]["date"].date()),
                    "stock_code": row["stock_code"],
                    "score": float(row["score"]),
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "shares": shares,
                    "profit": profit,
                    "return": ret,
                })
            cash += day_profit
            equity_points.append({"date": trade_date, "equity": cash})

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_points)
        if equity_df.empty:
            equity_df = pd.DataFrame([{"date": pd.Timestamp.today().normalize(), "equity": initial_capital}])
        equity_df = equity_df.sort_values("date").reset_index(drop=True)
        equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0.0)

        total_return = float(equity_df.iloc[-1]["equity"] / initial_capital - 1)
        sharpe = 0.0
        if equity_df["daily_return"].std() not in (0, np.nan):
            std = float(equity_df["daily_return"].std())
            if std > 0:
                sharpe = float(equity_df["daily_return"].mean() / std * np.sqrt(252))
        win_rate = float((trades_df["profit"] > 0).mean()) if not trades_df.empty else 0.0
        profit_factor = float(
            trades_df.loc[trades_df["profit"] > 0, "profit"].sum() /
            abs(trades_df.loc[trades_df["profit"] < 0, "profit"].sum())
        ) if not trades_df.empty and (trades_df.loc[trades_df["profit"] < 0, "profit"].sum() != 0) else 0.0

        summary = {
            "initial_capital": float(initial_capital),
            "final_equity": float(equity_df.iloc[-1]["equity"]),
            "total_return": total_return,
            "max_drawdown": self._max_drawdown(equity_df["equity"]),
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trade_count": int(len(trades_df)),
        }

        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        trades_path = self.config.results_dir / "backtest_trades.csv"
        equity_path = self.config.results_dir / "backtest_equity.csv"
        summary_path = self.config.results_dir / "backtest_summary.json"
        trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")
        pd.Series(summary).to_json(summary_path, force_ascii=False, indent=2)

        return BacktestResult(trades=trades_df, daily_equity=equity_df, summary=summary)
