"""Filter 完整区间回测 (直接读取 backfill_predictions.csv，不走 patch)"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("d:/qcr/Stock/stock_analysis_system")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/financial_transformer/src"))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.backtest import TransformerStrategyBacktester

PRED_PATH = ROOT / "experiments/financial_transformer/results/formal_filtered_v5_e80_bsstop_seed33/backfill_predictions.csv"
CONFIG_PATH = ROOT / "experiments/financial_transformer/config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml"

FAIR = {
    "short_up_threshold": 0.254,
    "take_profit_threshold": 0.572,
    "stop_loss_threshold": 0.50,
    "top_k_per_day": 2,
    "hold_days": 35,
    "take_profit": 0.14,
    "stop_loss": -0.06,
}

# 读取预测数据
df = pd.read_csv(PRED_PATH, dtype={"stock_code": str})
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
df["date"] = pd.to_datetime(df["date"])
for col in ["pred_short_up", "pred_take_profit", "pred_stop_loss", "pred_max_gain", "pred_max_drawdown"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["date", "pred_short_up", "pred_take_profit", "pred_stop_loss"])

# 用 strategy_score 计算 signal_score
from financial_transformer_experiment.scoring import strategy_score_frame
config = load_config(str(CONFIG_PATH))
df["signal_score"] = strategy_score_frame(df, exit_label_mode=config.multitask.exit_label_mode)

windows = [
    ("full_jan_jun", "2026-01-01", "2026-06-04"),
    ("old_jan_mar", "2026-01-01", "2026-03-04"),
    ("recent_mar_jun", "2026-03-05", "2026-06-04"),
]

for label, start, end in windows:
    print(f"\n{'='*60}")
    print(f"  Filter: {label} ({start} ~ {end})")
    print(f"{'='*60}")

    window_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    # 应用阈值过滤
    passed = window_df[
        (window_df["pred_short_up"] > FAIR["short_up_threshold"])
        & (window_df["pred_take_profit"] > FAIR["take_profit_threshold"])
        & (window_df["pred_stop_loss"] < FAIR["stop_loss_threshold"])
    ].copy()

    # top_k
    passed = passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
    passed = passed.groupby("date", group_keys=False).head(FAIR["top_k_per_day"]).reset_index(drop=True)

    print(f"  window rows: {len(window_df)}, passed: {len(passed)}, unique dates: {passed['date'].nunique()}")
    if passed.empty:
        print(f"  No signals in this window")
        continue

    # 手工回测（支持多持仓）
    import sqlite3

    conn = sqlite3.connect(config.databases.trade_db)
    trades = []
    cash = 1_000_000.0
    positions = {}  # stock_code -> {shares, entry_price, entry_date, signal_date, label}

    # 先按日期排序所有信号
    passed = passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).reset_index(drop=True)

    # 获取所有相关股票的历史价格（缓存）
    all_codes = sorted(passed["stock_code"].unique())
    price_cache = {}
    for sc in all_codes:
        pdf = pd.read_sql_query(
            "SELECT 日期, 开盘, 最高, 最低, 收盘 FROM stock_data WHERE stock_code=? ORDER BY 日期",
            conn, params=[sc]
        )
        pdf.columns = ["date", "open", "high", "low", "close"]
        pdf["date"] = pd.to_datetime(pdf["date"])
        for c in ["open", "high", "low", "close"]:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
        price_cache[sc] = pdf.dropna(subset=["open", "close"]).reset_index(drop=True)

    signal_idx = 0
    signals_list = passed.to_dict("records")

    # 遍历交易日（用预测日期 + 往后推的最晚退出日期）
    all_signal_dates = sorted(passed["date"].unique())
    first_date = all_signal_dates[0]
    last_date = pd.Timestamp("2026-06-04") + pd.Timedelta(days=FAIR["hold_days"] + 5)

    # 生成所有需要检查的交易日
    trade_dates = pd.date_range(first_date, last_date, freq="B")  # business days

    for check_date in trade_dates:
        # 1. 检查持仓退出
        to_remove = []
        for sc, pos in list(positions.items()):
            pdf = price_cache[sc]
            bar = pdf[pdf["date"] == check_date]
            if bar.empty:
                continue
            bar = bar.iloc[0]
            current_close = float(bar["close"])
            pct = (current_close - pos["entry_price"]) / pos["entry_price"]
            days_held = (check_date - pos["entry_date"]).days

            should_exit = False
            exit_reason = None
            if days_held >= FAIR["hold_days"]:
                should_exit = True
                exit_reason = "hold_days"
            elif pct >= FAIR["take_profit"]:
                should_exit = True
                exit_reason = "take_profit"
            elif pct <= FAIR["stop_loss"]:
                should_exit = True
                exit_reason = "stop_loss"

            if should_exit:
                pnl = pos["shares"] * (current_close - pos["entry_price"])
                cash += pos["shares"] * current_close
                trades.append({
                    "stock_code": sc,
                    "signal_date": pos["signal_date"],
                    "entry_date": str(pos["entry_date"].date()),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_date": str(check_date.date()),
                    "exit_price": round(current_close, 2),
                    "return": round(pct, 6),
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                })
                to_remove.append(sc)

        for sc in to_remove:
            del positions[sc]

        # 2. 检查新信号入场
        positions_value = sum(
            pos["shares"] * float(price_cache[sc][price_cache[sc]["date"] == check_date].iloc[0]["close"])
            for sc, pos in positions.items()
            if not price_cache[sc][price_cache[sc]["date"] == check_date].empty
        )
        total_assets = cash + positions_value

        while signal_idx < len(signals_list):
            sig = signals_list[signal_idx]
            sig_date = pd.Timestamp(sig["date"])
            if sig_date > check_date:
                break

            sc = sig["stock_code"]
            if sc in positions:
                signal_idx += 1
                continue  # 已持仓，跳过

            pdf = price_cache[sc]
            # 下一交易日入场
            next_bar = pdf[pdf["date"] >= sig_date + pd.Timedelta(days=1)]
            if next_bar.empty:
                signal_idx += 1
                continue
            next_bar = next_bar.iloc[0]
            if next_bar["date"] > check_date:
                break  # 还没到入场日

            # 在 check_date 这天入场
            entry_price = float(next_bar["open"])
            weight = min(0.20, max(0.08, max(float(sig.get("signal_score", 0.5)), 0.3) * 0.15))
            capital = max(total_assets, cash) * weight
            shares = max(100, int(capital / entry_price / 100) * 100)
            cost = shares * entry_price

            if cost > cash:
                # 现金不够，按可买数量调整
                shares = max(100, int(cash / entry_price / 100) * 100)
                cost = shares * entry_price

            if cost > cash or shares < 100:
                signal_idx += 1
                continue

            cash -= cost
            positions[sc] = {
                "shares": shares,
                "entry_price": entry_price,
                "entry_date": pd.Timestamp(next_bar["date"]),
                "signal_date": str(sig_date.date()),
            }
            signal_idx += 1

    # 清仓剩余持仓（按最后一天收盘价）
    for sc, pos in list(positions.items()):
        pdf = price_cache[sc]
        last_bar = pdf.iloc[-1]
        current_close = float(last_bar["close"])
        pct = (current_close - pos["entry_price"]) / pos["entry_price"]
        pnl = pos["shares"] * (current_close - pos["entry_price"])
        cash += pos["shares"] * current_close
        trades.append({
            "stock_code": sc,
            "signal_date": pos["signal_date"],
            "entry_date": str(pos["entry_date"].date()),
            "entry_price": round(pos["entry_price"], 2),
            "exit_date": str(last_bar["date"].date()),
            "exit_price": round(current_close, 2),
            "return": round(pct, 6),
            "pnl": round(pnl, 2),
            "exit_reason": "end_of_period",
        })

    conn.close()

    if not trades:
        print("  No trades")
        continue

    tdf = pd.DataFrame(trades)
    win_trades = len(tdf[tdf["return"] > 0])
    total_trades = len(tdf)
    total_return = total_assets / 1_000_000.0 - 1.0

    # Sharpe (简化)
    daily_rets = []
    for t in trades:
        days = (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
        daily_rets.append(t["return"] / max(days, 1))
    if daily_rets:
        sharpe = np.mean(daily_rets) / max(np.std(daily_rets), 1e-9) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        cumulative *= (1 + t["return"])
        peak = max(peak, cumulative)
        max_dd = min(max_dd, (cumulative - peak) / peak)

    # Profit factor
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)) or 1e-9
    pf = gross_profit / gross_loss

    print(f"  trades={total_trades}, win={win_trades}, win_rate={win_trades/total_trades:.2%}")
    print(f"  total_return={total_return:.4%}, sharpe={sharpe:.2f}, max_dd={max_dd:.4%}, pf={pf:.2f}")

print("\nDone.")
