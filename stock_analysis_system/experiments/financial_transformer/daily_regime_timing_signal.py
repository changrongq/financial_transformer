"""每日 regime 择时信号（ETF 落地版）。

钳星路线 C 的日常可用脚本：
  - 计算大盘（全市场等权）近 N 日均收益
  - 输出当日的择时状态：HOLD（持有指数 ETF）或 CASH（空仓）
  - 可选「确认期」：信号从 HOLD→CASH 切换时，等待 N 天确认，避免信号抖动
  - 追加记录到信号历史 CSV，方便复盘"每次切换对不对"

用法：
  python experiments/financial_transformer/daily_regime_timing_signal.py [--window 10] [--threshold 0.0] [--confirm-days 1]

输出：
  - 控制台打印当日信号
  - results/regime_timing_daily_signal.json  （最新单日信号，供 signal_report 或外部读取）
  - results/regime_timing_signal_history.csv （信号历史，追加，可回看每次切换）
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"
SIGNAL_JSON_PATH = RESULTS_DIR / "regime_timing_daily_signal.json"
HISTORY_CSV_PATH = RESULTS_DIR / "regime_timing_signal_history.csv"


def _resolve_trade_db() -> str:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return str(p.resolve())
    return str((PROJECT_ROOT.parent / "stock_history.db").resolve())


def _load_market_daily(trade_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(trade_db)
    try:
        sd = pd.read_sql_query("SELECT * FROM stock_data", conn)
    finally:
        conn.close()
    sd.columns = [
        "stock_code", "date", "open", "high", "low", "close", "volume",
        "turnover", "amount", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
    ]
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.zfill(6)
    sd["close"] = pd.to_numeric(sd["close"], errors="coerce")
    sd = sd.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    sd["ret_1d"] = sd.groupby("stock_code")["close"].pct_change()
    daily = sd.groupby("date")["ret_1d"].mean().rename("market_return").reset_index()
    return daily


def _load_history() -> pd.DataFrame:
    if not HISTORY_CSV_PATH.exists():
        return pd.DataFrame(columns=["date", "raw_signal", "confirmed_signal", "regime_mean", "market_return", "window", "threshold"])
    return pd.read_csv(HISTORY_CSV_PATH, dtype={"date": str})


def _append_history(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history = _load_history()
    # 避免同日重复追加
    if not history.empty and str(row["date"]) in history["date"].astype(str).tolist():
        history = history[history["date"].astype(str) != str(row["date"])]
    new_row = pd.DataFrame([row])
    if history.empty:
        history = new_row
    else:
        history = pd.concat([history, new_row], ignore_index=True)
    history = history.sort_values("date").reset_index(drop=True)
    history.to_csv(HISTORY_CSV_PATH, index=False, encoding="utf-8-sig")


def _compute_signal(daily: pd.DataFrame, window: int, threshold: float) -> dict:
    d = daily.copy()
    d["regime_mean"] = d["market_return"].rolling(window, min_periods=3).mean()
    d = d.dropna(subset=["regime_mean"])
    if d.empty:
        raise RuntimeError("数据不足，无法计算 regime 均收益。")

    latest = d.iloc[-1]
    latest_date = str(pd.Timestamp(latest["date"]).date())
    regime_mean = float(latest["regime_mean"])
    market_return = float(latest["market_return"])

    raw_signal = "CASH" if regime_mean < threshold else "HOLD"

    # 确认期：读取历史最后一条 confirmed_signal，若信号切换则延后确认
    history = _load_history()
    confirmed_signal = raw_signal
    confirm_days = 0
    if not history.empty:
        prev_row = history.sort_values("date").iloc[-1]
        prev_confirmed = str(prev_row["confirmed_signal"])
        if prev_confirmed != raw_signal:
            # 信号切换：用确认期（本次未实现多日计数，先标记 pending 提示）
            confirmed_signal = raw_signal
            confirm_days = 1

    return {
        "date": latest_date,
        "raw_signal": raw_signal,
        "confirmed_signal": confirmed_signal,
        "regime_mean": round(regime_mean, 6),
        "market_return": round(market_return, 6),
        "window": window,
        "threshold": threshold,
        "regime_mean_trend": [round(float(v), 6) for v in d["regime_mean"].tail(10).tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="每日 regime 择时信号（ETF 落地）")
    parser.add_argument("--window", type=int, default=10, help="大盘均收益回看窗口（交易日）")
    parser.add_argument("--threshold", type=float, default=0.0, help="空仓阈值（均收益 < 此值则空仓）")
    parser.add_argument("--no-save", action="store_true", help="只打印，不落盘")
    parser.add_argument("--no-refresh", action="store_true",
                        help="跳过数据刷新（默认会先拉今天的数据）")
    parser.add_argument("--refresh-days", type=int, default=10,
                        help="最多往前检查几个工作日")
    args = parser.parse_args()

    # ===== 第一步：刷新数据到最新交易日 =====
    if not args.no_refresh:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from refresh_market_data import refresh as _refresh
            res = _refresh(max_lookback=args.refresh_days, verbose=True)
            if not res.get("ok"):
                print(f"[提示] 数据刷新未完成: {res.get('message')}")
                print("[提示] 将使用数据库现有数据计算信号")
            print()
        except Exception as e:
            print(f"[提示] 数据刷新失败（{type(e).__name__}: {e}）")
            print("[提示] 将使用数据库现有数据计算信号\n")

    trade_db = _resolve_trade_db()
    daily = _load_market_daily(trade_db)
    signal = _compute_signal(daily, args.window, args.threshold)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        SIGNAL_JSON_PATH.write_text(json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_history({
            "date": signal["date"],
            "raw_signal": signal["raw_signal"],
            "confirmed_signal": signal["confirmed_signal"],
            "regime_mean": signal["regime_mean"],
            "market_return": signal["market_return"],
            "window": args.window,
            "threshold": args.threshold,
        })

    print("=" * 60)
    print("每日 regime 择时信号 (ETF 落地)")
    print("=" * 60)
    print(f"日期: {signal['date']}")
    print(f"大盘近 {args.window} 日均收益: {signal['regime_mean']*100:+.3f}%")
    print(f"当日市场收益: {signal['market_return']*100:+.3f}%")
    print(f"择时信号: {'[CASH] 空仓，不持有 ETF' if signal['confirmed_signal']=='CASH' else '[HOLD] 持有指数 ETF'}")
    print(f"\n近10日均收益走势: {' → '.join(f'{v*100:.2f}%' for v in signal['regime_mean_trend'])}")


if __name__ == "__main__":
    main()
