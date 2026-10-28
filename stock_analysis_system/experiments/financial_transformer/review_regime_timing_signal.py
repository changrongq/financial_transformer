"""regime 择时信号复盘 —— 统计每次 HOLD/CASH 切换的质量。

回答钳星复盘时要问的问题：
  - 每次「转 CASH（空仓）」后，接下来 N 天市场是涨还是跌？（跌=躲对了）
  - 每次「转 HOLD（持有）」后，接下来 N 天市场是涨还是跌？（涨=接对了）
  - 整体择时 vs 一直持有，谁更好？

模式：
  --source db    从数据库回算整个历史区间的每日信号（默认，数据完整）
  --source csv   读 regime_timing_signal_history.csv（未来跑久后按实际记录复盘）

用法：
  python experiments/financial_transformer/review_regime_timing_signal.py [--window 10] [--forward-days 5] [--start 2024-10-01]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RESULTS_DIR = ROOT / "results"
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


def _max_drawdown(series: pd.Series) -> float:
    cum = (1.0 + series.fillna(0.0)).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _review_from_db(daily: pd.DataFrame, window: int, threshold: float, start: str, forward_days: int) -> dict:
    d = daily[daily["date"] >= start].copy()
    d["regime_mean"] = d["market_return"].rolling(window, min_periods=3).mean()
    d = d.dropna(subset=["regime_mean"]).reset_index(drop=True)
    d["signal"] = np.where(d["regime_mean"] < threshold, "CASH", "HOLD")

    # 未来 forward_days 累计收益（从次日开始）
    d["fwd_cum_ret"] = d["market_return"].shift(-1).rolling(forward_days, min_periods=1).sum().shift(-(forward_days - 1))

    # ===== 维度1：切换对错率 =====
    d["prev_signal"] = d["signal"].shift(1)
    d["switched"] = d["signal"] != d["prev_signal"]
    switches = d[d["switched"] & d["prev_signal"].notna()].copy()

    switch_rows = []
    for _, s in switches.iterrows():
        switch_date = s["date"]
        to_signal = s["signal"]
        fwd_val = s["fwd_cum_ret"]
        switch_rows.append({
            "date": str(pd.Timestamp(switch_date).date()),
            "to": to_signal,
            "forward_ret": None if pd.isna(fwd_val) else round(float(fwd_val), 6),
        })

    to_cash = [r for r in switch_rows if r["to"] == "CASH"]
    to_hold = [r for r in switch_rows if r["to"] == "HOLD"]
    cash_correct = sum(1 for r in to_cash if r["forward_ret"] is not None and r["forward_ret"] < 0)
    hold_correct = sum(1 for r in to_hold if r["forward_ret"] is not None and r["forward_ret"] > 0)
    cash_avg = float(np.mean([r["forward_ret"] for r in to_cash if r["forward_ret"] is not None])) if any(r["forward_ret"] is not None for r in to_cash) else None
    hold_avg = float(np.mean([r["forward_ret"] for r in to_hold if r["forward_ret"] is not None])) if any(r["forward_ret"] is not None for r in to_hold) else None

    # ===== 维度2：暴跌躲避率 =====
    crash_threshold = -0.02  # 单日跌幅超过 2% 视为暴跌
    crash_days = d[d["market_return"] < crash_threshold].copy()
    crash_dodged = crash_days[crash_days["signal"] == "CASH"]
    crash_hit = crash_days[crash_days["signal"] == "HOLD"]
    # 被躲开的暴跌：CASH 期的暴跌日，当日收益=0（空仓），实则避开了损失
    dodged_loss = float(-crash_dodged["market_return"].sum()) if not crash_dodged.empty else 0.0
    hit_loss = float(-crash_hit["market_return"].sum()) if not crash_hit.empty else 0.0

    # ===== 维度3：累计收益/回撤对比 =====
    d["strategy_return"] = np.where(d["signal"] == "HOLD", d["market_return"], 0.0)
    buy_hold_ret = float((1.0 + d["market_return"]).cumprod().iloc[-1] - 1.0)
    strategy_ret = float((1.0 + d["strategy_return"]).cumprod().iloc[-1] - 1.0)
    buy_hold_dd = _max_drawdown(d["market_return"])
    strategy_dd = _max_drawdown(d["strategy_return"])

    return {
        # 维度1：切换对错
        "total_switches": len(switch_rows),
        "to_cash_count": len(to_cash),
        "to_hold_count": len(to_hold),
        "to_cash_correct_ratio": f"{cash_correct}/{len(to_cash)}" if to_cash else "N/A",
        "to_hold_correct_ratio": f"{hold_correct}/{len(to_hold)}" if to_hold else "N/A",
        "avg_forward_ret_after_cash": cash_avg,
        "avg_forward_ret_after_hold": hold_avg,
        # 维度2：暴跌躲避
        "crash_threshold": crash_threshold,
        "crash_days_total": int(len(crash_days)),
        "crash_days_dodged": int(len(crash_dodged)),
        "crash_days_hit": int(len(crash_hit)),
        "crash_dodge_ratio": f"{len(crash_dodged)}/{len(crash_days)}" if len(crash_days) else "N/A",
        "dodged_loss_amount": round(dodged_loss, 6),
        "hit_loss_amount": round(hit_loss, 6),
        # 维度3：累计收益/回撤
        "buy_hold_return": round(buy_hold_ret, 6),
        "strategy_return": round(strategy_ret, 6),
        "buy_hold_max_drawdown": round(buy_hold_dd, 6),
        "strategy_max_drawdown": round(strategy_dd, 6),
        # 明细
        "switch_detail": switch_rows,
        "forward_days": forward_days,
        "start": start,
        "end": str(pd.Timestamp(d["date"].iloc[-1]).date()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="regime 择时信号复盘")
    parser.add_argument("--source", choices=["db", "csv"], default="db")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--forward-days", type=int, default=5, help="切换后看未来 N 日累计收益")
    parser.add_argument("--start", default="2024-10-01")
    args = parser.parse_args()

    if args.source == "db":
        trade_db = _resolve_trade_db()
        daily = _load_market_daily(trade_db)
        review = _review_from_db(daily, args.window, args.threshold, args.start, args.forward_days)
    else:
        review = {"message": "csv 模式暂未实现，请用 --source db"}

    # 输出
    print("=" * 64)
    print(f"regime 择时信号复盘（窗口{args.window}日，区间 {review.get('start','?')}~{review.get('end','?')}）")
    print("=" * 64)
    if "message" in review:
        print(review["message"])
        return

    print("\n【维度1：切换对错率】（提醒：单看会误导，择时不靠每次对，靠躲开暴跌）")
    print(f"  信号切换总次数: {review['total_switches']}")
    print(f"  转 CASH 躲对(未来{args.forward_days}日下跌): {review['to_cash_correct_ratio']}")
    print(f"  转 HOLD 接对(未来{args.forward_days}日上涨): {review['to_hold_correct_ratio']}")

    print("\n【维度2：暴跌躲避率】（择时真正的价值）")
    print(f"  暴跌日(单日跌幅>2%)总数: {review['crash_days_total']}")
    print(f"  被躲开(落在CASH期): {review['crash_days_dodged']}  ({review['crash_dodge_ratio']})")
    print(f"  没躲开(落在HOLD期): {review['crash_days_hit']}")
    print(f"  躲开的累计损失: {review['dodged_loss_amount']*100:+.2f}%")
    print(f"  没躲开的累计损失: {review['hit_loss_amount']*100:+.2f}%")

    print("\n【维度3：累计收益/回撤对比】")
    print(f"  一直满仓: 收益 {review['buy_hold_return']*100:+.2f}%, 最大回撤 {review['buy_hold_max_drawdown']*100:.2f}%")
    print(f"  择时策略: 收益 {review['strategy_return']*100:+.2f}%, 最大回撤 {review['strategy_max_drawdown']*100:.2f}%")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "regime_timing_review_summary.json"
    out.write_text(json.dumps(review, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
