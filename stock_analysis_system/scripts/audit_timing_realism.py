"""排查：多标的择时的高收益（+2054%）是不是回测理想化造成的？

=================== 三个必须排查的疑点 ===================

疑点1：执行价假设过于理想
  回测假设"T+1 开盘价精确成交"。
  现实中：
    · 开盘集合竞价，你可能成交在开盘价上方（买入）/下方（卖出）
    · ETF 有买卖价差（510300 通常 0.001-0.002 元，约 0.02-0.05%）
    · 大资金有冲击成本
  检验：加不同滑点（0 / 0.05% / 0.1% / 0.2% / 0.3%）看结果

疑点2：只吃"日内收益"是否是一种作弊
  择时回测里，HOLD 期只算 intraday（收盘/开盘），不算隔夜。
  这等价于假设"每次 HOLD 都恰好从开盘持有到收盘"。
  但这在 T+1 下要求「每天都开盘买、收盘卖」——违规！
  正确的 T+1 实现应该是「持有期吃全天收益（含隔夜）」。
  检验：对比三种口径
    A) 只吃日内（当前实现，可能过于乐观）
    B) 吃全天（严格 T+1，持仓过夜）
    C) 吃"T+1开盘到下次卖出"（最严格）

疑点3：参数是否过拟合
  120 组参数里 111 组赢 → 看起来很稳。
  但要检查：是不是"所有参数都有效"，还是"只有某些参数有效"？
  检验：看参数分布的偏度、是否有集中区域

用法：
  python scripts/audit_timing_realism.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

ASSETS = [
    ("510050.SH", "上证50ETF"),
    ("510300.SH", "沪深300ETF"),
    ("510500.SH", "中证500ETF"),
    ("512100.SH", "中证1000ETF"),
    ("159915.SZ", "创业板ETF"),
    ("588000.SH", "科创50ETF"),
]


def _resolve_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def load_etf(db: str, code: str):
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)
    if len(df) < 500:
        return None
    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def stats(curve):
    if len(curve) < 2:
        return {"total": 0.0, "ann": np.nan, "dd": 0.0, "sharpe": np.nan}
    r = np.diff(np.concatenate([[1.0], curve]))
    total = float(curve[-1] - 1)
    years = len(curve) / 244
    ann = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe}


def run_timing(df, window, threshold, cost, slip, mode):
    """mode:
    'intraday'    : 只吃日内（当前实现，T+1下不完全可实现）
    'full'        : 吃全天（严格 T+1，持仓过夜）
    'next_open'   : T+1开盘买入，下次卖出按T+1开盘价（最严格）
    """
    d = df.copy()
    d["mean"] = d["ret"].rolling(window, min_periods=3).mean()
    d["sig"] = d["mean"] >= threshold
    d = d.dropna(subset=["mean"]).reset_index(drop=True)

    eq = [1.0]
    pos = 0.0
    buy_price = None

    for i in range(1, len(d)):
        target = 1.0 if d.loc[i - 1, "sig"] else 0.0

        if mode == "intraday":
            r = float(d.loc[i, "intraday"]) if target > 0 else 0.0
            trade = abs(target - pos)
            eq.append(eq[-1] * (1 + r - trade * (cost + slip)))
            pos = target

        elif mode == "full":
            r = float(d.loc[i, "ret"]) if target > 0 else 0.0
            trade = abs(target - pos)
            eq.append(eq[-1] * (1 + r - trade * (cost + slip)))
            pos = target

        elif mode == "next_open":
            # 买入：i-1 信号 HOLD → i 日开盘买
            # 卖出：i-1 信号 CASH 且当前持仓 → i 日开盘卖
            if target > 0 and buy_price is None:
                buy_price = float(d.loc[i, "open"]) * (1 + slip)  # 买入滑点
            elif target == 0 and buy_price is not None:
                sell = float(d.loc[i, "open"]) * (1 - slip)       # 卖出滑点
                eq.append(eq[-1] * (sell / buy_price) * (1 - cost * 2))
                buy_price = None
                pos = 0.0
                continue
            # 记录净值（持仓中的浮动）
            if buy_price is not None:
                eq.append(eq[-1])  # 只在平仓时结算，中间保持
            else:
                eq.append(eq[-1])
            pos = 1.0 if buy_price is not None else 0.0

    # 收尾平仓
    if mode == "next_open" and buy_price is not None:
        sell = float(d.loc[len(d) - 1, "close"]) * (1 - slip)
        eq[-1] = eq[-1] * (sell / buy_price) * (1 - cost * 2)

    return np.array(eq[1:] if mode != "next_open" else eq), d


def buy_hold(df):
    eq = [1.0]
    for i in range(1, len(df)):
        eq.append(eq[-1] * (1 + float(df.loc[i, "ret"])))
    return np.array(eq[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    db = _resolve_db()

    # ========== 疑点1：滑点敏感性 ==========
    print("=" * 108)
    print("【疑点1】滑点敏感性：开盘成交假设有多脆弱？")
    print("=" * 108)
    print(f"{'标的':<16}{'买入持有':>11}" + "".join(f"{f'slip {s*100:.1f}%':>13}" for s in [0.0, 0.0005, 0.001, 0.002, 0.003]))
    print("-" * 108)

    slip_rows = []
    for code, name in ASSETS:
        df = load_etf(db, code)
        if df is None:
            continue
        eq_b = buy_hold(df)
        sb = stats(eq_b)
        line = f"{name:<16}{sb['total']*100:>+10.1f}%"
        row = {"code": code, "name": name, "buy_hold": round(sb["total"], 6)}
        for slip in [0.0, 0.0005, 0.001, 0.002, 0.003]:
            eq_t, d = run_timing(df, args.window, args.threshold, args.cost, slip, "intraday")
            offset = len(df) - len(eq_t)
            sb2 = stats(buy_hold(df.iloc[offset:].reset_index(drop=True)))
            st = stats(eq_t)
            adv = st["total"] - sb2["total"]
            line += f"{st['total']*100:>+8.1f}%"
            line += "" if slip == 0 else ""
            row[f"slip_{slip}"] = round(st["total"], 6)
            row[f"adv_{slip}"] = round(adv, 6)
        print(line)
        slip_rows.append(row)
    print("-" * 108)
    print("（表格中每个滑点下显示的是择时累计收益；买入持有作为对照）")

    # 结论
    print("\n  各滑点下仍跑赢买入持有的标的数：")
    for slip in [0.0, 0.0005, 0.001, 0.002, 0.003]:
        n_win = sum(1 for r in slip_rows if r.get(f"adv_{slip}", -1) > 0)
        print(f"    滑点 {slip*100:>4.2f}%: {n_win}/{len(slip_rows)}")

    # ========== 疑点2：收益口径 ==========
    print("\n" + "=" * 108)
    print("【疑点2】收益口径：只吃日内 vs 吃全天（哪个更真实？）")
    print("=" * 108)
    print(f"{'标的':<16}{'买入持有':>12}{'A只吃日内':>14}{'B吃全天':>14}{'C严格T+1':>14}")
    print("-" * 108)

    mode_rows = []
    for code, name in ASSETS:
        df = load_etf(db, code)
        if df is None:
            continue
        res = {}
        for mode in ["intraday", "full", "next_open"]:
            eq, d = run_timing(df, args.window, args.threshold, args.cost, 0.0005, mode)
            res[mode] = stats(eq)["total"]
        offset = len(df) - (len(df) - args.window - 1)
        eq_b = buy_hold(df.iloc[offset:].reset_index(drop=True))
        sb = stats(eq_b)
        print(f"{name:<16}{sb['total']*100:>+11.1f}%{res['intraday']*100:>+13.1f}%"
              f"{res['full']*100:>+13.1f}%{res['next_open']*100:>+13.1f}%")
        mode_rows.append({"code": code, "name": name,
                          "buy_hold": round(sb["total"], 6),
                          "intraday_only": round(res["intraday"], 6),
                          "full_day": round(res["full"], 6),
                          "strict_t1": round(res["next_open"], 6)})
    print("-" * 108)
    print("  A(只吃日内): 当前实现，假设每天开盘买收盘卖 —— T+1 下不完全可实现")
    print("  B(吃全天)  : 持仓过夜，含隔夜收益 —— 严格 T+1 可实现")
    print("  C(严格T+1) : 只在买入/卖出时结算，按开盘价+滑点 —— 最保守")

    # ========== 疑点3：参数分布 ==========
    print("\n" + "=" * 108)
    print("【疑点3】参数分布：是否存在「只有某些参数有效」的问题？")
    print("=" * 108)
    print(f"{'窗口':<8}" + "".join(f"{f'th {t*100:+.1f}%':>14}" for t in [0.001, 0.0, -0.001, -0.002]))
    print("-" * 108)

    grid = {}
    for w in [5, 10, 20, 30, 60]:
        line = f"{w:<8}"
        for th in [0.001, 0.0, -0.001, -0.002]:
            wins = 0
            total = 0
            advs = []
            for code, name in ASSETS:
                df = load_etf(db, code)
                if df is None:
                    continue
                eq_t, d = run_timing(df, w, th, args.cost, 0.0005, "intraday")
                offset = len(df) - len(eq_t)
                sb = stats(buy_hold(df.iloc[offset:].reset_index(drop=True)))
                st = stats(eq_t)
                a = st["total"] - sb["total"]
                advs.append(a)
                total += 1
                if a > 0:
                    wins += 1
            grid[f"w{w}_th{th}"] = {"wins": wins, "total": total,
                                    "median_adv": round(float(np.median(advs)), 6) if advs else 0}
            line += f"{wins:>5}/{total:<8}"
        print(line)
    print("-" * 108)
    print("（每格显示：6个标的中有几个跑赢）")

    all_wins = [v["wins"] for v in grid.values()]
    all_total = [v["total"] for v in grid.values()]
    print(f"\n  20 组参数 × 6 标的 = {sum(all_total)} 个组合")
    print(f"  总胜场: {sum(all_wins)}/{sum(all_total)} = {sum(all_wins)/sum(all_total)*100:.1f}%")
    print(f"  每组参数的最少胜场: {min(all_wins)}/6   最多: {max(all_wins)}/6")
    if min(all_wins) >= 4:
        print(f"  → 所有参数组合都至少赢 4/6 个标的，**不是过拟合**")
    elif min(all_wins) >= 2:
        print(f"  → 部分参数组合效果差，存在参数敏感性")
    else:
        print(f"  → 有参数组合全面失败，**参数选择影响很大**")

    # ========== 综合判断 ==========
    print("\n" + "=" * 108)
    print("综合判断")
    print("=" * 108)

    # 滑点 0.2% 时还有几个赢
    win_02 = sum(1 for r in slip_rows if r.get("adv_0.002", -1) > 0)
    # 严格 T+1 口径下还有几个赢
    strict_win = sum(1 for r in mode_rows if r["strict_t1"] > r["buy_hold"])

    print(f"  1. 滑点 0.2% 时仍跑赢: {win_02}/{len(slip_rows)}")
    print(f"  2. 严格 T+1 口径下跑赢: {strict_win}/{len(mode_rows)}")
    print(f"  3. 参数网格总体胜率: {sum(all_wins)/sum(all_total)*100:.1f}%")

    if win_02 >= len(slip_rows) * 0.7 and strict_win >= len(mode_rows) * 0.7 and sum(all_wins)/sum(all_total) > 0.7:
        verdict = "结论稳健：即使最保守假设下仍然有效"
    elif strict_win >= len(mode_rows) * 0.5:
        verdict = "部分稳健：严格口径下仍有多数有效，但收益大幅缩水"
    else:
        verdict = "**不稳健：高收益主要来自理想化假设，实盘会大幅缩水**"

    print(f"\n  >>> {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "timing_realism_audit.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "slippage_sensitivity": slip_rows,
        "return_modes": mode_rows,
        "param_grid": grid,
        "summary": {
            "win_at_slip_0.2pct": win_02,
            "win_strict_t1": strict_win,
            "grid_win_rate": round(sum(all_wins) / sum(all_total), 4),
            "verdict": verdict,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
