"""多标的稳健性验证：择时（T+1开盘执行）在中证1000等标的上是否也有效。

背景：
  510300 沪深300ETF 已验证：14年 +160.1% vs 买入持有 +57.9%，19/20参数胜。
  但这是单一标的。可能只是"沪深300特有"。

  本脚本在 6 个不同风格的 ETF 上跑同一套择时规则，检验普适性。

标的（按流动性/代表性挑选）：
  510050 上证50ETF     大盘蓝筹，20.5年（最长）
  510300 沪深300ETF    大盘，14.4年（已验证，作为基准）
  510500 中证500ETF    中盘，13.5年
  512100 中证1000ETF   小盘，9.9年  ← 船长特别提到
  159915 创业板ETF     成长，14.8年
  588000 科创50ETF     科技，5.9年

规则（全部 T+1 开盘执行，单边万5）：
  大盘/标的自身 10日均收益 >= 0 → HOLD
  < 0 → CASH

参数网格：窗口 5/10/20/30/60 × 阈值 +0.1%/0%/-0.1%/-0.2%

用法：
  python scripts/validate_timing_multi_assets.py
  python scripts/validate_timing_multi_assets.py --cost 0.001
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
    ("510050.SH", "上证50ETF", "大盘蓝筹"),
    ("510300.SH", "沪深300ETF", "大盘"),
    ("510500.SH", "中证500ETF", "中盘"),
    ("512100.SH", "中证1000ETF", "小盘"),
    ("159915.SZ", "创业板ETF", "成长"),
    ("588000.SH", "科创50ETF", "科技"),
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


def load_etf(db: str, code: str) -> pd.DataFrame | None:
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
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def stats(curve: np.ndarray) -> dict:
    if len(curve) < 2:
        return {"total": 0.0, "ann": np.nan, "dd": 0.0, "sharpe": np.nan}
    r = np.diff(np.concatenate([[1.0], curve]))
    total = float(curve[-1] - 1)
    years = len(curve) / 244
    ann = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe}


def run_timing(df: pd.DataFrame, window: int, threshold: float, cost: float):
    """T日收盘算信号 → T+1开盘执行 → 吃T+1日内收益。"""
    d = df.copy()
    d["mean"] = d["ret"].rolling(window, min_periods=3).mean()
    d["sig"] = d["mean"] >= threshold
    d = d.dropna(subset=["mean"]).reset_index(drop=True)

    eq = [1.0]
    pos = 0.0
    for i in range(1, len(d)):
        target = 1.0 if d.loc[i - 1, "sig"] else 0.0   # T-1 信号
        r = float(d.loc[i, "intraday"]) if target > 0 else 0.0  # T 日开盘买入，吃日内
        trade = abs(target - pos)
        eq.append(eq[-1] * (1 + r - trade * cost))
        pos = target
    return np.array(eq[1:]), d


def buy_hold_curve(df: pd.DataFrame) -> np.ndarray:
    """买入持有：T+1 开盘买入，之后一直持有（吃全天收益）。"""
    d = df.reset_index(drop=True)
    eq = [1.0]
    for i in range(1, len(d)):
        eq.append(eq[-1] * (1 + float(d.loc[i, "ret"])))
    return np.array(eq[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--start", default=None, help="统一起始日期 YYYY-MM-DD（对比用）")
    args = parser.parse_args()

    db = _resolve_db()
    print("=" * 104)
    print("多标的择时稳健性验证（T+1 开盘执行）")
    print("=" * 104)
    print(f"单边成本 {args.cost*100:.3f}%\n")

    summary = []
    detail = {}

    # ===== 第一部分：各标的默认参数（窗口10/阈值0）=====
    print(f"{'标的':<16}{'风格':<10}{'年数':>6}{'买入持有':>12}{'择时':>12}"
          f"{'优势':>11}{'回撤(择时/买入)':>18}{'夏普(择时/买入)':>18}")
    print("-" * 104)

    for code, name, style in ASSETS:
        df = load_etf(db, code)
        if df is None:
            print(f"{name:<16}{style:<10} 无数据或不足500天")
            continue
        if args.start:
            df = df[df["date"] >= args.start].reset_index(drop=True)
            if len(df) < 300:
                continue

        eq_t, d = run_timing(df, 10, 0.0, args.cost)
        eq_b = buy_hold_curve(df.iloc[d["date"].index[0] if False else 0:].reset_index(drop=True))
        # 对齐：择时起点比 df 晚 window 天，买入持有用同样起点
        offset = len(df) - len(eq_t)
        eq_b = buy_hold_curve(df.iloc[offset:].reset_index(drop=True))
        n = min(len(eq_t), len(eq_b))
        st, sb = stats(eq_t[:n]), stats(eq_b[:n])
        adv = st["total"] - sb["total"]
        flag = "WIN " if adv > 0 else "LOSE"

        print(f"{name:<16}{style:<10}{len(df)/244:>6.1f}{sb['total']*100:>+11.1f}%"
              f"{st['total']*100:>+11.1f}%{flag}{adv*100:>+7.1f}pp"
              f"{st['dd']*100:>9.1f}%/{sb['dd']*100:>6.1f}%"
              f"{st['sharpe']:>9.2f}/{sb['sharpe']:>6.2f}")

        summary.append({
            "code": code, "name": name, "style": style,
            "years": round(len(df) / 244, 1),
            "buy_hold": round(sb["total"], 6), "timing": round(st["total"], 6),
            "advantage": round(adv, 6),
            "timing_dd": round(st["dd"], 6), "bh_dd": round(sb["dd"], 6),
            "timing_sharpe": round(st["sharpe"], 4), "bh_sharpe": round(sb["sharpe"], 4),
        })

    print("-" * 104)
    if summary:
        n_win = sum(1 for s in summary if s["advantage"] > 0)
        print(f"\n跑赢买入持有: {n_win}/{len(summary)}")
        print(f"优势中位数: {np.median([s['advantage'] for s in summary])*100:+.1f}pp")
        print(f"回撤改善中位数: {np.median([s['timing_dd']-s['bh_dd'] for s in summary])*100:+.1f}pp")
        print(f"夏普改善中位数: {np.median([s['timing_sharpe']-s['bh_sharpe'] for s in summary]):+.2f}")

    # ===== 第二部分：每个标的的参数网格 =====
    print("\n" + "=" * 104)
    print("参数网格稳健性（每个标的 20 组参数，统计胜率）")
    print("=" * 104)
    print(f"{'标的':<16}{'胜场':>10}{'优势中位数':>14}{'最好':>12}{'最差':>12}")
    print("-" * 104)

    for code, name, style in ASSETS:
        df = load_etf(db, code)
        if df is None:
            continue
        if args.start:
            df = df[df["date"] >= args.start].reset_index(drop=True)
            if len(df) < 300:
                continue

        wins, advs = 0, []
        for w in [5, 10, 20, 30, 60]:
            for th in [0.001, 0.0, -0.001, -0.002]:
                eq_t, d = run_timing(df, w, th, args.cost)
                offset = len(df) - len(eq_t)
                eq_b = buy_hold_curve(df.iloc[offset:].reset_index(drop=True))
                n = min(len(eq_t), len(eq_b))
                st, sb = stats(eq_t[:n]), stats(eq_b[:n])
                a = st["total"] - sb["total"]
                advs.append(a)
                if a > 0:
                    wins += 1
        if advs:
            print(f"{name:<16}{wins:>4}/{len(advs):<5}{np.median(advs)*100:>+13.1f}pp"
                  f"{max(advs)*100:>+11.1f}pp{min(advs)*100:>+11.1f}pp")
            detail[name] = {"wins": wins, "total": len(advs),
                            "median_adv": round(float(np.median(advs)), 6),
                            "best": round(float(max(advs)), 6),
                            "worst": round(float(min(advs)), 6)}

    print("-" * 104)

    # ===== 第三部分：分年度一致性（中证1000 + 上证50）=====
    print("\n" + "=" * 104)
    print("分年度一致性（默认参数，验证是否每年都有效）")
    print("=" * 104)

    for code, name, style in [("512100.SH", "中证1000ETF", "小盘"),
                              ("510050.SH", "上证50ETF", "大盘蓝筹")]:
        df = load_etf(db, code)
        if df is None:
            continue
        eq_t, d = run_timing(df, 10, 0.0, args.cost)
        offset = len(df) - len(eq_t)
        sub = df.iloc[offset:].reset_index(drop=True)
        eq_b = buy_hold_curve(sub)
        n = min(len(eq_t), len(eq_b))
        eq_t, eq_b = eq_t[:n], eq_b[:n]
        dd = sub.iloc[:n].reset_index(drop=True)
        dd["year"] = dd["date"].dt.year

        print(f"\n  {name}（{len(dd)/244:.1f}年）")
        print(f"  {'年份':<8}{'买入持有':>12}{'择时':>12}{'差额':>12}")
        print("  " + "-" * 46)
        wins = tot = 0
        for y in sorted(dd["year"].unique()):
            mask = (dd["year"] == y).values
            idx = np.where(mask)[0]
            if len(idx) < 60:
                continue
            prev_t = eq_t[idx[0] - 1] if idx[0] > 0 else 1.0
            prev_b = eq_b[idx[0] - 1] if idx[0] > 0 else 1.0
            rb = eq_b[idx[-1]] / prev_b - 1
            rt = eq_t[idx[-1]] / prev_t - 1
            wins += int(rt > rb)
            tot += 1
            print(f"  {y:<8}{rb*100:>+11.1f}%{rt*100:>+11.1f}%{(rt-rb)*100:>+11.1f}pp")
        print("  " + "-" * 46)
        if tot:
            print(f"  逐年胜率: {wins}/{tot} = {wins/tot*100:.0f}%")

    # ===== 结论 =====
    print("\n" + "=" * 104)
    print("结论")
    print("=" * 104)
    if summary:
        n_win = sum(1 for s in summary if s["advantage"] > 0)
        print(f"  {len(summary)} 个不同风格标的（大盘/中盘/小盘/成长/科技）")
        print(f"  跑赢买入持有: {n_win}/{len(summary)}")
        if n_win == len(summary):
            print(f"  >>> 全部胜出，择时规则具有普适性")
        elif n_win >= len(summary) * 0.7:
            print(f"  >>> 多数胜出，规则基本普适")
        else:
            print(f"  >>> 只在部分标的有效，需要区分对待")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "timing_multi_assets.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "assets_summary": summary,
        "grid_detail": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
