"""验证「次日开盘执行」的择时是否稳健（换窗口 + 换阈值 + 分段）。

背景：
  发现原回测有未来函数（当日收盘出信号、当日成交），
  改成"次日开盘执行"后收益从 +149.9% 掉到 +27.4%，跑不过买入持有 +28.1%。

本脚本检查：
  1. 换不同窗口/阈值，次日开盘执行是否还能跑赢买入持有？
  2. 最大回撤是否确实改善？
  3. 分年度看，哪些年份赢、哪些年份输？

用法：
  python scripts/verify_timing_next_open.py
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


def load_market(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    sd = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [收盘] AS close FROM stock_data", conn
    )
    conn.close()
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    sd["close"] = pd.to_numeric(sd["close"], errors="coerce")
    sd = sd.dropna(subset=["close"])
    sd = sd[sd["stock_code"].str.len() == 6]
    sd = sd.sort_values(["stock_code", "date"])
    sd["ret"] = sd.groupby("stock_code")["close"].pct_change(fill_method=None)
    daily = sd.groupby("date")["ret"].mean().rename("market_ret").reset_index()
    return daily.dropna().sort_values("date").reset_index(drop=True)


def stats(eq: pd.Series) -> dict:
    ret = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    years = len(eq) / 244.0
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    sh = float(ret.mean() / ret.std() * np.sqrt(244)) if ret.std() > 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sh}


def run_variant(d: pd.DataFrame, window: int, threshold: float, cost: float, lag: int):
    """lag=0 当日收盘(有未来函数)；lag=1 次日开盘；lag=2 次日收盘。"""
    m = d["market_ret"].rolling(window, min_periods=3).mean()
    sig = (m >= threshold)
    held = sig.shift(lag).fillna(False).astype(bool)
    sw = held.ne(held.shift(1)) & held.shift(1).notna()
    gross = np.where(held, d["market_ret"], 0.0)
    net = gross - np.where(sw.values, cost, 0.0)
    eq = pd.Series(np.cumprod(1 + net), index=d["date"])
    return eq, held, int(sw.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="验证次日开盘执行的稳健性")
    parser.add_argument("--cost", type=float, default=0.0005)
    args = parser.parse_args()

    d = load_market(_resolve_db())
    d = d[d["date"] >= "2024-10-01"].reset_index(drop=True)
    bh_eq = pd.Series(np.cumprod(1 + d["market_ret"].values), index=d["date"])
    bh = stats(bh_eq)
    print(f"区间: {d['date'].min().date()} ~ {d['date'].max().date()}（{len(d)} 天）")
    print(f"买入持有: {bh['total']*100:+.1f}%（年化 {bh['ann']*100:+.1f}%, 回撤 {bh['dd']*100:.1f}%）\n")

    print("=" * 100)
    print(f"{'窗口':>5}{'阈值':>9} | {'A当日(未来函数)':>17}{'B次日开盘':>15}{'C次日收盘':>15} | {'B vs 买入':>11}{'B回撤':>9}")
    print("=" * 100)

    rows = []
    for w in [5, 10, 20, 30, 60]:
        for th in [0.0, -0.001, -0.002, 0.001]:
            eA, _, _ = run_variant(d, w, th, args.cost, 0)
            eB, hB, nB = run_variant(d, w, th, args.cost, 1)
            eC, _, _ = run_variant(d, w, th, args.cost, 2)
            sA, sB, sC = stats(eA), stats(eB), stats(eC)
            beat = sB["total"] - bh["total"]
            rows.append({
                "window": w, "threshold": th,
                "A_total": sA["total"], "B_total": sB["total"], "C_total": sC["total"],
                "B_beat_bh_pp": beat, "B_dd": sB["dd"], "B_switches": nB,
            })
            flag = "WIN " if beat > 0 else "LOSE"
            print(f"{w:>5}{th*100:>8.1f}% | {sA['total']*100:>16.1f}%{sB['total']*100:>14.1f}%"
                  f"{sC['total']*100:>14.1f}% | {flag}{beat*100:>+8.1f}pp{sB['dd']*100:>8.1f}%")
    print("=" * 100)

    df = pd.DataFrame(rows)
    n_win = int((df["B_beat_bh_pp"] > 0).sum())
    print(f"\nB方案（次日开盘）跑赢买入持有的组合: {n_win}/{len(df)}")
    print(f"B方案收益中位数: {df['B_total'].median()*100:+.1f}%   "
          f"买入持有: {bh['total']*100:+.1f}%")
    print(f"B方案回撤中位数: {df['B_dd'].median()*100:.1f}%   "
          f"买入持有回撤: {bh['dd']*100:.1f}%")
    print(f"A方案（未来函数）收益中位数: {df['A_total'].median()*100:+.1f}%  "
          f"← 虚高 {(df['A_total'].median()-df['B_total'].median())*100:.1f}pp")

    # 默认参数的分年度
    print("\n默认参数（窗口10, 阈值0）分年度：")
    eB, hB, _ = run_variant(d, 10, 0.0, args.cost, 1)
    print(f"{'年份':<8}{'买入持有':>12}{'择时B':>12}{'差额':>12}")
    for y in sorted(set(d["date"].dt.year)):
        mask_b = d["date"].dt.year == y
        mb = float(np.prod(1 + d.loc[mask_b, "market_ret"].values) - 1)
        mask_e = eB.index.year == y
        sub = eB[mask_e]
        me = float(sub.iloc[-1] / sub.iloc[0] - 1) if len(sub) > 1 else 0.0
        print(f"{y:<8}{mb*100:>+11.1f}%{me*100:>+11.1f}%{(me-mb)*100:>+11.1f}pp")

    # 波动率对比
    rB = eB.pct_change().dropna()
    rbh = bh_eq.pct_change().dropna()
    print(f"\n日波动率: 择时B {rB.std()*100:.3f}%  vs  买入持有 {rbh.std()*100:.3f}%")
    print(f"夏普: 择时B {stats(eB)['sharpe']:.2f}  vs  买入持有 {bh['sharpe']:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "timing_next_open_verification.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "buy_hold": {k: round(v, 6) for k, v in bh.items()},
        "combos": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        "n_win": n_win, "n_total": len(df),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
