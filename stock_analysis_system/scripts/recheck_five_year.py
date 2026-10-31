# -*- coding: utf-8 -*-
"""独立复核 FIVE_YEAR_DATA_VERDICT.md 的关键结论。

文档声称：
  1. 逐年跑赢 2/5
  2. 逐月胜率 20.0%（10/50）
  3. 参数网格 8/16 为正
  4. 调仓 40 天全部为正（+23.9 ~ +27.4pp）
  5. Top20% 涨停比例 3.03% vs 全市场 1.33%

本脚本用**独立实现**重算这些数字，不复用 final_cross_section_verdict.py。

设计：
  · 用 close/close.shift(1) 算收益（口径自洽）
  · 严格 T+1：T 日收盘算信号，T+1 日开盘买入
  · 买入时剔除【次日开盘涨停】的股票（无涨跌幅的板块按 20%/30% 判断）
  · 成本：双边 0.3%/边 + 印花税 0.1%

用法：
  python scripts/recheck_five_year.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TD = 244


def resolve_db() -> str:
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            with cfg.open("r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            raw = c.get("databases", {}).get("trade_db")
            if raw:
                p = Path(raw)
                if not p.is_absolute():
                    p = ROOT / p
                if p.exists():
                    return str(p)
        except Exception:
            pass
    return str((ROOT.parent / "stock_history.db").resolve())


def limit_pct(code: str) -> float:
    """单日涨跌幅上限。"""
    c = str(code)
    if c.startswith(("30", "68")):
        return 0.20          # 创业板 / 科创板
    if c.startswith(("43", "83", "87", "88", "92")):
        return 0.30          # 北交所
    return 0.10              # 主板


def load_panel(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql_query("""
        SELECT 日期 AS date, stock_code, 开盘 AS open, 最高 AS high,
               最低 AS low, 收盘 AS close, 成交额 AS amount,
               换手率 AS turnover
        FROM stock_data
        WHERE 收盘 > 0 AND 开盘 > 0
    """, con)
    con.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "amount", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close", "open"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    g = df.groupby("stock_code", group_keys=False)

    # ★ 口径自洽：ret1 用 close 相对上一日 close
    df["prev_close"] = g["close"].shift(1)
    df["ret1"] = df["close"] / df["prev_close"] - 1.0
    # 次日开盘相对今日收盘（用于判断能否买入）
    df["next_open"] = g["open"].shift(-1)
    df["open_gap"] = df["next_open"] / df["close"] - 1.0

    # 特征
    lr = np.log1p(df["ret1"].clip(-0.95, 1.0))
    df["lr"] = lr
    for n in [5, 20]:
        df[f"ret{n}"] = np.exp(g["lr"].transform(
            lambda s: s.rolling(n, min_periods=max(3, n // 2)).sum())) - 1.0
    for n in [20, 60]:
        df[f"vol{n}"] = g["ret1"].transform(
            lambda s: s.rolling(n, min_periods=max(5, n // 2)).std())
    df["turnover_z"] = g["turnover"].transform(
        lambda s: (s - s.rolling(60, min_periods=20).mean())
        / s.rolling(60, min_periods=20).std().replace(0, np.nan))
    df["amplitude"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["high120"] = g["high"].transform(
        lambda s: s.rolling(120, min_periods=40).max())
    df["low120"] = g["low"].transform(
        lambda s: s.rolling(120, min_periods=40).min())
    df["pos120"] = ((df["close"] - df["low120"]) /
                    (df["high120"] - df["low120"]).replace(0, np.nan))

    return df.replace([np.inf, -np.inf], np.nan)


FACTORS = ["turnover_z", "vol20", "ret20", "amplitude", "pos120"]


def cs_z(df, col):
    mu = df.groupby("date_str")[col].transform("mean")
    sd = df.groupby("date_str")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).clip(-3, 3)


def build_score(df, signs):
    s = pd.Series(0.0, index=df.index)
    n = 0
    for f in FACTORS:
        if f not in df.columns or not np.isfinite(signs.get(f, np.nan)):
            continue
        s = s + cs_z(df, f).fillna(0.0) * np.sign(signs[f])
        n += 1
    return s / max(n, 1)


def train_signs(df, train_dates, h=20):
    """用训练段算各因子 IC 符号。"""
    print(f"  计算训练段 IC（前 {len(train_dates)} 天）...")
    # 未来 h 日收益（用 close）
    df = df.copy()
    df["fwd"] = df.groupby("stock_code", group_keys=False)["close"].transform(
        lambda s: s.shift(-h) / s - 1.0)
    sub = df[df["date_str"].isin(set(train_dates))]
    signs = {}
    for f in FACTORS:
        ics = []
        for d, g in sub.groupby("date_str"):
            s2 = g[[f, "fwd"]].dropna()
            if len(s2) < 300:
                continue
            a = s2[f].rank().values
            b = s2["fwd"].rank().values
            a -= a.mean()
            b -= b.mean()
            den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            if den > 0:
                ics.append((a * b).sum() / den)
        signs[f] = float(np.mean(ics)) if ics else 0.0
    return signs


def backtest(df, test_dates, rebalance, top_q, cost_side,
             exclude_limit_up=True):
    """T+1 开盘买入，每 rebalance 天调仓。"""
    # 索引
    ret_map = df.set_index(["date_str", "stock_code"])["ret1"].to_dict()
    gap_map = df.set_index(["date_str", "stock_code"])["open_gap"].to_dict()
    dmap = {d: g for d, g in df.groupby("date_str")}

    nav = 1.0
    curve = []
    w_prev, pending = {}, None
    turn_sum = 0.0

    for i, d in enumerate(test_dates):
        g = dmap.get(d)
        if g is None or len(g) < 500:
            continue

        # 执行昨日挂单
        cost = 0.0
        if pending is not None:
            keys = set(pending) | set(w_prev)
            turn = 0.5 * sum(abs(pending.get(k, 0.0) - w_prev.get(k, 0.0))
                             for k in keys)
            turn_sum += turn
            cost = turn * (2 * cost_side + 0.001)
            w_prev, pending = pending, None

        # 今日收益
        if w_prev:
            rs, ws = [], []
            for sc, w in w_prev.items():
                r = ret_map.get((d, sc), np.nan)
                if isinstance(r, (int, float)) and np.isfinite(r):
                    rs.append(w * r)
                    ws.append(w)
            r = sum(rs) / sum(ws) if ws and sum(ws) > 0 else 0.0
        else:
            r = 0.0

        nav *= (1 + r) * (1 - cost)
        curve.append(nav)

        # 收盘算分
        if i % rebalance == 0:
            s = g[["stock_code", "score"]].dropna()
            if len(s) < 500:
                continue
            thr = s["score"].quantile(1 - top_q)
            picks = s[s["score"] >= thr]["stock_code"].tolist()

            if exclude_limit_up and picks:
                keep = []
                for c in picks:
                    gp = gap_map.get((d, c), np.nan)
                    lim = limit_pct(c)
                    # 次日开盘涨幅接近涨停 → 买不进
                    if np.isfinite(gp) and gp >= lim - 0.005:
                        continue
                    keep.append(c)
                picks = keep

            pending = {c: 1.0 / len(picks) for c in picks} if picks else {}

    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan,
                "days": len(curve), "turnover": 0.0}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    years = max(len(arr) / TD, 1e-9)
    return {
        "total": float(arr[-1] - 1),
        "dd": float((full / peak - 1).min()),
        "sharpe": float(rets.mean() / sd * np.sqrt(TD)) if sd > 1e-12 else np.nan,
        "days": len(arr),
        "turnover": float(turn_sum / years),
    }


def bench_eq(df, dates):
    ret_map = df.set_index(["date_str", "stock_code"])["ret1"].to_dict()
    dmap = {d: g for d, g in df.groupby("date_str")}
    nav, curve = 1.0, []
    for d in dates:
        g = dmap.get(d)
        if g is None or len(g) < 500:
            continue
        r = float(g["ret1"].mean())
        nav *= (1 + r)
        curve.append(nav)
    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    return {"total": float(arr[-1] - 1),
            "dd": float((full / peak - 1).min()),
            "sharpe": float(rets.mean() / sd * np.sqrt(TD)) if sd > 1e-12 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=250)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results/five_year_recheck.json")
    args = ap.parse_args()

    db = resolve_db()
    print("=" * 92)
    print("独立复核 5 年横截面结论")
    print("=" * 92)

    df = load_panel(db)
    nd = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[nd >= 250]
    df = df[df["turnover"].fillna(0) >= 0.1]
    print(f"样本 {len(df):,} 行  {df['stock_code'].nunique():,} 只  "
          f"{df['date_str'].nunique()} 天")
    print(f"区间 {df['date'].min().date()} ~ {df['date'].max().date()}")

    dates = sorted(df["date_str"].unique())
    train, test = dates[:args.train_days], dates[args.train_days:]
    print(f"训练 {len(train)} 天 -> 测试 {len(test)} 天")

    signs = train_signs(df, train)
    print("\n训练段 IC 符号:")
    for f, v in signs.items():
        print(f"  {f:<14}{v:+.4f}")

    df["score"] = build_score(df, signs)
    b_all = bench_eq(df, test)
    print(f"\n市场等权（测试段）: {b_all['total']*100:+.1f}%  "
          f"回撤 {b_all['dd']*100:.1f}%  夏普 {b_all['sharpe']:.2f}")

    # ------------------------------------------------------------
    # 1. 逐年
    # ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("[1] 逐年（top20%, 调仓20天, 成本0.3%/边, 剔除次日开盘涨停）")
    print("=" * 92)
    ymap = {}
    for d in test:
        ymap.setdefault(d[:4], []).append(d)

    print(f"  {'年':<6}{'天':>5}{'策略':>11}{'市场':>11}{'超额':>11}{'夏普':>8}")
    yrows = []
    for y in sorted(ymap):
        ds = ymap[y]
        if len(ds) < 30:
            continue
        st = backtest(df, ds, 20, 0.2, args.cost)
        bh = bench_eq(df, ds)
        ex = st["total"] - bh["total"]
        yrows.append({"year": y, "days": len(ds), "strategy": st["total"],
                      "bench": bh["total"], "excess": ex,
                      "sharpe": st["sharpe"]})
        print(f"  {y:<6}{len(ds):>5}{st['total']*100:>10.1f}%"
              f"{bh['total']*100:>10.1f}%{ex*100:>+10.1f}pp"
              f"{st['sharpe']:>8.2f}")

    n_beat_y = sum(1 for r in yrows if r["excess"] > 0)
    print(f"\n  逐年跑赢: {n_beat_y}/{len(yrows)}")

    # ------------------------------------------------------------
    # 2. 逐月
    # ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("[2] 逐月（同参数）")
    print("=" * 92)
    mmap = {}
    for d in test:
        mmap.setdefault(d[:7], []).append(d)
    n_beat_m = 0
    mrows = []
    for m in sorted(mmap):
        ds = mmap[m]
        if len(ds) < 8:
            continue
        st = backtest(df, ds, 20, 0.2, args.cost)
        bh = bench_eq(df, ds)
        ex = st["total"] - bh["total"]
        mrows.append({"month": m, "excess": ex})
        if ex > 0:
            n_beat_m += 1
    print(f"  共 {len(mrows)} 个月，跑赢 {n_beat_m} "
          f"({n_beat_m/max(len(mrows),1)*100:.1f}%)")

    # ------------------------------------------------------------
    # 3. 参数网格
    # ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("[3] 参数网格（超额 pp，成本 0.3%/边）")
    print("=" * 92)
    print(f"  {'top%':<6}" + "".join(f"{f'rb{n}':>11}" for n in [5, 10, 20, 40]))
    grid = {}
    n_pos = 0
    n_tot = 0
    for top in [0.1, 0.2, 0.3]:
        line = f"  {top*100:>5.0f}"
        grid[str(top)] = {}
        for rb in [5, 10, 20, 40]:
            st = backtest(df, test, rb, top, args.cost)
            ex = st["total"] - b_all["total"]
            grid[str(top)][str(rb)] = ex
            n_pos += 1 if ex > 0 else 0
            n_tot += 1
            line += f"{ex*100:>+10.1f}pp"
        print(line)
    print(f"\n  正超额: {n_pos}/{n_tot}")

    # ------------------------------------------------------------
    # 4. 涨停比例验证
    # ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("[4] Top20% 组的次日开盘涨停比例")
    print("=" * 92)
    dte = df[df["date_str"].isin(set(test))]
    n_top_lu = n_top = n_all_lu = n_all = 0
    for d, g in dte.groupby("date_str"):
        g2 = g[["stock_code", "score", "open_gap"]].dropna(subset=["score"])
        if len(g2) < 500:
            continue
        thr = g2["score"].quantile(0.8)
        topg = g2[g2["score"] >= thr]
        lim = topg["stock_code"].apply(limit_pct).values
        lu = (topg["open_gap"].values >= lim - 0.005)
        n_top_lu += int(lu.sum())
        n_top += len(topg)
        lam = g2["stock_code"].apply(limit_pct).values
        lua = (g2["open_gap"].values >= lam - 0.005)
        n_all_lu += int(lua.sum())
        n_all += len(g2)
    print(f"  Top20%: {n_top_lu:,}/{n_top:,} = {n_top_lu/max(n_top,1)*100:.2f}%")
    print(f"  全市场: {n_all_lu:,}/{n_all:,} = {n_all_lu/max(n_all,1)*100:.2f}%")
    print(f"  倍数: {n_top_lu/max(n_top,1) / max(n_all_lu/max(n_all,1), 1e-9):.2f}x")

    # ------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------
    print("\n" + "=" * 92)
    print("与 FIVE_YEAR_DATA_VERDICT.md 对比")
    print("=" * 92)
    checks = [
        ("逐年跑赢 2/5", f"{n_beat_y}/{len(yrows)}"),
        ("逐月胜率 20.0%", f"{n_beat_m/max(len(mrows),1)*100:.1f}%"),
        ("参数网格 8/16", f"{n_pos}/{n_tot}"),
        ("Top20% 涨停 3.03% vs 全市场 1.33%",
         f"{n_top_lu/max(n_top,1)*100:.2f}% vs "
         f"{n_all_lu/max(n_all,1)*100:.2f}%"),
    ]
    for name, val in checks:
        print(f"  {name:<40} 实测: {val}")

    out = {"n_rows": int(len(df)), "n_test": len(test),
           "signs": signs, "bench": b_all,
           "yearly": yrows, "n_beat_year": n_beat_y,
           "n_months": len(mrows), "n_beat_month": n_beat_m,
           "grid": grid, "n_pos": n_pos, "n_tot": n_tot,
           "limit_up": {"top_pct": n_top_lu/max(n_top,1),
                        "all_pct": n_all_lu/max(n_all,1)}}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "five_year_recheck.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 results/five_year_recheck.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
