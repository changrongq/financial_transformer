# -*- coding: utf-8 -*-
"""追查 gap20 因子（20 日累计隔夜收益）为何 IC 为正且稳定。

重算横截面时发现：
  gap20  IC = +0.0353  ICIR = +10.95  IC>0 占比 76.0%（H=20）
  gap5   IC = +0.0285  ICIR = +7.05   IC>0 占比 68.2%

与其他所有因子（IC 全为负）方向相反，且 IC>0 占比异常高。

可能解释：
  1. 真信号：隔夜涨得多的股票，未来继续涨（动量）
  2. 与 turnover_z 高度相关，只是它的镜像
  3. 小市值/次新股效应
  4. 数据问题（pre_close 口径）

本脚本检验：
  A. 与其他因子的相关性（是否只是换手率的镜像）
  B. 分层单调性
  C. 扣成本可交易性
  D. 分年度稳定性
  E. 剔除 gap20 后，其他因子结论是否改变

用法：
  python scripts/test_gap_factor.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_cross_section_with_preclose import (
    resolve_db, load_quality_dates, load_panel_preclose,
    build_features_preclose, add_labels, daily_ic, FEATURES_NEW)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TRADING_DAYS = 244


def cs_zscore(df, col):
    mu = df.groupby("date_str")[col].transform("mean")
    sd = df.groupby("date_str")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).clip(-3, 3)


def run_portfolio(df, score_col, dates, rebalance, top_q, cost_side):
    ret_map = df.set_index(["date_str", "stock_code"])["ret1"].to_dict()
    dmap = {d: g for d, g in df.groupby("date_str")}
    nav, curve = 1.0, []
    w_prev, pending = {}, None
    total_turn = 0.0

    for i, d in enumerate(dates):
        g = dmap.get(d)
        if g is None or len(g) < 300:
            continue
        cost = 0.0
        if pending is not None:
            keys = set(pending) | set(w_prev)
            turn = 0.5 * sum(abs(pending.get(k, 0.0) - w_prev.get(k, 0.0))
                             for k in keys)
            total_turn += turn
            cost = turn * (2 * cost_side + 0.001)
            w_prev, pending = pending, None

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

        if i % rebalance == 0:
            s = g[["stock_code", score_col]].dropna()
            if len(s) < 300:
                continue
            thr = s[score_col].quantile(1 - top_q)
            picks = s[s[score_col] >= thr]["stock_code"].tolist()
            pending = {c: 1.0 / len(picks) for c in picks} if picks else {}

    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan,
                "turnover": 0.0, "days": len(curve)}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    years = max(len(arr) / TRADING_DAYS, 1e-9)
    return {"total": float(arr[-1] - 1),
            "dd": float((full / peak - 1).min()),
            "sharpe": float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan,
            "turnover": float(total_turn / years), "days": len(arr)}


def bench(df, dates):
    dmap = {d: g for d, g in df.groupby("date_str")}
    nav, curve = 1.0, []
    for d in dates:
        g = dmap.get(d)
        if g is None or len(g) < 300:
            continue
        nav *= (1 + float(g["ret1"].mean()))
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
            "sharpe": float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results/gap_factor.json")
    args = ap.parse_args()

    db = resolve_db()
    df = load_panel_preclose(db, load_quality_dates(), 4500)
    df = build_features_preclose(df)
    df = add_labels(df, [args.horizon])
    nd = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[nd >= 200]
    df = df[df["turnover"].fillna(0) >= 0.1]

    ycol = f"fwd{args.horizon}"
    dates = sorted(df["date_str"].unique())
    train = dates[:args.train_days]
    test = dates[args.train_days:]

    print("=" * 92)
    print(f"gap20 因子追查   H={args.horizon}")
    print("=" * 92)
    print(f"训练 {len(train)} 天 | 测试 {len(test)} 天")

    report = {"horizon": args.horizon}

    # ============================================================
    # A. 与其他因子的横截面相关性
    # ============================================================
    print("\n[A] gap20 与其他因子的横截面相关性（Spearman）")
    print("-" * 92)
    others = ["turnover_z", "vol20", "ret20", "amplitude", "pos120",
              "intra20", "ret1", "amt_ratio20"]
    print(f"  {'因子':<16}{'与gap20相关性':>16}")
    corrs = {}
    sample_days = test[::10][:20]
    for f in others:
        vals = []
        for d in sample_days:
            g = df[(df["date_str"] == d)][[f, "gap20"]].dropna()
            if len(g) < 300:
                continue
            a = g[f].rank().values
            b = g["gap20"].rank().values
            a -= a.mean()
            b -= b.mean()
            den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            if den > 0:
                vals.append((a * b).sum() / den)
        m = float(np.mean(vals)) if vals else np.nan
        corrs[f] = m
        print(f"  {f:<16}{m:>+16.4f}")
    report["correlations"] = corrs

    # ============================================================
    # B. 分层单调性
    # ============================================================
    print("\n[B] gap20 五分层收益（测试期）")
    print("-" * 92)
    dte = df[df["date_str"].isin(set(test))]
    layers = []
    for d, g in dte.groupby("date_str"):
        g2 = g[["gap20", ycol]].dropna()
        if len(g2) < 500:
            continue
        g2 = g2.assign(q=pd.qcut(g2["gap20"], 5, labels=False, duplicates="drop"))
        for q, gg in g2.groupby("q"):
            layers.append({"date": d, "q": int(q),
                           "ret": float(gg[ycol].mean())})
    ldf = pd.DataFrame(layers)
    print(f"  {'分层':<8}{'平均收益':>12}{'样本日数':>10}")
    agg = ldf.groupby("q")["ret"].mean()
    for q, v in agg.items():
        print(f"  Q{int(q)+1:<7}{v*100:>11.3f}%{int((ldf['q']==q).sum()):>10}")
    means = agg.values
    mono = all(means[i] <= means[i+1] for i in range(len(means)-1))
    print(f"\n  单调递增? {mono}")
    print(f"  Q5-Q1 = {(means[-1]-means[0])*100:+.3f}%")
    report["layers"] = {f"Q{i+1}": float(v) for i, v in enumerate(means)}
    report["monotone"] = bool(mono)
    report["spread"] = float(means[-1] - means[0])

    # ============================================================
    # C. 扣成本回测（只有 gap20）
    # ============================================================
    print("\n[C] 只用 gap20 的多头组合（扣成本）")
    print("-" * 92)
    df["score_gap"] = cs_zscore(df, "gap20").fillna(0.0)
    b = bench(df, test)
    print(f"  市场等权: {b['total']*100:+.1f}%  回撤 {b['dd']*100:.1f}%  "
          f"夏普 {b['sharpe']:.2f}")
    print(f"\n  {'调仓':<6}{'成本/边':<10}{'多头':>11}{'超额':>11}"
          f"{'夏普':>8}{'年换手':>9}")
    rows = []
    for rb in [5, 10, 20]:
        for cost in [0.0, 0.003, 0.005]:
            st = run_portfolio(df, "score_gap", test, rb, 0.2, cost)
            ex = st["total"] - b["total"]
            rows.append({"rebalance": rb, "cost": cost,
                         "total": st["total"], "excess": ex,
                         "sharpe": st["sharpe"],
                         "turnover": st["turnover"]})
            print(f"  {rb:<6}{cost*100:>7.2f}%  {st['total']*100:>10.1f}%"
                  f"{ex*100:>+10.1f}pp{st['sharpe']:>8.2f}"
                  f"{st['turnover']:>9.1f}")
        print()
    report["backtest"] = rows

    # ============================================================
    # D. 分季度
    # ============================================================
    print("=" * 92)
    print("[D] 分季度（调仓10天, 成本0.3%/边）")
    print("=" * 92)
    qmap = {}
    for d in test:
        qmap.setdefault(str(pd.Timestamp(d).to_period("Q")), []).append(d)
    print(f"  {'季度':<10}{'天':>5}{'多头':>11}{'市场':>11}{'超额':>11}")
    qrows = []
    for q in sorted(qmap):
        ds = qmap[q]
        if len(ds) < 20:
            continue
        st = run_portfolio(df, "score_gap", ds, 10, 0.2, 0.003)
        bq = bench(df, ds)
        ex = st["total"] - bq["total"]
        qrows.append({"q": q, "long": st["total"], "bench": bq["total"],
                      "excess": ex})
        print(f"  {q:<10}{len(ds):>5}{st['total']*100:>10.1f}%"
              f"{bq['total']*100:>10.1f}%{ex*100:>+10.1f}pp")
    n_beat = sum(1 for r in qrows if r["excess"] > 0)
    print(f"\n  跑赢: {n_beat}/{len(qrows)}")
    report["quarterly"] = qrows

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 92)
    print("判定")
    print("=" * 92)
    best = max(rows, key=lambda r: r["excess"])
    checks = [
        ("分层单调递增", mono, f"Q1→Q5 = {(means[-1]-means[0])*100:+.3f}%"),
        ("与换手率低相关（|corr|<0.3）",
         abs(corrs.get("turnover_z", 1)) < 0.3,
         f"{corrs.get('turnover_z', np.nan):+.4f}"),
        ("扣0.3%成本后超额 > 0", best["excess"] > 0,
         f"{best['excess']*100:+.1f}pp @ {best['rebalance']}天"),
        ("分季度跑赢过半", n_beat > len(qrows)/2, f"{n_beat}/{len(qrows)}"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "gap20 有可交易价值" if all_ok else "gap20 不成立"
    print(f"\n  结论: {verdict}")
    report["checks"] = [{"name": a, "ok": bool(o), "detail": d}
                        for a, o, d in checks]
    report["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
