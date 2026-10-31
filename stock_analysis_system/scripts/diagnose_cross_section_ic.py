# -*- coding: utf-8 -*-
"""诊断：横截面 IC 显著但分季度组合亏损，矛盾在哪？

已知：
  · 5 个因子训练期 IC 全部为负（-0.065 ~ -0.096），ICIR -9 ~ -14
  · 扣 0.3%/边后多头超额 +3.1pp（调仓10天）
  · 但分季度只有 1/5 跑赢市场

可能的矛盾来源：
  1. IC 是用【训练期】算的符号，但测试期符号可能翻转
  2. IC 是「秩相关」，而组合收益依赖「幅度」——秩相关对极端值不敏感
  3. 因子在小市值股上有效，但组合等权买入后，收益被大市值股拖累
  4. 分层非单调：Q1 和 Q5 都好，中间差

本脚本：
  A. 测试期逐月 IC（看符号是否稳定）
  B. 五分层收益（看是否单调）
  C. 按市值分组看 IC（是否只在小盘有效）
  D. 用「测试期前 60 天」重估符号，再用剩余时间检验

用法：
  python scripts/diagnose_cross_section_ic.py
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
from test_cross_section import (resolve_db, load_quality_dates, load_panel,
                                build_features, add_labels)
from test_cross_section_v2 import cs_zscore

ROOT = Path(__file__).resolve().parents[1]

FACTORS = ["turnover_z", "vol20", "amplitude", "ret20", "pos120"]


def daily_ic_series(df, f, ycol, min_n=300):
    out = {}
    for d, g in df.groupby("date_str"):
        s = g[[f, ycol]].dropna()
        if len(s) < min_n:
            continue
        a = s[f].rank().values
        b = s[ycol].rank().values
        a -= a.mean()
        b -= b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if den > 0:
            out[d] = float((a * b).sum() / den)
    return pd.Series(out).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--min-history-days", type=int, default=200)
    ap.add_argument("--out", default="results/cross_section_ic_diagnosis.json")
    args = ap.parse_args()

    db = resolve_db()
    usable = load_quality_dates()
    df = load_panel(db, usable, 4500)
    df = build_features(df)
    df = add_labels(df, [args.horizon])

    n_days_stock = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[n_days_stock >= args.min_history_days]
    ycol = f"fwd{args.horizon}"

    all_dates = sorted(df["date_str"].unique())
    train = all_dates[:args.train_days]
    test = all_dates[args.train_days:]
    print("=" * 92)
    print(f"横截面 IC 矛盾诊断   H={args.horizon}")
    print("=" * 92)
    print(f"训练 {len(train)} 天 | 测试 {len(test)} 天（{test[0]} ~ {test[-1]}）")

    report = {"horizon": args.horizon, "n_train": len(train),
              "n_test": len(test)}

    # ============================================================
    # A. 逐月 IC（训练 vs 测试）
    # ============================================================
    print("\n[A] 逐月 IC（关键：测试期符号是否翻转）")
    print("-" * 92)

    df_tr = df[df["date_str"].isin(set(train))]
    df_te = df[df["date_str"].isin(set(test))]

    print(f"  {'因子':<14}{'训练IC':>10}{'测试IC':>10}{'同号?':>8}")
    sign_rows = []
    for f in FACTORS:
        ic_tr = daily_ic_series(df_tr, f, ycol)
        ic_te = daily_ic_series(df_te, f, ycol)
        m_tr = float(ic_tr.mean()) if len(ic_tr) else np.nan
        m_te = float(ic_te.mean()) if len(ic_te) else np.nan
        same = np.isfinite(m_tr) and np.isfinite(m_te) and np.sign(m_tr) == np.sign(m_te)
        sign_rows.append({"factor": f, "ic_train": m_tr, "ic_test": m_te,
                          "same_sign": bool(same)})
        print(f"  {f:<14}{m_tr:>+10.4f}{m_te:>+10.4f}"
              f"{'  是' if same else '  否 <<<':>8}")
    report["sign_stability"] = sign_rows

    # 逐月
    print(f"\n  逐月测试期 IC（{FACTORS[0]}）:")
    ics = daily_ic_series(df_te, FACTORS[0], ycol)
    monthly = ics.groupby(pd.Series(pd.to_datetime(ics.index)).dt.to_period("M").astype(str)).mean()
    for m, v in monthly.items():
        print(f"    {m}: {v:+.4f}  {'正' if v > 0 else '负'}")
    report["monthly_ic"] = {str(k): float(v) for k, v in monthly.items()}

    # ============================================================
    # B. 五分层收益（是否单调）
    # ============================================================
    print("\n[B] 测试期五分层收益（等权综合分数）")
    print("-" * 92)

    # 用训练期符号合成分数
    signs = {r["factor"]: r["ic_train"] for r in sign_rows}
    df["score"] = 0.0
    n_used = 0
    for f in FACTORS:
        if not np.isfinite(signs[f]) or signs[f] == 0:
            continue
        df["score"] = df["score"] + cs_zscore(df, f).fillna(0.0) * np.sign(signs[f])
        n_used += 1
    df["score"] /= max(n_used, 1)

    dte = df[df["date_str"].isin(set(test))]
    layer_rows = []
    for d, g in dte.groupby("date_str"):
        g2 = g[["score", ycol]].dropna()
        if len(g2) < 500:
            continue
        g2 = g2.assign(q=pd.qcut(g2["score"], 5, labels=False, duplicates="drop"))
        for q, gg in g2.groupby("q"):
            layer_rows.append({"date": d, "q": int(q), "ret": float(gg[ycol].mean())})
    ldf = pd.DataFrame(layer_rows)
    if len(ldf):
        agg = ldf.groupby("q")["ret"].agg(["mean", "count"])
        print(f"  {'分层':<8}{'平均收益':>12}{'样本日数':>10}")
        for q, r in agg.iterrows():
            print(f"  Q{int(q)+1:<7}{r['mean']*100:>11.3f}%{r['count']:>10.0f}")
        means = agg["mean"].values
        mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
        mono_r = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
        print(f"\n  单调递增? {mono}   单调递减? {mono_r}")
        print(f"  Q5-Q1 = {(means[-1]-means[0])*100:+.3f}%")
        report["layers"] = {f"Q{i+1}": float(v) for i, v in enumerate(means)}
        report["monotone"] = bool(mono or mono_r)

    # ============================================================
    # C. 按换手率（流动性）分组看 IC
    # ============================================================
    print("\n[C] 按流动性分组看 IC（是否只在小盘/低流动性有效）")
    print("-" * 92)
    print(f"  {'流动性组':<10}{'股票数':>8}{'IC(turnover_z)':>16}"
          f"{'IC(vol20)':>12}{'IC(ret20)':>12}")

    liq_rows = []
    for d, g in dte.groupby("date_str"):
        g2 = g.dropna(subset=["turnover"])
        if len(g2) < 500:
            continue
        g2 = g2.assign(liq=pd.qcut(g2["turnover"], 3, labels=False,
                                   duplicates="drop"))
        for lq, gg in g2.groupby("liq"):
            for f in ["turnover_z", "vol20", "ret20"]:
                s = gg[[f, ycol]].dropna()
                if len(s) < 100:
                    continue
                a = s[f].rank().values
                b = s[ycol].rank().values
                a -= a.mean()
                b -= b.mean()
                den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
                v = (a * b).sum() / den if den > 0 else np.nan
                liq_rows.append({"date": d, "liq": int(lq), "factor": f,
                                 "ic": float(v), "n": len(s)})
    if liq_rows:
        ldf2 = pd.DataFrame(liq_rows)
        names = {0: "低流动性", 1: "中流动性", 2: "高流动性"}
        piv = ldf2.groupby(["liq", "factor"])["ic"].mean().unstack()
        for lq in sorted(piv.index):
            row = piv.loc[lq]
            print(f"  {names.get(lq, lq):<10}"
                  f"{'':>8}"
                  f"{row.get('turnover_z', np.nan):>+16.4f}"
                  f"{row.get('vol20', np.nan):>+12.4f}"
                  f"{row.get('ret20', np.nan):>+12.4f}")
        report["by_liquidity"] = {
            names.get(int(k), str(k)): {f: float(v) for f, v in r.items()}
            for k, r in piv.iterrows()}

    # ============================================================
    # D. 只用测试期前 60 天重估符号
    # ============================================================
    print("\n[D] 用测试期前 60 天重估符号，再用剩余时间检验")
    print("-" * 92)
    if len(test) > 80:
        te_early = set(test[:60])
        te_late = test[60:]
        df_e = df[df["date_str"].isin(te_early)]
        df_l = df[df["date_str"].isin(set(te_late))]

        print(f"  {'因子':<14}{'前60天IC':>12}{'后续IC':>12}{'同号?':>8}")
        rows2 = []
        for f in FACTORS:
            a = daily_ic_series(df_e, f, ycol)
            b = daily_ic_series(df_l, f, ycol)
            ma = float(a.mean()) if len(a) else np.nan
            mb = float(b.mean()) if len(b) else np.nan
            same = np.isfinite(ma) and np.isfinite(mb) and np.sign(ma) == np.sign(mb)
            rows2.append({"factor": f, "ic_early": ma, "ic_late": mb,
                          "same_sign": bool(same)})
            print(f"  {f:<14}{ma:>+12.4f}{mb:>+12.4f}"
                  f"{'  是' if same else '  否 <<<':>8}")
        report["reestimate_sign"] = rows2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
