# -*- coding: utf-8 -*-
"""决定性检验：创业板「20日调仓」策略是不是过拟合。

已有证据的矛盾：
  支持：
    · 测试期两半 IC 同向（+0.150 / +0.286）
    · 4 种训练/测试切分全部跑赢（+29 ~ +72pp）
    · 超出随机对照 95% 分位
  反对：
    · 调仓周期 20 天最好，5/10/40/60/120 天都差很多 → 参数尖峰
    · 随机对照里 300 次有 5 次超过它
    · 6 只标的只有创业板成立

最关键的检验：**样本外时间**
  我们一直用 train 60% / test 20%（末段）。但末段是创业板大牛市（+76%）。
  如果信号只是"牛市里满仓、熊市里空仓"的滞后版本，
  那么在一段没有单边趋势的时期就会失效。

本脚本：
  1. 用最早的数据训练，测试【完全不同】的历史区间（2013-2019）
  2. 逐个自然年检验（每一年用当年之前的数据训练）
  3. 极端压力测试：只用 2015 股灾前训练，测 2015 股灾
  4. 与"20日均线规则"这种零参数基准对比

用法：
  python scripts/test_chinext_out_of_sample.py
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
from test_feature_value_strict import spearman
from test_feature_value_extended import (resolve_db, load_etf_ext,
                                         load_market_breadth,
                                         PRICE_FEATS, PV_FEATS,
                                         VOLSTRUCT_FEATS, make_label)
from rl_ledger import run_ledger, ledger_stats

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATS = PRICE_FEATS + PV_FEATS + VOLSTRUCT_FEATS


def bt(df, targets, s, e, cost, start_pos=0.0):
    if e - s < 3:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan, "trades": 0}
    r = run_ledger(df["date"].values[s:e],
                   df["open"].values[s:e].astype(float),
                   df["close"].values[s:e].astype(float),
                   df["pre_close"].values[s:e].astype(float),
                   targets[s:e], cost=cost, slip=0.0, start_pos=start_pos)
    st = ledger_stats(r.nav)
    st["trades"] = r.trades
    return st


def make_period_target(pred, te, n, period=20):
    """每 period 天按预测中位数调仓。"""
    pos = np.zeros(len(pred))
    med = np.median(pred)
    cur = 1.0
    for i in range(len(pred)):
        if i % period == 0:
            cur = 1.0 if pred[i] > med else 0.0
        pos[i] = cur
    t = np.zeros(n)
    t[te] = pos
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--period", type=int, default=20)
    ap.add_argument("--out", default="results/chinext_oos.json")
    args = ap.parse_args()

    db = resolve_db()
    breadth = load_market_breadth(db)
    df = load_etf_ext(db, "159915.SZ", breadth)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    n = len(df)
    X = df[FEATS].values.astype(np.float64)
    dates = pd.DatetimeIndex(df["date"].values)
    years = dates.year

    print("=" * 88)
    print(f"创业板 20日调仓策略：真正的样本外检验")
    print(f"数据 {dates.min().date()} ~ {dates.max().date()}  n={n}")
    print("=" * 88)

    report = {"cost": args.cost, "period": args.period, "n": n}

    # ============================================================
    # 1. 逐年滚动：每年用之前所有数据训练
    # ============================================================
    print("\n[1] 逐年检验（每年用【之前所有年】训练，预测当年）")
    print("-" * 88)
    print(f"  {'年份':<8}{'训练n':>8}{'测试n':>7}{'IC':>9}{'策略':>11}"
          f"{'持有':>11}{'差':>11}{'市场':>9}")

    yearly = []
    all_years = sorted(set(years))
    for yy in all_years:
        if yy < 2015:
            continue
        tr_mask = years < yy
        te_mask = years == yy
        tr = np.where(tr_mask)[0]
        te = np.where(te_mask)[0]
        if len(tr) < 400 or len(te) < 60:
            continue

        yh = make_label(df, 20)
        ok_tr = np.isfinite(yh[tr])
        tr = tr[ok_tr]
        if len(tr) < 400:
            continue

        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), yh[tr])

        # 预测当年（用当年特征，但模型只用过去训练）
        pred = m.predict(sc.transform(X[te]))
        ic = spearman(pred, yh[te]) if np.isfinite(yh[te]).sum() > 30 else np.nan

        t = make_period_target(pred, te, n, args.period)
        s, e = int(te[0]), int(te[-1]) + 1
        st = bt(df, t, s, e, args.cost)
        bh = bt(df, np.ones(n), s, e, args.cost, start_pos=1.0)
        diff = st["total"] - bh["total"]

        yearly.append({"year": int(yy), "n_train": len(tr), "n_test": len(te),
                       "ic": ic, "strategy": st["total"], "buyhold": bh["total"],
                       "diff": diff, "trades": st["trades"]})
        print(f"  {yy:<8}{len(tr):>8}{len(te):>7}{ic:>+9.3f}"
              f"{st['total']*100:>10.1f}%{bh['total']*100:>10.1f}%"
              f"{diff*100:>+10.1f}pp{bh['total']*100:>8.1f}%")

    n_beat = sum(1 for r in yearly if r["diff"] > 0)
    n_ic_pos = sum(1 for r in yearly if np.isfinite(r["ic"]) and r["ic"] > 0)
    print(f"\n  跑赢买入持有: {n_beat}/{len(yearly)}")
    print(f"  IC 为正: {n_ic_pos}/{len(yearly)}")
    report["yearly"] = yearly

    # ============================================================
    # 2. 与零参数基准对比（20日均线）
    # ============================================================
    print("\n[2] 零参数基准对比：20日均线规则（同样的调仓周期）")
    print("-" * 88)

    ma20 = df["close"].rolling(20).mean()
    ma_rule = (df["close"] > ma20).astype(float).values

    print(f"  {'区间':<22}{'ML策略':>11}{'MA20规则':>12}{'持有':>11}")
    ml_tot = ma_tot = bh_tot = 0.0
    segs = []
    for yy in sorted(set(years)):
        if yy < 2015:
            continue
        te = np.where(years == yy)[0]
        if len(te) < 60:
            continue
        s, e = int(te[0]), int(te[-1]) + 1
        st_ma = bt(df, ma_rule, s, e, args.cost, start_pos=float(ma_rule[s - 1]) if s > 0 else 0.0)
        st_bh = bt(df, np.ones(n), s, e, args.cost, start_pos=1.0)
        yv = next((r for r in yearly if r["year"] == yy), None)
        ml = yv["strategy"] if yv else np.nan
        segs.append({"year": int(yy), "ml": ml, "ma20": st_ma["total"],
                     "bh": st_bh["total"]})
        print(f"  {yy:<22}{ml*100:>10.1f}%{st_ma['total']*100:>11.1f}%"
              f"{st_bh['total']*100:>10.1f}%")
    report["ma20_compare"] = segs

    # ============================================================
    # 3. 2015 股灾压力测试
    # ============================================================
    print("\n[3] 2015 股灾压力测试（只用 2015-06 之前训练）")
    print("-" * 88)
    crash_start = pd.Timestamp("2015-06-12")
    crash_end = pd.Timestamp("2016-02-29")
    tr_mask = dates < crash_start
    te_mask = (dates >= crash_start) & (dates <= crash_end)
    tr = np.where(tr_mask)[0]
    te = np.where(te_mask)[0]
    if len(tr) > 300 and len(te) > 30:
        yh = make_label(df, 20)
        ok_tr = np.isfinite(yh[tr])
        tr = tr[ok_tr]
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), yh[tr])
        pred = m.predict(sc.transform(X[te]))
        t = make_period_target(pred, te, n, args.period)
        s, e = int(te[0]), int(te[-1]) + 1
        st = bt(df, t, s, e, args.cost)
        bh = bt(df, np.ones(n), s, e, args.cost, start_pos=1.0)
        st_ma = bt(df, ma_rule, s, e, args.cost, start_pos=1.0)
        print(f"  区间 {dates[te[0]].date()} ~ {dates[te[-1]].date()}")
        print(f"  ML策略:   {st['total']*100:+8.1f}%  换手 {st['trades']}")
        print(f"  MA20规则: {st_ma['total']*100:+8.1f}%")
        print(f"  买入持有: {bh['total']*100:+8.1f}%")
        report["crash_2015"] = {"ml": st["total"], "ma20": st_ma["total"],
                                "buyhold": bh["total"]}
    else:
        print("  样本不足")
        report["crash_2015"] = None

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 88)
    print("判定")
    print("=" * 88)

    if yearly:
        med_diff = float(np.median([r["diff"] for r in yearly]))
        worst = min(yearly, key=lambda r: r["diff"])
    else:
        med_diff, worst = np.nan, None

    checks = [
        ("逐年跑赢过半", n_beat > len(yearly) / 2,
         f"{n_beat}/{len(yearly)}"),
        ("逐年 IC 过半为正", n_ic_pos > len(yearly) / 2,
         f"{n_ic_pos}/{len(yearly)}"),
        ("逐年超额中位数为正", med_diff > 0,
         f"{med_diff*100:+.1f}pp" if np.isfinite(med_diff) else "nan"),
        ("最差年份不是巨亏", worst is not None and worst["diff"] > -0.20,
         f"{worst['year']} {worst['diff']*100:+.1f}pp" if worst else "nan"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "策略在样本外成立" if all_ok else "策略在样本外不成立（过拟合）"
    print(f"\n  结论: {verdict}")
    report["checks"] = [{"name": a_, "ok": bool(o), "detail": d_}
                        for a_, o, d_ in checks]
    report["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
