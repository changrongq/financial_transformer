# -*- coding: utf-8 -*-
"""创业板信号：IC 为真但回测亏钱，问题在哪？

现象：
  H=20 IC=+0.2295  p=0.005（单看 p 很好）
  但扣成本后跑输买入持有 -47.7pp，换手 171 次

疑点：
  1. 「pred > 中位 就满仓」等价于仓位在 0/1 之间频繁切换
     → 每 2 天换一次，成本吃掉所有收益
  2. IC 是全样本秩相关，回测是时序路径，两者不等价
  3. IC 显著可能只是「成交额高 → 波动大 → 收益分布更宽」
     而不是「成交额高 → 未来涨」

本脚本诊断：
  A. 分层收益：按预测分 5 组，看是否单调（真信号应单调）
  B. 换手成本：算不同成本假设下的净收益
  C. 持有期衰减：预测 1/5/10/20 天后还有效吗
  D. 交易频率：用"只在信号强度超过阈值时才动"能不能救
  E. 与买入持有的真实比较（同仓位暴露）

用法：
  python scripts/diagnose_chinext_signal.py
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

CODE = "159915.SZ"
FEATS = PRICE_FEATS + PV_FEATS + VOLSTRUCT_FEATS


def backtest(df, targets, s, e, cost, start_pos=0.0):
    r = run_ledger(df["date"].values[s:e],
                   df["open"].values[s:e].astype(float),
                   df["close"].values[s:e].astype(float),
                   df["pre_close"].values[s:e].astype(float),
                   targets[s:e], cost=cost, slip=0.0, start_pos=start_pos)
    st = ledger_stats(r.nav)
    st["trades"] = r.trades
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/chinext_diagnosis.json")
    args = ap.parse_args()

    db = resolve_db()
    breadth = load_market_breadth(db)
    df = load_etf_ext(db, CODE, breadth)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    n = len(df)
    i_tr, i_va = int(n * 0.60), int(n * 0.80)

    X = df[FEATS].values.astype(np.float64)
    y = make_label(df, 20)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    tr = idx[idx < i_tr]
    te = idx[idx >= i_va]

    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
    pred = m.predict(sc.transform(X[te]))
    yte = y[te]

    print("=" * 86)
    print(f"创业板信号诊断  n={n}  测试段 {len(te)} 天")
    print(f"IC = {spearman(pred, yte):+.4f}")
    print("=" * 86)

    report = {"code": CODE, "ic": float(spearman(pred, yte))}

    # ============================================================
    # A. 分层收益（真信号必须单调）
    # ============================================================
    print("\n[A] 五分层收益（按预测值分组）")
    print("-" * 86)
    qs = np.quantile(pred, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"  {'分组':<8}{'样本数':>7}{'平均未来20日收益':>18}{'胜率':>9}")
    layers = []
    for k in range(5):
        lo, hi = qs[k], qs[k + 1]
        mask = (pred >= lo) & (pred <= hi) if k == 4 else (pred >= lo) & (pred < hi)
        if mask.sum() < 10:
            continue
        ret = float(yte[mask].mean())
        win = float((yte[mask] > 0).mean())
        layers.append({"q": k + 1, "n": int(mask.sum()), "ret": ret, "win": win})
        print(f"  Q{k+1:<7}{mask.sum():>7}{ret*100:>17.2f}%{win*100:>8.1f}%")

    rets = [l["ret"] for l in layers]
    monotone = all(rets[i] <= rets[i + 1] for i in range(len(rets) - 1))
    print(f"\n  单调递增? {monotone}")
    print(f"  Q5 - Q1 = {(rets[-1]-rets[0])*100:+.2f}%")
    report["layers"] = layers
    report["monotone"] = bool(monotone)

    # ============================================================
    # B. 不同成本假设
    # ============================================================
    print("\n[B] 成本敏感性（pred > 中位 就满仓）")
    print("-" * 86)
    med = np.median(pred)
    tgt = np.zeros(n)
    tgt[te] = (pred > med).astype(float)
    s, e = int(te[0]), int(te[-1]) + 1

    print(f"  {'成本/边':<10}{'总收益':>12}{'换手':>8}{'年化成本拖累':>16}")
    cost_rows = []
    for c in [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]:
        st = backtest(df, tgt, s, e, c)
        yrs = (e - s) / 244
        drag = (1 - (1 - c) ** (st["trades"] * 2)) if c > 0 else 0.0
        cost_rows.append({"cost": c, "total": st["total"],
                          "trades": st["trades"], "drag": float(drag)})
        print(f"  {c*100:>8.3f}%{st['total']*100:>11.1f}%{st['trades']:>8}"
              f"{drag*100:>15.1f}%")
    report["cost_sensitivity"] = cost_rows

    # ============================================================
    # C. 持有期衰减
    # ============================================================
    print("\n[C] 信号衰减（用同一模型预测不同持有期）")
    print("-" * 86)
    decay = []
    for h in [1, 5, 10, 20, 40, 60]:
        yh = make_label(df, h)
        okh = np.isfinite(yh)
        idxh = np.where(okh)[0]
        trh = idxh[idxh < i_tr]
        teh = idxh[idxh >= i_va]
        if len(teh) < 80:
            continue
        sch = StandardScaler().fit(X[trh])
        mh = Ridge(alpha=10.0).fit(sch.transform(X[trh]), yh[trh])
        ph = mh.predict(sch.transform(X[teh]))
        v = spearman(ph, yh[teh])
        decay.append({"h": h, "ic": float(v)})
        print(f"  H={h:>3} 日  IC={v:+.4f}")
    report["decay"] = decay

    # ============================================================
    # D. 低换手版本（只在预测强度大时才满仓）
    # ============================================================
    print("\n[D] 降低换手：只在预测超过高分位时才满仓")
    print("-" * 86)
    print(f"  {'阈值':<10}{'持仓天数占比':>14}{'总收益':>12}{'换手':>8}{'vs持有':>12}")
    thr_rows = []
    bh = backtest(df, np.ones(n), s, e, 0.0003, start_pos=1.0)
    for q in [0.5, 0.7, 0.8, 0.9, 0.95]:
        thr = np.quantile(pred, q)
        t = np.zeros(n)
        t[te] = (pred > thr).astype(float)
        st = backtest(df, t, s, e, 0.0003)
        frac = float(t[te].mean())
        thr_rows.append({"q": q, "frac": frac, "total": st["total"],
                         "trades": st["trades"]})
        print(f"  分位{q*100:>4.0f}%{frac*100:>13.1f}%{st['total']*100:>11.1f}%"
              f"{st['trades']:>8}{(st['total']-bh['total'])*100:>+11.1f}pp")
    print(f"  买入持有{'':<10}{100.0:>13.1f}%{bh['total']*100:>11.1f}%"
          f"{bh['trades']:>8}")
    report["threshold_sweep"] = thr_rows
    report["buyhold"] = bh

    # ============================================================
    # E. 同期与买入持有的风险调整比较
    # ============================================================
    print("\n[E] 同期风险对比")
    print("-" * 86)
    st_full = backtest(df, tgt, s, e, 0.0003)
    print(f"  模型(0/1)   收益 {st_full['total']*100:+7.1f}%  "
          f"回撤 {st_full['dd']*100:6.1f}%  夏普 {st_full['sharpe']:5.2f}")
    print(f"  买入持有    收益 {bh['total']*100:+7.1f}%  "
          f"回撤 {bh['dd']*100:6.1f}%  夏普 {bh['sharpe']:5.2f}")
    print(f"  持有时间占比 {float(tgt[te].mean())*100:.1f}%")

    report["summary"] = {
        "model": st_full, "buyhold": bh,
        "exposure": float(tgt[te].mean()),
        "layer_spread": float(rets[-1] - rets[0]),
        "monotone": bool(monotone),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
