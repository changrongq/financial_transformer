# -*- coding: utf-8 -*-
"""检验 H=20 时 5/6 标的 IC 为负是否是真信号（均值回归），而不仅是噪声。

严格版发现：
  H=20 日：中位 IC = -0.0692，5/6 标的为负，但只有 2/6 显著。

这个模式值得单独追：
  · 如果只是噪声，正负应该各半（二项分布 5/6 同向的 p 约 0.11）
  · 如果真存在均值回归，那么"特征预测低收益 → 实际高收益"是可交易的

本脚本做三件事：
  1. 二项检验：6 只标的中 5 只同向，这个概率有多大
  2. 反正向策略回测：把模型预测**取反**做仓位，看扣成本后是否赚钱
  3. 分年度稳定性：如果只是某两年贡献，不是真信号

用法：
  python scripts/test_mean_reversion_signal.py
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
from test_feature_value import load_etf, resolve_db, make_label, FEATURES
from test_feature_value_strict import (CODES, NAMES, spearman,
                                       block_bootstrap_ci)
from rl_ledger import run_ledger, ledger_stats

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """精确二项检验（双尾）。"""
    from math import comb
    def pmf(i):
        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    obs = pmf(k)
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12)
    return float(min(1.0, total))


def run_one(code: str, db: str, h: int, cost: float) -> dict:
    name = NAMES.get(code, code)
    df = load_etf(db, code)
    if df.empty:
        return {"code": code, "name": name, "error": "no data"}
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    n = len(df)
    if n < 800:
        return {"code": code, "name": name, "error": f"too few {n}"}

    X = df[FEATURES].values.astype(np.float64)
    y = make_label(df, h).values.astype(np.float64)
    ok = np.isfinite(y)

    idx = np.where(ok)[0]
    i_tr, i_va = int(n * 0.60), int(n * 0.80)
    tr = idx[idx < i_tr]
    te = idx[idx >= i_va]
    if len(tr) < 300 or len(te) < 100:
        return {"code": code, "name": name, "error": "sample too small"}

    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=10.0)
    m.fit(sc.transform(X[tr]), y[tr])

    # 在测试段做样本外预测
    Xte_s = sc.transform(X[te])
    pred = m.predict(Xte_s)
    real_ic = spearman(pred, y[te])

    # ---- 反向策略：pred 低 → 买；pred 高 → 空 ----
    # 仓位 = -sign(pred - median)，映射到 {0, 1}
    med = np.median(pred)
    target = (pred < med).astype(float)

    # 构造全长的 target 数组，只在 te 上有效
    tgt_full = np.zeros(n)
    tgt_full[te] = target

    # 账本：从 te 第一天开始
    s = int(te[0])
    e = int(te[-1]) + 1
    r = run_ledger(df["date"].values[s:e],
                   df["open"].values[s:e].astype(float),
                   df["close"].values[s:e].astype(float),
                   df["pre_close"].values[s:e].astype(float),
                   tgt_full[s:e], cost=cost, slip=0.0)
    st = ledger_stats(r.nav)

    # ---- 买入持有对照（同区间） ----
    tgt_bh = np.ones(n)
    r_bh = run_ledger(df["date"].values[s:e],
                      df["open"].values[s:e].astype(float),
                      df["close"].values[s:e].astype(float),
                      df["pre_close"].values[s:e].astype(float),
                      tgt_bh[s:e], cost=cost, slip=0.0, start_pos=1.0)
    st_bh = ledger_stats(r_bh.nav)

    # ---- 分年度 ----
    te_dates = df["date"].values[te]
    yrs = pd.DatetimeIndex(te_dates).year
    yearly = []
    for yy in sorted(set(yrs)):
        mask = yrs == yy
        if mask.sum() < 30:
            continue
        # 该年的策略收益：用 pred 与 y 的相关方向
        yy_ic = spearman(pred[mask], y[te][mask])
        # 反向策略在该年的理论收益
        yy_ret = float(np.mean(y[te][mask][target[mask] > 0])) if (target[mask] > 0).sum() > 0 else np.nan
        yearly.append({"year": int(yy), "ic": yy_ic, "n": int(mask.sum()),
                       "long_ret": yy_ret})

    lo, hi = block_bootstrap_ci(pred, y[te])

    return {
        "code": code, "name": name,
        "n": n, "n_test": int(len(te)),
        "ic": real_ic,
        "ic_ci95": [lo, hi],
        "reverse_total": st["total"],
        "reverse_dd": st["dd"],
        "reverse_sharpe": st["sharpe"],
        "reverse_trades": r.trades,
        "bh_total": st_bh["total"],
        "bh_dd": st_bh["dd"],
        "yearly": yearly,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--out", default="results/mean_reversion_signal.json")
    args = ap.parse_args()

    db = resolve_db()
    print("=" * 84)
    print(f"H={args.horizon} 日 均值回归信号检验")
    print("=" * 84)

    results = []
    for code in CODES:
        r = run_one(code, db, args.horizon, args.cost)
        results.append(r)
        if "error" in r:
            print(f"{code} {NAMES.get(code,'')}: {r['error']}")
            continue
        ci = r["ic_ci95"]
        print(f"\n{code} {r['name']}")
        print(f"  IC = {r['ic']:+.4f}  CI95=[{ci[0]:+.3f},{ci[1]:+.3f}]  n_test={r['n_test']}")
        print(f"  反向策略: {r['reverse_total']*100:+8.1f}%  回撤 {r['reverse_dd']*100:6.1f}%  "
              f"夏普 {r['reverse_sharpe']:5.2f}  换手 {r['reverse_trades']}")
        print(f"  买入持有: {r['bh_total']*100:+8.1f}%  回撤 {r['bh_dd']*100:6.1f}%")

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\n没有有效结果")
        return 1

    # ---- 二项检验 ----
    print("\n" + "=" * 84)
    print("二项检验：多标的同向是否是偶然")
    print("=" * 84)

    ics = np.array([r["ic"] for r in valid])
    n_neg = int((ics < 0).sum())
    n_pos = int((ics > 0).sum())
    n_tot = len(ics)

    p_neg = binom_test_two_sided(n_neg, n_tot, 0.5)
    p_pos = binom_test_two_sided(n_pos, n_tot, 0.5)
    print(f"  正 {n_pos} / 负 {n_neg} / 共 {n_tot}")
    print(f"  二项检验 p（负向） = {p_neg:.4f}")
    print(f"  二项检验 p（正向） = {p_pos:.4f}")
    print(f"  → {'方向一致不是偶然' if min(p_neg, p_pos) < 0.10 else '方向分布与随机无显著差异'}")

    # ---- 反向策略汇总 ----
    print("\n" + "=" * 84)
    print("反向策略（做多 IC 的负向）扣成本后")
    print("=" * 84)
    rev = np.array([r["reverse_total"] for r in valid])
    bh = np.array([r["bh_total"] for r in valid])
    print(f"  反向策略: 中位 {np.median(rev)*100:+.1f}%  平均 {rev.mean()*100:+.1f}%  "
          f"跑赢持有 {int((rev > bh).sum())}/{len(valid)}")
    for r in valid:
        print(f"    {r['name']:<8} 反向 {r['reverse_total']*100:+8.1f}%  "
              f"持有 {r['bh_total']*100:+8.1f}%  "
              f"差 {(r['reverse_total']-r['bh_total'])*100:+8.1f}pp")

    # ---- 分年度稳定性 ----
    print("\n" + "=" * 84)
    print("分年度 IC 稳定性（每年 IC 是否为负）")
    print("=" * 84)
    all_years = {}
    for r in valid:
        for y in r["yearly"]:
            all_years.setdefault(y["year"], []).append(y["ic"])
    neg_years = 0
    for yy in sorted(all_years):
        vals = np.array([v for v in all_years[yy] if np.isfinite(v)])
        if len(vals) == 0:
            continue
        m = float(np.median(vals))
        if m < 0:
            neg_years += 1
        print(f"  {yy}: 中位 IC {m:+.4f}  ({len(vals)} 只标的)")
    print(f"\n  IC 中位为负的年份: {neg_years}/{len(all_years)}")

    report = {
        "horizon": args.horizon, "cost": args.cost,
        "results": results,
        "binom": {"n_pos": n_pos, "n_neg": n_neg, "n": n_tot,
                  "p_neg": p_neg, "p_pos": p_pos},
        "reverse_vs_bh": {"median": float(np.median(rev)),
                          "win": int((rev > bh).sum()), "n": len(valid)},
        "yearly_ic_median": {str(k): float(np.median([v for v in vs if np.isfinite(v)]))
                             for k, vs in all_years.items()
                             if any(np.isfinite(v) for v in vs)},
        "neg_years": neg_years, "total_years": len(all_years),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
