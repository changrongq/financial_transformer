# -*- coding: utf-8 -*-
"""验证创业板在扩展特征下的显著性是真是假。

观察：
  纯价格       H=10 IC=+0.108 p=0.23   H=20 IC=+0.013 p=0.90
  价格+量价     H=10 IC=+0.185 p=0.03   H=20 IC=+0.230 p=0.02
  价格+量价+波动 H=5  IC=+0.159 p=0.00   H=10 IC=+0.186 p=0.03

问题：这是真信号，还是 6 标的 × 3 特征集 × 3 持有期 = 54 次检验里的
      多重比较产物？（54 次检验，期望约 2.7 个 p<0.05 纯属偶然）

本脚本做五件事：
  1. 精确统计一共做了多少次检验，Bonferroni / FDR 校正
  2. 逐个特征单变量 IC（看是不是某个特征在驱动）
  3. 扣成本的真实回测（用统一账本）
  4. 滚动窗口逐年看
  5. 打乱标签的"最好结果"分布（模拟多重比较）

用法：
  python scripts/verify_chinext_signal.py
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
from test_feature_value_strict import spearman, block_bootstrap_ci
from test_feature_value_extended import (resolve_db, load_etf_ext,
                                         load_market_breadth,
                                         PRICE_FEATS, PV_FEATS,
                                         VOLSTRUCT_FEATS, make_label)
from rl_ledger import run_ledger, ledger_stats

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CODE = "159915.SZ"
FEATS = PRICE_FEATS + PV_FEATS + VOLSTRUCT_FEATS


def split_idx(n, h):
    i_tr, i_va = int(n * 0.60), int(n * 0.80)
    return i_tr, i_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--out", default="results/chinext_signal_check.json")
    args = ap.parse_args()

    db = resolve_db()
    breadth = load_market_breadth(db)
    df = load_etf_ext(db, CODE, breadth)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    n = len(df)

    print("=" * 88)
    print(f"创业板 {CODE} 显著性验证")
    print(f"区间 {df['date'].min().date()} ~ {df['date'].max().date()}  n={n}")
    print(f"特征 {len(FEATS)} 个   置换 {args.perms} 次   成本 {args.cost*100:.3f}%/边")
    print("=" * 88)

    i_tr, i_va = split_idx(n, 0)
    X = df[FEATS].values.astype(np.float64)

    report = {"code": CODE, "n": n, "features": FEATS,
              "n_perm": args.perms, "cost": args.cost}

    # ============================================================
    # 1. 多重比较校正
    # ============================================================
    print("\n" + "-" * 88)
    print("[1] 多重比较校正")
    print("-" * 88)

    # 我们实际做过的检验数：6 标的 × 3 特征集 × 3 持有期
    # 加上这次的第一批：6 标的 × 4 持有期（纯价格 12 特征）
    n_tests_total = 6 * 3 * 3 + 6 * 4
    print(f"  历史检验总数（两批合计）: {n_tests_total}")
    print(f"  纯偶然期望 p<0.05 个数: {n_tests_total * 0.05:.1f}")
    print(f"  实际观测到 p<0.05 个数（创业板+量价）: 3")
    bonf = 0.05 / n_tests_total
    print(f"  Bonferroni 阈值: p < {bonf:.5f}")
    print(f"  → 创业板最好的 p=0.02，{'通过' if 0.02 < bonf else '未通过'} Bonferroni")

    # ============================================================
    # 2. 单变量 IC（找驱动特征）
    # ============================================================
    print("\n" + "-" * 88)
    print("[2] 单变量 IC（H=20，测试段）")
    print("-" * 88)

    h = 20
    y = make_label(df, h)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    tr = idx[idx < i_tr]
    te = idx[idx >= i_va]

    uni = []
    for f in FEATS:
        if not np.isfinite(df[f].values[te]).all():
            continue
        v = spearman(df[f].values[te], y[te])
        if np.isfinite(v):
            uni.append({"feature": f, "ic": float(v)})
    uni.sort(key=lambda x: -abs(x["ic"]))

    print(f"  {'特征':<20}{'单变量IC':>12}")
    for u in uni[:10]:
        print(f"  {u['feature']:<20}{u['ic']:>+12.4f}")
    report["univariate"] = uni

    # ============================================================
    # 3. 多变量 + 置换检验 + 回测
    # ============================================================
    print("\n" + "-" * 88)
    print("[3] 多变量模型 + 置换零分布")
    print("-" * 88)

    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    m = Ridge(alpha=10.0).fit(Xtr, y[tr])
    pred = m.predict(Xte)
    real_ic = spearman(pred, y[te])

    rng = np.random.RandomState(0)
    null = []
    for _ in range(args.perms):
        yp = y[tr].copy()
        rng.shuffle(yp)
        v = spearman(Ridge(alpha=10.0).fit(Xtr, yp).predict(Xte), y[te])
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)

    p_val = float((np.abs(null) >= abs(real_ic)).mean())
    lo, hi = block_bootstrap_ci(pred, y[te])

    print(f"  H={h}  IC={real_ic:+.4f}  p={p_val:.4f}  "
          f"零分布 std={null.std():.4f}  CI95=[{lo:+.3f},{hi:+.3f}]")
    print(f"  零分布 95% 分位: {np.percentile(null, 95):+.4f}")
    print(f"  零分布最大:      {np.abs(null).max():+.4f}")

    report["multivariate"] = {
        "h": h, "ic": real_ic, "p": p_val,
        "null_std": float(null.std()),
        "null_p95": float(np.percentile(null, 95)),
        "null_max_abs": float(np.abs(null).max()),
        "ci95": [lo, hi],
    }

    # ---- 回测：正向策略 ----
    print("\n  扣成本回测（预测 > 中位 就满仓）")
    med = np.median(pred)
    tgt_full = np.zeros(n)
    tgt_full[te] = (pred > med).astype(float)

    s, e = int(te[0]), int(te[-1]) + 1
    r = run_ledger(df["date"].values[s:e],
                   df["open"].values[s:e].astype(float),
                   df["close"].values[s:e].astype(float),
                   df["pre_close"].values[s:e].astype(float),
                   tgt_full[s:e], cost=args.cost, slip=0.0)
    st = ledger_stats(r.nav)

    r_bh = run_ledger(df["date"].values[s:e],
                      df["open"].values[s:e].astype(float),
                      df["close"].values[s:e].astype(float),
                      df["pre_close"].values[s:e].astype(float),
                      np.ones(e - s), cost=args.cost, slip=0.0, start_pos=1.0)
    st_bh = ledger_stats(r_bh.nav)

    print(f"    模型策略: {st['total']*100:+8.1f}%  回撤 {st['dd']*100:6.1f}%  "
          f"夏普 {st['sharpe']:5.2f}  换手 {r.trades}")
    print(f"    买入持有: {st_bh['total']*100:+8.1f}%  回撤 {st_bh['dd']*100:6.1f}%  "
          f"夏普 {st_bh['sharpe']:5.2f}")
    print(f"    差值: {(st['total']-st_bh['total'])*100:+.1f}pp")

    report["backtest"] = {"model": st, "buyhold": st_bh,
                          "trades": r.trades}

    # ============================================================
    # 4. 逐年稳定性
    # ============================================================
    print("\n" + "-" * 88)
    print("[4] 逐年 IC（测试段内）")
    print("-" * 88)

    te_years = pd.DatetimeIndex(df["date"].values[te]).year
    yearly = []
    for yy in sorted(set(te_years)):
        mask = te_years == yy
        if mask.sum() < 40:
            continue
        v = spearman(pred[mask], y[te][mask])
        yearly.append({"year": int(yy), "ic": v, "n": int(mask.sum())})
        if np.isfinite(v):
            print(f"  {yy}: IC={v:+.4f}  n={mask.sum()}")

    neg = sum(1 for r in yearly if np.isfinite(r["ic"]) and r["ic"] < 0)
    tot = sum(1 for r in yearly if np.isfinite(r["ic"]))
    print(f"  负 IC 年份: {neg}/{tot}")
    report["yearly"] = yearly

    # ============================================================
    # 5. 模拟多重比较：随机挑 54 次检验，看能否也找到 p<0.03
    # ============================================================
    print("\n" + "-" * 88)
    print("[5] 多重比较模拟：随机数据能刷出多少显著结果")
    print("-" * 88)

    rng2 = np.random.RandomState(7)
    n_sim = 200
    best_ps = []
    for _ in range(n_sim):
        # 生成 54 组纯随机 IC（用零分布的近似正态）
        fake = rng2.normal(0, 0.13, size=54)   # 与实测零分布 std 一致
        # 用 100 个置换样本估 p
        best_p = min(
            float((np.abs(rng2.normal(0, 0.13, 100)) >= abs(v)).mean())
            for v in fake)
        best_ps.append(best_p)
    best_ps = np.array(best_ps)

    print(f"  模拟 {n_sim} 次（每次 54 组随机检验）")
    print(f"  出现至少一个 p<0.05 的比例: {(best_ps < 0.05).mean()*100:.1f}%")
    print(f"  出现至少一个 p<0.03 的比例: {(best_ps < 0.03).mean()*100:.1f}%")
    print(f"  出现至少一个 p<0.02 的比例: {(best_ps < 0.02).mean()*100:.1f}%")
    print(f"\n  → 我们观测到的最好 p=0.02，"
          f"纯随机情况下的发生概率约 {(best_ps < 0.02).mean()*100:.1f}%")
    report["multiple_comparison_sim"] = {
        "p_any_005": float((best_ps < 0.05).mean()),
        "p_any_003": float((best_ps < 0.03).mean()),
        "p_any_002": float((best_ps < 0.02).mean()),
    }

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 88)
    print("判定")
    print("=" * 88)

    checks = [
        ("Bonferroni 校正后仍显著", p_val < bonf, f"p={p_val:.4f} vs {bonf:.5f}"),
        ("置换零分布未见过这么大的 IC",
         abs(real_ic) > np.abs(null).max(),
         f"|IC|={abs(real_ic):.4f} vs null_max={np.abs(null).max():.4f}"),
        ("扣成本后跑赢买入持有", st["total"] > st_bh["total"],
         f"{(st['total']-st_bh['total'])*100:+.1f}pp"),
        ("逐年 IC 至少 75% 同向",
         tot > 0 and max(neg, tot - neg) / tot >= 0.75,
         f"{max(neg, tot-neg)}/{tot}"),
        ("多重比较下不太可能是偶然",
         (best_ps < p_val).mean() < 0.20,
         f"随机达到 p<{p_val:.2f} 的概率 {(best_ps < p_val).mean()*100:.1f}%"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "真信号" if all_ok else "不足以判定为真信号"
    print(f"\n  结论: {verdict}")
    report["checks"] = [{"name": n_, "ok": bool(o), "detail": d}
                        for n_, o, d in checks]
    report["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
