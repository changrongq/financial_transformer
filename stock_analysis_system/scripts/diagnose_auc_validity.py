# -*- coding: utf-8 -*-
"""检验 test AUC 0.55~0.59 是真信号还是评估假象。

可疑点
------
1. 同一天的股票高度相关（市场 beta），
   导致 AUC 被少数几天主导，有效样本量远小于名义样本量。

2. test 期只有约 40 个交易日。如果模型学到"某几天的市场方向"，
   AUC 会虚高，但不是选股能力。

3. 如果 tag=正例的日期集中在少数几天，
   模型只需要预测"今天是不是好日子"就能拿到高 AUC。

本脚本做三件事：
  A. 按日拆解 test AUC（每天的 AUC 及样本数）
  B. 日内 AUC vs 跨日 AUC 的对比
  C. 日期聚类 bootstrap（每天为一块）估计 AUC 置信区间

用法：
  python scripts/diagnose_auc_validity.py --max-stocks 200
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
sys.path.insert(0, str(ROOT / "scripts"))


def auc(y, p):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--out", default="results/auc_validity.json")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from train_single_task import (load_panel, add_label_end, build_sequences,
                                   SmallTransformer, FEATURE_COLS, LOG_COLS,
                                   spearman)
    from transformer_fixes import ColumnScaler, snapshot_state_dict
    from sklearn.linear_model import Ridge

    db = str((ROOT.parent / "stock_history.db").resolve())
    print("=" * 90)
    print("AUC 有效性诊断")
    print("=" * 90)

    df = load_panel(args.max_stocks, db)
    df = df.dropna(subset=["target_up_5d"]).reset_index(drop=True)
    df = add_label_end(df, horizon=5)
    avail = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=avail).reset_index(drop=True)

    ds = build_sequences(df, avail, args.seq_len)
    Xtr, ytr, frtr, _ = ds["train"]
    Xva, yva, frva, _ = ds["val"]
    Xte, yte, frte, mte = ds["test"]
    print(f"\ntrain {len(Xtr):,}  val {len(Xva):,}  test {len(Xte):,}")

    # 标准化
    tr_df = pd.DataFrame(Xtr[:, -1, :], columns=avail)
    sc = ColumnScaler(log_cols=[c for c in LOG_COLS if c in avail]).fit(tr_df)

    def scale(X3):
        n, t, f = X3.shape
        flat = pd.DataFrame(X3.reshape(-1, f), columns=avail)
        return sc.transform(flat).values.astype(np.float32).reshape(n, t, f)

    Xtr_s, Xva_s, Xte_s = scale(Xtr), scale(Xva), scale(Xte)

    dates = np.array([m[1] for m in mte])
    codes = np.array([m[0] for m in mte])

    # ---- Ridge 预测 ----
    m = Ridge(alpha=1.0).fit(Xtr_s[:, -1, :], ytr)
    p_ridge = m.predict(Xte_s[:, -1, :])
    overall = auc(yte, p_ridge)
    print(f"\n[0] Ridge test AUC（全体）= {overall:.4f}")

    # ---- A. 按日拆解 ----
    print("\n" + "=" * 90)
    print("[A] 按交易日拆解 AUC")
    print("=" * 90)
    ud = sorted(set(dates))
    print(f"  test 共 {len(ud)} 个交易日")
    print(f"  {'日期':<12}{'样本':>7}{'正例':>7}{'总AUC':>10}"
          f"{'仅日内AUC':>12}")
    rows = []
    for d in ud:
        mask = dates == d
        if mask.sum() < 10:
            continue
        yd = yte[mask]
        pd_ = p_ridge[mask]
        a_all = auc(yd, pd_)
        rows.append({"date": d, "n": int(mask.sum()),
                     "pos": int(yd.sum()), "auc": a_all})
        print(f"  {d:<12}{int(mask.sum()):>7}{int(yd.sum()):>7}"
              f"{a_all:>10.4f}")

    rdf = pd.DataFrame(rows)
    print(f"\n  日期数 {len(rdf)}  平均日内 AUC = {rdf['auc'].mean():.4f}"
          f"  (中位 {rdf['auc'].median():.4f})")
    print(f"  有 AUC 的日期 {rdf['auc'].notna().sum()} / {len(rdf)}")
    print(f"  正例集中度: 单日最多 {rdf['pos'].max()} 个正例 / "
          f"总 {rdf['pos'].sum()} 个")

    # ---- B. 跨日 vs 日内 ----
    print("\n" + "=" * 90)
    print("[B] 全体 AUC 是否被「跨日差异」夸大")
    print("=" * 90)
    # ★ 正确做法：二分类标签不能减均值。
    #   要消除"哪天整体好/坏"的影响，应该：
    #   (1) 只在有足够正例和负例的日期上算 AUC，再对日期取平均
    #   (2) 用 future_return（连续值）做日内秩相关
    valid_days = []
    for d in ud:
        mk = dates == d
        if mk.sum() < 20:
            continue
        yd = yte[mk]
        if yd.sum() >= 5 and (1 - yd).sum() >= 5:
            valid_days.append(d)

    per_day_auc = []
    per_day_ic = []
    for d in valid_days:
        mk = dates == d
        a = auc(yte[mk], p_ridge[mk])
        # 日内 IC：预测 vs 实际未来收益（连续）
        icd = spearman(p_ridge[mk], frte[mk])
        if np.isfinite(a):
            per_day_auc.append(a)
        if np.isfinite(icd):
            per_day_ic.append(icd)

    a_within = float(np.mean(per_day_auc)) if per_day_auc else np.nan
    ic_within = float(np.mean(per_day_ic)) if per_day_ic else np.nan
    print(f"  全体 AUC                     = {overall:.4f}")
    print(f"  有效日期（正负例各 >= 5）     = {len(valid_days)} / {len(ud)}")
    print(f"  【日内 AUC 的日期平均】       = {a_within:.4f}"
          f"  ← 这才是纯选股能力")
    print(f"  【日内 IC 的日期平均】        = {ic_within:.4f}"
          f"  ← 预测 vs 真实收益的秩相关")
    print(f"  日内 IC > 0 的日期占比        = "
          f"{np.mean(np.array(per_day_ic) > 0)*100:.1f}%")

    # ---- C. 日期聚类 bootstrap ----
    print("\n" + "=" * 90)
    print("[C] 按日期聚类 bootstrap（每日为一块）")
    print("=" * 90)
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(500):
        pick = rng.choice(ud, size=len(ud), replace=True)
        idx = np.concatenate([np.where(dates == d)[0] for d in pick])
        a = auc(yte[idx], p_ridge[idx])
        if np.isfinite(a):
            boot.append(a)
    boot = np.array(boot)
    print(f"  bootstrap {len(boot)} 次")
    print(f"  中位 {np.median(boot):.4f}  "
          f"95%CI [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")
    print(f"  P(AUC > 0.5) = {(boot > 0.5).mean()*100:.1f}%")

    # ---- 判定 ----
    print("\n" + "=" * 90)
    print("[D] 判定")
    print("=" * 90)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    checks = [
        ("日均 AUC 显著 > 0.5", rdf["auc"].mean() > 0.53,
         f"{rdf['auc'].mean():.4f}"),
        ("日内去均值后 AUC 仍 > 0.52", a_within > 0.52,
         f"{a_within:.4f}"),
        ("互信区间下界 > 0.50", lo > 0.50, f"[{lo:.4f}, {hi:.4f}]"),
        ("正例不集中在少数日", rdf["pos"].max() / max(rdf["pos"].sum(), 1) < 0.15,
         f"最多 {rdf['pos'].max()}/{rdf['pos'].sum()}"),
    ]
    ok_all = True
    for n, o, d in checks:
        print(f"  [{'PASS' if o else 'FAIL'}] {n}  ({d})")
        ok_all &= o

    verdict = "AUC 可信，确有选股信号" if ok_all else "AUC 可能是评估假象，需谨慎"
    print(f"\n  结论: {verdict}")

    out = {
        "n_test": int(len(yte)), "n_days": len(ud),
        "ridge_test_auc": overall, "auc_within_day": a_within,
        "ic_within_day": ic_within,
        "daily_auc_mean": float(rdf["auc"].mean()),
        "daily_auc_median": float(rdf["auc"].median()),
        "bootstrap_median": float(np.median(boot)),
        "bootstrap_ci": [float(lo), float(hi)],
        "p_auc_gt_05": float((boot > 0.5).mean()),
        "checks": [{"name": n, "ok": bool(o), "detail": d}
                   for n, o, d in checks],
        "verdict": verdict,
        "daily": rows,
    }
    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
