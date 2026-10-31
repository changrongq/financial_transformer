# -*- coding: utf-8 -*-
"""解释矛盾：AUC 0.59 但 IC -0.03。

现象
----
  · 日内 AUC（分类：未来5日是否涨≥3%） = 0.5934
  · 日内 IC（排序：预测 vs 未来5日实际收益） = -0.0311

两者不必然一致，但符号相反很反常。可能的解释：

  H1. 模型学到的是「波动/风险」而非「收益」：
      高波动股既容易涨超 3%，也容易跌超 3%，
      所以预测「会涨超3%」对高波动股给高分 → AUC 高；
      但这些股的中位收益并不高 → 排序 IC 为负甚至为 0。

  H2. 阈值标签的固有特性：
      y=1 只要求「≥3%」，不区分涨 3% 还是涨 30%。
      模型可能只是找出「振幅大的股票」。

  H3. 我的 IC 计算有问题。

本脚本检验 H1（最可能）。

用法：
  python scripts/explain_auc_ic_divergence.py --max-stocks 200
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


def spearman(a, b):
    if len(a) < 20:
        return np.nan
    ra = pd.Series(a).rank().values.astype(float)
    rb = pd.Series(b).rank().values.astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--out", default="results/auc_ic_divergence.json")
    args = ap.parse_args()

    from train_single_task import (load_panel, add_label_end, build_sequences,
                                   FEATURE_COLS, LOG_COLS)
    from transformer_fixes import ColumnScaler
    from sklearn.linear_model import Ridge

    db = str((ROOT.parent / "stock_history.db").resolve())
    print("=" * 90)
    print("AUC 与 IC 符号矛盾的解释")
    print("=" * 90)

    df = load_panel(args.max_stocks, db)
    df = df.dropna(subset=["target_up_5d"]).reset_index(drop=True)
    df = add_label_end(df, horizon=5)
    avail = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=avail).reset_index(drop=True)
    ds = build_sequences(df, avail, args.seq_len)

    Xtr, ytr, frtr, _ = ds["train"]
    Xte, yte, frte, mte = ds["test"]
    dates = np.array([m[1] for m in mte])

    tr_df = pd.DataFrame(Xtr[:, -1, :], columns=avail)
    sc = ColumnScaler(log_cols=[c for c in LOG_COLS if c in avail]).fit(tr_df)

    def scale(X3):
        n, t, f = X3.shape
        flat = pd.DataFrame(X3.reshape(-1, f), columns=avail)
        return sc.transform(flat).values.astype(np.float32).reshape(n, t, f)

    Xtr_s, Xte_s = scale(Xtr), scale(Xte)
    m = Ridge(alpha=1.0).fit(Xtr_s[:, -1, :], ytr)
    p = m.predict(Xte_s[:, -1, :])

    print(f"\ntest {len(yte):,} 样本  {len(set(dates))} 天")
    print(f"  全体 AUC = {auc(yte, p):.4f}")
    print(f"  全体 IC  = {spearman(p, frte):.4f}")

    # ---- H1：预测值是否只是在预测「波动」 ----
    print("\n" + "=" * 90)
    print("[H1] 预测值是否在预测「波动」而不是「收益」")
    print("=" * 90)

    # 从特征里取波动率和振幅作为代理
    vol_idx = avail.index("volatility_20d") if "volatility_20d" in avail else None
    amp_idx = avail.index("gap_open") if "gap_open" in avail else None

    # 用原始（未标准化前的）特征
    last_raw = Xtr[:, -1, :]
    te_raw = Xte[:, -1, :]

    if vol_idx is not None:
        # 用训练集中位数还原 log/z 之前的量级不方便，直接用标准化后的值排序
        vol_te = Xte_s[:, -1, vol_idx]
        ic_vol = spearman(p, vol_te)
        print(f"  预测值 rank 与 volatility_20d 的秩相关: {ic_vol:+.4f}")
        print(f"  → {'预测主要在押波动' if abs(ic_vol) > 0.3 else '预测与波动关系不大'}")

    # 关键检验：把未来收益按 |收益| 分组，看 AUC 在各组内是否还在
    print(f"\n  按「未来5日收益绝对值」分 4 组，组内 AUC 与 IC:")
    print(f"  {'组':<10}{'样本':>7}{'平均收益':>11}{'AUC':>9}{'IC':>9}")
    absr = np.abs(frte)
    qs = np.quantile(absr, [0, 0.25, 0.5, 0.75, 1.0])
    group_rows = []
    for k in range(4):
        lo, hi = qs[k], qs[k + 1]
        mk = (absr >= lo) & (absr <= hi) if k == 3 else (absr >= lo) & (absr < hi)
        if mk.sum() < 50:
            continue
        a = auc(yte[mk], p[mk])
        icg = spearman(p[mk], frte[mk])
        print(f"  Q{k+1:<9}{int(mk.sum()):>7}{frte[mk].mean()*100:>10.2f}%"
              f"{a:>9.4f}{icg:>9.4f}")
        group_rows.append({"q": k + 1, "n": int(mk.sum()),
                           "mean_ret": float(frte[mk].mean()),
                           "auc": a, "ic": icg})

    # ---- H2：阈值标签的信息损失 ----
    print("\n" + "=" * 90)
    print("[H2] 阈值标签 y=1 内部收益差异巨大")
    print("=" * 90)
    pos = frte[yte == 1]
    neg = frte[yte == 0]
    print(f"  正例 (y=1): n={len(pos):,}  收益 中位 {np.median(pos)*100:+.2f}%  "
          f"p10 {np.percentile(pos,10)*100:+.2f}%  p90 {np.percentile(pos,90)*100:+.2f}%")
    print(f"  负例 (y=0): n={len(neg):,}  收益 中位 {np.median(neg)*100:+.2f}%  "
          f"p90 {np.percentile(neg,90)*100:+.2f}%")
    print(f"\n  正例内部的收益跨度: "
          f"{np.percentile(pos,90)*100 - np.percentile(pos,10)*100:.1f} 个百分点")
    print(f"  → 标签只区分「是否 ≥3%」，不区分「3% 还是 30%」")

    # ---- 用连续收益做标签再训一次 ----
    print("\n" + "=" * 90)
    print("[H3] 换成连续收益标签（回归），AUC/IC 是否恢复一致")
    print("=" * 90)
    okr = np.isfinite(frtr)
    m_reg = Ridge(alpha=1.0).fit(Xtr_s[okr, -1, :], frtr[okr])
    p_reg = m_reg.predict(Xte_s[:, -1, :])

    # 两种评估
    # 1) 用连续预测去分类「≥3%」
    y_pred_bin = (p_reg > np.quantile(p_reg, 1 - yte.mean())).astype(int)
    a_reg = auc(yte, p_reg)
    ic_reg = spearman(p_reg, frte)
    print(f"  回归模型的 AUC（预测 vs ≥3% 标签）= {a_reg:.4f}")
    print(f"  回归模型的 IC （预测 vs 连续收益）  = {ic_reg:.4f}")
    print(f"\n  对比分类模型:")
    print(f"  分类模型 AUC = {auc(yte, p):.4f}   IC = {spearman(p, frte):.4f}")

    # ---- 结论 ----
    print("\n" + "=" * 90)
    print("[结论]")
    print("=" * 90)
    a_cls = auc(yte, p)
    ic_cls = spearman(p, frte)
    if a_cls > 0.55 and ic_cls < 0:
        verdict = ("分类信号存在（能判断会不会涨≥3%），"
                   "但不等于排序信号（判断不了谁涨得多）")
    elif ic_reg > 0.03:
        verdict = "连续标签下有正向排序信号"
    else:
        verdict = "两种标签下都没有稳定信号"
    print(f"  {verdict}")

    out = {
        "n_test": int(len(yte)),
        "cls_auc": a_cls, "cls_ic": ic_cls,
        "reg_auc": a_reg, "reg_ic": ic_reg,
        "by_abs_return_groups": group_rows,
        "pos_ret_median": float(np.median(pos)),
        "verdict": verdict,
    }
    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
