# -*- coding: utf-8 -*-
"""特征增量价值检验 —— 严格版（含显著性检验与多标的）。

上一版 `test_feature_value.py` 发现：
  · 所有持有期的真实 IC 都不显著
  · 但【打乱标签】的 IC 也能到 ±0.10 —— 说明单次 IC 波动本身就有这么大
  · 结论"无显著信息"是对的，但证据不够硬

本版加强：
  1. **多标的重复**：6 只 ETF 各自独立检验，看方向是否一致
  2. **置换检验分布**：打乱标签 100 次，得到 IC 的零分布，
     再看真实 IC 落在多少分位数
  3. **Newey-West / 分块 bootstrap**：处理时间序列自相关
  4. **滚动窗口**：不只看一次 test，看 IC 在多个时间窗口是否稳定

判据（严格）：
  · |真实 IC| 超过零分布的 95% 分位 → 显著
  · 且 6 只标的中至少 4 只方向一致 → 不是单标的偶然
  · 且 滚动窗口中至少 60% 同向 → 时间上稳定

用法：
  python scripts/test_feature_value_strict.py
  python scripts/test_feature_value_strict.py --horizons 5 10 20 --perms 200
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
from test_feature_value import (load_etf, resolve_db, make_label, FEATURES)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CODES = ["510050.SH", "510300.SH", "510500.SH",
         "512100.SH", "159915.SZ", "588000.SH"]
NAMES = {"510050.SH": "上证50", "510300.SH": "沪深300", "510500.SH": "中证500",
         "512100.SH": "中证1000", "159915.SZ": "创业板", "588000.SH": "科创50"}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 20:
        return np.nan
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def fit_ridge_ic(Xtr, ytr, Xte, yte) -> float:
    """训练 Ridge，返回测试段 IC。"""
    if len(Xtr) < 100 or len(Xte) < 50:
        return np.nan
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=10.0)
    m.fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    return spearman(pred, yte)


def analytic_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """同一模型在这批数据上的 IC（用于分层，不需要重训）。"""
    return spearman(pred, actual)


def permutation_null(X, y, tr, te, n_perm: int, seed: int = 0) -> np.ndarray:
    """打乱训练段标签，重训，得到测试 IC 的零分布。"""
    rng = np.random.RandomState(seed)
    out = []
    sc = StandardScaler().fit(X[tr])
    Xtr_s = sc.transform(X[tr])
    Xte_s = sc.transform(X[te])
    for _ in range(n_perm):
        yp = y[tr].copy()
        rng.shuffle(yp)
        m = Ridge(alpha=10.0)
        m.fit(Xtr_s, yp)
        out.append(spearman(m.predict(Xte_s), y[te]))
    return np.array([v for v in out if np.isfinite(v)])


def block_bootstrap_ci(pred: np.ndarray, actual: np.ndarray,
                       block: int = 20, n_boot: int = 300,
                       seed: int = 0) -> tuple:
    """分块 bootstrap 得到 IC 的 95% 置信区间（处理时序自相关）。"""
    rng = np.random.RandomState(seed)
    n = len(pred)
    if n < block * 2:
        return (np.nan, np.nan)
    n_blocks = max(2, n // block)
    vals = []
    for _ in range(n_boot):
        starts = rng.randint(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        idx = idx[idx < n]
        v = spearman(pred[idx], actual[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def rolling_ic(X, y, n_splits: int = 5, train_frac: float = 0.5) -> list:
    """滚动窗口 IC：每个窗口用前 50% 训练、后 50% 测试。"""
    n = len(X)
    out = []
    win = n // n_splits
    for k in range(n_splits):
        s = k * win
        e = min(s + win, n)
        if e - s < 200:
            continue
        mid = s + int((e - s) * train_frac)
        tr = np.arange(s, mid)
        te = np.arange(mid, e)
        v = fit_ridge_ic(X[tr], y[tr], X[te], y[te])
        out.append({"window": k, "start": int(s), "end": int(e), "ic": v})
    return out


def analyze_one(code: str, db: str, horizons: list, n_perm: int) -> dict:
    name = NAMES.get(code, code)
    df = load_etf(db, code)
    if df.empty:
        return {"code": code, "name": name, "error": "no data"}

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    n = len(df)
    if n < 800:
        return {"code": code, "name": name, "error": f"too few rows {n}"}

    X = df[FEATURES].values.astype(np.float64)
    i_tr, i_va = int(n * 0.60), int(n * 0.80)

    res = {"code": code, "name": name, "n": n,
           "period": [str(df["date"].min().date()), str(df["date"].max().date())],
           "horizons": {}}

    for h in horizons:
        y = make_label(df, h).values.astype(np.float64)
        ok = np.isfinite(y)
        idx = np.where(ok)[0]
        tr = idx[idx < i_tr]
        te = idx[idx >= i_va]

        if len(tr) < 300 or len(te) < 100:
            res["horizons"][str(h)] = {"error": "sample too small"}
            continue

        # 真实 IC
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=10.0)
        m.fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        real_ic = spearman(pred, y[te])

        # 零分布
        null = permutation_null(X, y, tr, te, n_perm)
        if len(null) == 0:
            pct = np.nan
            p = np.nan
        else:
            pct = float((np.abs(null) < abs(real_ic)).mean() * 100)
            # 双尾 p 值
            p = float((np.abs(null) >= abs(real_ic)).mean())

        # 置信区间
        lo, hi = block_bootstrap_ci(pred, y[te])

        # 滚动稳定性
        roll = rolling_ic(X, y, n_splits=5)

        res["horizons"][str(h)] = {
            "ic": real_ic,
            "null_std": float(np.std(null)) if len(null) else np.nan,
            "null_pctile": pct,
            "p_value": p,
            "ic_ci95": [lo, hi],
            "n_test": int(len(te)),
            "rolling": roll,
            "rolling_same_sign": (sum(1 for r in roll
                                      if np.isfinite(r["ic"]) and np.sign(r["ic"]) == np.sign(real_ic))
                                  if np.isfinite(real_ic) else 0),
            "rolling_total": sum(1 for r in roll if np.isfinite(r["ic"])),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--perms", type=int, default=150)
    ap.add_argument("--out", default="results/feature_value_strict.json")
    args = ap.parse_args()

    db = resolve_db()
    print("=" * 88)
    print("特征增量价值检验（严格版）")
    print(f"标的 {len(CODES)} 只   持有期 {args.horizons}   置换次数 {args.perms}")
    print("=" * 88)

    report = {"codes": CODES, "horizons": args.horizons,
              "n_perm": args.perms, "results": {}}

    for code in CODES:
        print(f"\n--- {code} {NAMES.get(code, '')} ---")
        r = analyze_one(code, db, args.horizons, args.perms)
        report["results"][code] = r
        if "error" in r:
            print(f"  {r['error']}")
            continue
        print(f"  区间 {r['period'][0]} ~ {r['period'][1]}  n={r['n']}")
        for h in args.horizons:
            d = r["horizons"].get(str(h), {})
            if "error" in d:
                print(f"  H={h:>2}: {d['error']}")
                continue
            ci = d["ic_ci95"]
            print(f"  H={h:>2}: IC={d['ic']:+.4f}  "
                  f"零分布std={d['null_std']:.4f}  "
                  f"分位={d['null_pctile']:5.1f}%  p={d['p_value']:.3f}  "
                  f"CI95=[{ci[0]:+.3f},{ci[1]:+.3f}]  "
                  f"滚动同向={d['rolling_same_sign']}/{d['rolling_total']}")

    # ============================================================
    # 汇总判定
    # ============================================================
    print("\n" + "=" * 88)
    print("汇总判定")
    print("=" * 88)

    summary = {}
    for h in args.horizons:
        rows = []
        for code in CODES:
            d = report["results"].get(code, {}).get("horizons", {}).get(str(h), {})
            if "ic" not in d:
                continue
            rows.append({
                "code": code,
                "name": NAMES.get(code, code),
                "ic": d["ic"],
                "p": d["p_value"],
                "ci": d["ic_ci95"],
                "roll_ok": d["rolling_same_sign"],
                "roll_n": d["rolling_total"],
            })

        if not rows:
            continue

        ics = np.array([r["ic"] for r in rows])
        ps = np.array([r["p"] for r in rows])
        # 三选法（Fisher 合并 p 值过于乐观，这里用最保守的）
        n_sig = int((ps < 0.05).sum())
        n_neg = int((ics < 0).sum())
        n_pos = int((ics > 0).sum())
        median_ic = float(np.median(ics))

        # 稳定性
        stable = sum(1 for r in rows
                     if r["roll_n"] > 0 and r["roll_ok"] / r["roll_n"] >= 0.60)

        print(f"\n  H={h} 日:")
        print(f"    中位 IC = {median_ic:+.4f}")
        print(f"    显著标的数(p<0.05) = {n_sig}/{len(rows)}")
        print(f"    方向: 正 {n_pos} / 负 {n_neg}")
        print(f"    滚动窗口 60% 同向的标的数 = {stable}/{len(rows)}")

        # 判定
        # 有信息 = 多数标的显著 + 方向一致 + 滚动稳定
        informative = (n_sig >= max(3, len(rows) // 2)
                       and max(n_pos, n_neg) >= max(4, int(len(rows) * 0.75))
                       and stable >= max(3, len(rows) // 2))
        summary[str(h)] = {
            "median_ic": median_ic,
            "n_significant": n_sig,
            "n_total": len(rows),
            "n_positive": n_pos, "n_negative": n_neg,
            "n_rolling_stable": stable,
            "informative": bool(informative),
            "detail": rows,
        }
        print(f"    → {'【有增量信息】' if informative else '【无可靠增量信息】'}")

    report["summary"] = summary
    any_info = any(v["informative"] for v in summary.values())
    report["verdict"] = "有增量信息" if any_info else "无增量信息"

    print("\n" + "=" * 88)
    print(f"总体结论: {report['verdict']}")
    print("=" * 88)
    print("""
判据说明（严格版）：
  · 多数标的 p < 0.05            → 真实 IC 超出置换零分布
  · 至少 75% 标的同向            → 不是单标的偶然
  · 多数标的滚动窗口 60% 同向    → 时间上稳定
  三条全满足才算"有增量信息"。
""")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
