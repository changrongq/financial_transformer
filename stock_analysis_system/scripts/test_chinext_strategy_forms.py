# -*- coding: utf-8 -*-
"""创业板信号：分层检验 + 测试期偏差 + 正确的策略形式。

上次诊断发现：
  · Q5-Q1 分层价差 +6.27%（信号真实存在）
  · 但所有 0/1 择时都跑输买入持有
  · IC 随持有期增大（H40 达 +0.27），像是趋势/动量

两个可能：
  (a) 信号真实，但策略形式错了（不该在牛市中空仓）
  (b) 测试期只是创业板大牛市，任何降低暴露都亏

本脚本判定：
  1. 测试期偏差：把测试段拆成两半，看是否都成立
  2. 正确的策略形式：
     - 满仓 + 只在预测最差时降到 50%（不是 0/1）
     - 用 20 日持有期，但每月才调一次仓（降低换手）
  3. 对照：同期其他标的（不是只看创业板）

用法：
  python scripts/test_chinext_strategy_forms.py
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
from test_feature_value_strict import spearman, CODES, NAMES
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
    r = run_ledger(df["date"].values[s:e],
                   df["open"].values[s:e].astype(float),
                   df["close"].values[s:e].astype(float),
                   df["pre_close"].values[s:e].astype(float),
                   targets[s:e], cost=cost, slip=0.0, start_pos=start_pos)
    st = ledger_stats(r.nav)
    st["trades"] = r.trades
    return st


def build_pred(df, feats, h, train_frac=0.60, test_frac=0.80):
    n = len(df)
    X = df[feats].values.astype(np.float64)
    y = make_label(df, h)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    i_tr, i_va = int(n * train_frac), int(n * test_frac)
    tr = idx[idx < i_tr]
    te = idx[idx >= i_va]
    if len(tr) < 200 or len(te) < 80:
        return None
    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
    return {"pred_te": m.predict(sc.transform(X[te])),
            "te": te, "y": y, "n": n, "i_va": i_va}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--out", default="results/chinext_strategy_forms.json")
    args = ap.parse_args()

    db = resolve_db()
    breadth = load_market_breadth(db)
    report = {"cost": args.cost, "results": {}}

    print("=" * 88)
    print("创业板信号：策略形式与测试期偏差")
    print("=" * 88)

    # ============================================================
    # 1. 测试期拆两半
    # ============================================================
    print("\n[1] 测试期拆成前后两半")
    print("-" * 88)

    df = load_etf_ext(db, "159915.SZ", breadth)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    n = len(df)

    # 把测试段再对半分，检验两半是否同向
    b = build_pred(df, FEATS, 20, 0.60, 0.80)
    if b is None:
        print("  样本不足")
        return 1
    te = b["te"]
    pred = b["pred_te"]
    mid = len(te) // 2
    te1, te2 = te[:mid], te[mid:]
    p1, p2 = pred[:mid], pred[mid:]
    y1, y2 = b["y"][te1], b["y"][te2]

    ic1 = spearman(p1, y1)
    ic2 = spearman(p2, y2)
    d1s = pd.Timestamp(df["date"].values[te1[0]]).date()
    d1e = pd.Timestamp(df["date"].values[te1[-1]]).date()
    d2s = pd.Timestamp(df["date"].values[te2[0]]).date()
    d2e = pd.Timestamp(df["date"].values[te2[-1]]).date()
    print(f"  前半 {d1s} ~ {d1e}  IC={ic1:+.4f}  n={len(te1)}")
    print(f"  后半 {d2s} ~ {d2e}  IC={ic2:+.4f}  n={len(te2)}")
    print(f"  → 两半{'同向' if np.sign(ic1) == np.sign(ic2) else '反向，不稳定'}")

    report["results"]["halves"] = {"ic_first": ic1, "ic_second": ic2,
                                   "period_first": [str(d1s), str(d1e)],
                                   "period_second": [str(d2s), str(d2e)]}

    # ============================================================
    # 2. 真实敞口对比：测试期创业板涨了多少
    # ============================================================
    print("\n[2] 测试期市场环境")
    print("-" * 88)
    s, e = int(te[0]), int(te[-1]) + 1
    bh = bt(df, np.ones(n), s, e, args.cost, start_pos=1.0)
    print(f"  买入持有: {bh['total']*100:+.1f}%  回撤 {bh['dd']*100:.1f}%  "
          f"夏普 {bh['sharpe']:.2f}")
    print(f"  测试段天数: {e-s}")
    report["results"]["buyhold"] = bh

    # ============================================================
    # 3. 不同策略形式
    # ============================================================
    print("\n[3] 策略形式对比（H=20 预测）")
    print("-" * 88)

    q = np.quantile(pred, [0.1, 0.3, 0.5, 0.7, 0.9])
    forms = {}

    def make_from_pred(pred_te, rule):
        t = np.zeros(n)
        t[te] = rule(pred_te)
        return t

    # (a) 0/1 按中位
    forms["0/1 中位"] = make_from_pred(pred, lambda p: (p > np.median(p)).astype(float))
    # (b) 极端降仓：最差 20% 降半仓
    def rule_b(p):
        out = np.ones(len(p))
        out[p < np.quantile(p, 0.2)] = 0.5
        return out
    forms["满仓+最差20%降半仓"] = make_from_pred(pred, rule_b)
    # (c) 最差 10% 全空
    def rule_c(p):
        out = np.ones(len(p))
        out[p < np.quantile(p, 0.1)] = 0.0
        return out
    forms["满仓+最差10%全空"] = make_from_pred(pred, rule_c)
    # (d) 线性映射：把预测分位映射到 0.3~1.0 仓位
    def rule_d(p):
        r = pd.Series(p).rank(pct=True).values
        return 0.3 + 0.7 * r
    forms["仓位=0.3+0.7*分位"] = make_from_pred(pred, rule_d)
    # (e) 每月调仓版（H=20 但每 20 天更新一次）
    def rule_e(p):
        out = np.zeros(len(p))
        cur = 1.0
        for i in range(len(p)):
            if i % 20 == 0:
                cur = 1.0 if p[i] > np.median(p) else 0.0
            out[i] = cur
        return out
    forms["0/1 但每20日调一次"] = make_from_pred(pred, rule_e)
    # (f) 买入持有
    forms["买入持有"] = np.ones(n)

    print(f"  {'策略形式':<24}{'总收益':>12}{'回撤':>9}{'夏普':>7}"
          f"{'换手':>7}{'平均仓位':>10}{'vs持有':>11}")
    rows = []
    for name, t in forms.items():
        sp = 1.0 if name == "买入持有" else 0.0
        st = bt(df, t, s, e, args.cost, start_pos=sp)
        expo = float(t[te].mean())
        diff = st["total"] - bh["total"]
        rows.append({"form": name, "total": st["total"], "dd": st["dd"],
                     "sharpe": st["sharpe"], "trades": st["trades"],
                     "exposure": expo, "vs_bh": diff})
        print(f"  {name:<24}{st['total']*100:>11.1f}%{st['dd']*100:>8.1f}%"
              f"{st['sharpe']:>7.2f}{st['trades']:>7}{expo*100:>9.1f}%"
              f"{diff*100:>+10.1f}pp")
    report["results"]["forms"] = rows

    # ============================================================
    # 4. 换到其他标的：看同样的策略形式是否有效
    # ============================================================
    print("\n[4] 同一策略形式（满仓+最差20%降半仓）在其他标的上")
    print("-" * 88)
    print(f"  {'标的':<12}{'IC':>9}{'总收益':>11}{'持有':>11}{'差':>11}")
    others = []
    for code in CODES:
        d2 = load_etf_ext(db, code, breadth)
        d2 = d2.dropna(subset=FEATS).reset_index(drop=True)
        if len(d2) < 800:
            continue
        b2 = build_pred(d2, FEATS, 20, 0.60, 0.80)
        if b2 is None:
            continue
        te2, p2 = b2["te"], b2["pred_te"]
        n2 = b2["n"]

        # ★ 修正：先在测试段内构造仓位序列，再整体写入全长数组
        pos_te = np.ones(len(p2))
        pos_te[p2 < np.quantile(p2, 0.2)] = 0.5
        t2 = np.ones(n2)
        t2[te2] = pos_te

        s2, e2 = int(te2[0]), int(te2[-1]) + 1
        st2 = bt(d2, t2, s2, e2, args.cost, start_pos=1.0)
        bh2 = bt(d2, np.ones(n2), s2, e2, args.cost, start_pos=1.0)
        ic2 = spearman(p2, b2["y"][te2])
        others.append({"code": code, "name": NAMES.get(code, ""),
                       "ic": ic2, "total": st2["total"],
                       "bh": bh2["total"],
                       "diff": st2["total"] - bh2["total"]})
        print(f"  {NAMES.get(code,code):<12}{ic2:>+9.3f}{st2['total']*100:>10.1f}%"
              f"{bh2['total']*100:>10.1f}%{(st2['total']-bh2['total'])*100:>+10.1f}pp")
    report["results"]["other_assets"] = others

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 88)
    print("判定")
    print("=" * 88)

    best = max(rows, key=lambda r: r["total"])
    n_beat = sum(1 for r in others if r["diff"] > 0)
    checks = [
        ("信号在测试期两半都同向", np.sign(ic1) == np.sign(ic2),
         f"{ic1:+.3f} / {ic2:+.3f}"),
        ("存在跑赢买入持有的策略形式", any(r["vs_bh"] > 0 for r in rows),
         f"最好的是 {best['form']} {best['vs_bh']*100:+.1f}pp"),
        ("该形式在其他标的也跑赢", n_beat >= 3,
         f"{n_beat}/{len(others)}"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "值得继续研究" if all_ok else "不成立，停止这条线"
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
