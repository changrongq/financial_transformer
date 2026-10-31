# -*- coding: utf-8 -*-
"""严格检验「0/1 每 20 日调一次」这个形式，+131.9% 是否可信。

可疑点：
  1. 收益 131.9% vs 买入持有 76%，换手仅 19 次 —— 太好
  2. 平均仓位 45%，却比 100% 满仓赚得多 —— 要求择时非常准
  3. 训练/测试只切了一次

必须做的检验：
  A. 时序核对：targets[i-1] 的执行时点是否真的正确
  B. 参数敏感性：调仓周期改成 5/10/20/40/60 天，看是否只有 20 天好
  C. 滚动窗口：多次训练/测试切分，看是否稳定
  D. 逐笔交易检查：19 笔交易的进出点是否合理
  E. 与「同仓位暴露的随机策略」对比

用法：
  python scripts/audit_chinext_best_form.py
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
    r = run_ledger(df["date"].values[s:e],
                   df["open"].values[s:e].astype(float),
                   df["close"].values[s:e].astype(float),
                   df["pre_close"].values[s:e].astype(float),
                   targets[s:e], cost=cost, slip=0.0, start_pos=start_pos)
    st = ledger_stats(r.nav)
    st["trades"] = r.trades
    st["_res"] = r
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--out", default="results/chinext_best_form_audit.json")
    args = ap.parse_args()

    db = resolve_db()
    breadth = load_market_breadth(db)
    df = load_etf_ext(db, "159915.SZ", breadth)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    n = len(df)
    X = df[FEATS].values.astype(np.float64)

    print("=" * 88)
    print("严格审查：0/1 每 20 日调一次")
    print("=" * 88)

    report = {"cost": args.cost}

    # ============================================================
    # A. 时序核对：重跑一次并打印前 10 天的账本
    # ============================================================
    print("\n[A] 时序核对（账本前 8 天）")
    print("-" * 88)

    h = 20
    y = make_label(df, h)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    i_tr, i_va = int(n * 0.60), int(n * 0.80)
    tr = idx[idx < i_tr]
    te = idx[idx >= i_va]

    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
    pred = m.predict(sc.transform(X[te]))

    # 每 20 日调一次
    pos_te = np.zeros(len(pred))
    cur = 1.0
    med = np.median(pred)
    for i in range(len(pred)):
        if i % 20 == 0:
            cur = 1.0 if pred[i] > med else 0.0
        pos_te[i] = cur

    tgt = np.zeros(n)
    tgt[te] = pos_te
    s, e = int(te[0]), int(te[-1]) + 1

    st = bt(df, tgt, s, e, args.cost)
    print(f"  总收益 {st['total']*100:+.1f}%  换手 {st['trades']}  回撤 {st['dd']*100:.1f}%")

    print(f"\n  {'日期':<12}{'pre_close':>10}{'open':>8}{'close':>8}"
          f"{'prev_pos':>9}{'new_pos':>9}{'隔夜':>9}{'日内':>9}{'nav':>10}")
    for d in st["_res"].days[:8]:
        ds = pd.Timestamp(d.date).date()
        print(f"  {str(ds):<12}{'':>10}{'':>8}{'':>8}"
              f"{d.prev_pos:>9.0f}{d.new_pos:>9.0f}"
              f"{d.overnight_ret*100:>8.2f}%{d.intraday_ret*100:>8.2f}%"
              f"{d.nav:>10.4f}")

    print("\n  核对要点：new_pos 应该等于【前一天】算出的目标仓位")
    print("  （run_ledger 内部用 targets[i-1] 作为第 i 天的执行目标）")
    ok_time = True
    for k, d in enumerate(st["_res"].days[:5]):
        i_in_df = s + k + 1
        expect = tgt[i_in_df - 1]
        if abs(d.new_pos - expect) > 1e-9:
            ok_time = False
            print(f"  MISMATCH at day {k}: new_pos={d.new_pos} expect={expect}")
    print(f"  时序正确? {ok_time}")
    report["timing_ok"] = bool(ok_time)

    # ============================================================
    # B. 参数敏感性：调仓周期
    # ============================================================
    print("\n[B] 调仓周期敏感性（如果是真信号，不应只在 20 天好）")
    print("-" * 88)
    print(f"  {'周期':<8}{'总收益':>12}{'回撤':>9}{'夏普':>7}{'换手':>7}{'vs持有':>11}")

    bh = bt(df, np.ones(n), s, e, args.cost, start_pos=1.0)
    rows = []
    for period in [1, 5, 10, 20, 40, 60, 120]:
        p = np.zeros(len(pred))
        c = 1.0
        for i in range(len(pred)):
            if i % period == 0:
                c = 1.0 if pred[i] > med else 0.0
            p[i] = c
        t = np.zeros(n)
        t[te] = p
        stx = bt(df, t, s, e, args.cost)
        d = stx["total"] - bh["total"]
        rows.append({"period": period, "total": stx["total"],
                     "dd": stx["dd"], "sharpe": stx["sharpe"],
                     "trades": stx["trades"], "vs_bh": d})
        print(f"  {period:<8}{stx['total']*100:>11.1f}%{stx['dd']*100:>8.1f}%"
              f"{stx['sharpe']:>7.2f}{stx['trades']:>7}{d*100:>+10.1f}pp")
    print(f"  {'买入持有':<8}{bh['total']*100:>11.1f}%{bh['dd']*100:>8.1f}%"
          f"{bh['sharpe']:>7.2f}{bh['trades']:>7}")
    report["period_sensitivity"] = rows

    # ============================================================
    # C. 滚动窗口（多次切分）
    # ============================================================
    print("\n[C] 滚动窗口：换成不同训练/测试切分")
    print("-" * 88)
    print(f"  {'切分':<22}{'IC':>9}{'策略':>11}{'持有':>11}{'差':>11}")

    splits = []
    for tf in [0.40, 0.50, 0.60, 0.70]:
        i_tr2 = int(n * tf)
        idx2 = idx
        tr2 = idx2[idx2 < i_tr2]
        te2 = idx2[idx2 >= i_tr2]
        if len(tr2) < 200 or len(te2) < 100:
            continue
        sc2 = StandardScaler().fit(X[tr2])
        m2 = Ridge(alpha=10.0).fit(sc2.transform(X[tr2]), y[tr2])
        p2 = m2.predict(sc2.transform(X[te2]))
        ic2 = spearman(p2, y[te2])

        pp = np.zeros(len(p2))
        c2 = 1.0
        med2 = np.median(p2)
        for i in range(len(p2)):
            if i % 20 == 0:
                c2 = 1.0 if p2[i] > med2 else 0.0
            pp[i] = c2
        t2 = np.zeros(n)
        t2[te2] = pp
        s2, e2 = int(te2[0]), int(te2[-1]) + 1
        st2 = bt(df, t2, s2, e2, args.cost)
        bh2 = bt(df, np.ones(n), s2, e2, args.cost, start_pos=1.0)
        d2 = st2["total"] - bh2["total"]
        label = f"train {tf*100:.0f}%"
        splits.append({"train_frac": tf, "ic": ic2, "total": st2["total"],
                       "bh": bh2["total"], "diff": d2})
        print(f"  {label:<22}{ic2:>+9.3f}{st2['total']*100:>10.1f}%"
              f"{bh2['total']*100:>10.1f}%{d2*100:>+10.1f}pp")
    report["rolling_splits"] = splits

    # ============================================================
    # D. 逐笔交易
    # ============================================================
    print("\n[D] 逐笔交易明细（周期=20）")
    print("-" * 88)
    days = st["_res"].days
    trades = []
    prev = 0.0
    for k, d in enumerate(days):
        if abs(d.new_pos - d.prev_pos) > 1e-9:
            trades.append({
                "date": str(pd.Timestamp(d.date).date()),
                "from": d.prev_pos, "to": d.new_pos,
                "nav_before": days[k - 1].nav if k > 0 else 1.0,
            })
    print(f"  共 {len(trades)} 笔")
    for tk in trades[:12]:
        print(f"    {tk['date']}  {tk['from']:.0f} → {tk['to']:.0f}  "
              f"nav={tk['nav_before']:.4f}")
    report["trades"] = trades

    # ============================================================
    # E. 同仓位暴露的随机策略
    # ============================================================
    print("\n[E] 对照：同平均仓位的随机择时")
    print("-" * 88)
    expo = float(pos_te.mean())
    rng = np.random.RandomState(0)
    sims = []
    for _ in range(300):
        p = np.zeros(len(pred))
        c = 1.0
        for i in range(len(pred)):
            if i % 20 == 0:
                c = 1.0 if rng.rand() < expo else 0.0
            p[i] = c
        t = np.zeros(n)
        t[te] = p
        stx = bt(df, t, s, e, args.cost)
        sims.append(stx["total"])
    sims = np.array(sims)
    print(f"  实测策略收益: {st['total']*100:+.1f}%  平均仓位 {expo*100:.1f}%")
    print(f"  随机对照(300次): 中位 {np.median(sims)*100:+.1f}%  "
          f"p95 {np.percentile(sims,95)*100:+.1f}%  最大 {sims.max()*100:+.1f}%")
    pct = float((sims < st["total"]).mean() * 100)
    print(f"  实测策略在随机分布中的分位: {pct:.1f}%")
    report["random_control"] = {
        "exposure": expo, "actual": st["total"],
        "median": float(np.median(sims)),
        "p95": float(np.percentile(sims, 95)),
        "max": float(sims.max()), "pctile": pct,
    }

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 88)
    print("判定")
    print("=" * 88)

    n_period_beat = sum(1 for r in rows if r["vs_bh"] > 0)
    n_split_beat = sum(1 for r in splits if r["diff"] > 0)
    checks = [
        ("账本时序正确", ok_time, ""),
        ("调仓周期不敏感（>=3 个周期跑赢）", n_period_beat >= 3,
         f"{n_period_beat}/{len(rows)}"),
        ("滚动切分稳定（>=3/4 跑赢）", n_split_beat >= 3,
         f"{n_split_beat}/{len(splits)}"),
        ("超出随机对照 95% 分位", st["total"] > np.percentile(sims, 95),
         f"{st['total']*100:+.1f}% vs {np.percentile(sims,95)*100:+.1f}%"),
    ]
    all_ok = True
    for name, okk, detail in checks:
        print(f"  [{'PASS' if okk else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        all_ok &= okk

    verdict = "值得继续" if all_ok else "不稳定，不能作为策略"
    print(f"\n  结论: {verdict}")
    report["checks"] = [{"name": a_, "ok": bool(o), "detail": d_}
                        for a_, o, d_ in checks]
    report["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in report.items() if k != "trades"},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
