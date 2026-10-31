# -*- coding: utf-8 -*-
"""横截面最终检验：分层不单调，那"能用"的到底是哪一段？

诊断结论：
  · 5 个因子测试期 IC 全部同号（-0.04 ~ -0.064），信号方向稳定
  · 但五分层【不单调】：Q1 -0.056%, Q2 +0.148%, Q3 +0.206%,
    Q4 +0.262%, Q5 +0.221%
  · IC 只在高流动性组显著（-0.061 vs 低流动性 -0.019）

不单调意味着：
  Q1（预测最差）并不是收益最差 → 反向做空最差组会亏
  Q5（预测最好）也不是最好 → 买最好组会亏

真正可能是：Q2~Q4 都比 Q1/Q5 好，也就是【极端值都是坏的】。
这在 A 股很常见：
  · Q1 = 极端高换手/高波动 → 游资炒作，后续回落
  · Q5 = 极端低换手 → 僵尸股，无人问津

本脚本检验四种选股口径：
  A. Top20%（买 Q5）        —— 常规做法
  B. Q2~Q4 中间 60%         —— 避开两端
  C. Q1 反向做多（买"最差"）—— 看 Q1 是不是真的差
  D. Q2~Q4 等权 + 剔除 Q5   —— 只剔除最不活跃

每种都扣 0.3%/边成本，看分季度是否稳定。

用法：
  python scripts/test_cross_section_final.py
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
from test_cross_section_v2 import cs_zscore, run_portfolio, bench_equal_weight

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["turnover_z", "vol20", "amplitude", "ret20", "pos120"]


def build_score(df, signs):
    s = pd.Series(0.0, index=df.index)
    n = 0
    for f in FACTORS:
        if not np.isfinite(signs.get(f, np.nan)) or signs[f] == 0:
            continue
        s = s + cs_zscore(df, f).fillna(0.0) * np.sign(signs[f])
        n += 1
    return s / max(n, 1)


def run_quantile_portfolio(df, dates, rebalance, lo_q, hi_q, cost_side):
    """买 score 在 [lo_q, hi_q] 分位之间的股票，等权。

    lo_q=0.8, hi_q=1.0 → Top20%
    lo_q=0.2, hi_q=0.8 → 中间 60%
    """
    ret_map = df.set_index(["date_str", "stock_code"])["ret1"].to_dict()
    dmap = {d: g for d, g in df.groupby("date_str")}

    nav = 1.0
    curve = []
    w_prev = {}
    pending = None
    total_turn = 0.0

    for i, d in enumerate(dates):
        g = dmap.get(d)
        if g is None or len(g) < 300:
            continue

        cost = 0.0
        if pending is not None:
            keys = set(pending) | set(w_prev)
            turn = 0.5 * sum(abs(pending.get(k, 0.0) - w_prev.get(k, 0.0))
                             for k in keys)
            total_turn += turn
            cost = turn * (2 * cost_side + 0.001)
            w_prev = pending
            pending = None

        if w_prev:
            rs, ws = [], []
            for sc, w in w_prev.items():
                r = ret_map.get((d, sc), np.nan)
                if isinstance(r, (int, float)) and np.isfinite(r):
                    rs.append(w * r)
                    ws.append(w)
            r = sum(rs) / sum(ws) if ws and sum(ws) > 0 else 0.0
        else:
            r = 0.0

        nav *= (1 + r) * (1 - cost)
        curve.append(nav)

        if i % rebalance == 0:
            s = g[["stock_code", "score"]].dropna()
            if len(s) < 300:
                continue
            ql = s["score"].quantile(lo_q)
            qh = s["score"].quantile(hi_q)
            picks = s[(s["score"] >= ql) & (s["score"] <= qh)]["stock_code"].tolist()
            if picks:
                w = 1.0 / len(picks)
                pending = {c: w for c in picks}
            else:
                pending = {}

    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan,
                "turnover_annual": 0.0, "days": len(curve)}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    years = max(len(arr) / 244, 1e-9)
    return {
        "total": float(arr[-1] - 1),
        "dd": float((full / peak - 1).min()),
        "sharpe": float(rets.mean() / sd * np.sqrt(244)) if sd > 1e-12 else np.nan,
        "turnover_annual": float(total_turn / years),
        "days": len(arr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--reb-early", type=int, default=5,
                    help="评分所用的持有期（决定调仓间隔）")
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results/cross_section_final.json")
    args = ap.parse_args()

    db = resolve_db()
    df = load_panel(db, load_quality_dates(), 4500)
    df = build_features(df)
    df = add_labels(df, [args.horizon])
    nd = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[nd >= 200]

    dates = sorted(df["date_str"].unique())
    train, test = dates[:args.train_days], dates[args.train_days:]
    ycol = f"fwd{args.horizon}"

    # 训练期符号
    df_tr = df[df["date_str"].isin(set(train))]
    signs = {}
    for f in FACTORS:
        ics = []
        for d, g in df_tr.groupby("date_str"):
            s = g[[f, ycol]].dropna()
            if len(s) < 300:
                continue
            a, b = s[f].rank().values, s[ycol].rank().values
            a -= a.mean()
            b -= b.mean()
            den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            if den > 0:
                ics.append((a * b).sum() / den)
        signs[f] = float(np.mean(ics)) if ics else 0.0

    df["score"] = build_score(df, signs)

    print("=" * 90)
    print("横截面最终检验：分层不单调，哪一段能用？")
    print("=" * 90)
    print(f"训练 {len(train)} 天 | 测试 {len(test)} 天 ({test[0]} ~ {test[-1]})")
    print(f"成本 {args.cost*100:.2f}%/边（含印花税+滑点）")
    print(f"调仓间隔 {args.reb_early} 天")

    bench = bench_equal_weight(df, test)
    print(f"\n市场等权: {bench['total']*100:+.1f}%  "
          f"回撤 {bench['dd']*100:.1f}%  夏普 {bench['sharpe']:.2f}")

    # 四种口径
    variants = {
        "A. Top20%(Q5)": (0.80, 1.00),
        "B. 中间60%(Q2-Q4)": (0.20, 0.80),
        "C. Bottom20%(Q1反向)": (0.00, 0.20),
        "D. 剔除Q5(0-80%)": (0.00, 0.80),
        "E. 剔除Q1(20-100%)": (0.20, 1.00),
    }

    print("\n" + "-" * 90)
    print(f"  {'口径':<24}{'收益':>11}{'vs市场':>11}{'回撤':>9}"
          f"{'夏普':>8}{'年换手':>9}")
    print("-" * 90)
    rows = {}
    for name, (lo, hi) in variants.items():
        st = run_quantile_portfolio(df, test, args.reb_early, lo, hi, args.cost)
        ex = st["total"] - bench["total"]
        rows[name] = {"total": st["total"], "excess": ex, "dd": st["dd"],
                      "sharpe": st["sharpe"],
                      "turnover": st["turnover_annual"]}
        print(f"  {name:<24}{st['total']*100:>10.1f}%{ex*100:>+10.1f}pp"
              f"{st['dd']*100:>8.1f}%{st['sharpe']:>8.2f}"
              f"{st['turnover_annual']:>9.1f}")

    # 分季度：只对最好的两个口径
    print("\n" + "-" * 90)
    print("分季度（最好的两个口径 + 市场对比）")
    print("-" * 90)

    ranked = sorted(rows.items(), key=lambda kv: -kv[1]["excess"])
    top2 = [r[0] for r in ranked[:2]]

    qmap = {}
    for d in test:
        qmap.setdefault(str(pd.Timestamp(d).to_period("Q")), []).append(d)

    print(f"  {'季度':<10}{'天':>5}" + "".join(f"{n[:12]:>14}" for n in top2)
          + f"{'市场':>11}")
    qres = []
    for q in sorted(qmap):
        ds = qmap[q]
        if len(ds) < 20:
            continue
        vals = {}
        for n in top2:
            lo, hi = variants[n]
            st = run_quantile_portfolio(df, ds, args.reb_early, lo, hi, args.cost)
            vals[n] = st["total"]
        bq = bench_equal_weight(df, ds)
        qres.append({"quarter": q, "days": len(ds), "market": bq["total"],
                     **vals})
        print(f"  {q:<10}{len(ds):>5}"
              + "".join(f"{vals[n]*100:>13.1f}%" for n in top2)
              + f"{bq['total']*100:>10.1f}%")

    n_beat = {n: sum(1 for r in qres if r.get(n, 0) > r["market"]) for n in top2}
    for n in top2:
        print(f"  {n}: 跑赢市场 {n_beat[n]}/{len(qres)}")

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 90)
    print("判定")
    print("=" * 90)

    best_name, best = ranked[0]
    checks = [
        ("最优口径扣成本后超额 > 5pp", best["excess"] > 0.05,
         f"{best_name} {best['excess']*100:+.1f}pp"),
        ("最优口径分季度跑赢过半",
         n_beat.get(best_name, 0) > len(qres) / 2,
         f"{n_beat.get(best_name, 0)}/{len(qres)}"),
        ("夏普高于市场", best["sharpe"] > bench["sharpe"],
         f"{best['sharpe']:.2f} vs {bench['sharpe']:.2f}"),
        ("换手可接受（<100倍/年）", best["turnover"] < 100,
         f"{best['turnover']:.1f}倍/年"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "有可交易价值" if all_ok else "不成立"
    print(f"\n  结论: {verdict}")
    print(f"\n  注意：这是 259 天测试期、1 个市场环境的结果。")
    print(f"  即使全部 PASS，也需要更长时间样本外验证才能用于实盘。")

    out = {"horizon": args.horizon, "n_train": len(train),
           "n_test": len(test), "cost": args.cost,
           "benchmark": bench, "variants": rows,
           "top2": top2, "quarterly": qres, "n_beat": n_beat,
           "checks": [{"name": a_, "ok": bool(o), "detail": d_}
                      for a_, o, d_ in checks],
           "verdict": verdict}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
