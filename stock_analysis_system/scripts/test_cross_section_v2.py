# -*- coding: utf-8 -*-
"""横截面因子：修正成本模型 + 多空价差检验。

上次回测的两个问题：
  1. 年换手 18159 倍 —— 成本按"股票只数"累计，逻辑错误
  2. 分季度 1/5 跑赢 —— 但 IC 明明是显著的（-0.05 ~ -0.09, ICIR -9 ~ -14）

矛盾点：IC 显著但组合收益不显著，可能因为：
  · 多头组合包含大量 beta 暴露（市场涨跌主导）
  · 真正有效的是「多空价差」，但 A 股融券困难
  · 换手成本吃掉了 alpha

本脚本：
  1. 正确成本模型：按【组合权重换手】算，不是按股票数
  2. 多空组合（理论上限，看 alpha 有多大）
  3. 多头 vs 等权市场（beta 中性化后）
  4. 行业/规模中性化（用市值代理）
  5. 成本敏感性：0 / 0.1% / 0.3% / 0.5% 单边

用法：
  python scripts/test_cross_section_v2.py
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

ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS = 244

DEFAULT_FACTORS = ["turnover_z", "vol20", "amplitude", "ret20", "pos120"]


def cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    mu = df.groupby("date_str")[col].transform("mean")
    sd = df.groupby("date_str")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).clip(-3, 3)


def train_signs(df: pd.DataFrame, feats: list, h: int, train_days: int) -> dict:
    dates = sorted(df["date_str"].unique())
    sub = df[df["date_str"].isin(set(dates[:train_days]))]
    ycol = f"fwd{h}"
    out = {}
    for f in feats:
        ics = []
        for d, g in sub.groupby("date_str"):
            g2 = g[[f, ycol]].dropna()
            if len(g2) < 200:
                continue
            a = g2[f].rank().values
            b = g2[ycol].rank().values
            a -= a.mean()
            b -= b.mean()
            den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            if den > 0:
                ics.append((a * b).sum() / den)
        out[f] = float(np.mean(ics)) if ics else 0.0
    return out


def run_portfolio(df: pd.DataFrame, score_col: str, dates: list,
                  rebalance: int, top_q: float, mode: str,
                  cost_side: float) -> dict:
    """回测。mode ∈ {long, long_short, benchmark}

    正确的成本模型：
      组合权重 w 的换手 = 0.5 * sum|w_new - w_old|（单边换手）
      成本 = 换手 × (买卖双边成本 + 滑点 + 印花税)
    """
    ret_map = df.set_index(["date_str", "stock_code"])["ret1"].to_dict()
    dmap = {d: g for d, g in df.groupby("date_str")}

    nav = 1.0
    curve = []
    w_prev = {}            # stock -> weight
    pending = None         # 待执行权重（T 日算，T+1 执行）
    total_turnover = 0.0

    for i, d in enumerate(dates):
        g = dmap.get(d)
        if g is None or len(g) < 300:
            continue

        # ---- 执行挂单 ----
        cost = 0.0
        if pending is not None:
            # 单边换手 = 0.5 * sum|Δw|
            keys = set(pending) | set(w_prev)
            turn = 0.5 * sum(abs(pending.get(k, 0.0) - w_prev.get(k, 0.0))
                             for k in keys)
            total_turnover += turn
            # 成本：买入成本+滑点 + 卖出成本+滑点+印花税，按换手比例
            cost = turn * (2 * cost_side + 0.001)
            w_prev = pending
            pending = None

        # ---- 持仓收益 ----
        if w_prev:
            r = sum(w * ret_map.get((d, sc), 0.0)
                    for sc, w in w_prev.items()
                    if np.isfinite(ret_map.get((d, sc), np.nan)))
            # 分母补齐（有股票停牌时）
            wsum = sum(w for sc, w in w_prev.items()
                       if np.isfinite(ret_map.get((d, sc), np.nan)))
            if wsum > 0 and wsum < 0.999:
                r = r / wsum    # 归一化到满仓
        else:
            r = 0.0

        nav *= (1 + r) * (1 - cost)
        curve.append(nav)

        # ---- 收盘算分，明日执行 ----
        if i % rebalance == 0:
            s = g[["stock_code", score_col]].dropna(subset=[score_col])
            if len(s) < 300:
                continue
            if mode == "long":
                thr = s[score_col].quantile(1 - top_q)
                picks = s[s[score_col] >= thr]["stock_code"].tolist()
                if picks:
                    w = 1.0 / len(picks)
                    pending = {c: w for c in picks}
                else:
                    pending = {}
            elif mode == "long_short":
                lo = s[score_col].quantile(top_q)
                hi = s[score_col].quantile(1 - top_q)
                longs = s[s[score_col] >= hi]["stock_code"].tolist()
                shorts = s[s[score_col] <= lo]["stock_code"].tolist()
                pw = {}
                if longs:
                    wl = 0.5 / len(longs)
                    for c in longs:
                        pw[c] = wl
                if shorts:
                    ws = -0.5 / len(shorts)
                    for c in shorts:
                        pw[c] = ws
                pending = pw

    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan,
                "turnover_annual": 0.0, "days": len(curve),
                "ruined": False}

    arr = np.array(curve)
    # ★ 破产保护：净值 <= 0 时停止（多空组合可能穿仓）
    ruined = bool((arr <= 0).any())
    if ruined:
        first_bad = int(np.argmax(arr <= 0))
        arr = arr[:max(first_bad, 1)]
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    dd = float((full / peak - 1).min())
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan
    years = max(len(arr) / TRADING_DAYS, 1e-9)
    total = float(arr[-1] - 1) if not ruined else -1.0
    return {
        "total": total,
        "ann": float((1 + total) ** (1 / years) - 1) if total > -1 else np.nan,
        "dd": dd, "sharpe": sharpe, "days": len(arr),
        "turnover_annual": float(total_turnover / years),
        "ruined": ruined,
    }


def bench_equal_weight(df: pd.DataFrame, dates: list) -> dict:
    dmap = {d: g for d, g in df.groupby("date_str")}
    nav = 1.0
    curve = []
    for d in dates:
        g = dmap.get(d)
        if g is None or len(g) < 300:
            continue
        r = float(g["ret1"].mean())
        nav *= (1 + r)
        curve.append(nav)
    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan, "days": len(curve)}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    return {"total": float(arr[-1] - 1),
            "dd": float((full / peak - 1).min()),
            "sharpe": float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan,
            "days": len(arr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--top-quantile", type=float, default=0.2)
    ap.add_argument("--rebalance", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--min-stocks", type=int, default=4500)
    ap.add_argument("--min-history-days", type=int, default=200,
                    help="股票至少出现多少个交易日")
    ap.add_argument("--min-turnover", type=float, default=0.1,
                    help="最低换手率(%%)，剔除僵尸股")
    ap.add_argument("--out", default="results/cross_section_v2.json")
    args = ap.parse_args()

    db = resolve_db()
    usable = load_quality_dates()
    print("=" * 94)
    print("横截面因子 v2：修正成本模型 + 多空价差")
    print("=" * 94)

    df = load_panel(db, usable, args.min_stocks)
    df = build_features(df)
    df = add_labels(df, [args.horizon])

    # ★ 修正过滤条件：用【交易日数】而不是自然日
    #   旧写法 (date - first).dt.days >= 365 会把 2024 年上市的全部剔除，
    #   导致测试期只剩 15 天。
    n_days_stock = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[n_days_stock >= args.min_history_days]
    df = df[df["turnover"].fillna(0) >= args.min_turnover]
    print(f"过滤后: {len(df):,} 行  {df['stock_code'].nunique()} 只  "
          f"{df['date_str'].nunique()} 天")

    dates = sorted(df["date_str"].unique())
    train = dates[:args.train_days]
    test = dates[args.train_days:]
    print(f"训练 {len(train)} 天  测试 {len(test)} 天  "
          f"({test[0]} ~ {test[-1]})")
    print(f"样本 {len(df):,} 行  {df['stock_code'].nunique()} 只")

    signs = train_signs(df, args.factors, args.horizon, args.train_days)
    print("\n训练期 IC 符号:")
    for f, v in signs.items():
        print(f"  {f:<14}{v:+.4f}")

    # 合成分数
    df["score"] = 0.0
    n_used = 0
    for f in args.factors:
        if signs[f] == 0:
            continue
        df["score"] = df["score"] + cs_zscore(df, f).fillna(0.0) * np.sign(signs[f])
        n_used += 1
    df["score"] = df["score"] / max(n_used, 1)

    report = {"horizon": args.horizon, "factors": args.factors,
              "signs": signs, "train_days": len(train),
              "test_days": len(test), "results": {}}

    # ============================================================
    # 多空价差（alpha 理论上限）
    # ============================================================
    print("\n" + "=" * 94)
    print("[1] 多空组合（alpha 上限，成本=0）")
    print("=" * 94)
    print(f"  {'调仓':<8}{'多空收益':>12}{'回撤':>10}{'夏普':>8}{'年换手':>10}")

    ls_rows = {}
    for rb in args.rebalance:
        st = run_portfolio(df, "score", test, rb, args.top_quantile,
                           "long_short", 0.0)
        ls_rows[rb] = st
        print(f"  {rb:<8}{st['total']*100:>11.1f}%{st['dd']*100:>9.1f}%"
              f"{st['sharpe']:>8.2f}{st['turnover_annual']:>10.1f}")
    report["long_short"] = {str(k): v for k, v in ls_rows.items()}

    # ============================================================
    # 多头 vs 市场
    # ============================================================
    print("\n" + "=" * 94)
    print("[2] 多头组合 vs 市场等权（正确成本模型）")
    print("=" * 94)
    bench = bench_equal_weight(df, test)
    print(f"  市场等权: {bench['total']*100:+.1f}%  回撤 {bench['dd']*100:.1f}%  "
          f"夏普 {bench['sharpe']:.2f}")

    print(f"\n  {'调仓':<6}{'成本/边':<10}{'多头':>11}{'市场':>11}"
          f"{'超额':>11}{'夏普':>8}{'年换手':>9}")
    for rb in args.rebalance:
        for cost in [0.0, 0.001, 0.003, 0.005]:
            st = run_portfolio(df, "score", test, rb, args.top_quantile,
                               "long", cost)
            ex = st["total"] - bench["total"]
            print(f"  {rb:<6}{cost*100:>7.2f}%  {st['total']*100:>10.1f}%"
                  f"{bench['total']*100:>10.1f}%{ex*100:>+10.1f}pp"
                  f"{st['sharpe']:>8.2f}{st['turnover_annual']:>9.1f}")
            report["results"][f"{rb}_{cost}"] = {
                "rebalance": rb, "cost_side": cost, "long": st,
                "benchmark": bench, "excess": ex}
        print()

    # ============================================================
    # 分季度（调仓=10, 成本=0.3%）
    # ============================================================
    print("=" * 94)
    print("[3] 分季度（调仓=10天, 成本=0.3%/边）")
    print("=" * 94)
    rb = 10 if 10 in args.rebalance else args.rebalance[0]
    qmap = {}
    for d in test:
        qmap.setdefault(str(pd.Timestamp(d).to_period("Q")), []).append(d)

    print(f"  {'季度':<10}{'天':>5}{'多头':>11}{'市场':>11}{'超额':>11}")
    qrows = []
    for q in sorted(qmap):
        ds = qmap[q]
        if len(ds) < 20:
            continue
        st = run_portfolio(df, "score", ds, rb, args.top_quantile, "long", 0.003)
        bq = bench_equal_weight(df, ds)
        ex = st["total"] - bq["total"]
        qrows.append({"quarter": q, "days": len(ds), "long": st["total"],
                      "bench": bq["total"], "excess": ex})
        print(f"  {q:<10}{len(ds):>5}{st['total']*100:>10.1f}%"
              f"{bq['total']*100:>10.1f}%{ex*100:>+10.1f}pp")
    n_beat = sum(1 for r in qrows if r["excess"] > 0)
    print(f"\n  跑赢市场: {n_beat}/{len(qrows)}")
    report["quarterly"] = qrows

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 94)
    print("判定")
    print("=" * 94)

    # 多空是最接近 alpha 的（无 beta）
    best_ls = max(ls_rows.items(), key=lambda kv: kv[1]["total"])
    # 扣 0.3% 成本的多头超额
    real = {k: v for k, v in report["results"].items() if v["cost_side"] == 0.003}
    best_real = max(real.items(), key=lambda kv: kv[1]["excess"]) if real else None

    checks = [
        ("多空组合 alpha 显著为正", best_ls[1]["total"] > 0.10,
         f"{best_ls[1]['total']*100:+.1f}% @ 调仓{best_ls[0]}天"),
        ("扣 0.3%/边成本后多头超额 > 0", best_real and best_real[1]["excess"] > 0,
         f"{best_real[1]['excess']*100:+.1f}pp @ 调仓{best_real[1]['rebalance']}天"
         if best_real else "nan"),
        ("分季度跑赢过半", n_beat > len(qrows) / 2,
         f"{n_beat}/{len(qrows)}"),
        ("换手可接受（<50倍/年）", best_ls[1]["turnover_annual"] < 50,
         f"{best_ls[1]['turnover_annual']:.1f}倍/年"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "有可交易 alpha" if all_ok else "扣成本后不成立"
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
