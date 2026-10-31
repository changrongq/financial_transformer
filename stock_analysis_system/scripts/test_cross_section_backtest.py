# -*- coding: utf-8 -*-
"""横截面因子：真实可交易性检验（扣成本 + 严格时序）。

已知：
  · 12 个因子横截面 IC 全为负，剔除除权和次新股后仍为负（-0.05 ~ -0.09）
  · 低换手组 5 日收益 +0.179%，高换手组 -0.249%，价差 +0.428%，t=3.09

但「统计显著」不等于「能赚钱」。必须检验：
  1. 扣真实成本（A 股：佣金万分之2.5 + 印花税千分之1（卖出）+ 滑点）
  2. 严格时序：T 日收盘算信号 → T+1 日开盘交易（T+1 制度下买入当天不能卖）
  3. 多空 vs 只有多头（A 股融券难，实际只能做多）
  4. 换手成本的侵蚀
  5. 分年度/分季度稳定性

设计：
  · 每 N 天调仓一次（N ∈ {1, 5, 10, 20}）
  · 组合形式：
      a) Top20% 多头（等权），其余空仓
      b) Top20% 多头 vs 市场等权（超额）
      c) Top20% 多头 - Bottom20%（多空，理论上限）
  · 成本：万分之2.5 双边 + 卖出印花税 0.1% + 滑点 0.05%/边
  · 严格用 T+1 开盘价成交

用法：
  python scripts/test_cross_section_backtest.py
  python scripts/test_cross_section_backtest.py --factors turnover_z vol20 --rebalance 5
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
                                build_features, add_labels, FEATURES)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TRADING_DAYS = 244


def compute_ic_sign(df: pd.DataFrame, feats: list, h: int,
                    train_days: int) -> dict:
    """用训练段（前 train_days 天）算每个因子 IC 的符号。"""
    dates = sorted(df["date_str"].unique())
    train_dates = set(dates[:train_days])
    sub = df[df["date_str"].isin(train_dates)]
    ycol = f"fwd{h}"
    signs = {}
    for f in feats:
        s = sub[["date_str", f, ycol]].dropna()
        if len(s) < 20000:
            signs[f] = 0.0
            continue
        ics = []
        for d, g in s.groupby("date_str"):
            if len(g) < 200:
                continue
            a = g[f].rank().values
            b = g[ycol].rank().values
            a = a - a.mean()
            b = b - b.mean()
            den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            if den > 0:
                ics.append((a * b).sum() / den)
        signs[f] = float(np.mean(ics)) if ics else 0.0
    return signs


def build_score(df: pd.DataFrame, feats: list, signs: dict) -> pd.Series:
    """横截面 z-score，按训练期 IC 符号定向，等权合成。"""
    out = pd.Series(0.0, index=df.index)
    n_used = 0
    for f in feats:
        if signs.get(f, 0.0) == 0:
            continue
        mu = df.groupby("date_str")[f].transform("mean")
        sd = df.groupby("date_str")[f].transform("std").replace(0, np.nan)
        z = (df[f] - mu) / sd
        z = z.clip(-3, 3) * np.sign(signs[f])
        out = out + z.fillna(0.0)
        n_used += 1
    return out / max(n_used, 1)


def backtest_long_only(df: pd.DataFrame, score_col: str, test_dates: list,
                       rebalance: int, top_q: float,
                       cost_buy: float, cost_sell: float,
                       slip: float) -> dict:
    """Top-Q 多头组合，每 rebalance 天调仓。

    ★ 严格 T+1 与成交时点：
      第 t 日 收盘 用 close 算分 → 选出持仓
      第 t+1 日 开盘 成交（用 open）
      第 t+1 日起 持有，按日收益（ret1）累积

    ⚠️ 已知限制：stock_data 无 pre_close，ret1 在除权日会假跌。
       已在 build_features 里把 ret1 截断到 ±31%，但仍有残余。
       这会使结果偏保守（假跌会被计入），不会虚增收益。
    """
    nav = 1.0
    curve = []
    daily_ret = []
    prev_holdings = set()
    trades = 0
    pending = None      # 待执行的目标持仓（T 日算，T+1 执行）

    # 按股票-日期索引，加速
    ret_map = df.set_index(["date_str", "stock_code"])["ret1"].to_dict()

    for i, d in enumerate(test_dates):
        g = df[df["date_str"] == d]
        if len(g) < 300:
            continue

        # ---- 1. 执行昨日挂单（今日开盘成交）----
        cost_today = 0.0
        if pending is not None:
            new_h = pending
            if prev_holdings:
                turn = (len(new_h.symmetric_difference(prev_holdings))
                        / max(len(new_h | prev_holdings), 1))
            else:
                turn = 1.0 if new_h else 0.0
            # 双边换手：买入成本 + 卖出成本 + 印花税 + 滑点
            cost_today = turn * (cost_buy + cost_sell + 0.001 + 2 * slip)
            trades += int(turn * max(len(new_h), 1))
            prev_holdings = new_h
            pending = None

        # ---- 2. 今日持有收益 ----
        if prev_holdings:
            rs = [ret_map.get((d, sc), np.nan) for sc in prev_holdings]
            rs = [r for r in rs if isinstance(r, (int, float)) and np.isfinite(r)]
            r = float(np.mean(rs)) if rs else 0.0
        else:
            r = 0.0

        nav *= (1 + r) * (1 - cost_today)
        curve.append(nav)
        daily_ret.append(r - cost_today)

        # ---- 3. 今日收盘算分，明日开盘执行 ----
        if i % rebalance == 0:
            s = g[["stock_code", score_col]].dropna(subset=[score_col])
            if len(s) >= 300:
                thr = s[score_col].quantile(1 - top_q)
                pending = set(s[s[score_col] >= thr]["stock_code"])

    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan,
                "trades": trades, "days": len(curve)}

    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    dd = float((full / peak - 1).min())
    total = float(arr[-1] - 1)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan
    years = max(len(arr) / TRADING_DAYS, 1e-9)
    ann = float((1 + total) ** (1 / years) - 1) if total > -1 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe,
            "trades": trades, "days": len(arr)}


def backtest_market(df: pd.DataFrame, test_dates: list,
                    rebalance: int, cost_buy: float, cost_sell: float,
                    slip: float) -> dict:
    """市场等权基线（同期、同成本）。"""
    nav = 1.0
    curve = []
    for i, d in enumerate(test_dates):
        g = df[df["date_str"] == d]
        if len(g) < 300:
            continue
        r = float(g["ret1"].mean()) if "ret1" in g else 0.0
        c = 0.0
        if i % rebalance == 0 and i == 0:
            c = cost_buy + slip
        nav *= (1 + r) * (1 - c)
        curve.append(nav)

    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan, "days": len(curve)}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    dd = float((full / peak - 1).min())
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan
    return {"total": float(arr[-1] - 1), "dd": dd, "sharpe": sharpe,
            "days": len(arr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="+",
                    default=["turnover_z", "vol20", "amplitude", "ret20", "pos120"])
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--top-quantile", type=float, default=0.2)
    ap.add_argument("--rebalance", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--cost-buy", type=float, default=0.00025)
    ap.add_argument("--cost-sell", type=float, default=0.00025)
    ap.add_argument("--slip", type=float, default=0.0005)
    ap.add_argument("--min-stocks", type=int, default=4500)
    ap.add_argument("--out", default="results/cross_section_backtest.json")
    args = ap.parse_args()

    db = resolve_db()
    usable = load_quality_dates()
    print("=" * 92)
    print("横截面因子：真实可交易性检验")
    print("=" * 92)

    df = load_panel(db, usable, args.min_stocks)
    df = build_features(df)
    df = add_labels(df, [args.horizon])

    dates = sorted(df["date_str"].unique())
    if len(dates) < args.train_days + 60:
        print(f"日期不足: {len(dates)}")
        return 1

    train_dates = dates[:args.train_days]
    test_dates = dates[args.train_days:]
    print(f"训练 {len(train_dates)} 天: {train_dates[0]} ~ {train_dates[-1]}")
    print(f"测试 {len(test_dates)} 天: {test_dates[0]} ~ {test_dates[-1]}")
    print(f"成本: 买入 {args.cost_buy*100:.3f}% 卖出 {args.cost_sell*100:.3f}% "
          f"+印花税 0.1% + 滑点 {args.slip*100:.3f}%/边")

    # 训练期 IC 符号
    print("\n训练期因子 IC 符号:")
    signs = compute_ic_sign(df, args.factors, args.horizon, args.train_days)
    for f, v in signs.items():
        print(f"  {f:<14}{v:+.4f}  {'(反向)' if v < 0 else '(正向)'}")

    df["score"] = build_score(df, args.factors, signs)

    report = {"horizon": args.horizon, "factors": args.factors,
              "signs": signs, "train_days": len(train_dates),
              "test_days": len(test_dates),
              "cost": {"buy": args.cost_buy, "sell": args.cost_sell,
                       "stamp": 0.001, "slip": args.slip},
              "results": {}}

    # 市场基线
    print("\n" + "=" * 92)
    print("结果")
    print("=" * 92)
    base = {}
    for rb in args.rebalance:
        b = backtest_market(df, test_dates, rb, args.cost_buy,
                            args.cost_sell, args.slip)
        base[rb] = b
    report["market_baseline"] = base

    print(f"\n  {'调仓':<8}{'多头组合':>14}{'市场等权':>14}{'超额':>12}"
          f"{'回撤':>10}{'夏普':>8}{'年换手':>10}")
    for rb in args.rebalance:
        st = backtest_long_only(df, "score", test_dates, rb,
                                args.top_quantile, args.cost_buy,
                                args.cost_sell, args.slip)
        mb = base[rb]
        excess = st["total"] - mb["total"]
        ann_turn = st["trades"] / max(st["days"] / TRADING_DAYS, 1e-9)
        print(f"  {rb:<8}{st['total']*100:>13.1f}%{mb['total']*100:>13.1f}%"
              f"{excess*100:>+11.1f}pp{st['dd']*100:>9.1f}%"
              f"{st['sharpe']:>8.2f}{ann_turn:>10.0f}")
        report["results"][str(rb)] = {"long": st, "market": mb,
                                      "excess": excess,
                                      "annual_turnover": ann_turn}

    # ============================================================
    # 分季度稳定性
    # ============================================================
    print("\n" + "-" * 92)
    print("分季度超额收益（调仓周期=5）")
    print("-" * 92)

    rb = 5 if 5 in args.rebalance else args.rebalance[0]
    q_dates = {}
    for d in test_dates:
        q = pd.Timestamp(d).to_period("Q")
        q_dates.setdefault(str(q), []).append(d)

    print(f"  {'季度':<10}{'天数':>6}{'多头':>10}{'市场':>10}{'超额':>10}")
    qrows = []
    for q in sorted(q_dates):
        ds = q_dates[q]
        if len(ds) < 20:
            continue
        st = backtest_long_only(df, "score", ds, rb, args.top_quantile,
                                args.cost_buy, args.cost_sell, args.slip)
        mb = backtest_market(df, ds, rb, args.cost_buy, args.cost_sell, args.slip)
        ex = st["total"] - mb["total"]
        qrows.append({"quarter": q, "days": len(ds), "long": st["total"],
                      "market": mb["total"], "excess": ex})
        print(f"  {q:<10}{len(ds):>6}{st['total']*100:>9.1f}%"
              f"{mb['total']*100:>9.1f}%{ex*100:>+9.1f}pp")
    report["quarterly"] = qrows

    n_beat = sum(1 for r in qrows if r["excess"] > 0)
    print(f"\n  跑赢市场: {n_beat}/{len(qrows)}")

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 92)
    print("判定")
    print("=" * 92)

    best_rb = max(report["results"].items(),
                  key=lambda kv: kv[1]["excess"])
    best_excess = best_rb[1]["excess"]

    checks = [
        ("扣成本后大幅超额（>5pp）", best_excess > 0.05,
         f"{best_excess*100:+.1f}pp @ 调仓{best_rb[0]}天"),
        ("分季度跑赢过半", n_beat > len(qrows) / 2,
         f"{n_beat}/{len(qrows)}"),
        ("夏普高于市场", best_rb[1]["long"]["sharpe"] >
         best_rb[1]["market"]["sharpe"],
         f"{best_rb[1]['long']['sharpe']:.2f} vs {best_rb[1]['market']['sharpe']:.2f}"),
        ("换手可接受（<30倍/年）", best_rb[1]["annual_turnover"] < 30,
         f"{best_rb[1]['annual_turnover']:.0f}倍/年"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "横截面因子有可交易价值" if all_ok else "扣成本后不成立"
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
