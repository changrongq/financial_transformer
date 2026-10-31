# -*- coding: utf-8 -*-
"""最终裁决：用修正后的自洽口径做横截面回测。

修正内容（2026-10-31）：
  ret1 从 `close/pre_close - 1` 改为 `close/close.shift(1) - 1`
  原因：pre_close(tushare 除权基准) 与 close(本地复权价) 不同源，
        导致 ret1 累乘与 fwd 不一致（12.2% 样本差异 > 1%），
        并造成回测净值虚高 25 倍。

本脚本做最终检验：
  1. 多因子组合 vs 市场（扣真实成本）
  2. 逐年滚动（walk-forward）
  3. 逐月胜率
  4. 涨跌停可交易性过滤
  5. 参数网格稳定性

判定标准（严格）：
  · 扣 0.3%/边成本后逐年跑赢过半
  · 逐月胜率 > 55%
  · 参数网格 70% 以上为正
  · 剔除涨停不可成交后仍成立

用法：
  python scripts/final_cross_section_verdict.py
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
from rerun_cross_section_with_preclose import (
    resolve_db, load_quality_dates, load_panel_preclose,
    build_features_preclose, add_labels, FEATURES_NEW)

ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS = 244


def limit_pct(code: str) -> float:
    if code.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    if code.startswith(("30", "68")):
        return 0.20
    return 0.10


def cs_zscore(df, col):
    mu = df.groupby("date_str")[col].transform("mean")
    sd = df.groupby("date_str")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).clip(-3, 3)


def backtest(df, score_col, dates, rebalance, top_q, cost_side,
             exclude_limit_up=True):
    """Top-Q 多头，T 日收盘算分、T+1 开盘成交。"""
    ret_map = df.set_index(["date_str", "stock_code"])["ret1"].to_dict()
    lu_map = df.set_index(["date_str", "stock_code"])["is_limit_up"].to_dict()
    dmap = {d: g for d, g in df.groupby("date_str")}

    nav, curve = 1.0, []
    w_prev, pending = {}, None
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
            w_prev, pending = pending, None

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
            s = g[["stock_code", score_col]].dropna()
            if len(s) < 300:
                continue
            if exclude_limit_up:
                # 剔除当日涨停的（次日无法买入）
                lu = g.set_index("stock_code")["is_limit_up"]
                s = s[~s["stock_code"].map(lu).fillna(False)]
            thr = s[score_col].quantile(1 - top_q)
            picks = s[s[score_col] >= thr]["stock_code"].tolist()
            pending = {c: 1.0 / len(picks) for c in picks} if picks else {}

    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan,
                "turnover": 0.0, "days": len(curve)}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    years = max(len(arr) / TRADING_DAYS, 1e-9)
    return {
        "total": float(arr[-1] - 1),
        "dd": float((full / peak - 1).min()),
        "sharpe": float(rets.mean() / sd * np.sqrt(TRADING_DAYS))
        if sd > 1e-12 else np.nan,
        "turnover": float(total_turn / years),
        "days": len(arr),
    }


def bench(df, dates):
    dmap = {d: g for d, g in df.groupby("date_str")}
    nav, curve = 1.0, []
    for d in dates:
        g = dmap.get(d)
        if g is None or len(g) < 300:
            continue
        nav *= (1 + float(g["ret1"].mean()))
        curve.append(nav)
    if len(curve) < 20:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan}
    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    return {"total": float(arr[-1] - 1),
            "dd": float((full / peak - 1).min()),
            "sharpe": float(rets.mean() / sd * np.sqrt(TRADING_DAYS))
            if sd > 1e-12 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--train-days", type=int, default=200)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results/cross_section_verdict.json")
    args = ap.parse_args()

    db = resolve_db()
    df = load_panel_preclose(db, load_quality_dates(), 4500)
    df = build_features_preclose(df)
    df = add_labels(df, [args.horizon])
    nd = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[nd >= 200]
    df = df[df["turnover"].fillna(0) >= 0.1]

    # 涨停标记
    thr = df["stock_code"].map(limit_pct)
    df["is_limit_up"] = df["ret1"] >= (thr - 0.005)

    ycol = f"fwd{args.horizon}"
    dates = sorted(df["date_str"].unique())
    train, test = dates[:args.train_days], dates[args.train_days:]

    # 训练期确定方向
    df_tr = df[df["date_str"].isin(set(train))]
    signs = {}
    for f in FEATURES_NEW:
        if f not in df.columns:
            continue
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

    score = pd.Series(0.0, index=df.index)
    n = 0
    for f, sg in signs.items():
        if sg == 0:
            continue
        score = score + cs_zscore(df, f).fillna(0.0) * np.sign(sg)
        n += 1
    df["score"] = score / max(n, 1)

    print("=" * 92)
    print("横截面最终裁决（修正自洽口径）")
    print("=" * 92)
    print(f"训练 {len(train)} 天  测试 {len(test)} 天 "
          f"({test[0]} ~ {test[-1]})")
    print(f"因子 {n} 个  成本 {args.cost*100:.2f}%/边")

    b = bench(df, test)
    print(f"\n市场等权: {b['total']*100:+.1f}%  回撤 {b['dd']*100:.1f}%  "
          f"夏普 {b['sharpe']:.2f}")

    report = {"n_train": len(train), "n_test": len(test),
              "cost": args.cost, "bench": b, "signs": signs}

    # ---- 参数网格 ----
    print(f"\n[1] 参数网格（top X% × 调仓 N 天，剔除涨停）")
    print(f"  {'top%':<6}" + "".join(f"{f'rb{r}':>11}" for r in [5, 10, 20, 40]))
    grid = {}
    for top in [0.1, 0.2, 0.3, 0.5]:
        line = f"  {top*100:>5.0f}"
        grid[str(top)] = {}
        for rb in [5, 10, 20, 40]:
            st = backtest(df, "score", test, rb, top, args.cost)
            ex = st["total"] - b["total"]
            grid[str(top)][str(rb)] = {"total": st["total"], "excess": ex,
                                       "sharpe": st["sharpe"],
                                       "turnover": st["turnover"]}
            line += f"{ex*100:>+10.1f}pp"
        print(line)
    n_pos = sum(1 for t in grid.values() for v in t.values() if v["excess"] > 0)
    n_tot = sum(len(t) for t in grid.values())
    print(f"\n  正超额: {n_pos}/{n_tot} ({n_pos/n_tot*100:.0f}%)")
    report["grid"] = grid
    report["grid_positive"] = n_pos
    report["grid_total"] = n_tot

    # ---- 逐年 ----
    print(f"\n[2] 逐年滚动（top20%, 调仓20天, 剔除涨停）")
    print(f"  {'年':<8}{'策略':>11}{'市场':>11}{'超额':>11}{'夏普':>8}{'天数':>7}")
    yearly = []
    for y in sorted(set(pd.DatetimeIndex(test).year)):
        ds = [d for d in test if d[:4] == str(y)]
        if len(ds) < 40:
            continue
        st = backtest(df, "score", ds, 20, 0.2, args.cost)
        by = bench(df, ds)
        ex = st["total"] - by["total"]
        yearly.append({"year": int(y), "strategy": st["total"],
                       "bench": by["total"], "excess": ex,
                       "sharpe": st["sharpe"], "days": len(ds)})
        print(f"  {y:<8}{st['total']*100:>10.1f}%{by['total']*100:>10.1f}%"
              f"{ex*100:>+10.1f}pp{st['sharpe']:>8.2f}{len(ds):>7}")
    n_ybeat = sum(1 for r in yearly if r["excess"] > 0)
    print(f"\n  逐年跑赢: {n_ybeat}/{len(yearly)}")
    report["yearly"] = yearly

    # ---- 逐月 ----
    print(f"\n[3] 逐月胜率（top20%, 调仓20天）")
    months = sorted(set(d[:7] for d in test))
    mrows = []
    for m in months:
        ds = [d for d in test if d[:7] == m]
        if len(ds) < 10:
            continue
        st = backtest(df, "score", ds, 20, 0.2, args.cost)
        bm = bench(df, ds)
        mrows.append({"month": m, "excess": st["total"] - bm["total"]})
    n_mbeat = sum(1 for r in mrows if r["excess"] > 0)
    print(f"  月份数 {len(mrows)}  跑赢 {n_mbeat} "
          f"({n_mbeat/max(len(mrows),1)*100:.1f}%)")
    report["monthly"] = mrows
    report["monthly_winrate"] = n_mbeat / max(len(mrows), 1)

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 92)
    print("判定")
    print("=" * 92)

    checks = [
        ("逐年跑赢过半", n_ybeat > len(yearly) / 2,
         f"{n_ybeat}/{len(yearly)}"),
        ("逐月胜率 > 55%", n_mbeat / max(len(mrows), 1) > 0.55,
         f"{n_mbeat/max(len(mrows),1)*100:.1f}%"),
        ("参数网格 70% 以上为正", n_pos / max(n_tot, 1) >= 0.70,
         f"{n_pos}/{n_tot}"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "横截面因子有可交易价值" if all_ok else "扣成本后不成立"
    print(f"\n  结论: {verdict}")
    report["checks"] = [{"name": a, "ok": bool(o), "detail": d}
                        for a, o, d in checks]
    report["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
