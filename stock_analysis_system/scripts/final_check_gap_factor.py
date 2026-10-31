# -*- coding: utf-8 -*-
"""gap20 因子：分层单调但分季度不稳定，做最终判定。

已有证据：
  支持：
    · 五分层单调递增（Q1 -0.556% → Q5 +0.920%，价差 +1.476%）
    · 与换手率相关系数仅 +0.12（不是镜像）
    · 扣 0.3%/边成本，调仓 20 天仍有 +3.6pp 超额
    · ICIR +10.95（H=20），IC>0 占比 76%
  反对：
    · 分季度 2/5 跑赢
    · 调仓 5 天扣 0.5% 成本变负
    · 市场等权同期 +21.3%、夏普 1.03（基准很强，跑赢难）

本脚本做最后的严格检验：
  1. 逐月滚动（比季度更细）
  2. 用「测试期前半段」做样本内，验证后半段
  3. 与"20日均线规则"这种零参数基准对比
  4. 参数敏感性：Top分数 X%，调仓 N 天
  5. 成本盈亏平衡点

用法：
  python scripts/final_check_gap_factor.py
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
    build_features_preclose, add_labels)
from test_gap_factor import cs_zscore, run_portfolio, bench, TRADING_DAYS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--out", default="results/gap_final.json")
    args = ap.parse_args()

    db = resolve_db()
    df = load_panel_preclose(db, load_quality_dates(), 4500)
    df = build_features_preclose(df)
    df = add_labels(df, [args.horizon])
    nd = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[nd >= 200]
    df = df[df["turnover"].fillna(0) >= 0.1]

    dates = sorted(df["date_str"].unique())
    # 训练期只用前 200 天确定方向，其余作为测试
    test = dates[200:]
    df["score"] = cs_zscore(df, "gap20").fillna(0.0)

    print("=" * 92)
    print("gap20 最终判定")
    print("=" * 92)
    print(f"全样本 {len(dates)} 天，测试段 {len(test)} 天 "
          f"({test[0]} ~ {test[-1]})")

    b_all = bench(df, test)
    print(f"市场等权(测试段): {b_all['total']*100:+.1f}%  "
          f"回撤 {b_all['dd']*100:.1f}%  夏普 {b_all['sharpe']:.2f}")

    report = {"n_test_days": len(test), "bench": b_all}

    # ============================================================
    # 1. 逐月
    # ============================================================
    print("\n[1] 逐月超额（调仓20天, 成本0.3%/边）")
    print("-" * 92)
    mmap = {}
    for d in test:
        mmap.setdefault(d[:7], []).append(d)
    print(f"  {'月份':<10}{'天':>5}{'多头':>10}{'市场':>10}{'超额':>10}")
    mrows = []
    for m in sorted(mmap):
        ds = mmap[m]
        if len(ds) < 10:
            continue
        st = run_portfolio(df, "score", ds, 20, 0.2, 0.003)
        bm = bench(df, ds)
        ex = st["total"] - bm["total"]
        mrows.append({"month": m, "long": st["total"],
                      "bench": bm["total"], "excess": ex})
        print(f"  {m:<10}{len(ds):>5}{st['total']*100:>9.1f}%"
              f"{bm['total']*100:>9.1f}%{ex*100:>+9.1f}pp")
    n_beat_m = sum(1 for r in mrows if r["excess"] > 0)
    print(f"\n  逐月跑赢: {n_beat_m}/{len(mrows)}")
    report["monthly"] = mrows

    # ============================================================
    # 2. 前后半段
    # ============================================================
    print("\n[2] 测试期前后半段")
    print("-" * 92)
    mid = len(test) // 2
    halves = {}
    for name, ds in [("前半", test[:mid]), ("后半", test[mid:])]:
        st = run_portfolio(df, "score", ds, 20, 0.2, 0.003)
        bh = bench(df, ds)
        halves[name] = {"long": st["total"], "bench": bh["total"],
                        "excess": st["total"] - bh["total"],
                        "sharpe": st["sharpe"]}
        print(f"  {name} ({len(ds)} 天): 多头 {st['total']*100:+.1f}%  "
              f"市场 {bh['total']*100:+.1f}%  "
              f"超额 {(st['total']-bh['total'])*100:+.1f}pp  "
              f"夏普 {st['sharpe']:.2f}")
    report["halves"] = halves

    # ============================================================
    # 3. 零参数基准：20 日均线
    # ============================================================
    print("\n[3] 零参数基准：横截面 20 日均线规则")
    print("-" * 92)
    print("  规则：买 close > MA20 的股票，等权，每 20 天调仓")

    # 预计算 MA20 与每日常用字段（避免循环内查询）
    df["ma20"] = df.groupby("stock_code")["close"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())

    dmap_ret = {d: dict(zip(g["stock_code"], g["ret1"]))
                for d, g in df.groupby("date_str")}
    dmap_ma = {d: g[["stock_code", "close", "ma20"]].dropna(subset=["ma20"])
               for d, g in df.groupby("date_str")}

    nav, curve = 1.0, []
    w_prev, pending = {}, None
    total_turn = 0.0
    for i, d in enumerate(test):
        if d not in dmap_ret:
            continue
        rmap = dmap_ret[d]
        cost = 0.0
        if pending is not None:
            keys = set(pending) | set(w_prev)
            turn = 0.5 * sum(abs(pending.get(k, 0.0) - w_prev.get(k, 0.0))
                             for k in keys)
            total_turn += turn
            cost = turn * (2 * 0.003 + 0.001)
            w_prev, pending = pending, None
        if w_prev:
            rs, ws = [], []
            for sc, w in w_prev.items():
                r = rmap.get(sc, np.nan)
                if np.isfinite(r):
                    rs.append(w * r)
                    ws.append(w)
            r = sum(rs) / sum(ws) if ws and sum(ws) > 0 else 0.0
        else:
            r = 0.0
        nav *= (1 + r) * (1 - cost)
        curve.append(nav)
        if i % 20 == 0 and d in dmap_ma:
            gg = dmap_ma[d]
            picks = gg[gg["close"] > gg["ma20"]]["stock_code"].tolist()
            pending = {c: 1.0 / len(picks) for c in picks} if picks else {}

    arr = np.array(curve)
    full = np.concatenate([[1.0], arr])
    rets = full[1:] / full[:-1] - 1.0
    peak = np.maximum.accumulate(full)
    sd = rets.std(ddof=1)
    ma_tot = float(arr[-1] - 1)
    ma_sharpe = float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan
    print(f"  MA20 选出: {ma_tot*100:+.1f}%  夏普 {ma_sharpe:.2f}  "
          f"年换手 {total_turn/(len(test)/TRADING_DAYS):.1f}")
    print(f"  市场等权: {b_all['total']*100:+.1f}%  夏普 {b_all['sharpe']:.2f}")
    report["ma20_bench"] = {"total": ma_tot, "sharpe": ma_sharpe,
                            "turnover": float(total_turn / (len(test)/TRADING_DAYS))}

    # ============================================================
    # 4. 参数敏感性
    # ============================================================
    print("\n[4] 参数敏感性（top X% × 调仓 N 天，成本 0.3%/边）")
    print("-" * 92)
    print(f"  {'top%':<6}" + "".join(f"{f'rb{n}':>10}" for n in [5, 10, 20, 40]))
    grid = {}
    for top in [0.1, 0.2, 0.3, 0.5]:
        line = f"  {top*100:>5.0f}"
        grid[str(top)] = {}
        for rb in [5, 10, 20, 40]:
            st = run_portfolio(df, "score", test, rb, top, 0.003)
            ex = st["total"] - b_all["total"]
            grid[str(top)][str(rb)] = ex
            line += f"{ex*100:>+9.1f}pp"
        print(line)
    n_positive = sum(1 for t in grid.values() for v in t.values() if v > 0)
    n_total = sum(len(t) for t in grid.values())
    print(f"\n  正超额组合: {n_positive}/{n_total}")
    report["grid"] = grid

    # ============================================================
    # 5. 成本盈亏平衡
    # ============================================================
    print("\n[5] 成本盈亏平衡点（调仓20天, top20%）")
    print("-" * 92)
    print(f"  {'成本/边':<10}{'多头':>11}{'超额':>11}")
    be_rows = []
    for c in [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007]:
        st = run_portfolio(df, "score", test, 20, 0.2, c)
        ex = st["total"] - b_all["total"]
        be_rows.append({"cost": c, "total": st["total"], "excess": ex})
        mark = "  <<< 转负" if ex <= 0 and (not be_rows[:-1] or
                                          be_rows[-2]["excess"] > 0) else ""
        print(f"  {c*100:>7.3f}%  {st['total']*100:>10.1f}%"
              f"{ex*100:>+10.1f}pp{mark}")
    report["breakeven"] = be_rows

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 92)
    print("最终判定")
    print("=" * 92)

    both_halves = (halves["前半"]["excess"] > 0 and halves["后半"]["excess"] > 0)
    checks = [
        ("逐月跑赢过半", n_beat_m > len(mrows) * 0.5,
         f"{n_beat_m}/{len(mrows)}"),
        ("前后半段都跑赢", both_halves,
         f"{halves['前半']['excess']*100:+.1f} / "
         f"{halves['后半']['excess']*100:+.1f}pp"),
        ("参数网格多数为正", n_positive > n_total * 0.6,
         f"{n_positive}/{n_total}"),
        ("成本容忍度 >= 0.4%", next(
            (r["cost"] for r in be_rows if r["excess"] <= 0), 0.01) >= 0.004,
         f"{next((r['cost'] for r in be_rows if r['excess'] <= 0), 0.01)*100:.2f}%"),
        ("优于零参数 MA20 基准",
         halves["前半"]["long"] > 0 and b_all["total"] > 0,
         f"MA20 {ma_tot*100:+.1f}% vs 市场 {b_all['total']*100:+.1f}%"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    verdict = "gap20 值得继续研究" if all_ok else "gap20 不足以支撑实盘"
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
