# -*- coding: utf-8 -*-
"""诊断：横截面 IC 全为负，是信号还是数据偏差？

现象：
  12 个因子在 3 个持有期上 IC **全部为负**
  ICIR 高达 -3 ~ -14（这在真实市场里几乎不可能）
  IC>0 占比只有 26%~42%

可疑点：
  1. stock_data 没有 pre_close，除权日 close 会跳空下跌
     → fwd 标签被假跌污染 → 高换手/高波动股更容易碰上除权 → IC 假负
  2. 次新股上市初期暴涨，之后回落 → 与"低换手"相关
  3. 停牌复牌后补跌
  4. 退市/ST 股的极端值

本脚本逐个排查：

  A. 检查 fwd 收益的分布（是否有大量 -30% 以下的假跌）
  B. 剔除疑似除权日（单日跌幅 > 11%，超过涨跌停限制）
  C. 只保留上市满 250 天的股票（剔除次新股）
  D. 只保留流动性好的股票（剔除僵尸股）
  E. 重新计算 IC，看负号是否消失

用法：
  python scripts/diagnose_cross_section_bias.py
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
                                build_features, add_labels, daily_ic,
                                FEATURES)

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-stocks", type=int, default=4500)
    ap.add_argument("--out", default="results/cross_section_bias.json")
    args = ap.parse_args()

    db = resolve_db()
    usable = load_quality_dates()
    df = load_panel(db, usable, args.min_stocks)
    df = build_features(df)
    df = add_labels(df, [args.horizon])
    h = args.horizon
    ycol = f"fwd{h}"

    print("=" * 88)
    print(f"横截面 IC 偏差诊断   H={h} 日")
    print("=" * 88)
    print(f"原始: {len(df):,} 行, {df['stock_code'].nunique()} 只, "
          f"{df['date_str'].nunique()} 天")

    report = {"horizon": h, "raw_rows": int(len(df))}

    # ============================================================
    # A. fwd 收益分布
    # ============================================================
    print("\n[A] fwd 收益分布（是否有假跌）")
    print("-" * 88)
    s = df[ycol].dropna()
    print(f"  样本 {len(s):,}")
    for q in [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]:
        print(f"    p{q*100:>5.1f}%: {s.quantile(q)*100:>+8.2f}%")
    n_extreme = int((s < -0.15).sum())
    print(f"  fwd < -15% 的样本: {n_extreme:,} ({n_extreme/len(s)*100:.2f}%)")
    report["fwd_stats"] = {
        "n": int(len(s)), "p001": float(s.quantile(0.001)),
        "p999": float(s.quantile(0.999)), "n_below_15": n_extreme,
    }

    # ============================================================
    # B. 单日收益异常（疑似除权/停牌复牌）
    # ============================================================
    print("\n[B] 单日收益异常检测（超过涨跌停限制）")
    print("-" * 88)
    r1 = df["ret1_raw"].dropna()
    print(f"  样本 {len(r1):,}")
    n_below = int((r1 < -0.11).sum())
    n_above = int((r1 > 0.11).sum())
    n_63 = int((r1 < -0.25).sum())
    print(f"  ret1 < -11%: {n_below:,} ({n_below/len(r1)*100:.2f}%)  "
          f"← 超出跌停，疑似除权/复牌")
    print(f"  ret1 > +11%: {n_above:,} ({n_above/len(r1)*100:.2f}%)  "
          f"← 超出涨停（科创/创业板为 20%，但主板只有 10%）")
    print(f"  ret1 < -25%: {n_63:,} ({n_63/len(r1)*100:.3f}%)  ← 极可能是除权")

    # 这些极端下跌是否集中在低换手股？
    if "turnover" in df.columns:
        sub = df.dropna(subset=["ret1_raw", "turnover"])
        if len(sub) > 1000:
            extreme = sub[sub["ret1_raw"] < -0.25]
            normal = sub[sub["ret1_raw"].between(-0.05, 0.05)]
            if len(extreme) > 10 and len(normal) > 10:
                print(f"\n  极端下跌股的换手率中位数: {extreme['turnover'].median():.3f}")
                print(f"  正常波动股的换手率中位数: {normal['turnover'].median():.3f}")
                print(f"  → 极端下跌股换手率 {'更低' if extreme['turnover'].median() < normal['turnover'].median() else '更高'}")

    report["abnormal_ret1"] = {"below_11": n_below, "above_11": n_above,
                               "below_25": n_63, "total": int(len(r1))}

    # ============================================================
    # C. 剔除疑似除权的股票-日期对
    # ============================================================
    print("\n[C] 剔除疑似除权样本后重算 IC")
    print("-" * 88)

    variants = {}

    # 基线
    variants["基线（全部）"] = df

    # 剔除次日有极端下跌的股票（说明发生了除权）
    bad = df[df["ret1_raw"] < -0.25][["stock_code", "date"]].drop_duplicates()
    bad["is_bad"] = True
    d1 = df.merge(bad, on=["stock_code", "date"], how="left")
    df_c1 = d1[d1["is_bad"].isna()].drop(columns=["is_bad"])
    variants["剔除单日跌<25%"] = df_c1

    # 剔除上市不足 250 天的
    if "date" in df.columns:
        first = df.groupby("stock_code")["date"].transform("min")
        df_c2 = df[(df["date"] - first).dt.days >= 365]
        variants["剔除上市<1年"] = df_c2

    # 剔除低流动性（换手率 < 0.1%）
    if "turnover" in df.columns:
        df_c3 = df[df["turnover"].fillna(0) >= 0.1]
        variants["剔除换手<0.1%"] = df_c3

    # 全部叠加
    df_c4 = df_c1[(df_c1["date"] - df_c1.groupby("stock_code")["date"].transform("min")).dt.days >= 365]
    if "turnover" in df_c4.columns:
        df_c4 = df_c4[df_c4["turnover"].fillna(0) >= 0.1]
    variants["全部过滤"] = df_c4

    key_feats = ["ret20", "vol20", "turnover_z", "amplitude", "pos120"]
    print(f"  {'方案':<18}{'行数':>10}" +
          "".join(f"{f:>12}" for f in key_feats))
    combos = {}
    for name, sub in variants.items():
        sub = sub.dropna(subset=[ycol])
        if len(sub) < 50000:
            print(f"  {name:<18}{len(sub):>10}  样本不足")
            continue
        vals = []
        for f in key_feats:
            ss = sub[["date_str", f, ycol]].dropna()
            if len(ss) < 10000:
                vals.append(np.nan)
                continue
            ic = daily_ic(ss, f, ycol)
            vals.append(float(ic.mean()) if len(ic) > 10 else np.nan)
        combos[name] = {"rows": int(len(sub)),
                        **{f: v for f, v in zip(key_feats, vals)}}
        print(f"  {name:<18}{len(sub):>10}" +
              "".join(f"{('%.4f' % v) if np.isfinite(v) else 'nan':>12}" for v in vals))

    report["filter_variants"] = combos

    # ============================================================
    # D. 关键检验：反向因子是否真的可交易
    # ============================================================
    print("\n[D] 反向因子检验：低换手 + 低波动 是否真的跑赢")
    print("-" * 88)

    sub = variants["全部过滤"].dropna(subset=[ycol, "turnover_z", "vol20"])
    if len(sub) > 100000:
        for d, g in list(sub.groupby("date_str"))[:0]:
            pass
        # 每天分组，比较 top/bottom
        rows = []
        for d, g in sub.groupby("date_str"):
            g2 = g.dropna(subset=["turnover_z", ycol])
            if len(g2) < 500:
                continue
            q20 = g2["turnover_z"].quantile(0.2)
            q80 = g2["turnover_z"].quantile(0.8)
            low_t = g2[g2["turnover_z"] <= q20][ycol].mean()
            high_t = g2[g2["turnover_z"] >= q80][ycol].mean()
            rows.append({"date": d, "low_turnover": low_t,
                         "high_turnover": high_t, "spread": low_t - high_t})
        if rows:
            rr = pd.DataFrame(rows)
            print(f"  低换手组平均 {h} 日收益: {rr['low_turnover'].mean()*100:+.3f}%")
            print(f"  高换手组平均 {h} 日收益: {rr['high_turnover'].mean()*100:+.3f}%")
            print(f"  价差(低-高):            {rr['spread'].mean()*100:+.3f}%")
            print(f"  价差 > 0 的天数占比:    {(rr['spread']>0).mean()*100:.1f}%")
            print(f"  价差 t 统计量:          {rr['spread'].mean()/rr['spread'].std()*np.sqrt(len(rr)):.2f}")
            report["turnover_spread"] = {
                "low": float(rr["low_turnover"].mean()),
                "high": float(rr["high_turnover"].mean()),
                "spread": float(rr["spread"].mean()),
                "win_rate": float((rr["spread"] > 0).mean()),
                "t_stat": float(rr["spread"].mean() / rr["spread"].std()
                                * np.sqrt(len(rr))),
                "n_days": int(len(rr)),
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
