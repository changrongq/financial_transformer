# -*- coding: utf-8 -*-
"""核心验证：ret1 累乘 vs close 比值，是否一致？

如果一致 → fwd20 与逐日累计应该相等
如果不一致 → pre_close(复权基准) 与 close 不匹配，是 bug

对一个具体案例逐个数字核对。
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_cross_section_with_preclose import (
    resolve_db, load_quality_dates, load_panel_preclose,
    build_features_preclose, add_labels)
from test_gap_factor import cs_zscore

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_panel_preclose(resolve_db(), load_quality_dates(), 4500)
    df = build_features_preclose(df)
    df = add_labels(df, [20])
    nd = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[nd >= 200]
    df = df[df["turnover"].fillna(0) >= 0.1]

    print("=" * 92)
    print("核心验证：ret1 累乘 是否等于 close 比值")
    print("=" * 92)

    # 取一只股票，连续 25 天逐日核对
    code = "002299"
    sub = df[df["stock_code"] == code].sort_values("date_str").reset_index(drop=True)
    print(f"\n[1] {code} 逐日数据（取 20 天窗口）")

    # 找 2022-08-24 的位置
    idx = sub[sub["date_str"] == "2022-08-24"].index
    if len(idx) == 0:
        print("  找不到起始日")
        return 1
    p0 = idx[0]
    win = sub.iloc[p0:p0 + 21]
    print(f"  {'日期':<12}{'close':>9}{'pre_close':>11}{'ret1':>9}"
          f"{'累乘':>10}{'close比值':>11}")
    cum = 1.0
    for k, (_, r) in enumerate(win.iterrows()):
        cum *= (1 + r["ret1"])
        close_ratio = r["close"] / win.iloc[0]["close"]
        print(f"  {r['date_str']:<12}{r['close']:>9.2f}"
              f"{r['pre_close']:>11.4f}{r['ret1']*100:>+8.3f}%"
              f"{cum:>10.4f}{close_ratio:>11.4f}")

    print(f"\n  20 天后：累乘 {cum:.6f}  vs  close比值 "
          f"{win.iloc[-1]['close']/win.iloc[0]['close']:.6f}")
    print(f"  差异: {abs(cum - win.iloc[-1]['close']/win.iloc[0]['close']):.8f}")

    # ============================================================
    # 2. 全样本检验
    # ============================================================
    print(f"\n[2] 全样本：ret1 累乘 20 天 vs fwd20")
    df2 = df.sort_values(["stock_code", "date_str"]).copy()
    g = df2.groupby("stock_code", group_keys=False)
    df2["lr"] = np.log1p(df2["ret1"])
    df2["cum20"] = np.exp(g["lr"].transform(
        lambda s: s.shift(-1).rolling(20, min_periods=20).sum().shift(-19))) - 1

    ok = df2["cum20"].notna() & df2["fwd20"].notna()
    diff = (df2.loc[ok, "cum20"] - df2.loc[ok, "fwd20"]).abs()
    print(f"  可比样本: {int(ok.sum()):,}")
    print(f"  差异中位: {diff.median():.8f}")
    print(f"  差异 p99: {diff.quantile(0.99):.8f}")
    print(f"  差异 > 1%: {int((diff > 0.01).sum()):,} "
          f"({(diff>0.01).mean()*100:.3f}%)")

    bad = df2.loc[ok][diff > 0.01]
    if len(bad):
        print(f"\n  不一致样例（前 5）:")
        for _, r in bad.head(5).iterrows():
            print(f"    {r['stock_code']} {r['date_str']}: "
                  f"cum20 {r['cum20']*100:+.3f}% vs fwd20 {r['fwd20']*100:+.3f}%")

    # ============================================================
    # 3. 关键：找出为什么 Top20% 的 fwd20 为负而逐日复利为正
    # ============================================================
    print(f"\n[3] 关键差异：为什么 Top20% 的 fwd20 为负、逐日复利为正？")
    print(f"    （因为 Top20% 是【每天重新选】，而 fwd20 是【持有 20 天】）")

    dates = sorted(df["date_str"].unique())
    test = dates[200:]
    df["score"] = cs_zscore(df, "gap20").fillna(0.0)
    dmap = {d: gg for d, gg in df.groupby("date_str")}

    # 方法 A：每天调仓（等同于每天重新按 gap20 选 Top20%）
    print(f"\n  方法 A：每日调仓（用 ret1 均值）")
    rets_a = []
    for d in test[:250]:
        gg = dmap.get(d)
        if gg is None or len(gg) < 300:
            continue
        s = gg[["stock_code", "score", "ret1"]].dropna(subset=["score"])
        if len(s) < 300:
            continue
        thr = s["score"].quantile(0.8)
        top = s[s["score"] >= thr]
        rets_a.append(top["ret1"].mean())
    rets_a = np.array(rets_a)
    print(f"    平均日收益: {rets_a.mean()*100:+.4f}%  "
          f"年化 {((1+rets_a.mean())**244-1)*100:+.0f}%")

    # 方法 B：每 20 天调仓
    print(f"\n  方法 B：每 20 天调仓（持仓期内不换股）")
    nav = 1.0
    w_prev, pending = {}, None
    for i, d in enumerate(test[:250]):
        gg = dmap.get(d)
        if gg is None:
            continue
        if w_prev:
            rs = []
            for sc in w_prev:
                row = gg[gg["stock_code"] == sc]
                if len(row):
                    v = row["ret1"].iloc[0]
                    if np.isfinite(v):
                        rs.append(v)
            r = np.mean(rs) if rs else 0.0
        else:
            r = 0.0
        nav *= (1 + r)
        if i % 20 == 0:
            s = gg[["stock_code", "score"]].dropna()
            if len(s) >= 300:
                thr = s["score"].quantile(0.8)
                picks = s[s["score"] >= thr]["stock_code"].tolist()
                pending = {c: 1.0 / len(picks) for c in picks} if picks else {}
        if pending is not None:
            w_prev, pending = pending, None
    print(f"    250 天净值: {nav:.4f}（+{(nav-1)*100:.1f}%）")

    print(f"\n  方法 A 250 天净值（复利）: "
          f"{np.prod(1+rets_a):.4f}（+{(np.prod(1+rets_a)-1)*100:.1f}%）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
