"""挖：个股层面是否也是「隔夜负、日内正」？

⚠️⚠️ 数据与口径限制（知微审查第5/6条）：

  1. 【无复权】stock_data 没有 pre_close 字段，本脚本只能用 close.shift(1) 算隔夜。
     -> 个股拆股/分红时会产生假隔夜收益。
     -> 与 ETF/指数分析（已改用 pre_close）口径不一致，不可直接对比。

  2. 【无历史ST状态】本脚本读取了 stock_basic_info 的名称，但：
     - 未实施 ST/退市名称过滤
     - 即便实施，也是用【今天的名称】倒填历史，不是当时真实的 ST 状态
     -> 横截面结果包含当日已ST/后来退市的股票

  3. 【成交额不可用】stock_data 的成交额字段有 18 个月空洞，
     兜底值 close x volume 在多个时期严重偏离真实值。
     -> 任何涉及流动性筛选的结论都不可靠

  4. 【描述统计 != 投资规则】按平均换手/波动分组是描述统计，
     用的是全样本信息，不能直接当作事前可执行的投资规则。

  因此：本文的 73.8%、5倍差距等数字应视为【描述性观察】，
  不是可交易策略的验证结果。因果（T+1）只是候选机制，未识别。

用法：
  python scripts/analyze_stock_overnight_cross_section.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _resolve_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def load_data(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [开盘] AS open, [最高] AS high, "
        "[最低] AS low, [收盘] AS close, [成交量] AS volume, [换手率] AS turnover "
        "FROM stock_data", conn)
    bi = pd.read_sql_query("SELECT stock_code, 名称, 行业 FROM stock_basic_info", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close", "open"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    df = df[df["stock_code"].str.len() == 6]
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    g = df.groupby("stock_code")
    df["prev_close"] = g["close"].shift(1)
    df["gap"] = df["open"] / df["prev_close"] - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["ret"] = df["close"] / df["prev_close"] - 1.0

    bi["stock_code"] = bi["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    df = df.merge(bi[["stock_code", "名称", "行业"]], on="stock_code", how="left")
    return df.dropna(subset=["gap"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-days", type=int, default=200, help="个股最少交易日数")
    args = parser.parse_args()

    df = load_data(_resolve_db())
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}  "
          f"{df['stock_code'].nunique()} 只股票, {len(df):,} 行")

    # ===== Q1: 个股层面符合比例 =====
    print("\n" + "=" * 96)
    print("【Q1】个股层面：多少比例的股票符合「隔夜负、日内正」？")
    print("=" * 96)

    per = df.groupby("stock_code").agg(
        n=("gap", "size"),
        gap_mean=("gap", "mean"),
        intra_mean=("intraday", "mean"),
        ret_mean=("ret", "mean"),
        gap_win=("gap", lambda x: (x > 0).mean()),
        vol=("ret", "std"),
        turnover=("turnover", "mean"),
    )
    per = per[per["n"] >= args.min_days]
    # 用累计收益判断"年化隔夜/日内"
    per["gap_cum"] = df.groupby("stock_code").apply(
        lambda x: (1 + x["gap"]).prod() - 1, include_groups=False).reindex(per.index)
    per["intra_cum"] = df.groupby("stock_code").apply(
        lambda x: (1 + x["intraday"]).prod() - 1, include_groups=False).reindex(per.index)

    n = len(per)
    n_gap_neg = int((per["gap_cum"] < 0).sum())
    n_intra_pos = int((per["intra_cum"] > 0).sum())
    n_both = int(((per["gap_cum"] < 0) & (per["intra_cum"] > 0)).sum())

    print(f"  样本股票数（≥{args.min_days}天）: {n}")
    print(f"  隔夜累计为负:   {n_gap_neg}/{n} = {n_gap_neg/n*100:.1f}%")
    print(f"  日内累计为正:   {n_intra_pos}/{n} = {n_intra_pos/n*100:.1f}%")
    print(f"  两者同时成立:   {n_both}/{n} = {n_both/n*100:.1f}%")

    if n_both / n > 0.85:
        verdict1 = "市场级现象：几乎所有个股都符合 → 不可从中选股"
    elif n_both / n > 0.60:
        verdict1 = "多数符合但有显著例外 → 可能有选股空间"
    else:
        verdict1 = "个股差异极大 → 指数规律在个股上不成立"
    print(f"\n  >>> {verdict1}")

    # 整体平均
    print(f"\n  全市场平均: 隔夜日均 {df['gap'].mean()*10000:+.2f}bp, "
          f"日内日均 {df['intraday'].mean()*10000:+.2f}bp")
    print(f"  隔夜胜率 {float((df['gap']>0).mean())*100:.1f}%, "
          f"日内胜率 {float((df['intraday']>0).mean())*100:.1f}%")

    # ===== Q2: 特征画像 =====
    print("\n" + "=" * 96)
    print("【Q2】哪些股票隔夜最负？特征横截面分析")
    print("=" * 96)

    # 分位数分组
    for feat, label in [("turnover", "平均换手率"), ("vol", "日波动率")]:
        print(f"\n  按{label}分5组（组1=最低，组5=最高）：")
        print(f"  {'组':<6}{'股票数':>8}{'隔夜日均bp':>14}{'日内日均bp':>14}{'隔夜负比例':>12}")
        print("  " + "-" * 56)
        try:
            q = pd.qcut(per[feat], 5, labels=False, duplicates="drop")
        except Exception:
            continue
        for gi in sorted(q.dropna().unique()):
            idx = per.index[q == gi]
            sub = df[df["stock_code"].isin(idx)]
            gm = sub["gap"].mean() * 10000
            im = sub["intraday"].mean() * 10000
            nr = float((per.loc[idx, "gap_cum"] < 0).mean())
            print(f"  {int(gi)+1:<6}{len(idx):>8}{gm:>+13.2f}{im:>+13.2f}{nr*100:>11.1f}%")

    # 行业维度
    print(f"\n  按行业（取样本≥8只的行业，按隔夜日均排序，最负的10个）：")
    ind = df.groupby("行业").agg(
        n_stock=("stock_code", "nunique"),
        gap_bp=("gap", lambda x: x.mean() * 10000),
        intra_bp=("intraday", lambda x: x.mean() * 10000),
    )
    ind = ind[ind["n_stock"] >= 8].sort_values("gap_bp")
    print(f"  {'行业':<16}{'股票数':>8}{'隔夜bp':>12}{'日内bp':>12}")
    print("  " + "-" * 48)
    for name, r in ind.head(10).iterrows():
        print(f"  {str(name):<16}{int(r['n_stock']):>8}{r['gap_bp']:>+11.2f}{r['intra_bp']:>+11.2f}")
    print(f"  ...（共 {len(ind)} 个行业）")
    print(f"\n  隔夜最正的5个行业：")
    print(f"  {'行业':<16}{'股票数':>8}{'隔夜bp':>12}{'日内bp':>12}")
    print("  " + "-" * 48)
    for name, r in ind.tail(5).iloc[::-1].iterrows():
        print(f"  {str(name):<16}{int(r['n_stock']):>8}{r['gap_bp']:>+11.2f}{r['intra_bp']:>+11.2f}")

    # ===== Q3: 稳定性检验（前后半段）=====
    print("\n" + "=" * 96)
    print("【Q3】稳定性：前半段 vs 后半段，隔夜最负的股票是否一致？")
    print("=" * 96)

    dates = sorted(df["date"].unique())
    mid = dates[len(dates) // 2]
    print(f"  前半段: {pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(mid).date()}")
    print(f"  后半段: {pd.Timestamp(mid).date()} ~ {pd.Timestamp(dates[-1]).date()}")

    h1 = df[df["date"] < mid].groupby("stock_code").agg(
        n=("gap", "size"), gap=("gap", "mean"))
    h2 = df[df["date"] >= mid].groupby("stock_code").agg(
        n=("gap", "size"), gap=("gap", "mean"))
    both = h1.join(h2, lsuffix="_h1", rsuffix="_h2", how="inner")
    both = both[(both["n_h1"] >= 80) & (both["n_h2"] >= 80)]

    if len(both) > 30:
        corr = both["gap_h1"].corr(both["gap_h2"])
        print(f"\n  两段隔夜均值的相关性: {corr:+.3f}")
        if abs(corr) < 0.15:
            print("  → 几乎没有持续性，无法据此选股")
        elif corr > 0.15:
            print("  → 有正持续性，隔夜负的股票倾向继续负（可用于选股）")
        else:
            print("  → 负持续性（均值回归），隔夜负的倾向转正")

        # 前20%最负 vs 后20%
        q20 = both["gap_h1"].quantile(0.2)
        q80 = both["gap_h1"].quantile(0.8)
        worst = both[both["gap_h1"] <= q20]
        best = both[both["gap_h1"] >= q80]
        print(f"\n  前半段隔夜最负的20%（{len(worst)}只）在后半段: {worst['gap_h2'].mean()*10000:+.2f}bp")
        print(f"  前半段隔夜最正的20%（{len(best)}只）在后半段: {best['gap_h2'].mean()*10000:+.2f}bp")
        spread = (worst['gap_h2'].mean() - best['gap_h2'].mean()) * 10000
        print(f"  价差（最负 - 最正）: {spread:+.2f}bp  "
              f"{'→ 可做多最负组（继续负）' if spread < 0 else '→ 反转（最正组反而更负）'}")

    # ===== 结论 =====
    print("\n" + "=" * 96)
    print("结论")
    print("=" * 96)
    print(f"  1. 个股符合比例: {n_both}/{n} = {n_both/n*100:.1f}%")
    print(f"     {verdict1}")
    print(f"  2. 全市场个股平均: 隔夜 {df['gap'].mean()*10000:+.2f}bp/日, "
          f"日内 {df['intraday'].mean()*10000:+.2f}bp/日")
    print(f"     与指数结论 {'一致（同向）' if df['gap'].mean() < 0 and df['intraday'].mean() > 0 else '不一致'}")
    print(f"  3. 个股间差异是否可利用: 见 Q3 稳定性检验")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "stock_overnight_cross_section.json"
    out.write_text(json.dumps({
        "n_stocks": n,
        "n_gap_negative": n_gap_neg, "n_intraday_positive": n_intra_pos,
        "n_both": n_both, "both_ratio": round(n_both / n, 4),
        "market_avg_gap_bp": round(float(df["gap"].mean() * 10000), 3),
        "market_avg_intra_bp": round(float(df["intraday"].mean() * 10000), 3),
        "verdict": verdict1,
        "industry_worst": [
            {"industry": str(k), "n": int(r["n_stock"]),
             "gap_bp": round(float(r["gap_bp"]), 2), "intra_bp": round(float(r["intra_bp"]), 2)}
            for k, r in ind.head(10).iterrows()],
        "industry_best": [
            {"industry": str(k), "n": int(r["n_stock"]),
             "gap_bp": round(float(r["gap_bp"]), 2), "intra_bp": round(float(r["intra_bp"]), 2)}
            for k, r in ind.tail(5).iterrows()],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
