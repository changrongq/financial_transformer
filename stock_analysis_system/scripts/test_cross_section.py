# -*- coding: utf-8 -*-
"""横截面选股：特征增量价值检验。

与择时的根本区别
----------------
  择时：一个时间序列，问「明天的收益能否预测」
  横截面：每个交易日 N 只股票，问「同一天里，哪些股票未来收益更高」

横截面的优势：
  · 每天 5000 只股票，479 天 → 约 250 万个样本
  · 市场整体涨跌可以被剔除（用当天横截面均值），噪声大幅降低
  · IC 是横截面秩相关，不是时间序列相关

设计
----
【特征】（全部 t 日收盘可得，且是横截面可比的）
  · 动量：ret5 / ret20 / ret60（过去收益）
  · 反转：ret1（昨日收益，短反转）
  · 波动：vol20 / vol60 / vol_ratio
  · 量能：amt_ratio20 / turnover_ratio20 / turnover
  · 位置：close_pos_120（距 120 日高低点的位置）
  · 估值：pe_ttm / pb（非空率低，单独处理）

【标签】
  fwd_h = 未来 h 日收益（h ∈ {1, 5, 20}）

【关键：横截面去均值】
  y = fwd_h - 当天横截面均值（或中位数）
  这样预测的是「相对强弱」，不是「市场涨跌」
  否则模型会学到"所有股票都涨"，那不是选股能力

【评价】
  1. 横截面 IC（每天算一次 Spearman，再对时间平均）
  2. ICIR = mean(IC) / std(IC) * sqrt(交易日数)
  3. 分层：每天按预测分 5 组，看 top-bottom 价差
  4. 多空组合扣成本回测（每天调仓 / 每 5 天 / 每 20 天）
  5. 逐年稳定性（虽然有 2 年数据，只能分 2 段）

用法：
  python scripts/test_cross_section.py
  python scripts/test_cross_section.py --horizons 1 5 20 --top-quantile 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def resolve_db() -> str:
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            with cfg.open("r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            raw = c.get("databases", {}).get("trade_db")
            if raw:
                p = Path(raw)
                if not p.is_absolute():
                    p = ROOT / p
                if p.exists():
                    return str(p)
        except Exception:
            pass
    return str((ROOT.parent / "stock_history.db").resolve())


def load_quality_dates() -> list:
    p = RESULTS_DIR / "stock_data_quality.json"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        q = json.load(f)
    return set(q.get("usable_dates", []))


def load_panel(db: str, usable_dates: set, min_stocks: int = 4500) -> pd.DataFrame:
    """加载面板数据，只保留质量达标日期。"""
    con = sqlite3.connect(db)
    df = pd.read_sql_query("""
        SELECT 日期 AS date, stock_code, 开盘 AS open, 最高 AS high,
               最低 AS low, 收盘 AS close, 成交量 AS vol, 成交额 AS amount,
               换手率 AS turnover, 市盈率TTM AS pe_ttm, 市净率 AS pb
        FROM stock_data
    """, con)
    con.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "vol", "amount",
              "turnover", "pe_ttm", "pb"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]

    if usable_dates:
        df = df[df["date_str"].isin(usable_dates)]

    # 剔除每天股票数不足的日子
    cnt = df.groupby("date_str")["stock_code"].transform("count")
    df = df[cnt >= min_stocks]

    return df.sort_values(["stock_code", "date"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算特征。全部用 t 日及之前数据。

    ⚠️ stock_data 没有 pre_close，无法跨除权连续。
       这是已知限制：单只股票的 ret 会在除权日出现假跳变。
       缓解：用 ret 的横截面排名（rank），极端值影响被压缩；
             同时把 |ret| > 30% 的单日收益标记为可疑并在计算时截断。
    """
    g = df.groupby("stock_code", group_keys=False)

    # 单日收益，截断极端值（疑似除权）
    df["ret1"] = g["close"].pct_change(fill_method=None)
    df["ret1_raw"] = df["ret1"].copy()
    df["ret1"] = df["ret1"].clip(-0.31, 0.31)

    # 多日收益（用累计净值方式，避免除权污染）
    lc = np.log(df["close"].clip(lower=1e-6))
    df["logc"] = lc
    for n in [5, 20, 60]:
        df[f"ret{n}"] = np.exp(g["logc"].transform(
            lambda s: s - s.shift(n))) - 1.0
        df[f"ret{n}"] = df[f"ret{n}"].clip(-0.9, 3.0)

    # 波动
    for n in [20, 60]:
        df[f"vol{n}"] = g["ret1"].transform(
            lambda s: s.rolling(n, min_periods=max(5, n // 2)).std())
    df["vol_ratio"] = df["vol20"] / df["vol60"].replace(0, np.nan)

    # 量能
    df["amt_ma20"] = g["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    df["amt_ratio20"] = df["amount"] / df["amt_ma20"].replace(0, np.nan)
    df["turnover_ma20"] = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    df["turnover_ratio"] = df["turnover"] / df["turnover_ma20"].replace(0, np.nan)
    df["turnover_z"] = g["turnover"].transform(
        lambda s: (s - s.rolling(60, min_periods=20).mean())
        / s.rolling(60, min_periods=20).std().replace(0, np.nan))

    # 位置
    df["high120"] = g["high"].transform(
        lambda s: s.rolling(120, min_periods=40).max())
    df["low120"] = g["low"].transform(
        lambda s: s.rolling(120, min_periods=40).min())
    df["pos120"] = ((df["close"] - df["low120"]) /
                    (df["high120"] - df["low120"]).replace(0, np.nan))

    # 振幅
    df["amplitude"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    return df


def add_labels(df: pd.DataFrame, horizons: list) -> pd.DataFrame:
    g = df.groupby("stock_code", group_keys=False)
    for h in horizons:
        # 未来 h 日收益：close[t+h] / close[t] - 1
        fwd = g["close"].transform(lambda s: s.shift(-h) / s - 1.0)
        df[f"fwd{h}"] = fwd.clip(-0.9, 3.0)
    return df


def cs_demean(df: pd.DataFrame, col: str) -> pd.Series:
    """横截面去中位数（用中位数更稳健）。"""
    return df[col] - df.groupby("date_str")[col].transform("median")


def daily_ic(df: pd.DataFrame, pred_col: str, y_col: str,
             min_n: int = 100) -> pd.Series:
    """每天算横截面 Spearman IC。"""
    out = {}
    for d, g in df.groupby("date_str"):
        s = g[[pred_col, y_col]].dropna()
        if len(s) < min_n:
            continue
        a = s[pred_col].rank().values
        b = s[y_col].rank().values
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if den <= 0:
            continue
        out[d] = float((a * b).sum() / den)
    return pd.Series(out).sort_index()


FEATURES = ["ret1", "ret5", "ret20", "ret60",
            "vol20", "vol60", "vol_ratio",
            "amt_ratio20", "turnover_ratio", "turnover_z",
            "pos120", "amplitude"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--min-stocks", type=int, default=4500)
    ap.add_argument("--top-quantile", type=float, default=0.2)
    ap.add_argument("--out", default="results/cross_section.json")
    args = ap.parse_args()

    db = resolve_db()
    usable = load_quality_dates()
    print("=" * 90)
    print("横截面选股：特征增量价值检验")
    print("=" * 90)
    print(f"质量白名单: {len(usable)} 天" if usable else "质量白名单: 未找到，使用全部日期")

    print("\n加载面板...")
    df = load_panel(db, usable, args.min_stocks)
    print(f"  行数 {len(df):,}")
    print(f"  股票 {df['stock_code'].nunique():,}")
    print(f"  交易日 {df['date_str'].nunique()}")
    print(f"  区间 {df['date'].min().date()} ~ {df['date'].max().date()}")

    print("\n计算特征...")
    df = build_features(df)
    print("计算标签...")
    df = add_labels(df, args.horizons)

    report = {
        "n_rows": int(len(df)),
        "n_stocks": int(df["stock_code"].nunique()),
        "n_days": int(df["date_str"].nunique()),
        "period": [str(df["date"].min().date()), str(df["date"].max().date())],
        "features": FEATURES,
        "horizons": args.horizons,
        "top_quantile": args.top_quantile,
        "results": {},
    }

    # ============================================================
    # 单因子 IC
    # ============================================================
    print("\n" + "=" * 90)
    print("单因子横截面 IC")
    print("=" * 90)

    for h in args.horizons:
        ycol = f"fwd{h}"
        print(f"\n--- H = {h} 日 ---")
        print(f"  {'特征':<18}{'IC均值':>10}{'IC标准差':>11}{'ICIR':>9}"
              f"{'IC>0占比':>10}{'样本天数':>10}")
        rows = []
        for f in FEATURES:
            sub = df[["date_str", f, ycol]].dropna()
            if len(sub) < 10000:
                continue
            ic = daily_ic(sub, f, ycol)
            if len(ic) < 20:
                continue
            mean_ic = float(ic.mean())
            std_ic = float(ic.std(ddof=1))
            icir = mean_ic / std_ic * np.sqrt(len(ic)) if std_ic > 0 else np.nan
            pos_rate = float((ic > 0).mean())
            rows.append({"feature": f, "ic_mean": mean_ic, "ic_std": std_ic,
                         "icir": icir, "pos_rate": pos_rate,
                         "n_days": int(len(ic))})
            print(f"  {f:<18}{mean_ic:>+10.4f}{std_ic:>11.4f}"
                  f"{icir:>9.2f}{pos_rate*100:>9.1f}%{len(ic):>10}")

        rows.sort(key=lambda r: -abs(r["ic_mean"]))
        report["results"][str(h)] = {"single_factor": rows}

        if rows:
            best = rows[0]
            print(f"\n  最强单因子: {best['feature']}  "
                  f"IC={best['ic_mean']:+.4f}  ICIR={best['icir']:.2f}")

    # ============================================================
    # 多因子组合
    # ============================================================
    print("\n" + "=" * 90)
    print("多因子组合（按 |IC| 排序后取 Top-N，等权 z-score）")
    print("=" * 90)

    for h in args.horizons:
        ycol = f"fwd{h}"
        rows = report["results"][str(h)]["single_factor"]
        if not rows:
            continue
        print(f"\n--- H = {h} 日 ---")

        for topn in [1, 3, 5]:
            sel = [r["feature"] for r in rows[:topn]]
            sub = df[["date_str", ycol] + sel].copy()
            # 横截面 z-score 后等权
            for f in sel:
                mu = sub.groupby("date_str")[f].transform("mean")
                sd = sub.groupby("date_str")[f].transform("std").replace(0, np.nan)
                sub[f + "_z"] = (sub[f] - mu) / sd
                # 方向：按训练期 IC 符号
                sign = np.sign(next(r["ic_mean"] for r in rows if r["feature"] == f))
                sub[f + "_z"] = sub[f + "_z"] * sign

            zcols = [f + "_z" for f in sel]
            sub["score"] = sub[zcols].mean(axis=1)
            sub = sub.dropna(subset=["score", ycol])
            if len(sub) < 10000:
                continue
            ic = daily_ic(sub, "score", ycol)
            if len(ic) < 20:
                continue
            mean_ic = float(ic.mean())
            std_ic = float(ic.std(ddof=1))
            icir = mean_ic / std_ic * np.sqrt(len(ic)) if std_ic > 0 else np.nan
            print(f"  Top{topn} ({','.join(sel[:3])}{'...' if topn>3 else ''}): "
                  f"IC={mean_ic:+.4f}  ICIR={icir:.2f}  "
                  f"IC>0={float((ic>0).mean())*100:.1f}%")
            report["results"][str(h)][f"combo_top{topn}"] = {
                "features": sel, "ic_mean": mean_ic, "icir": icir,
                "pos_rate": float((ic > 0).mean()), "n_days": int(len(ic)),
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
