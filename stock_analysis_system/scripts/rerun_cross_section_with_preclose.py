# -*- coding: utf-8 -*-
"""用 pre_close 重算横截面 IC，对比之前的结论。

之前（用 close.pct_change()）：
  · 12 个因子 IC 全为负，ICIR -3 ~ -14
  · turnover_z: IC -0.0651, ICIR -13.95
  · 五分层不单调
  · 扣成本后不成立

现在用 `ret = 收盘 / pre_close - 1`（真实收益），
看结论是否改变。

用法：
  python scripts/rerun_cross_section_with_preclose.py
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


def load_quality_dates() -> set:
    p = RESULTS_DIR / "stock_data_quality.json"
    if not p.exists():
        return set()
    with open(p, encoding="utf-8") as f:
        q = json.load(f)
    return set(q.get("usable_dates", []))


def load_panel_preclose(db: str, usable: set, min_stocks: int = 4500):
    con = sqlite3.connect(db)
    df = pd.read_sql_query("""
        SELECT 日期 AS date, stock_code, 开盘 AS open, 最高 AS high,
               最低 AS low, 收盘 AS close, pre_close,
               成交量 AS vol, 成交额 AS amount, 换手率 AS turnover
        FROM stock_data
        WHERE pre_close IS NOT NULL AND pre_close > 0
    """, con)
    con.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "pre_close",
              "vol", "amount", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "pre_close", "close"])
    df = df[(df["close"] > 0) & (df["pre_close"] > 0)]

    if usable:
        df = df[df["date_str"].isin(usable)]

    cnt = df.groupby("date_str")["stock_code"].transform("count")
    df = df[cnt >= min_stocks]
    return df.sort_values(["stock_code", "date"]).reset_index(drop=True)


def build_features_preclose(df: pd.DataFrame) -> pd.DataFrame:
    """用 pre_close 计算真实收益，再构造特征。"""
    g = df.groupby("stock_code", group_keys=False)

    # ★ 核心：真实日收益
    df["ret1"] = df["close"] / df["pre_close"] - 1.0
    df["ret1"] = df["ret1"].clip(-0.25, 0.35)   # 只截断退市级异常

    # 多日收益：用真实收益连乘，避免价格拼接问题
    lr = np.log1p(df["ret1"])
    df["lr"] = lr
    for n in [5, 20, 60]:
        df[f"ret{n}"] = np.exp(g["lr"].transform(
            lambda s: s.rolling(n, min_periods=max(3, n // 2)).sum())) - 1.0

    # 隔夜 / 日内（这是 pre_close 才能算对的）
    df["gap"] = df["open"] / df["pre_close"] - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0

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

    df["amplitude"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # 隔夜/日内累计（新特征，只有 pre_close 能算）
    for n in [5, 20]:
        df[f"gap{n}"] = np.exp(g["gap"].transform(
            lambda s: np.log1p(s.clip(-0.3, 0.3)).rolling(n, min_periods=max(3, n // 2)).sum())) - 1
        df[f"intra{n}"] = np.exp(g["intraday"].transform(
            lambda s: np.log1p(s.clip(-0.3, 0.3)).rolling(n, min_periods=max(3, n // 2)).sum())) - 1

    return df.replace([np.inf, -np.inf], np.nan)


FEATURES_OLD = ["ret1", "ret5", "ret20", "ret60",
                "vol20", "vol60", "vol_ratio",
                "amt_ratio20", "turnover_ratio", "turnover_z",
                "pos120", "amplitude"]
FEATURES_NEW = FEATURES_OLD + ["gap5", "gap20", "intra5", "intra20"]


def add_labels(df, horizons):
    """未来 h 日收益。

    用 close 比值是可靠的：因为 close 已经是除权调整后的连通序列
    （已验证 close[t]/pre_close[t] 与 tushare 真实收益一致）。
    """
    g = df.groupby("stock_code", group_keys=False)
    for h in horizons:
        df[f"fwd{h}"] = g["close"].transform(
            lambda s: s.shift(-h) / s - 1.0)
    return df


def daily_ic(df, f, ycol, min_n=300):
    out = {}
    for d, g in df.groupby("date_str"):
        s = g[[f, ycol]].dropna()
        if len(s) < min_n:
            continue
        a = s[f].rank().values
        b = s[ycol].rank().values
        a -= a.mean()
        b -= b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if den > 0:
            out[d] = float((a * b).sum() / den)
    return pd.Series(out).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--min-stocks", type=int, default=4500)
    ap.add_argument("--out", default="results/cross_section_preclose.json")
    args = ap.parse_args()

    db = resolve_db()
    usable = load_quality_dates()
    print("=" * 94)
    print("用 pre_close 重算横截面 IC（对比之前结论）")
    print("=" * 94)

    df = load_panel_preclose(db, usable, args.min_stocks)
    print(f"行数 {len(df):,}  股票 {df['stock_code'].nunique():,}  "
          f"交易日 {df['date_str'].nunique()}")
    print(f"区间 {df['date'].min().date()} ~ {df['date'].max().date()}")

    df = build_features_preclose(df)
    df = add_labels(df, args.horizons)

    # 过滤：上市满 200 个交易日、换手率 >= 0.1%
    nd = df.groupby("stock_code")["date_str"].transform("nunique")
    df = df[nd >= 200]
    df = df[df["turnover"].fillna(0) >= 0.1]
    print(f"过滤后 {len(df):,} 行")

    report = {"n_rows": int(len(df)),
              "n_stocks": int(df["stock_code"].nunique()),
              "n_days": int(df["date_str"].nunique()),
              "horizons": args.horizons, "results": {}}

    for h in args.horizons:
        ycol = f"fwd{h}"
        print(f"\n{'='*94}")
        print(f"H = {h} 日")
        print("=" * 94)
        print(f"  {'特征':<18}{'IC均值':>10}{'IC标准差':>11}{'ICIR':>9}"
              f"{'IC>0':>8}{'天数':>7}   vs 之前(用pct_change)")

        prev = {"turnover_z": -0.0651, "vol20": -0.0715, "ret20": -0.0710,
                "amplitude": -0.0711, "pos120": -0.0499}  # H=5 的数

        rows = []
        for f in FEATURES_NEW:
            if f not in df.columns:
                continue
            sub = df[["date_str", f, ycol]].dropna()
            if len(sub) < 20000:
                continue
            ic = daily_ic(sub, f, ycol)
            if len(ic) < 20:
                continue
            m = float(ic.mean())
            sd = float(ic.std(ddof=1))
            icir = m / sd * np.sqrt(len(ic)) if sd > 0 else np.nan
            before = prev.get(f, None)
            mark = ""
            if before is not None:
                mark = f"   {before:+.4f} → {m:+.4f}"
            rows.append({"feature": f, "ic": m, "ic_std": sd,
                         "icir": icir, "pos_rate": float((ic > 0).mean()),
                         "n_days": int(len(ic)), "ic_before": before})
            print(f"  {f:<18}{m:>+10.4f}{sd:>11.4f}{icir:>9.2f}"
                  f"{float((ic>0).mean())*100:>7.1f}%{len(ic):>7}{mark}")

        rows.sort(key=lambda r: -abs(r["ic"]))
        report["results"][str(h)] = rows

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
