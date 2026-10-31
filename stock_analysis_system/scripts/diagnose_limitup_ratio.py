# -*- coding: utf-8 -*-
"""核查涨停比例：文档说 Top20% 是市场 2.3 倍，实测是 0.43 倍。

可能原因：
  1. 统计的是「当日收盘涨停」还是「次日开盘涨停」
  2. 是否用 pct_chg 判断（本地无此字段）
  3. 涨跌停判定阈值不同（主板 10% / 创业 20% / 北交所 30%）
  4. 样本期不同（文档可能只用了部分年份）

本脚本同时统计多种口径，找出差异来源。

用法：
  python scripts/diagnose_limitup_ratio.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())


def limit_pct(code: str) -> float:
    c = str(code)
    if c.startswith(("30", "68")):
        return 0.20
    if c.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    return 0.10


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("""
        SELECT 日期 AS date, stock_code, 开盘 AS open, 收盘 AS close,
               最高 AS high, 最低 AS low, pre_close
        FROM stock_data
        WHERE 收盘 > 0 AND 开盘 > 0 AND pre_close > 0
    """, con)
    con.close()

    df["date_str"] = df["date"].str[:4] + "-" + df["date"].str[5:7] + "-" + df["date"].str[8:10]
    for c in ["open", "close", "high", "low", "pre_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close", "pre_close"])
    df = df.sort_values(["stock_code", "date_str"]).reset_index(drop=True)

    g = df.groupby("stock_code", group_keys=False)
    df["next_open"] = g["open"].shift(-1)
    df["lim"] = df["stock_code"].apply(limit_pct)

    # 口径 1：当日收盘相对 pre_close 涨停
    df["ret_pc"] = df["close"] / df["pre_close"] - 1.0
    df["lu_close"] = df["ret_pc"] >= (df["lim"] - 0.005)

    # 口径 2：当日收盘相对 prev close 涨停
    df["prev_close"] = g["close"].shift(1)
    df["ret_c"] = df["close"] / df["prev_close"] - 1.0
    df["lu_close2"] = df["ret_c"] >= (df["lim"] - 0.005)

    # 口径 3：次日开盘相对今日收盘涨停
    df["open_gap"] = df["next_open"] / df["close"] - 1.0
    df["lu_open"] = df["open_gap"] >= (df["lim"] - 0.005)

    print("=" * 90)
    print("涨停比例的多口径统计")
    print("=" * 90)

    # 分年统计
    df["year"] = df["date_str"].str[:4]
    for col, label in [("lu_close", "当日收盘涨停(vs pre_close)"),
                       ("lu_close2", "当日收盘涨停(vs prev close)"),
                       ("lu_open", "次日开盘涨停")]:
        sub = df.dropna(subset=[col])
        print(f"\n{label}:")
        print(f"  {'年':<6}{'样本':>10}{'涨停数':>10}{'比例':>9}")
        for y, gg in sub.groupby("year"):
            n = len(gg)
            k = int(gg[col].sum())
            print(f"  {y:<6}{n:>10,}{k:>10,}{k/n*100:>8.2f}%")
        n = len(sub)
        k = int(sub[col].sum())
        print(f"  合计  {n:>10,}{k:>10,}{k/n*100:>8.2f}%")

    # 检查 2022 年（首个测试年）的特殊性
    print("\n" + "=" * 90)
    print("为什么 2022 年样本只有 37 天？（文档逐年表里 2022 只有 37 天）")
    print("=" * 90)
    d = df.groupby("date_str")["stock_code"].count()
    print(f"  每日股票数中位 {d.median():.0f}，最少 {d.min()}")
    # 训练 250 天后，测试从哪天开始
    dates = sorted(df["date_str"].unique())
    print(f"  总交易日 {len(dates)}，第 250 天是 {dates[249]}")
    print(f"  测试期第一天 {dates[250]}")
    print(f"  2022 年剩余交易日: {sum(1 for x in dates[250:] if x.startswith('2022'))}")

    print("\n" + "=" * 90)
    print("量化对比：如果训练 250 天，测试期首个年份只有不到 1 个月")
    print("=" * 90)
    print("  → 2022 那一行的 +1.7pp 几乎没有统计意义")
    print("  → 真正有效的年份是 2023 / 2024 / 2025 / 2026 四年")


if __name__ == "__main__":
    main()
