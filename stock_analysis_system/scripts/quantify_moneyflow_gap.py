# -*- coding: utf-8 -*-
"""量化资金流数据时间覆盖缺口的影响。

发现：
  stock_data     : 2021-11-01 ~ 2026-10-29  (1206 个交易日)
  stock_moneyflow: 2025-01-02 ~ 2026-10-28  (只有约 435 个交易日)

  → 前 771 个交易日（2021-11 ~ 2024-12）的 7 个资金流特征全是常数 0

这会造成两个问题：
  1. 训练集里 64% 的样本，资金流特征是常数 0（零信息）
  2. 更严重：如果 z-score 用全长计算，
     常数 0 段的 std 被稀释，标准化后的值分布严重失真

本脚本量化：
  · 缺口占总样本的比例
  · 分段（有资金流 / 无资金流）的资金流特征统计
  · 标准化后两段分布的差异

用法：
  python scripts/quantify_moneyflow_gap.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())

MF_COLS = ["net_mf_amount_ratio", "net_lg_amount_ratio",
           "net_elg_amount_ratio", "net_mf_ratio_5d",
           "net_lg_ratio_5d", "net_elg_ratio_5d", "mf_acceleration"]


def main():
    con = sqlite3.connect(DB)

    print("=" * 90)
    print("资金流时间覆盖缺口的量化影响")
    print("=" * 90)

    # 每日样本量
    days = pd.read_sql_query("""
        SELECT 日期 AS date, COUNT(*) n FROM stock_data
        GROUP BY 日期 ORDER BY 日期
    """, con)
    days["date"] = pd.to_datetime(days["date"])

    mf = pd.read_sql_query(
        "SELECT MIN(日期) a, MAX(日期) b FROM stock_moneyflow", con)
    mf_start = pd.to_datetime(mf.iloc[0]["a"])

    days["has_mf"] = days["date"] >= mf_start
    n_total = int(days["n"].sum())
    n_no_mf = int(days.loc[~days["has_mf"], "n"].sum())
    n_mf = int(days.loc[days["has_mf"], "n"].sum())

    print(f"\n[1] 样本覆盖")
    print(f"  总样本      : {n_total:,} 行 / {len(days)} 天")
    print(f"  有资金流    : {n_mf:,} 行 / {int(days['has_mf'].sum())} 天 "
          f"({n_mf/n_total*100:.1f}%)")
    print(f"  **无资金流** : {n_no_mf:,} 行 / {int((~days['has_mf']).sum())} 天 "
          f"({n_no_mf/n_total*100:.1f}%)")
    print(f"  资金流起点  : {mf_start.date()}")

    # 逐年
    print(f"\n[2] 逐年资金流覆盖")
    days["year"] = days["date"].dt.year
    print(f"  {'年':<6}{'交易日':>7}{'样本':>12}{'有资金流天数':>14}"
          f"{'覆盖%':>9}")
    for y, g in days.groupby("year"):
        has = int(g["has_mf"].sum())
        tot = len(g)
        print(f"  {y:<6}{tot:>7}{int(g['n'].sum()):>12}{has:>14}"
              f"{has/tot*100:>8.1f}%")

    # ------------------------------------------------------------
    # 3. 资金流特征在两段的分布
    # ------------------------------------------------------------
    print(f"\n[3] 资金流特征在「有/无数据」两段的真实值")
    print("-" * 90)
    mf_start_str = mf_start.strftime("%Y-%m-%d")
    q = f"""
        SELECT 日期 AS date, stock_code FROM stock_data LIMIT 0
    """
    # 直接从 stock_moneyflow 看：缺口期间这些表根本没有对应行
    r = con.execute("""
        SELECT COUNT(*) FROM stock_moneyflow
        WHERE 日期 < ?
    """, (mf_start_str,)).fetchone()
    print(f"  stock_moneyflow 中早于 {mf_start_str} 的行数: {r[0]:,}")
    print(f"  → 缺口期间 {n_no_mf:,} 行的资金流 merge 结果是 NaN")
    print(f"  → build_layer_features 第 106-109 行 mask(..., 0.0) 把它们填成 0.0")

    # ------------------------------------------------------------
    # 4. 标准化后的影响
    # ------------------------------------------------------------
    print(f"\n[4] 对 z-score 标准化的影响（模拟）")
    print("-" * 90)
    # 模拟：64% 是 0，36% 是有真实分布的值
    rng = np.random.RandomState(0)
    nA = n_no_mf
    nB = n_mf
    real = rng.randn(nB) * 0.001      # 真实资金流比率量级
    zeros = np.zeros(nA)
    full = np.concatenate([zeros, real])
    mu, sd = full.mean(), full.std(ddof=0)
    print(f"  模拟: {nA:,} 个 0 + {nB:,} 个真实值")
    print(f"  全长 mean = {mu:.6g}   std = {sd:.6g}")
    z_zeros = (zeros - mu) / sd
    z_real = (real - mu) / sd
    print(f"  标准化后:")
    print(f"    缺口段(0 值)  z = {z_zeros[0]:+.4f}（常数）")
    print(f"    真实段        z 范围 [{z_real.min():+.2f}, {z_real.max():+.2f}]")
    print(f"\n  → 缺口段在标准化后是同一个常数，模型无法区分")
    print(f"  → 且真实段的 z 分布被压缩（因为 std 被大量 0 拉小/拉大）")

    # ------------------------------------------------------------
    # 5. 结论
    # ------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[5] 结论与影响")
    print("=" * 90)
    print(f"""
  事实：
    · 32 个特征配置**全部有数据**（我上一版说"只有 24 个"是错的）
    · 但资金流 7 列只覆盖最近 {int(days['has_mf'].sum())} 天
    · 占全部样本的 {n_mf/n_total*100:.1f}%

  影响：
    1. 缺口期间（{int((~days['has_mf']).sum())} 天，{n_no_mf/n_total*100:.0f}% 样本）
       这 7 列全部是常数 0 → 零信息
    2. 如果训练/验证切分跨过 {mf_start.date()}，
       模型在训练时学到"资金流=0"，在验证/测试时看到真实值 → 分布漂移
    3. 反过来如果训练集全在 2024 之前，模型从未见过资金流信号

  建议（按优先级）：
    A. 若只研究近期：把训练窗口限制在资金流覆盖范围内
    B. 若要用全历史：从配置中移除 7 个资金流特征，避免常数污染
    C. 补 2021-11 ~ 2024-12 的资金流数据（tushare moneyflow 接口，需另拉）
    D. 使用 missing-indicator + mask，让模型知道"这个特征今天不可用"
""")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
