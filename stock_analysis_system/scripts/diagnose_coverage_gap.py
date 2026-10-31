# -*- coding: utf-8 -*-
"""诊断 2026-09-24 ~ 2026-10-27 覆盖缺失问题。

现象：
  2026-09-22  只有 1975 只
  2026-09-24 ~ 2026-10-27  只有约 2320 只
  2026-10-28 起 恢复 5460+

而 2026-10-28 新增的 3138 只是 002/300 开头的深市股票，
说明缺的是【深市】数据。

本脚本确认：
  1. 缺失期间的股票代码分布
  2. 与正常日期的对比
  3. 每个日期具体缺哪些代码

用法：
  python scripts/diagnose_coverage_gap.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())


def main():
    con = sqlite3.connect(DB)

    print("=" * 88)
    print("覆盖缺失诊断")
    print("=" * 88)

    dates = [r[0] for r in con.execute(
        "SELECT DISTINCT 日期 FROM stock_data ORDER BY 日期").fetchall()]

    # 找出股票数异常的日子
    rows = con.execute("""
        SELECT 日期, COUNT(*) n FROM stock_data GROUP BY 日期 ORDER BY 日期
    """).fetchall()
    bad = [(d, n) for d, n in rows if n < 4500]
    print(f"\n股票数 < 4500 的日期: {len(bad)} 天")
    for d, n in bad:
        print(f"  {d}: {n:>5}")

    if not bad:
        return 0

    # 取一个正常日和一个异常日对比
    good_date = "2026-09-18"
    bad_date = "2026-10-15"

    print(f"\n{'='*88}")
    print(f"对比: {good_date}（正常） vs {bad_date}（异常）")
    print("=" * 88)

    for d in [good_date, bad_date]:
        codes = [r[0] for r in con.execute(
            "SELECT stock_code FROM stock_data WHERE 日期=?", (d,)).fetchall()]
        pfx = pd.Series([c[:2] for c in codes if c]).value_counts().sort_index()
        print(f"\n{d}: {len(codes)} 只")
        print(f"  前缀分布: {dict(pfx.head(20))}")

    # 差集
    good = set(r[0] for r in con.execute(
        "SELECT stock_code FROM stock_data WHERE 日期=?", (good_date,)).fetchall())
    bad_set = set(r[0] for r in con.execute(
        "SELECT stock_code FROM stock_data WHERE 日期=?", (bad_date,)).fetchall())
    missing = good - bad_set
    extra = bad_set - good
    print(f"\n缺失 {len(missing)} 只，多余 {len(extra)} 只")
    if missing:
        mp = pd.Series([c[:2] for c in missing if c]).value_counts().sort_index()
        print(f"  缺失股票前缀: {dict(mp)}")
        print(f"  样例: {sorted(missing)[:15]}")
    if extra:
        print(f"  多余样例: {sorted(extra)[:15]}")

    # 检查这些缺失股票在其他日期有没有数据
    if missing:
        sample = sorted(missing)[:5]
        q = ",".join("?" * len(sample))
        print(f"\n抽查缺失股票的历史数据:")
        for c in sample:
            r = con.execute("""
                SELECT MIN(日期), MAX(日期), COUNT(*) FROM stock_data
                WHERE stock_code=?
            """, (c,)).fetchone()
            print(f"  {c}: {r[0]} ~ {r[1]}  共 {r[2]} 天")

    # 检查异常期间的实际日期列表
    print(f"\n异常期间的交易日:")
    bad_dates = sorted(d for d, n in bad)
    print(f"  {bad_dates}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
