# -*- coding: utf-8 -*-
"""验证覆盖补齐与成交额修复的结果。"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())
con = sqlite3.connect(DB)

print("=" * 92)
print("覆盖补齐验证")
print("=" * 92)

print("\n[1] 每日股票数（应全部 >= 5000）")
rows = con.execute("""
    SELECT 日期, COUNT(*) n FROM stock_data GROUP BY 日期 ORDER BY 日期
""").fetchall()
ns = np.array([n for _, n in rows])
print(f"  总 {len(rows)} 个交易日")
print(f"  中位 {np.median(ns):.0f}  最少 {ns.min()}  最多 {ns.max()}")
bad = [(d, n) for d, n in rows if n < 4500]
print(f"  股票数 < 4500 的日期: {len(bad)} 天")
for d, n in bad:
    print(f"    {d}: {n}")

print("\n[2] 之前异常日期的现状")
for d in ["2026-09-22", "2026-09-24", "2026-10-15", "2026-10-27"]:
    r = con.execute("SELECT COUNT(*) FROM stock_data WHERE 日期=?", (d,)).fetchone()
    r2 = con.execute("""
        SELECT COUNT(*), SUM(CASE WHEN pre_close IS NOT NULL THEN 1 ELSE 0 END)
        FROM stock_data WHERE 日期=?
    """, (d,)).fetchone()
    print(f"  {d}: {r[0]:>5} 只  pre_close {r2[1] or 0}/{r2[0]}")

print("\n[3] 单位正确性（成交额/成交量/收盘 应 ≈ 1）")
for d in ["2026-09-18", "2026-10-15", "2026-10-28", "2026-10-29"]:
    rr = con.execute("""
        SELECT 收盘, 成交量, 成交额 FROM stock_data
        WHERE 日期=? AND 收盘>0 AND 成交量>0 AND 成交额>0
    """, (d,)).fetchall()
    if not rr:
        print(f"  {d}: 无数据")
        continue
    ratio = np.array([a / v / c for c, v, a in rr])
    ok = np.mean((ratio > 0.9) & (ratio < 1.1))
    print(f"  {d}: n={len(rr):>5}  单位正确率 {ok*100:5.1f}%  "
          f"中位比值 {np.median(ratio):.4f}")

print("\n[4] 用 pre_close 计算的日收益分布（全表）")
df = pd.read_sql_query("""
    SELECT stock_code, 日期, 收盘, pre_close FROM stock_data
    WHERE pre_close IS NOT NULL AND pre_close > 0 AND 收盘 > 0
""", con)
df["ret"] = df["收盘"] / df["pre_close"] - 1
print(f"  样本 {len(df):,} 行")
for q in [0.001, 0.01, 0.5, 0.99, 0.999]:
    print(f"    p{q*100:>6.1f}%: {df['ret'].quantile(q)*100:>+8.3f}%")

print("\n[5] pre_close 总覆盖率")
r = con.execute("""
    SELECT COUNT(*), SUM(CASE WHEN pre_close IS NOT NULL THEN 1 ELSE 0 END)
    FROM stock_data""").fetchone()
print(f"  {r[1] or 0:,}/{r[0]:,} = {(r[1] or 0)/r[0]*100:.2f}%")

print("\n[6] 抽查 2026-10-15 新增的沪市股票")
rows = con.execute("""
    SELECT stock_code, 开盘, 收盘, 成交量, 成交额, pre_close
    FROM stock_data WHERE 日期='2026-10-15' AND stock_code LIKE '60%'
    ORDER BY stock_code LIMIT 5
""").fetchall()
for row in rows:
    imp = row[4] / row[3] if row[3] else 0
    print(f"  {row[0]}  开{row[1]:>7.2f} 收{row[2]:>7.2f} "
          f"量{row[3]:>13,.0f} 额{row[4]:>15,.0f} 隐含价{imp:>7.3f}")

con.close()
