# -*- coding: utf-8 -*-
"""全面体检：确认当前数据库真实状态（不做任何假设）。"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())
con = sqlite3.connect(DB)

print("=" * 88)
print("数据库全面体检")
print("=" * 88)

# 1. 规模
r = con.execute("""
    SELECT COUNT(*), COUNT(DISTINCT stock_code), COUNT(DISTINCT 日期),
           MIN(日期), MAX(日期) FROM stock_data
""").fetchone()
print(f"\n[1] 规模")
print(f"  行数     : {r[0]:,}")
print(f"  股票数   : {r[1]:,}")
print(f"  交易日   : {r[2]:,}")
print(f"  日期范围 : {r[3]} ~ {r[4]}")

# 2. 字段
cols = [c[1] for c in con.execute("PRAGMA table_info(stock_data)")]
print(f"\n[2] 字段: {cols}")

# 3. 各字段覆盖率
print(f"\n[3] 字段覆盖率")
for c in cols:
    if c in ("stock_code", "日期"):
        continue
    r2 = con.execute(f"""
        SELECT COUNT(*), SUM(CASE WHEN [{c}] IS NOT NULL THEN 1 ELSE 0 END)
        FROM stock_data
    """).fetchone()
    print(f"  {c:<12} {(r2[1] or 0)/r2[0]*100:6.2f}%  ({r2[1] or 0:,}/{r2[0]:,})")

# 4. 按年统计单位自洽性
print(f"\n[4] 单位自洽性（amount/(vol*close)，应为 1.0）")
print(f"  {'年':<6}{'行数':>10}{'中位':>10}{'p5':>9}{'p95':>9}{'自洽%':>8}")
df = pd.read_sql_query("""
    SELECT 日期, 收盘, 成交量, 成交额 FROM stock_data
    WHERE 收盘>0 AND 成交量>0 AND 成交额>0
""", con)
df["year"] = df["日期"].str[:4]
df["ratio"] = df["成交额"] / df["成交量"] / df["收盘"]
for y, g in df.groupby("year"):
    ok = np.mean((g["ratio"] > 0.9) & (g["ratio"] < 1.1))
    print(f"  {y:<6}{len(g):>10,}{g['ratio'].median():>10.4f}"
          f"{g['ratio'].quantile(0.05):>9.4f}{g['ratio'].quantile(0.95):>9.4f}"
          f"{ok*100:>7.1f}%")

# 5. 每日股票数异常
print(f"\n[5] 每日股票数（<3000 的日期）")
rows = con.execute("""
    SELECT 日期, COUNT(*) n FROM stock_data GROUP BY 日期 ORDER BY 日期
""").fetchall()
bad = [(d, n) for d, n in rows if n < 3000]
print(f"  共 {len(bad)} 天")
for d, n in bad[:30]:
    print(f"    {d}: {n}")

# 6. 每日换手率覆盖
print(f"\n[6] 换手率覆盖 <50% 的日期数")
rows = con.execute("""
    SELECT 日期, COUNT(*) n,
           SUM(CASE WHEN 换手率 IS NOT NULL THEN 1 ELSE 0 END) ok
    FROM stock_data GROUP BY 日期
""").fetchall()
bad2 = [(d, n, o or 0) for d, n, o in rows if (o or 0) / n < 0.5]
print(f"  共 {len(bad2)} 天")
for d, n, o in bad2[:10]:
    print(f"    {d}: {o}/{n} = {o/n*100:.1f}%")

# 7. pre_close 覆盖
r3 = con.execute("""
    SELECT COUNT(*), SUM(CASE WHEN pre_close IS NOT NULL THEN 1 ELSE 0 END)
    FROM stock_data""").fetchone()
print(f"\n[7] pre_close 覆盖: {(r3[1] or 0)/r3[0]*100:.2f}%")

con.close()
