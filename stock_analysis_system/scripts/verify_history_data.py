# -*- coding: utf-8 -*-
"""验证 5 年历史数据的完整性与质量。"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())
con = sqlite3.connect(DB)

print("=" * 94)
print("5 年历史数据验证")
print("=" * 94)

print("\n[1] 总体规模")
r = con.execute("""
    SELECT COUNT(*), COUNT(DISTINCT stock_code), COUNT(DISTINCT 日期),
           MIN(日期), MAX(日期)
    FROM stock_data""").fetchone()
print(f"  行数     : {r[0]:,}")
print(f"  股票数   : {r[1]:,}")
print(f"  交易日   : {r[2]:,}")
print(f"  区间     : {r[3]} ~ {r[4]}")

print("\n[2] 每日股票数分布")
rows = con.execute("""
    SELECT 日期, COUNT(*) n FROM stock_data GROUP BY 日期 ORDER BY 日期
""").fetchall()
ns = np.array([n for _, n in rows])
print(f"  中位 {np.median(ns):.0f}  最少 {ns.min()}  最多 {ns.max()}")
print(f"  股票数 < 3000 的天数: {(ns < 3000).sum()}")
print(f"  股票数 < 4500 的天数: {(ns < 4500).sum()}")
worst = sorted(rows, key=lambda x: x[1])[:8]
print(f"  最少 8 天: {worst}")

print("\n[3] 字段非空率")
for c in ["收盘", "开盘", "成交量", "成交额", "换手率", "pre_close", "adj_factor"]:
    try:
        r = con.execute(f"""
            SELECT COUNT(*), SUM(CASE WHEN [{c}] IS NULL THEN 1 ELSE 0 END)
            FROM stock_data""").fetchone()
        print(f"  {c:<12} 非空 {(r[0] - (r[1] or 0))/r[0]*100:6.2f}%")
    except Exception as e:
        print(f"  {c:<12} ERROR {e}")

print("\n[4] 按年份的字段覆盖")
print(f"  {'年份':<8}{'行数':>12}{'股票':>7}{'换手率':>9}{'pre_close':>11}")
df = pd.read_sql_query("""
    SELECT SUBSTR(日期,1,4) AS y, stock_code, 换手率, pre_close
    FROM stock_data
""", con)
for y, g in df.groupby("y"):
    tr = g["换手率"].notna().mean() * 100
    pc = g["pre_close"].notna().mean() * 100
    print(f"  {y:<8}{len(g):>12,}{g['stock_code'].nunique():>7}"
          f"{tr:>8.1f}%{pc:>10.1f}%")

print("\n[5] 单位正确性（成交额/成交量/收盘 ≈ 1）")
print(f"  {'年份':<8}{'样本':>10}{'单位OK率':>11}{'中位比值':>12}")
for y in sorted(df["y"].unique()):
    rr = con.execute("""
        SELECT 收盘, 成交量, 成交额 FROM stock_data
        WHERE 日期 LIKE ? AND 收盘>0 AND 成交量>0 AND 成交额>0
        LIMIT 300000
    """, (y + "%",)).fetchall()
    if not rr:
        continue
    ratio = np.array([a / v / c for c, v, a in rr])
    ok = np.mean((ratio > 0.9) & (ratio < 1.1))
    print(f"  {y:<8}{len(rr):>10,}{ok*100:>10.1f}%{np.median(ratio):>12.4f}")

print("\n[6] 用 pre_close 算的日收益分布")
rr = pd.read_sql_query("""
    SELECT 收盘, pre_close FROM stock_data
    WHERE pre_close IS NOT NULL AND pre_close>0 AND 收盘>0
""", con)
rr["ret"] = rr["收盘"] / rr["pre_close"] - 1
print(f"  样本 {len(rr):,}")
for q in [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]:
    print(f"    p{q*100:>6.1f}%: {rr['ret'].quantile(q)*100:>+8.3f}%")

print("\n[7] 退市股检查（最后交易日 < 2026-10-01 的股票）")
rows = con.execute("""
    SELECT stock_code, MAX(日期) AS last_day, COUNT(*) n
    FROM stock_data GROUP BY stock_code
""").fetchall()
dead = [(c, d, n) for c, d, n in rows if d < "2026-10-01"]
print(f"  疑似退市/停牌: {len(dead)} 只")
if dead:
    print(f"  样例: {sorted(dead, key=lambda x: x[1])[:5]}")

con.close()
