# -*- coding: utf-8 -*-
"""当前数据库全面体检（只读，不修改）。"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())
con = sqlite3.connect(DB)

print("=" * 88)
print("数据库当前状态")
print("=" * 88)

r = con.execute("""
    SELECT COUNT(*), COUNT(DISTINCT stock_code), COUNT(DISTINCT 日期),
           MIN(日期), MAX(日期)
    FROM stock_data
""").fetchone()
print(f"\n[1] 规模")
print(f"  行数   : {r[0]:,}")
print(f"  股票数 : {r[1]:,}")
print(f"  交易日 : {r[2]:,}")
print(f"  区间   : {r[3]} ~ {r[4]}")

# 字段
cols = [c[1] for c in con.execute("PRAGMA table_info(stock_data)")]
print(f"\n[2] 字段: {cols}")

# 覆盖率
print(f"\n[3] 字段覆盖率")
for c in ["收盘", "成交量", "成交额", "换手率", "pre_close", "adj_factor"]:
    if c not in cols:
        print(f"  {c:<12} 不存在")
        continue
    rr = con.execute(f"""
        SELECT COUNT(*), SUM(CASE WHEN [{c}] IS NOT NULL THEN 1 ELSE 0 END)
        FROM stock_data""").fetchone()
    print(f"  {c:<12} {(rr[1] or 0)/rr[0]*100:6.2f}%  ({rr[1] or 0:,}/{rr[0]:,})")

# 每日股票数
print(f"\n[4] 每日股票数异常（< 3000）")
rows = con.execute("""
    SELECT 日期, COUNT(*) n FROM stock_data GROUP BY 日期 ORDER BY 日期
""").fetchall()
ns = np.array([n for _, n in rows])
print(f"  中位 {np.median(ns):.0f}  最少 {ns.min()}  最多 {ns.max()}")
bad = [(d, n) for d, n in rows if n < 3000]
if bad:
    print(f"  异常 {len(bad)} 天:")
    for d, n in bad[:30]:
        print(f"    {d}: {n}")
else:
    print("  无")

# 单位自洽（按年）
print(f"\n[5] 单位自洽性（amount/(vol*close) 应 ≈ 1）")
df = pd.read_sql_query("""
    SELECT 日期, 收盘, 成交量, 成交额 FROM stock_data
    WHERE 收盘>0 AND 成交量>0 AND 成交额>0
""", con)
df["year"] = df["日期"].str[:4]
df["ratio"] = df["成交额"] / df["成交量"] / df["收盘"]
for y, g in df.groupby("year"):
    m = g["ratio"].median()
    ok = np.mean((g["ratio"] > 0.9) & (g["ratio"] < 1.1))
    print(f"  {y}: n={len(g):>9,}  中位 {m:>10.4f}  正确率 {ok*100:5.1f}%")

# pre_close vs close 一致性
print(f"\n[6] pre_close 与 close.shift(1) 一致性（抽查 200 只）")
sample = [r[0] for r in con.execute(
    "SELECT DISTINCT stock_code FROM stock_data LIMIT 200").fetchall()]
q = ",".join("?" * len(sample))
sub = pd.read_sql_query(f"""
    SELECT stock_code, 日期, 收盘, pre_close FROM stock_data
    WHERE stock_code IN ({q}) AND pre_close IS NOT NULL
    ORDER BY stock_code, 日期
""", con, params=sample)
sub["prev_close"] = sub.groupby("stock_code")["收盘"].shift(1)
sub["diff"] = (sub["pre_close"] / sub["prev_close"] - 1).abs()
sub = sub.dropna(subset=["diff"])
print(f"  样本 {len(sub):,}")
print(f"  |pre_close/prev_close - 1| 中位: {sub['diff'].median():.6f}")
print(f"  差异 > 1% 的比例: {(sub['diff'] > 0.01).mean()*100:.2f}%")
print(f"  差异 > 5% 的比例: {(sub['diff'] > 0.05).mean()*100:.2f}%")

con.close()
