# -*- coding: utf-8 -*-
"""精确核对 stock_moneyflow 列名与代码期望的差异。"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())

NEED = ["净流入额", "大单买入额", "大单卖出额", "特大单买入额",
        "特大单卖出额", "小单买入额", "小单卖出额", "中单买入额",
        "中单卖出额", "净流入量"]

con = sqlite3.connect(DB)
cols = [c[1] for c in con.execute("PRAGMA table_info(stock_moneyflow)")]

print("=" * 88)
print("stock_moneyflow 列名精确核对")
print("=" * 88)

print(f"\n实际列数: {len(cols)}")
print("\n[1] 实际列名（用 repr 显示隐藏字符）")
for i, c in enumerate(cols):
    print(f"  {i:>2}: {c!r}")

print("\n[2] 代码期望的列 vs 实际")
print(f"  {'期望列名':<16}{'精确匹配':>10}{'去空格后匹配':>14}  实际匹配到的")
norm = {c.replace(" ", ""): c for c in cols}
for n in NEED:
    exact = n in cols
    stripped = n in norm
    actual = norm.get(n, "-")
    mark = "OK" if exact else ("需清洗" if stripped else "缺失")
    print(f"  {n:<16}{'Y' if exact else 'N':>10}"
          f"{'Y' if stripped else 'N':>14}  {actual!r}  {mark}")

print("\n[3] 统计")
n_exact = sum(1 for n in NEED if n in cols)
n_strip = sum(1 for n in NEED if n in norm)
print(f"  精确匹配:  {n_exact}/{len(NEED)}")
print(f"  去空格后:  {n_strip}/{len(NEED)}")

if n_strip > n_exact:
    print(f"\n  ★ 有 {n_strip - n_exact} 个列名需要清洗空格才能用")

# 检查数据完整性
print("\n[4] 数据完整性（用清洗后的列名）")
r = con.execute("SELECT COUNT(*), COUNT(DISTINCT stock_code), "
                "MIN(日期), MAX(日期) FROM stock_moneyflow").fetchone()
print(f"  行数 {r[0]:,}  股票 {r[1]:,}  {r[2]} ~ {r[3]}")

key = norm.get("净流入额")
if key:
    r = con.execute(f"SELECT COUNT(*), "
                    f"SUM(CASE WHEN [{key}] IS NOT NULL THEN 1 ELSE 0 END) "
                    f"FROM stock_moneyflow").fetchone()
    print(f"  [{key}] 非空 {r[1]:,}/{r[0]:,} = {(r[1] or 0)/r[0]*100:.1f}%")

con.close()
