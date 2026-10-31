# -*- coding: utf-8 -*-
"""确认 tushare moneyflow 接口能力与现有表结构。"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())

tok = (ROOT / "config" / "tushare_token.local").read_text(
    encoding="utf-8").strip()
import tushare as ts
ts.set_token(tok)
pro = ts.pro_api()

print("=" * 86)
print("tushare moneyflow 接口能力确认")
print("=" * 86)

# 1. 现有表结构
con = sqlite3.connect(DB)
cols = [c[1] for c in con.execute("PRAGMA table_info(stock_moneyflow)")]
print(f"\n[1] 现有 stock_moneyflow 列（{len(cols)} 个）")
for i, c in enumerate(cols):
    print(f"  {i:>2}: {c!r}")

r = con.execute("SELECT MIN(日期), MAX(日期), COUNT(*), "
                "COUNT(DISTINCT stock_code) FROM stock_moneyflow").fetchone()
print(f"\n  范围 {r[0]} ~ {r[1]}  {r[2]:,} 行  {r[3]:,} 只")
con.close()

# 2. 测试接口：按 trade_date 拉全市场
print(f"\n[2] 测试 pro.moneyflow(trade_date=...)")
try:
    df = pro.moneyflow(trade_date="20240603")
    print(f"  返回 {len(df) if df is not None else 0} 行")
    if df is not None and not df.empty:
        print(f"  列: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 3. 测试按股票拉
print(f"\n[3] 测试 pro.moneyflow(ts_code=..., start_date=...)")
try:
    df2 = pro.moneyflow(ts_code="000001.SZ",
                        start_date="20240601", end_date="20240610")
    print(f"  返回 {len(df2) if df2 is not None else 0} 行")
    if df2 is not None and not df2.empty:
        print(f"  日期: {sorted(df2['trade_date'].tolist())}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 4. 最早能拉到哪天
print(f"\n[4] 数据起始日期探测")
for d in ["20100104", "20150105", "20180102", "20210104"]:
    try:
        t = pro.moneyflow(trade_date=d)
        n = len(t) if t is not None else 0
        print(f"  {d}: {n} 行")
    except Exception as e:
        print(f"  {d}: {type(e).__name__}: {str(e)[:60]}")

# 5. 需要补的交易日数
print(f"\n[5] 需要补的范围")
con = sqlite3.connect(DB)
rows = con.execute("""
    SELECT 日期, COUNT(*) FROM stock_data
    WHERE 日期 < '2025-01-02'
    GROUP BY 日期 ORDER BY 日期
""").fetchall()
con.close()
print(f"  2025-01-02 之前的交易日: {len(rows)} 天")
print(f"  最早 {rows[0][0]}  最晚 {rows[-1][0]}")
print(f"  这些天的样本总数: {sum(r[1] for r in rows):,}")
