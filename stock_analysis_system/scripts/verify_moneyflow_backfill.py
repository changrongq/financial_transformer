# -*- coding: utf-8 -*-
"""验证补入的资金流数据是否正确。"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())

tok = (ROOT / "config" / "tushare_token.local").read_text(
    encoding="utf-8").strip()
import tushare as ts
ts.set_token(tok)
pro = ts.pro_api()

con = sqlite3.connect(DB)

print("=" * 90)
print("资金流数据验证（新补 vs tushare 原始）")
print("=" * 90)

days = pd.read_sql_query("""
    SELECT DISTINCT 日期 FROM stock_moneyflow
    WHERE 日期 < '2025-01-02' ORDER BY 日期
""", con)["日期"].tolist()
print(f"\n新补的交易日: {len(days)} 天  {days[:5] if days else '-'}")

if not days:
    print("没有新补数据")
    raise SystemExit(0)

for d in days[:2]:
    d_ts = d.replace("-", "")
    raw = pro.moneyflow(trade_date=d_ts)
    raw["stock_code"] = raw["ts_code"].str.split(".").str[0]

    loc = pd.read_sql_query("""
        SELECT stock_code, "净流入额" AS loc_net, "小单买入额" AS loc_sm_buy
        FROM stock_moneyflow WHERE 日期 = ?
    """, con, params=(d,))

    m = raw.merge(loc, on="stock_code", how="inner")
    if len(m) < 100:
        print(f"\n{d}: 交集不足 {len(m)}")
        continue

    print(f"\n{d}: tushare {len(raw)} 只, 本地 {len(loc)} 只, 交集 {len(m)} 只")
    for col, loc_col in [("net_mf_amount", "loc_net"),
                         ("buy_sm_amount", "loc_sm_buy")]:
        r = (m[col] / m[loc_col].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(r):
            print(f"  {col:<18} 比值中位 {r.median():.6f}  "
                  f"p5 {r.quantile(0.05):.6f}  p95 {r.quantile(0.95):.6f}")

# 对比修复前后的资金流特征可用性
print("\n" + "=" * 90)
print("覆盖情况")
print("=" * 90)
r = con.execute("""
    SELECT substr(日期,1,4) y, COUNT(DISTINCT 日期) n, COUNT(*) rows
    FROM stock_moneyflow GROUP BY y ORDER BY y
""").fetchall()
for y, n, rows in r:
    print(f"  {y}: {n:>4} 天  {rows:>9,} 行")

# stock_data 覆盖
r2 = con.execute("""
    SELECT substr(日期,1,4) y, COUNT(DISTINCT 日期) n
    FROM stock_data GROUP BY y ORDER BY y
""").fetchall()
print(f"\n  stock_data 每年交易日:")
for y, n in r2:
    print(f"    {y}: {n:>4} 天")

con.close()
