"""验证成交额修复质量：修复后的数据是否合理。"""
import sqlite3

import pandas as pd

c = sqlite3.connect(r"D:\qcr\Stock\stock_history.db")

print("=" * 76)
print("1. 已修复的日期（2024-10-08 ~ 10-14）数据检查")
print("=" * 76)
q = """
SELECT [日期], COUNT(*) n,
       SUM(CASE WHEN [成交额]>0 THEN 1 ELSE 0 END) amt_ok,
       AVG([成交量]) avg_vol,
       AVG([成交额]/NULLIF([成交量],0)) avg_price
FROM stock_data
WHERE [日期] BETWEEN '2024-10-08' AND '2024-10-14'
GROUP BY [日期] ORDER BY [日期]
"""
df = pd.read_sql_query(q, c)
print(df.to_string(index=False))
print()
print("  验证：平均隐含价格（成交额/成交量）应接近真实股价（10-30元）")

print()
print("=" * 76)
print("2. 单只股票验证（000001 平安银行）")
print("=" * 76)
q2 = """
SELECT [日期], [收盘], [成交量], [成交额],
       [成交额]/NULLIF([成交量],0) AS implied_price
FROM stock_data
WHERE stock_code='000001' AND [日期] BETWEEN '2024-10-08' AND '2024-10-14'
ORDER BY [日期]
"""
df2 = pd.read_sql_query(q2, c)
print(df2.to_string(index=False))
print()
print("  验证：implied_price 应 ≈ 收盘价")

print()
print("=" * 76)
print("3. 与 tushare 原始值对比")
print("=" * 76)
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)

tok = os.environ.get("TUSHARE_TOKEN")
if not tok:
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        tok = p.read_text(encoding="utf-8").strip()

if tok:
    import tushare as ts
    pro = ts.pro_api(tok)
    t = pro.daily(trade_date="20241014")
    if t is not None and not t.empty:
        t1 = t[t["ts_code"] == "000001.SZ"]
        if not t1.empty:
            r = t1.iloc[0]
            print(f"  tushare 原始:")
            print(f"    vol={r['vol']} (手)  amount={r['amount']} (千元)")
            print(f"    -> 应为 成交量={r['vol']*100:,.0f} 股, 成交额={r['amount']*1000:,.0f} 元")
            db_row = pd.read_sql_query(
                "SELECT [成交量],[成交额] FROM stock_data "
                "WHERE stock_code='000001' AND [日期]='2024-10-14'", c).iloc[0]
            print(f"  数据库:")
            print(f"    成交量={db_row['成交量']:,.0f}  成交额={db_row['成交额']:,.0f}")
            ok1 = abs(db_row["成交量"] - r["vol"] * 100) < 1
            ok2 = abs(db_row["成交额"] - r["amount"] * 1000) < 100
            print(f"  匹配: 成交量={'OK' if ok1 else 'FAIL'}  成交额={'OK' if ok2 else 'FAIL'}")

c.close()
