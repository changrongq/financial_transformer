# -*- coding: utf-8 -*-
"""抽样验证补入的历史资金流数据质量。"""
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

print("=" * 92)
print("历史资金流数据抽样验证")
print("=" * 92)

# 抽查不同年份
samples = ["2021-11-15", "2022-06-15", "2023-06-15", "2024-06-17",
           "2025-06-16", "2026-06-15"]

print(f"\n{'日期':<12}{'本地行数':>9}{'tushare':>9}{'交集':>7}"
      f"{'净流入额比值':>14}{'小单买入额比值':>16}")
for d in samples:
    d_ts = d.replace("-", "")
    try:
        raw = pro.moneyflow(trade_date=d_ts)
    except Exception as e:
        print(f"  {d}: tushare 失败 {e}")
        continue
    if raw is None or raw.empty:
        print(f"  {d}: tushare 无数据")
        continue
    raw["stock_code"] = raw["ts_code"].str.split(".").str[0]

    loc = pd.read_sql_query("""
        SELECT stock_code, "净流入额" AS net, "小单买入额" AS sm_buy
        FROM stock_moneyflow WHERE 日期 = ?
    """, con, params=(d,))

    m = raw.merge(loc, on="stock_code", how="inner")
    if len(m) < 50:
        print(f"  {d}: 交集不足 {len(m)}")
        continue

    r1 = (m["net_mf_amount"] / m["net"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan).dropna()
    r2 = (m["buy_sm_amount"] / m["sm_buy"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan).dropna()

    print(f"  {d:<12}{len(loc):>9}{len(raw):>9}{len(m):>7}"
          f"{r1.median():>14.6f}{r2.median():>16.6f}")

# 一致性统计
print("\n" + "=" * 92)
print("全表一致性：净流入额 vs (买入-卖出) 应相等")
print("=" * 92)
r = con.execute("""
    SELECT COUNT(*),
           SUM(CASE WHEN "净流入额" IS NULL THEN 1 ELSE 0 END)
    FROM stock_moneyflow
""").fetchone()
print(f"  总行 {r[0]:,}  净流入额空 {r[1] or 0:,} "
      f"({(r[1] or 0)/r[0]*100:.2f}%)")

# 抽 10 万行检查 净流入额 ≈ 全部四类单的买入总额 - 卖出总额
# 注意：tushare 的 net_mf_amount 定义是"净流入额"，等于
#   (小单买 + 中单买 + 大单买 + 特大单买)
# - (小单卖 + 中单卖 + 大单卖 + 特大单卖)
# 我上一版只用了大单+特大单，所以算出来"99.73% 不符"——那是我的公式错。
df = pd.read_sql_query("""
    SELECT "小单买入额" sb, "中单买入额" mb, "大单买入额" lb,
           "特大单买入额" eb, "小单卖出额" ss, "中单卖出额" ms,
           "大单卖出额" ls, "特大单卖出额" es,
           "净流入额" net
    FROM stock_moneyflow WHERE "净流入额" IS NOT NULL LIMIT 100000
""", con)
calc = (df["sb"] + df["mb"] + df["lb"] + df["eb"]
        - df["ss"] - df["ms"] - df["ls"] - df["es"])
diff = (calc - df["net"]).abs()
rel = diff / df["net"].abs().replace(0, np.nan)
print(f"\n  抽查 {len(df):,} 行（四类单全口径）")
print(f"  |(全部买入 - 全部卖出) - 净流入额| 中位: {diff.median():.6f}")
print(f"  绝对误差 > 1 元的比例: {(diff > 1).mean()*100:.2f}%")
print(f"  相对误差 > 1% 的比例: {(rel > 0.01).mean()*100:.2f}%")

# 单独看大单+特大单（这两种是"主力"口径，与净流入额不是同一个东西）
big = (df["lb"] + df["eb"] - df["ls"] - df["es"])
print(f"\n  参考：主力净额(大单+特大单) vs 净流入额")
print(f"    相关系数: {np.corrcoef(big, df['net'])[0,1]:.6f}")
print(f"    （不相等是正常的，净流入额含中小单）")

# 逐年统计
print("\n" + "=" * 92)
print("逐年数据量")
print("=" * 92)
rows = con.execute("""
    SELECT substr(日期,1,4) y, COUNT(DISTINCT 日期) days, COUNT(*) rows,
           COUNT(DISTINCT stock_code) codes
    FROM stock_moneyflow GROUP BY y ORDER BY y
""").fetchall()
print(f"  {'年':<6}{'交易日':>8}{'行数':>12}{'股票':>8}")
for y, d, n, c in rows:
    print(f"  {y:<6}{d:>8}{n:>12,}{c:>8,}")

con.close()
