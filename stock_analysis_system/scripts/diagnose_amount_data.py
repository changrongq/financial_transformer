"""诊断 stock_data 成交额数据的完整问题，规划修复方案。

要查清：
  1. 成交额字段的空洞范围
  2. close × volume 这个兜底值到底错在哪（单位？还是 vol 本身有问题？）
  3. tushare daily 的 amount 字段单位是什么
  4. 需要补多少天的数据
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)


def get_token():
    t = os.environ.get("TUSHARE_TOKEN")
    if t:
        return t.strip()
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


db = str((ROOT.parent / "stock_history.db").resolve())
c = sqlite3.connect(db)

print("=" * 76)
print("1. 成交额空洞范围")
print("=" * 76)
q = """
SELECT [日期] AS d,
  COUNT(*) AS n,
  SUM(CASE WHEN [成交额] IS NOT NULL AND [成交额] > 0 THEN 1 ELSE 0 END) AS amt_ok,
  SUM(CASE WHEN [成交量] IS NOT NULL AND [成交量] > 0 THEN 1 ELSE 0 END) AS vol_ok
FROM stock_data GROUP BY [日期] ORDER BY [日期]
"""
g = pd.read_sql_query(q, c)
g["amt_pct"] = g["amt_ok"] / g["n"] * 100
g["vol_pct"] = g["vol_ok"] / g["n"] * 100

# 分月汇总
g["date"] = pd.to_datetime(g["d"], errors="coerce")
g["ym"] = g["date"].dt.to_period("M")
m = g.groupby("ym").agg(
    days=("d", "size"),
    amt_pct=("amt_pct", "mean"),
    vol_pct=("vol_pct", "mean"),
    avg_n=("n", "mean"),
)
print(f"{'月份':<12}{'交易日':>8}{'成交额有值%':>14}{'成交量有值%':>14}{'平均股票数':>12}")
print("-" * 60)
for ym, r in m.iterrows():
    flag = ""
    if r["amt_pct"] < 50:
        flag = "  <-- 空洞"
    print(f"{str(ym):<12}{int(r['days']):>8}{r['amt_pct']:>13.1f}%{r['vol_pct']:>13.1f}%"
          f"{r['avg_n']:>12.0f}{flag}")

print()
print(f"总计: {len(g)} 个交易日")
print(f"  成交额完全为空的日子: {int((g['amt_ok'] == 0).sum())}")
print(f"  成交额部分缺失的日子: {int(((g['amt_ok'] > 0) & (g['amt_pct'] < 99)).sum())}")

print()
print("=" * 76)
print("2. close × volume 兜底值与真实成交额的比值")
print("=" * 76)
# 找有真实成交额的日子，算 ratio
df = pd.read_sql_query(
    "SELECT [日期] AS d, [收盘] AS close, [成交量] AS vol, [成交额] AS amt "
    "FROM stock_data WHERE [成交额] > 0 AND [成交量] > 0 LIMIT 200000", c)
df["calc"] = df["close"] * df["vol"]
df["ratio"] = df["amt"] / df["calc"]
print(f"  样本: {len(df):,} 行")
print(f"  ratio (真实成交额 / close×vol):")
print(f"    中位数: {df['ratio'].median():.4f}")
print(f"    均值:   {df['ratio'].mean():.4f}")
print(f"    P10:    {df['ratio'].quantile(0.1):.4f}")
print(f"    P90:    {df['ratio'].quantile(0.9):.4f}")
print()
if 0.95 < df["ratio"].median() < 1.05:
    print("  → 中位数≈1，说明【成交量单位是股】，close×vol 单位是【元】")
    print("    那兜底值本身是对的，问题在别处")
else:
    print(f"  → 中位数偏离 1，说明单位不匹配")

print()
print("=" * 76)
print("3. tushare daily 的 amount 字段单位")
print("=" * 76)
tok = get_token()
if tok:
    try:
        import tushare as ts
        pro = ts.pro_api(tok)
        # 拉一个最近的交易日
        t = pro.daily(trade_date="20261028")
        if t is not None and not t.empty:
            print(f"  样本: {len(t)} 只股票")
            print(f"  列: {list(t.columns)}")
            print(f"\n  前3行:")
            print(t[["ts_code", "close", "vol", "amount"]].head(3).to_string(index=False))
            t["calc"] = t["close"] * t["vol"]
            t["ratio"] = t["amount"] / t["calc"]
            print(f"\n  amount / (close × vol):")
            print(f"    中位数: {t['ratio'].median():.4f}")
            print(f"    → tushare amount 单位应为【千元】（ratio≈0.001 时）")
    except Exception as e:
        print(f"  查询失败: {e}")
else:
    print("  无 token")

print()
print("=" * 76)
print("4. 修复方案估算")
print("=" * 76)
need_days = int((g["amt_pct"] < 99).sum())
print(f"  需要补的交易日: {need_days}")
print(f"  按每交易日 1 次 API 调用、限速约 500 次/分钟估算: 约 {need_days/500:.0f} 分钟")
print(f"  数据量: 约 {need_days * 5400:,} 行")

c.close()
