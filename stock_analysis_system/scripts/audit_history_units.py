# -*- coding: utf-8 -*-
"""全表单位审计：vol 是手还是股，amount 是元还是千元。

核心事实（已确认）：
  tushare daily: vol = 手（100 股），amount = 千元
  本地 2022-11-01 000001: vol=1,381,234（= tushare 原值，手）
                          amount=1,459,758.4（= tushare 原值，千元）

  所以 ratio = amount/(vol×close) = 千元/(手×元)
             = 1000元 / (100股 × 元/股)
             = 10
  但实测是 0.0990 ...

  等一下：千元/(手×元) = 1000/(100×1) = 10，不是 0.099。
  需要重新算：amount=1,459,758.4 千元 = 1,459,758,400 元
  vol=1,381,234 手 = 138,123,400 股
  成交量×价 = 138,123,400 × 10.67 = 1,473,776,678 元
  amount/成交额 = 1,459,758,400 / 1,473,776,678 = 0.9905  ← 接近 1

  所以：
    若 vol 解释为「股」：amount是千元 → ratio = 0.0990
    若 vol 解释为「手」：amount是千元 → ratio = 0.9905

  实测 0.0990 说明脚本把它当「股」算了，而真实 vol 单位是手。
  结论：ratio 判据本身有歧义，不能用来判断单位。

本脚本用绝对量级判据：
  成交额(元) / (成交量(股) × 收盘价) 应该 ≈ 1
  即 amount_元 / (vol_手 × 100 × close) ≈ 1
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())
con = sqlite3.connect(DB)

tok = (ROOT / "config" / "tushare_token.local").read_text(encoding="utf-8").strip()
import tushare as ts
ts.set_token(tok)
pro = ts.pro_api()

print("=" * 96)
print("全表单位审计：用绝对量级 + tushare 原值比对")
print("=" * 96)

# 选取跨年份的样本
samples = [
    ("000001.SZ", "20211115", "2021-11-15"),
    ("000001.SZ", "20221115", "2022-11-15"),
    ("000001.SZ", "20230615", "2023-06-15"),
    ("000001.SZ", "20240617", "2024-06-17"),
    ("000001.SZ", "20250616", "2025-06-16"),
    ("000001.SZ", "20261015", "2026-10-15"),
]

print(f"\n  {'date':<12}{'t_vol(手)':>13}{'l_vol':>15}{'vol比':>8}"
      f"{'t_amt(千元)':>15}{'l_amt':>16}{'amt比':>9}{'l_amt元/额':>13}")

for ts_code, tsd, d in samples:
    code = ts_code.split(".")[0]
    raw = pro.daily(ts_code=ts_code, trade_date=tsd,
                    fields="ts_code,trade_date,close,vol,amount")
    if raw is None or raw.empty:
        print(f"  {d}: tushare 无数据")
        continue
    r = raw.iloc[0]
    loc = con.execute("""
        SELECT 收盘, 成交量, 成交额 FROM stock_data
        WHERE stock_code=? AND 日期=?
    """, (code, d)).fetchone()
    if not loc:
        print(f"  {d}: 本地无数据")
        continue
    lc, lv, la = loc
    vol_ratio = lv / r["vol"] if r["vol"] else np.nan
    amt_ratio = la / r["amount"] if r["amount"] else np.nan
    # 假设本地 vol 是手、amount 是千元时，隐含成交额与「股×价」的比
    implied = (la * 1000) / (lv * 100 * lc) if lv and lc else np.nan
    print(f"  {d:<12}{r['vol']:>13,.0f}{lv:>15,.0f}{vol_ratio:>8.3f}"
          f"{r['amount']:>15,.1f}{la:>16,.1f}{amt_ratio:>9.3f}"
          f"{implied:>13.4f}")

print("\n" + "=" * 96)
print("判读")
print("=" * 96)
print("""
  vol比 ≈ 1.0   → 本地 vol 与 tushare 同单位（手）
  amt比 ≈ 1.0   → 本地 amount 与 tushare 同单位（千元）
  l_amt元/额 ≈ 1.0 → 假设「vol=手, amount=千元」自洽

  若某年份 l_amt元/额 ≈ 1000，说明该年 amount 单位是「元」而非「千元」。
""")

# 按年份统计 implied
print("\n[按年份统计 l_amt元/(vol股×close) 的中位数]")
for y in ["2021", "2022", "2023", "2024", "2025", "2026"]:
    rows = con.execute("""
        SELECT 收盘, 成交量, 成交额 FROM stock_data
        WHERE 日期 LIKE ? AND 收盘>0 AND 成交量>0 AND 成交额>0
        LIMIT 200000
    """, (y + "%",)).fetchall()
    if not rows:
        print(f"  {y}: 无数据")
        continue
    arr = np.array([(a * 1000) / (v * 100 * c) for c, v, a in rows])
    print(f"  {y}: n={len(arr):>7,}  中位 {np.median(arr):>10.4f}  "
          f"p5 {np.percentile(arr,5):>8.4f}  p95 {np.percentile(arr,95):>8.4f}")

con.close()
