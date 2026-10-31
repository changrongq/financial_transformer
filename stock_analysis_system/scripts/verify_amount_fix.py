# -*- coding: utf-8 -*-
"""核查 2026-10-29 的成交额是否真的修对了。

现象：
  2026-10-29 单位正确率 100%，但中位比值 1.0123（不是 1.0000）
  说明 amount/(vol*close) 系统性偏高 1.2%

可能原因：
  1. 换算系数用 2026-09-18 校准，但不同日期的成交均价略有不同（正常）
  2. 真的存在单位偏差

关键判据：如果只是"成交均价与收盘价不同"，那 ratio 的分布应该
是围绕 1.0 的对称分布，而不是系统性偏高。

用法：
  python scripts/verify_amount_fix.py
"""

from __future__ import annotations

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

print("=" * 90)
print("2026-10-29 成交额修复核查")
print("=" * 90)

for d_ts, d in [("20261028", "2026-10-28"), ("20261029", "2026-10-29")]:
    raw = pro.daily(trade_date=d_ts,
                    fields="ts_code,close,vol,amount")
    if raw is None or raw.empty:
        print(f"\n{d}: tushare 无数据")
        continue
    raw["stock_code"] = raw["ts_code"].str.split(".").str[0]

    loc = pd.read_sql_query("""
        SELECT stock_code, 收盘 AS c_l, 成交量 AS v_l, 成交额 AS a_l
        FROM stock_data WHERE 日期=?
    """, con, params=(d,))
    m = raw.merge(loc, on="stock_code", how="inner")
    if len(m) < 100:
        print(f"\n{d}: 交集太少 {len(m)}")
        continue

    m["r_c"] = m["close"] / m["c_l"]
    m["r_v"] = m["vol"] / m["v_l"].replace(0, np.nan)
    m["r_a"] = m["amount"] / m["a_l"].replace(0, np.nan)
    # 本地隐含价 vs tushare 隐含价
    m["imp_l"] = m["a_l"] / m["v_l"]
    m["imp_t"] = m["amount"] / m["vol"]
    m["imp_ratio"] = m["imp_l"] / m["imp_t"]

    print(f"\n{d}  交集 {len(m)} 只")
    print(f"  close  比: 中位 {m['r_c'].median():.6f}  "
          f"std {m['r_c'].std():.6f}")
    print(f"  vol    比: 中位 {m['r_v'].median():.6f}  "
          f"std {m['r_v'].std():.6f}")
    print(f"  amount 比: 中位 {m['r_a'].median():.6f}  "
          f"std {m['r_a'].std():.6f}")
    print(f"  隐含价比 (local/tushare): 中位 {m['imp_ratio'].median():.6f}  "
          f"p5 {m['imp_ratio'].quantile(0.05):.6f}  "
          f"p95 {m['imp_ratio'].quantile(0.95):.6f}")

    # 用 tushare 的 amount/vol/close 看它的隐含价偏差
    m["t_ratio"] = m["amount"] / m["vol"] / m["close"]
    print(f"  tushare 自身 amount/(vol*close): 中位 {m['t_ratio'].median():.6f}")

# 对比多个日期，看 1.0082 是否是正常波动
print("\n" + "=" * 90)
print("多个日期的 amount/(vol*close) 中位值对比")
print("=" * 90)
print(f"  {'日期':<14}{'n':>7}{'中位比值':>12}{'p5':>10}{'p95':>10}")

for d in ["2026-09-18", "2026-10-15", "2026-10-20", "2026-10-28",
          "2026-10-29"]:
    rr = con.execute("""
        SELECT 收盘, 成交量, 成交额 FROM stock_data
        WHERE 日期=? AND 收盘>0 AND 成交量>0 AND 成交额>0
    """, (d,)).fetchall()
    if not rr:
        continue
    ratio = np.array([a / v / c for c, v, a in rr])
    print(f"  {d:<14}{len(ratio):>7}{np.median(ratio):>12.6f}"
          f"{np.percentile(ratio,5):>10.6f}{np.percentile(ratio,95):>10.6f}")

con.close()
