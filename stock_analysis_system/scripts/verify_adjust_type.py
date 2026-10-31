# -*- coding: utf-8 -*-
"""确认 stock_data 的 close 是什么复权口径。

关键发现：
  000001 在 2025-06-11 本地 close=11.49，tushare daily close=11.85
  ratio = 1.0313 恒定不变（29 行完全相同的比例）
  → 这正是【前复权因子】

如果是前复权：
  · 所有历史价格都按当前基准调整
  · 除权日前后的 close 连续，pct_change 是对的
  · 那么我们其实不需要 tushare 的 pre_close！

本脚本验证：
  1. ratio 是否在除权日之后恒定（前复权特征）
  2. 本地 close.pct_change() 是否在除权日平滑
  3. 对比"用 close.shift(1) 当 pre_close" 的效果

用法：
  python scripts/verify_adjust_type.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())

tok = (ROOT / "config" / "tushare_token.local").read_text(encoding="utf-8").strip()
import tushare as ts
ts.set_token(tok)
pro = ts.pro_api()


def check(code: str, con) -> dict:
    ts_code = code + (".SH" if code.startswith("6") else ".SZ")
    try:
        raw = pro.daily(ts_code=ts_code, start_date="20241008",
                        end_date="20261031",
                        fields="ts_code,trade_date,close,pre_close")
    except Exception as e:
        return {"code": code, "error": str(e)}

    raw["date"] = raw["trade_date"].astype(str).apply(
        lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    loc = pd.read_sql_query(
        "SELECT 日期 AS date, 收盘 AS close_local FROM stock_data "
        "WHERE stock_code=? ORDER BY 日期", con, params=(code,))
    m = raw.merge(loc, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if len(m) < 50:
        return {"code": code, "n": len(m), "note": "too few"}

    m["ratio"] = m["close"] / m["close_local"]
    ratio_uniq = m["ratio"].round(4).nunique()

    # 本地 close 的单日收益
    m["ret_local"] = m["close_local"].pct_change()
    # tushare 未复权 close 的单日收益
    m["ret_raw"] = m["close"].pct_change()

    n_ext_local = int((m["ret_local"] < -0.11).sum())
    n_ext_raw = int((m["ret_raw"] < -0.11).sum())

    # ratio 变化点（=除权日）
    ratio_chg = m["ratio"].pct_change()
    div_days = m.loc[ratio_chg.abs() > 0.005, "date"].tolist()

    return {
        "code": code, "n": len(m),
        "ratio_unique": int(ratio_uniq),
        "ratio_min": float(m["ratio"].min()),
        "ratio_max": float(m["ratio"].max()),
        "ratio_std": float(m["ratio"].std()),
        "ext_local": n_ext_local, "ext_raw": n_ext_raw,
        "div_days": div_days[:5],
    }


def main():
    con = sqlite3.connect(DB)
    codes = ["000001", "000002", "600000", "600036", "000858",
             "002415", "300750", "601318", "600519", "000333"]

    print("=" * 94)
    print("stock_data 的 close 是什么复权口径？")
    print("=" * 94)
    print(f"  {'代码':<8}{'行数':>6}{'ratio种类':>10}{'ratio范围':>20}"
          f"{'本地跌>11%':>11}{'未复权跌>11%':>13}")

    rows = []
    for c in codes:
        r = check(c, con)
        rows.append(r)
        if "error" in r:
            print(f"  {c:<8} ERROR {r['error']}")
            continue
        print(f"  {c:<8}{r['n']:>6}{r['ratio_unique']:>10}"
              f"{r['ratio_min']:>9.4f}~{r['ratio_max']:<9.4f}"
              f"{r['ext_local']:>11}{r['ext_raw']:>13}")

    con.close()

    print("\n" + "=" * 94)
    print("解读")
    print("=" * 94)
    print("""
  如果 ratio 只有少数几个取值，且随时间阶跃变化 →
    本地 close 是【前复权】，ratio 的阶跃点就是除权日。

  关键指标「本地跌>11%」：
    如果本地 close 已是前复权，这个数应该很小（只有真实跌停）
    如果本地 close 是未复权，这个数会很大（每次分红都跌）

  结论决定下一步：
    · 若本地已是前复权 → 不需要 tushare 的 pre_close，
      直接用 close.shift(1) 即可，我们之前的分析并没有被除权污染！
    · 若本地是未复权   → 必须补 pre_close
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
