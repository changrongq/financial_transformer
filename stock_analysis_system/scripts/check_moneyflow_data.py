# -*- coding: utf-8 -*-
"""检查资金流数据表是否有数据，以及覆盖范围。

生产路径（data.py:_load_trade_data_with_concepts）：
    116: if not moneyflow_df.empty:
    121:     df = df.merge(moneyflow_df, on=["stock_code","date"], how="left")

如果 stock_moneyflow 为空 → 不 merge → 后续 build_layer_features
第 76 行 `pd.concat([pd.to_numeric(df.get("净流入额", np.nan)), ...])`
会因为 np.nan 是标量而抛 TypeError。

用法：
  python scripts/check_moneyflow_data.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())


def main():
    con = sqlite3.connect(DB)
    print("=" * 90)
    print("资金流数据（stock_moneyflow）检查")
    print("=" * 90)

    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"\n数据库表: {tables}")

    for t in ["stock_moneyflow", "stock_top_inst", "stock_margin_detail",
              "stock_basic_info"]:
        print(f"\n[{t}]")
        if t not in tables:
            print("  表不存在")
            continue
        cols = [c[1] for c in con.execute(f"PRAGMA table_info({t})")]
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  行数: {n:,}")
        print(f"  列: {cols[:15]}{'...' if len(cols) > 15 else ''}")
        if n > 0:
            # 日期范围
            date_col = "日期" if "日期" in cols else (
                "date" if "date" in cols else None)
            if date_col:
                r = con.execute(
                    f"SELECT MIN({date_col}), MAX({date_col}), "
                    f"COUNT(DISTINCT stock_code) FROM {t}").fetchone()
                print(f"  范围: {r[0]} ~ {r[1]}  "
                      f"股票 {r[2]:,} 只")

    # 检查"净流入额"等列名是否存在
    print("\n" + "=" * 90)
    print("关键列名检查（build_layer_features 需要的）")
    print("=" * 90)
    need = ["净流入额", "大单买入额", "大单卖出额", "特大单买入额",
            "特大单卖出额", "小单买入额", "小单卖出额", "中单买入额",
            "中单卖出额"]
    if "stock_moneyflow" in tables:
        cols = [c[1] for c in con.execute("PRAGMA table_info(stock_moneyflow)")]
        print(f"\n  stock_moneyflow 实际列: {cols}")
        for c in need:
            print(f"    {c:<14} {'✓ 存在' if c in cols else '✗ 缺失'}")
    else:
        print("  stock_moneyflow 表不存在")
        for c in need:
            print(f"    {c:<14} ✗ 缺失")

    con.close()


if __name__ == "__main__":
    main()
