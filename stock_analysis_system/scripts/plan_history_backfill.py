# -*- coding: utf-8 -*-
"""规划 5 年历史数据拉取。

现状：
  stock_data    : 2024-10-08 ~ 2026-10-29（479 天，2 年）
  historical_daily: 277 只行业股，1852 天（更早但覆盖窄）

目标：
  把全市场拉到 5 年（2021-11 ~ 2026-10），约 1200 个交易日

先确认：
  1. tushare 单只股票能拉多长历史
  2. 需要多少行（数据量估算）
  3. 接口限速与耗时
  4. 写入策略（新表 or 原地扩展）

用法：
  python scripts/plan_history_backfill.py
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve_db() -> str:
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            with cfg.open("r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            raw = c.get("databases", {}).get("trade_db")
            if raw:
                p = Path(raw)
                if not p.is_absolute():
                    p = ROOT / p
                if p.exists():
                    return str(p)
        except Exception:
            pass
    return str((ROOT.parent / "stock_history.db").resolve())


def load_token():
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return os.environ.get("TUSHARE_TOKEN")


def main():
    db = resolve_db()
    con = sqlite3.connect(db)

    print("=" * 90)
    print("5 年历史数据拉取规划")
    print("=" * 90)

    # ---- 1. 现状 ----
    print("\n[1] 现有数据")
    for table in ["stock_data", "historical_daily", "etf_daily", "index_daily"]:
        try:
            r = con.execute(f"""
                SELECT COUNT(*), COUNT(DISTINCT stock_code),
                       MIN(日期), MAX(日期)
                FROM {table}
            """).fetchone()
            print(f"  {table:<18} {r[0]:>10,} 行  {r[1]:>5} 只  "
                  f"{r[2]} ~ {r[3]}")
        except Exception:
            try:
                r = con.execute(f"""
                    SELECT COUNT(*), COUNT(DISTINCT ts_code),
                           MIN(trade_date), MAX(trade_date)
                    FROM {table}
                """).fetchone()
                print(f"  {table:<18} {r[0]:>10,} 行  {r[1]:>5} 只  "
                      f"{r[2]} ~ {r[3]}")
            except Exception as e:
                print(f"  {table:<18} {e}")

    # ---- 2. tushare 能拉多长 ----
    print("\n[2] tushare 历史深度测试")
    token = load_token()
    if not token:
        print("  无 token")
        con.close()
        return 1

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    for start in ["20150101", "20180101", "20210101"]:
        t0 = time.time()
        try:
            df = pro.daily(ts_code="000001.SZ", start_date=start,
                           end_date="20261031",
                           fields="ts_code,trade_date,close,pre_close,vol,amount")
            el = time.time() - t0
            if df is not None and not df.empty:
                print(f"  start={start}: {len(df):>5} 行  "
                      f"{df['trade_date'].min()} ~ {df['trade_date'].max()}  "
                      f"耗时 {el:.2f}s")
            else:
                print(f"  start={start}: 空")
        except Exception as e:
            print(f"  start={start}: 失败 {e}")

    # ---- 3. 数据量估算 ----
    print("\n[3] 数据量估算")
    n_stocks = con.execute(
        "SELECT COUNT(DISTINCT stock_code) FROM stock_data").fetchone()[0]
    rows_per_stock_5y = 1215   # 5 年约 243 交易日/年
    est_rows = n_stocks * rows_per_stock_5y
    current = con.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
    print(f"  股票数        : {n_stocks:,}")
    print(f"  5 年目标行数  : ≈ {est_rows:,}")
    print(f"  现有行数      : {current:,}")
    print(f"  需新增        : ≈ {est_rows - current:,}")
    print(f"  数据库估算增长: ≈ {(est_rows - current) * 400 / 1e9:.2f} GB")

    # ---- 4. 耗时估算 ----
    print("\n[4] 耗时估算（按实测 0.064s/只 @ 4 并发）")
    for workers, per in [(4, 0.256), (4, 0.064), (6, 0.05)]:
        print(f"  {workers} 并发, {per:.3f}s/只: "
              f"{n_stocks * per / 60:.1f} 分钟")

    # ---- 5. 历史股票清单 ----
    print("\n[5] 需要拉取的股票范围")
    r = con.execute("""
        SELECT SUBSTR(stock_code,1,2) p, COUNT(DISTINCT stock_code) n
        FROM stock_data GROUP BY p ORDER BY n DESC
    """).fetchall()
    total = 0
    for p, n in r:
        print(f"    {p}xxxx: {n:>5} 只")
        total += n
    print(f"    合计  : {total:>5} 只")

    # 注意：只拉现有股票会漏掉已退市/暂停上市的
    print("\n  注意：只拉现有 {n:,} 只会漏掉 2024-10 之前退市的股票。"
          .format(n=n_stocks))
    print("  若需要完整历史（含退市），需先拉 stock_basic 的 list_status=L/D/P")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
