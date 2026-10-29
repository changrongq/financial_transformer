"""拉取长周期 ETF / 指数日线数据（10-20 年），存入数据库。

支撑船长的需求：
  - 把择时标的从「全市场等权」换成真实可交易的 ETF（510300）
  - 把回测区间从 2 年延长到 10-15 年
  - 数据量从 478 天提升到 3500+ 天（对 RL 至关重要）

存储：
  - 新表 etf_daily：ETF 日线（ts_code, trade_date, open/high/low/close, vol, amount）
  - 复用 index_daily_sh 的思路，新建 index_daily（多指数，含 OHLC）

用法：
  python scripts/fetch_etf_history.py --years 20
  python scripts/fetch_etf_history.py --codes 510300.SH,510500.SH
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _resolve_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def get_token() -> str:
    tok = os.environ.get("TUSHARE_TOKEN")
    if tok:
        return tok.strip()
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("找不到 tushare token")


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS etf_daily (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            pre_close REAL, change REAL, pct_chg REAL,
            vol REAL, amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_etf_daily_date ON etf_daily(trade_date);

        CREATE TABLE IF NOT EXISTS index_daily (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            pre_close REAL, change REAL, pct_chg REAL,
            vol REAL, amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily(trade_date);
        """
    )
    conn.commit()


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount"]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    d = d[cols]
    ph = ",".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({ph})"
    conn.executemany(sql, d.itertuples(index=False, name=None))
    conn.commit()
    return len(d)


def main() -> None:
    parser = argparse.ArgumentParser(description="拉取长周期 ETF/指数日线")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--codes", default="510300.SH,510500.SH,159915.SZ,510050.SH,512100.SH,588000.SH")
    parser.add_argument("--index-codes", default="000300.SH,000905.SH,000001.SH,399006.SZ,000852.SH")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-etf", action="store_true")
    args = parser.parse_args()

    for k in list(os.environ):
        if "PROXY" in k.upper():
            os.environ.pop(k, None)

    import tushare as ts
    pro = ts.pro_api(get_token())

    start = f"{datetime.now().year - args.years}0101"
    end = datetime.now().strftime("%Y%m%d")
    print(f"拉取区间: {start} ~ {end}（{args.years} 年）")

    db = _resolve_db()
    conn = sqlite3.connect(db)
    ensure_tables(conn)

    if not args.skip_etf:
        print("\n" + "=" * 68)
        print("ETF 日线")
        print("=" * 68)
        for code in [c.strip() for c in args.codes.split(",") if c.strip()]:
            try:
                df = pro.fund_daily(ts_code=code, start_date=start, end_date=end)
                n = upsert(conn, "etf_daily", df)
                if n:
                    d = df.sort_values("trade_date")
                    print(f"  {code:<12} [OK] {n:>5} 行  {d['trade_date'].min()} ~ {d['trade_date'].max()}")
                else:
                    print(f"  {code:<12} [--] 无数据")
            except Exception as e:
                print(f"  {code:<12} [X] {type(e).__name__}: {str(e)[:80]}")
            time.sleep(0.3)

    if not args.skip_index:
        print("\n" + "=" * 68)
        print("指数日线")
        print("=" * 68)
        for code in [c.strip() for c in args.index_codes.split(",") if c.strip()]:
            try:
                df = pro.index_daily(ts_code=code, start_date=start, end_date=end)
                n = upsert(conn, "index_daily", df)
                if n:
                    d = df.sort_values("trade_date")
                    print(f"  {code:<12} [OK] {n:>5} 行  {d['trade_date'].min()} ~ {d['trade_date'].max()}")
                else:
                    print(f"  {code:<12} [--] 无数据")
            except Exception as e:
                print(f"  {code:<12} [X] {type(e).__name__}: {str(e)[:80]}")
            time.sleep(0.3)

    print("\n" + "=" * 68)
    print("数据库汇总")
    print("=" * 68)
    for t in ["etf_daily", "index_daily"]:
        try:
            r = pd.read_sql_query(
                f"SELECT ts_code, COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b "
                f"FROM {t} GROUP BY ts_code ORDER BY n DESC", conn)
            print(f"\n  {t}:")
            print(r.to_string(index=False))
        except Exception as e:
            print(f"  {t}: {e}")

    conn.close()
    print(f"\n数据库: {db}")


if __name__ == "__main__":
    main()
