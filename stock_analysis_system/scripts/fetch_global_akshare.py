"""用 akshare 拉取美股/港股数据（绕开 tushare 的严格限速）。

akshare 是免费数据源，港股/美股都能拿，限速宽松。

用法：
  python scripts/fetch_global_akshare.py
  python scripts/fetch_global_akshare.py --only us
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)

# akshare 美股代码（不带交易所后缀）
US = [
    ("SPY", "标普500ETF"),
    ("DIA", "道指ETF"),
    ("QQQ", "纳斯达克100ETF"),
    ("IWM", "罗素2000ETF"),
    ("EEM", "新兴市场ETF"),
    ("EFA", "发达国家ETF"),
]

# akshare 港股代码（5位数字）
HK = [
    ("02800", "盈富基金(恒指ETF)"),
    ("00700", "腾讯控股"),
    ("00005", "汇丰控股"),
    ("02318", "中国平安H"),
]


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


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS global_daily (
            ts_code TEXT NOT NULL,
            market TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            pre_close REAL, chg REAL, pct_chg REAL,
            vol REAL, amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_global_date ON global_daily(trade_date);
    """)
    conn.commit()


def upsert(conn, code, market, df) -> int:
    if df is None or df.empty:
        return 0
    d = df.copy()
    cols = ["trade_date", "open", "high", "low", "close",
            "pre_close", "chg", "pct_chg", "vol", "amount"]
    for c in cols:
        if c not in d.columns:
            d[c] = None
    d = d[cols]
    rows = [(code, market) + tuple(r) for r in d.itertuples(index=False, name=None)]
    conn.executemany(
        "INSERT OR REPLACE INTO global_daily "
        "(ts_code, market, trade_date, open, high, low, close, pre_close, chg, pct_chg, vol, amount) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def norm_us(df: pd.DataFrame) -> pd.DataFrame:
    """akshare 美股：date, open, high, low, close, volume, amount"""
    d = df.rename(columns={
        "date": "trade_date", "volume": "vol",
        "open": "open", "high": "high", "low": "low", "close": "close",
    })
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.strftime("%Y%m%d")
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("trade_date").reset_index(drop=True)
    d["pre_close"] = d["close"].shift(1)
    d["chg"] = d["close"] - d["pre_close"]
    d["pct_chg"] = d["close"] / d["pre_close"] - 1
    return d


def norm_hk(df: pd.DataFrame) -> pd.DataFrame:
    """akshare 港股（新版返回英文列：date/open/high/low/close/volume/amount）。"""
    d = df.rename(columns={
        "日期": "trade_date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "vol", "成交额": "amount",
        "date": "trade_date", "volume": "vol",
    })
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.strftime("%Y%m%d")
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("trade_date").reset_index(drop=True)
    d["pre_close"] = d["close"].shift(1)
    d["chg"] = d["close"] - d["pre_close"]
    d["pct_chg"] = d["close"] / d["pre_close"] - 1
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["us", "hk", "all"], default="all")
    args = parser.parse_args()

    import akshare as ak

    db = _resolve_db()
    conn = sqlite3.connect(db)
    ensure_table(conn)

    start = "2004-01-01"

    if args.only in ("us", "all"):
        print("=" * 70)
        print("美股")
        print("=" * 70)
        for sym, name in US:
            try:
                df = ak.stock_us_daily(symbol=sym, adjust="")
                if df is None or df.empty:
                    print(f"  {sym:<6} {name:<16} [--] 空")
                    continue
                df = df[df["date"].astype(str) >= start]
                d = norm_us(df)
                n = upsert(conn, sym, "US", d)
                print(f"  {sym:<6} {name:<16} [OK] {n:>5} 行  "
                      f"{d['trade_date'].min()} ~ {d['trade_date'].max()}")
            except Exception as e:
                print(f"  {sym:<6} {name:<16} [X] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(1.5)

    if args.only in ("hk", "all"):
        print()
        print("=" * 70)
        print("港股")
        print("=" * 70)
        for code, name in HK:
            try:
                df = ak.stock_hk_daily(symbol=code, adjust="")
                if df is None or df.empty:
                    print(f"  {code} {name:<16} [--] 空")
                    continue
                date_col = "日期" if "日期" in df.columns else "date"
                df = df[df[date_col].astype(str) >= start]
                d = norm_hk(df)
                n = upsert(conn, f"{code}.HK", "HK", d)
                print(f"  {code} {name:<16} [OK] {n:>5} 行  "
                      f"{d['trade_date'].min()} ~ {d['trade_date'].max()}")
            except Exception as e:
                print(f"  {code} {name:<16} [X] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(1.5)

    print()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    r = pd.read_sql_query(
        "SELECT ts_code, market, COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b "
        "FROM global_daily GROUP BY ts_code ORDER BY market, n DESC", conn)
    if r.empty:
        print("  （无数据）")
    else:
        r["years"] = (r["n"] / 244).round(1)
        print(r.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
