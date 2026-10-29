"""拉取美股/港股数据，用于验证「隔夜负、日内正」是否全球普适。

注意：tushare 的 us_daily / hk_daily 限速 1 次/分钟，务必耐心等待。

标的：
  美股（用 ETF 代理指数，流动性好、历史长）
    SPY  标普500ETF
    DIA  道指ETF
    QQQ  纳斯达克100ETF
    IWM  罗素2000ETF（小盘）
    EEM  新兴市场ETF
    EFA  发达国家ETF
  港股
    02800.HK 盈富基金（恒生指数ETF）
    00700.HK 腾讯

存表：
  global_daily(ts_code, market, trade_date, open/high/low/close, pre_close, vol, amount)

用法：
  python scripts/fetch_global_daily.py
  python scripts/fetch_global_daily.py --only us
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

US_CODES = [
    ("SPY", "标普500ETF"),
    ("DIA", "道指ETF"),
    ("QQQ", "纳斯达克100ETF"),
    ("IWM", "罗素2000ETF(小盘)"),
    ("EEM", "新兴市场ETF"),
    ("EFA", "发达国家ETF"),
]

HK_CODES = [
    ("02800.HK", "盈富基金(恒指ETF)"),
    ("00700.HK", "腾讯控股"),
    ("00005.HK", "汇丰控股"),
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


def get_token() -> str:
    t = os.environ.get("TUSHARE_TOKEN")
    if t:
        return t.strip()
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("无 token")


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
    rename = {"pct_change": "pct_chg", "change": "chg"}
    d = d.rename(columns=rename)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["us", "hk", "all"], default="all")
    parser.add_argument("--wait", type=int, default=62, help="每次请求间隔秒（限速1次/分钟）")
    args = parser.parse_args()

    for k in list(os.environ):
        if "PROXY" in k.upper():
            os.environ.pop(k, None)

    import tushare as ts
    pro = ts.pro_api(get_token())

    db = _resolve_db()
    conn = sqlite3.connect(db)
    ensure_table(conn)

    start = "20040101"
    end = datetime.now().strftime("%Y%m%d")

    tasks = []
    if args.only in ("us", "all"):
        tasks += [("us_daily", c, n, "US") for c, n in US_CODES]
    if args.only in ("hk", "all"):
        tasks += [("hk_daily", c, n, "HK") for c, n in HK_CODES]

    print(f"共 {len(tasks)} 个标的，每个间隔 {args.wait}s，预计 {len(tasks)*args.wait/60:.0f} 分钟\n")

    for i, (fn, code, name, market) in enumerate(tasks):
        for attempt in range(3):
            try:
                f = getattr(pro, fn)
                df = f(ts_code=code, start_date=start, end_date=end)
                n = upsert(conn, code, market, df)
                if n:
                    d = df.sort_values("trade_date")
                    print(f"[{i+1}/{len(tasks)}] {code:<10} {name:<20} [OK] {n:>5} 行  "
                          f"{d['trade_date'].min()} ~ {d['trade_date'].max()}")
                else:
                    print(f"[{i+1}/{len(tasks)}] {code:<10} {name:<20} [--] 空")
                break
            except Exception as e:
                msg = str(e)
                if "频率超限" in msg or "每分钟" in msg:
                    print(f"[{i+1}/{len(tasks)}] {code:<10} 限速，等待 {args.wait}s...")
                    time.sleep(args.wait)
                    continue
                print(f"[{i+1}/{len(tasks)}] {code:<10} [X] {type(e).__name__}: {msg[:80]}")
                break
        if i < len(tasks) - 1:
            time.sleep(args.wait)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    r = pd.read_sql_query(
        "SELECT ts_code, market, COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b, "
        "ROUND(COUNT(*)/244.0,1) years FROM global_daily GROUP BY ts_code ORDER BY market, n DESC",
        conn)
    print(r.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
