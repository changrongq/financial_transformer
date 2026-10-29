"""每日增量拉取 ETF 分钟线（用于未来积累日内交易数据）。

背景：
  做日内策略回测需要 500+ 交易日的分钟数据。
  目前 akshare 只能拿到最近 32 天的 60 分钟线，且每次只能拉近期。
  解决办法：每天跑一次，把当天的分钟线增量存库，慢慢积累。

数据源：
  akshare.fund_etf_hist_min_em（东方财富）
    - period="1"  1分钟（可能被限流）
    - period="5"  5分钟
    - period="15" 15分钟
    - period="30" 30分钟
    - period="60" 60分钟（最稳定）

存储：
  新表 etf_minute(ts_code, freq, trade_time, open/high/low/close/volume/amount)
  主键 (ts_code, freq, trade_time) → 重复拉取会自动覆盖

建议用法（每天收盘后跑）：
  python scripts/collect_etf_minutes_daily.py
  python scripts/collect_etf_minutes_daily.py --freqs 60,30,15

Windows 任务计划（每个交易日 15:30 跑）：
  schtasks /create /tn "CollectETFMinutes" /tr "d:\qcr\Stock\stock_analysis_system\.venv\Scripts\python.exe d:\qcr\Stock\stock_analysis_system\scripts\collect_etf_minutes_daily.py" /sc daily /st 15:30
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

for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)

DEFAULT_ETFS = ["510300", "510500", "159915", "510050"]


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
        CREATE TABLE IF NOT EXISTS etf_minute (
            ts_code TEXT NOT NULL,
            freq TEXT NOT NULL,
            trade_time TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL,
            collected_at TEXT,
            PRIMARY KEY (ts_code, freq, trade_time)
        );
        CREATE INDEX IF NOT EXISTS idx_etf_minute_time ON etf_minute(trade_time);
        CREATE INDEX IF NOT EXISTS idx_etf_minute_code_freq ON etf_minute(ts_code, freq);
    """)
    conn.commit()


def norm(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名。akshare 返回：时间/开盘/收盘/最高/最低/成交量/成交额 或英文。"""
    m = {
        "时间": "trade_time", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "datetime": "trade_time", "date": "trade_time",
    }
    d = df.rename(columns={k: v for k, v in m.items() if k in df.columns})
    if "trade_time" not in d.columns:
        # 尝试用第一列
        d = d.rename(columns={d.columns[0]: "trade_time"})
    d["trade_time"] = pd.to_datetime(d["trade_time"], errors="coerce")
    d = d.dropna(subset=["trade_time"])
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = None
    d["trade_time"] = d["trade_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return d[["trade_time", "open", "high", "low", "close", "volume", "amount"]]


def upsert(conn, code_full: str, freq: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(code_full, freq) + tuple(r) + (now,)
            for r in df.itertuples(index=False, name=None)]
    conn.executemany(
        "INSERT OR REPLACE INTO etf_minute "
        "(ts_code, freq, trade_time, open, high, low, close, volume, amount, collected_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def fetch_minute(ak, code: str, freq: str, retries: int = 3) -> pd.DataFrame | None:
    """多接口备用 + 重试。"""
    attempts = [
        lambda: ak.fund_etf_hist_min_em(symbol=code, period=freq, adjust=""),
        lambda: ak.fund_etf_hist_min_em(symbol=code, period=freq, adjust="qfq"),
        lambda: ak.stock_zh_a_hist_min_em(symbol=code, period=freq, adjust=""),
    ]
    for r in range(retries):
        for i, fn in enumerate(attempts):
            try:
                df = fn()
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
            time.sleep(1.5)
        time.sleep(3)
    return None


def to_full_code(code: str) -> str:
    code = code.strip()
    if "." in code:
        return code
    if code.startswith(("15", "16", "18")):
        return f"{code}.SZ"
    return f"{code}.SH"


def main() -> None:
    parser = argparse.ArgumentParser(description="每日增量拉取 ETF 分钟线")
    parser.add_argument("--codes", default=",".join(DEFAULT_ETFS))
    parser.add_argument("--freqs", default="60", help="逗号分隔，如 60 或 60,30,15")
    parser.add_argument("--sleep", type=float, default=2.0, help="每次请求后等待秒数")
    args = parser.parse_args()

    import akshare as ak

    db = _resolve_db()
    conn = sqlite3.connect(db)
    ensure_table(conn)

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    freqs = [f.strip() for f in args.freqs.split(",") if f.strip()]

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 开始拉取")
    print(f"  {len(codes)} 个标的 × {len(freqs)} 个频率 = {len(codes)*len(freqs)} 次请求")

    total = 0
    for code in codes:
        for freq in freqs:
            try:
                df = fetch_minute(ak, code, freq)
                if df is None or df.empty:
                    print(f"  {code} {freq}min: [--] 拉取失败（东财接口不稳定）")
                    continue
                d = norm(df)
                n = upsert(conn, to_full_code(code), f"{freq}min", d)
                total += n
                days = pd.to_datetime(d["trade_time"]).dt.date.nunique()
                print(f"  {code} {freq}min: [OK] {n:>5} 行, {days} 个交易日, "
                      f"{d['trade_time'].min()} ~ {d['trade_time'].max()}")
            except Exception as e:
                print(f"  {code} {freq}min: [X] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(args.sleep)

    # 汇总
    print("\n" + "=" * 70)
    print("数据库现状")
    print("=" * 70)
    r = pd.read_sql_query(
        "SELECT ts_code, freq, COUNT(*) n, COUNT(DISTINCT substr(trade_time,1,10)) days, "
        "MIN(trade_time) a, MAX(trade_time) b FROM etf_minute GROUP BY ts_code, freq "
        "ORDER BY ts_code, freq", conn)
    if r.empty:
        print("  （无数据）")
    else:
        print(r.to_string(index=False))
        max_days = int(r["days"].max())
        print(f"\n  最长覆盖: {max_days} 个交易日（约 {max_days/244:.2f} 年）")
        need = 500
        if max_days < need:
            remaining = need - max_days
            print(f"  距离做日内回测（{need}天）还差 {remaining} 个交易日")
        else:
            print(f"  已达 {need} 天门槛，可开始日内策略回测")

    conn.close()
    print(f"\n本次写入 {total} 行")


if __name__ == "__main__":
    main()
