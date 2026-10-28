from __future__ import annotations

from datetime import datetime, timedelta
import time
from pathlib import Path
import sqlite3
import pandas as pd

from .config import SystemConfig
from .logging_utils import setup_logger

logger = setup_logger("hierarchical.market_cap")

try:
    import tushare as ts
except Exception:  # pragma: no cover
    ts = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def normalize_ts_code(stock_code: str) -> str:
    code = str(stock_code).split(".")[0].zfill(6)
    if code.startswith(("5", "6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def ensure_market_cap_schema(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_caps (
                stock_code TEXT PRIMARY KEY,
                ts_code TEXT,
                total_mv REAL,
                circ_mv REAL,
                trade_date TEXT,
                updated_at TEXT,
                source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fetch_progress (
                stock_code TEXT PRIMARY KEY,
                status TEXT,
                message TEXT,
                updated_at TEXT
            )
            """
        )
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()}
        for column_name, column_type in {
            "ts_code": "TEXT",
            "trade_date": "TEXT",
            "updated_at": "TEXT",
            "source": "TEXT",
        }.items():
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE market_caps ADD COLUMN {column_name} {column_type}")
        conn.commit()
    finally:
        conn.close()


def get_all_stock_codes(trade_db: str) -> list[str]:
    conn = sqlite3.connect(trade_db)
    try:
        df = pd.read_sql_query("SELECT DISTINCT stock_code FROM stock_data ORDER BY stock_code", conn)
        return df["stock_code"].astype(str).tolist()
    finally:
        conn.close()


def _fetch_daily_basic(pro, ts_code: str, trade_date: str | None = None) -> pd.DataFrame:
    if trade_date:
        df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields="ts_code,trade_date,total_mv,circ_mv")
        if not df.empty:
            return df
    start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    return pro.daily_basic(ts_code=ts_code, start_date=start, end_date=end, fields="ts_code,trade_date,total_mv,circ_mv")


def fetch_and_store_market_caps(
    config: SystemConfig,
    trade_date: str | None = None,
    sleep_seconds: float = 0.25,
    refresh_existing: bool = False,
) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法抓取市值数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_market_cap_schema(config.databases.market_cap_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    codes = get_all_stock_codes(config.databases.trade_db)
    results: list[dict] = []

    conn = sqlite3.connect(config.databases.market_cap_db)
    try:
        processed = set(pd.read_sql_query("SELECT stock_code FROM market_caps", conn)["stock_code"].astype(str).tolist())
        todo_codes = codes if refresh_existing else [code for code in codes if code not in processed]
        logger.info("股票总数=%s, 已存在=%s, 待抓取=%s", len(codes), len(processed), len(todo_codes))

        for code in tqdm(todo_codes, desc="fetch market cap"):
            ts_code = normalize_ts_code(code)
            now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                df = _fetch_daily_basic(pro, ts_code, trade_date)
                if df.empty:
                    conn.execute(
                        "INSERT OR REPLACE INTO fetch_progress(stock_code, status, message, updated_at) VALUES (?, ?, ?, ?)",
                        (code, "empty", "daily_basic returned no rows", now_text),
                    )
                    conn.commit()
                    continue
                latest = df.sort_values("trade_date").iloc[-1]
                record = {
                    "stock_code": str(code),
                    "ts_code": ts_code,
                    "total_mv": float(latest["total_mv"]),
                    "circ_mv": float(latest["circ_mv"]),
                    "trade_date": str(latest["trade_date"]),
                    "updated_at": now_text,
                    "source": "tushare",
                }
                conn.execute("DELETE FROM market_caps WHERE stock_code = ?", (str(code),))
                pd.DataFrame([record]).to_sql("market_caps", conn, if_exists="append", index=False)
                conn.execute(
                    "INSERT OR REPLACE INTO fetch_progress(stock_code, status, message, updated_at) VALUES (?, ?, ?, ?)",
                    (code, "done", "ok", now_text),
                )
                conn.commit()
                results.append(record)
            except Exception as exc:
                logger.warning("抓取 %s 失败: %s", code, exc)
                conn.execute(
                    "INSERT OR REPLACE INTO fetch_progress(stock_code, status, message, updated_at) VALUES (?, ?, ?, ?)",
                    (code, "error", str(exc)[:500], now_text),
                )
                conn.commit()
            time.sleep(sleep_seconds)

        return pd.read_sql_query("SELECT * FROM market_caps ORDER BY stock_code", conn)
    finally:
        conn.close()
