"""回补目标行业股票的历史日线（2019-2026，日历因子验证用）。

按股票逐只拉 2019-2026 的日线，写入独立的 historical_daily 表（不动现有 stock_data）。
同时可拉上证指数作为全市场基准。

用法：
  python scripts/fetch_historical_daily.py --start 20190101 --end 20260820
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.tushare_config import TUSHARE_TOKEN

try:
    import tushare as ts
except Exception:
    ts = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


# 目标行业的行业关键词
GROUP_KEYWORDS = {
    "旅游": ["旅游景点", "旅游服务", "酒店餐饮"],
    "农业": ["农业综合", "农药化肥"],
    "电力": ["火力发电", "水力发电", "其他发电"],
    "家电": ["家用电器", "白色家电"],
}


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


def ensure_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_daily (
                stock_code TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (stock_code, date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_fetch_progress (
                stock_code TEXT PRIMARY KEY,
                fetched_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def build_target_codes(db_path: str) -> pd.DataFrame:
    """从 stock_basic_info 选目标行业股票，返回 stock_code+ts_code+行业+group。"""
    conn = sqlite3.connect(db_path)
    try:
        bi = pd.read_sql_query("SELECT stock_code, ts_code, 行业 FROM stock_basic_info", conn)
    except Exception:
        bi = pd.DataFrame(columns=["stock_code", "ts_code", "行业"])
    finally:
        conn.close()
    bi["行业"] = bi["行业"].astype(str)

    out = []
    for group, kws in GROUP_KEYWORDS.items():
        mask = bi["行业"].str.contains("|".join(kws), na=False)
        sub = bi[mask].copy()
        sub["group"] = group
        out.append(sub[["stock_code", "ts_code", "行业", "group"]])
    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["stock_code", "ts_code", "行业", "group"])
    result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)
    return result.drop_duplicates(subset=["stock_code"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="回补目标行业历史日线")
    parser.add_argument("--start", default="20190101")
    parser.add_argument("--end", default="20260820")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    if ts is None:
        raise RuntimeError("未安装 tushare")
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    db_path = _resolve_db()
    ensure_schema(db_path)

    targets = build_target_codes(db_path)
    print(f"目标行业股票数: {len(targets)}")
    print(targets.groupby("group").size().to_string())

    conn = sqlite3.connect(db_path)
    try:
        done = set(pd.read_sql_query("SELECT stock_code FROM historical_fetch_progress", conn)["stock_code"].tolist())
    except Exception:
        done = set()
    conn.close()

    todo = targets[~targets["stock_code"].isin(done)]
    print(f"已拉取: {len(done)} 只，待拉取: {len(todo)} 只")

    total_rows = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="fetch stock daily"):
        code = row["stock_code"]
        ts_code = str(row["ts_code"]) if pd.notna(row["ts_code"]) else None
        if not ts_code or ts_code == "None" or ts_code == "nan":
            if code.startswith(("6", "5", "9")):
                ts_code = code + ".SH"
            else:
                ts_code = code + ".SZ"
        try:
            df = pro.daily(ts_code=ts_code, start_date=args.start, end_date=args.end)
        except Exception as exc:
            print(f"  [失败] {code}: {exc}")
            time.sleep(args.sleep * 3)
            continue
        if df is None or df.empty:
            time.sleep(args.sleep)
            continue

        out = pd.DataFrame({
            "stock_code": code,
            "date": df["trade_date"],
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["vol"], errors="coerce"),
            "amount": pd.to_numeric(df["amount"], errors="coerce"),
        })
        with sqlite3.connect(db_path) as c:
            out.to_sql("historical_daily", c, if_exists="append", index=False)
            c.execute("INSERT OR REPLACE INTO historical_fetch_progress(stock_code, fetched_at) VALUES(?, ?)",
                      (code, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            c.commit()
        total_rows += len(out)
        time.sleep(args.sleep)

    print(f"\n完成。本次新增 {total_rows} 行。")


if __name__ == "__main__":
    main()
