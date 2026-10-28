from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import layer_from_market_cap_yi
from .logging_utils import setup_logger

logger = setup_logger("hierarchical.splitter")


def _load_market_caps(market_cap_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(market_cap_db)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()]
        date_col = "trade_date" if "trade_date" in columns else "update_date" if "update_date" in columns else None
        select_cols = ["stock_code", "total_mv", "circ_mv"]
        if date_col:
            select_cols.append(f"{date_col} AS trade_date")
        return pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM market_caps", conn)
    finally:
        conn.close()


def build_stock_splits(market_cap_db: str, features_db: str, output_csv: str, seed: int = 42) -> pd.DataFrame:
    caps = _load_market_caps(market_cap_db)
    if caps.empty:
        raise RuntimeError("market_caps 表为空，请先抓取市值数据。")

    sort_cols = ["stock_code"] + (["trade_date"] if "trade_date" in caps.columns else [])
    caps = caps.sort_values(sort_cols).drop_duplicates("stock_code", keep="last")
    caps["market_cap_yi"] = caps["total_mv"].astype(float) / 10000.0
    caps["layer"] = caps["market_cap_yi"].apply(layer_from_market_cap_yi)

    train_df, temp_df = train_test_split(caps, test_size=0.4, random_state=seed, shuffle=True, stratify=caps["layer"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True, stratify=temp_df["layer"])

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    overlap = (
        set(train_df["stock_code"]) & set(val_df["stock_code"])
        | set(train_df["stock_code"]) & set(test_df["stock_code"])
        | set(val_df["stock_code"]) & set(test_df["stock_code"])
    )
    if overlap:
        raise RuntimeError(f"数据泄露：发现重叠股票 {list(overlap)[:5]}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    conn = sqlite3.connect(features_db)
    try:
        split_df.to_sql("stock_splits", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    logger.info("分割完成 train=%s val=%s test=%s", len(train_df), len(val_df), len(test_df))
    return split_df
