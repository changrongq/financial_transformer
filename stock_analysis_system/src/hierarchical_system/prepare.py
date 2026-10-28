from __future__ import annotations

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

from .config import SystemConfig
from .features import LAYER_FEATURES, build_layer_features
from .logging_utils import setup_logger

logger = setup_logger("hierarchical.prepare")


def _latest_market_caps(market_cap_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(market_cap_db)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()]
        date_col = "trade_date" if "trade_date" in columns else "update_date" if "update_date" in columns else None
        select_cols = ["stock_code", "total_mv"]
        if date_col:
            select_cols.append(f"{date_col} AS trade_date")
        df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM market_caps", conn)
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError("market_caps 表为空，请先执行 fetch_market_cap.py")
    sort_cols = ["stock_code"] + (["trade_date"] if "trade_date" in df.columns else [])
    df = df.sort_values(sort_cols).drop_duplicates("stock_code", keep="last")
    df["market_cap_yi"] = df["total_mv"].astype(float) / 10000.0
    return df[["stock_code", "market_cap_yi"]]


def _load_splits(features_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(features_db)
    try:
        return pd.read_sql_query("SELECT stock_code, split, layer FROM stock_splits", conn)
    finally:
        conn.close()


def _load_trade_data(trade_db: str, stock_code: str) -> pd.DataFrame:
    conn = sqlite3.connect(trade_db)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM stock_data WHERE stock_code = ? ORDER BY 日期",
            conn,
            params=[stock_code],
        )
        # 真实换手率（修正股权结构污染，见 free_float.py）
        # 同一 trade_db 内 LEFT JOIN real_turnover_rate 表，获取 real_turnover_rate 列
        try:
            rto_df = pd.read_sql_query(
                "SELECT stock_code, date, real_turnover_rate "
                "FROM real_turnover_rate WHERE stock_code = ?",
                conn,
                params=[stock_code],
            )
        except Exception:
            rto_df = pd.DataFrame()
        try:
            moneyflow_df = pd.read_sql_query(
                "SELECT * FROM stock_moneyflow WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[stock_code],
            )
        except Exception:
            moneyflow_df = pd.DataFrame()
        try:
            top_inst_df = pd.read_sql_query(
                "SELECT * FROM stock_top_inst WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[stock_code],
            )
        except Exception:
            top_inst_df = pd.DataFrame()
        try:
            margin_df = pd.read_sql_query(
                "SELECT * FROM stock_margin_detail WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[stock_code],
            )
        except Exception:
            margin_df = pd.DataFrame()
        try:
            basic_info_df = pd.read_sql_query(
                "SELECT * FROM stock_basic_info WHERE stock_code = ?",
                conn,
                params=[stock_code],
            )
        except Exception:
            basic_info_df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns=rename_map)
    for col in ["open", "high", "low", "close", "volume", "turnover_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    missing_turnover_mask = pd.Series(False, index=df.index)
    if "volume" in df.columns:
        missing_turnover_mask = df["volume"].isna()
        if "amount" in df.columns:
            missing_turnover_mask = missing_turnover_mask & df["amount"].isna()
        df.loc[missing_turnover_mask, "volume"] = 0.0
    if "amount" in df.columns:
        df["amount"] = df["amount"].fillna(df["close"] * df["volume"])
    else:
        df["amount"] = df["close"] * df["volume"]
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = np.nan
    df.loc[df["turnover_rate"].isna() & missing_turnover_mask, "turnover_rate"] = 0.0
    # 真实换手率：用修正后的真实换手率替代名义换手率（缺失回退名义）
    # 这样 micro/small/mid/large 各层的 turnover_rate / turnover_change 特征
    # 统一从"名义"升级为"真实"，无需改 features.py。
    df["date"] = pd.to_datetime(df["date"])
    if not rto_df.empty:
        rto_df = rto_df.rename(columns={"date": "_rto_date"})
        rto_df["_rto_date"] = pd.to_datetime(rto_df["_rto_date"], errors="coerce")
        rto_df["real_turnover_rate"] = pd.to_numeric(rto_df["real_turnover_rate"], errors="coerce")
        df = df.merge(
            rto_df[["stock_code", "_rto_date", "real_turnover_rate"]],
            left_on=["stock_code", "date"],
            right_on=["stock_code", "_rto_date"],
            how="left",
        ).drop(columns=["_rto_date"])
        df["real_turnover_rate"] = df["real_turnover_rate"].fillna(df["turnover_rate"])
        df["turnover_rate"] = df["real_turnover_rate"].fillna(df["turnover_rate"])
    if not moneyflow_df.empty:
        moneyflow_df = moneyflow_df.rename(columns={"日期": "date"})
        moneyflow_df["date"] = pd.to_datetime(moneyflow_df["date"])
        for col in [c for c in moneyflow_df.columns if c not in {"stock_code", "date"}]:
            moneyflow_df[col] = pd.to_numeric(moneyflow_df[col], errors="coerce")
        df = df.merge(moneyflow_df, on=["stock_code", "date"], how="left")
    if not top_inst_df.empty:
        top_inst_df = top_inst_df.rename(columns={"日期": "date"})
        top_inst_df["date"] = pd.to_datetime(top_inst_df["date"])
        for col in [c for c in top_inst_df.columns if c not in {"stock_code", "date"}]:
            top_inst_df[col] = pd.to_numeric(top_inst_df[col], errors="coerce")
        df = df.merge(top_inst_df, on=["stock_code", "date"], how="left")
    if not margin_df.empty:
        margin_df = margin_df.rename(columns={"日期": "date"})
        margin_df["date"] = pd.to_datetime(margin_df["date"])
        for col in [c for c in margin_df.columns if c not in {"stock_code", "date"}]:
            margin_df[col] = pd.to_numeric(margin_df[col], errors="coerce")
        df = df.merge(margin_df, on=["stock_code", "date"], how="left")
    if not basic_info_df.empty:
        basic_info_df = basic_info_df.copy()
        for column_name in ["名称", "地域", "行业", "市场", "上市日期"]:
            if column_name in basic_info_df.columns:
                df[column_name] = basic_info_df.iloc[0][column_name]
    return df


def extract_hierarchical_features(config: SystemConfig) -> pd.DataFrame:
    market_caps = _latest_market_caps(config.databases.market_cap_db)
    splits = _load_splits(config.databases.features_db)
    merged = splits.merge(market_caps, on="stock_code", how="inner")
    all_frames: list[pd.DataFrame] = []
    horizon = config.training.prediction_horizon
    threshold = config.training.target_return_threshold

    for _, row in merged.iterrows():
        stock_code = str(row["stock_code"])
        split = row["split"]
        layer = row["layer"]
        market_cap_yi = float(row["market_cap_yi"])
        try:
            price_df = _load_trade_data(config.databases.trade_db, stock_code)
            if price_df.empty or len(price_df) < max(90, config.training.lookback_window + horizon):
                continue
            feats = build_layer_features(price_df, market_cap_yi)
            feats["future_return"] = feats["close"].shift(-horizon) / feats["close"] - 1
            feats["target_up"] = (feats["future_return"] >= threshold).astype(float)
            feats["split"] = split
            feats["stock_code"] = stock_code
            keep_cols = [
                "stock_code", "date", "layer", "split", "market_cap_yi", "future_return", "target_up",
                *sorted({col for cols in LAYER_FEATURES.values() for col in cols}),
            ]
            feats = feats[[col for col in keep_cols if col in feats.columns]].replace([np.inf, -np.inf], np.nan)
            feats = feats.dropna().reset_index(drop=True)
            all_frames.append(feats)
        except Exception as exc:
            logger.warning("特征提取失败 %s: %s", stock_code, exc)

    if not all_frames:
        raise RuntimeError("没有提取到任何特征数据。")

    feature_df = pd.concat(all_frames, ignore_index=True)
    Path(config.databases.features_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.databases.features_db)
    try:
        feature_df.to_sql("hierarchical_features", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    logger.info("特征提取完成，共 %s 行", len(feature_df))
    return feature_df
