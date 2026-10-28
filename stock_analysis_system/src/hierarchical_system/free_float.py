"""真实自由流通盘 · 换手率修正。

船长核心洞察：官方"流通股"定义 ≠ "真正会交易的股"。
国资控股、实控人锁定、战略投资者限售、员工持股等筹码"事实永不出售"，
应从流通盘剔除，得到"真实自由流通股"，进而修正换手率。

    真实自由流通股 = 总股本
      - 限售股（未解禁）
      - 国资控股持股（事实不流通）
      - 实控人及一致行动人持股
      - 战略投资者锁定持股

    真实换手率 = 当日成交量 / 真实自由流通股 × 100%

数据源（tushare）：
  - top10_holders       十大股东 + 持股比例 + 股东性质（holder_type）
  - top10_floatholders  十大流通股东（交叉验证）
  - share_float         限售解禁数据
  - pledge_stat         股权质押（辅助）

判定口径（保守默认）：
  仅剔除"事实不流通"筹码——限售股 + 国资控股 + 实控人/一致行动人。
  基金 / 散户 / 财务投资（会交易）不计入剔除，避免误伤。
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import SystemConfig
from .logging_utils import setup_logger
from .market_cap import get_all_stock_codes, normalize_ts_code

logger = setup_logger("hierarchical.free_float")

try:
    import tushare as ts
except Exception:  # pragma: no cover
    ts = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# ── "事实不流通"筹码判定规则（数值稳健版）──────────────────────
# 问题：tushare 返回的 holder_type 中文在 Windows 下常见乱码（GBK 被错误解码），
# 无法可靠地用中文关键词识别"国资/实控人"。因此改用**不依赖中文**的数值规则。
#
# 核心信号（可量化、稳定）：
#   1. hold_change == 0  —— 长期零变动，说明大股东从不交易（实控人/国资典型特征）
#   2. hold_ratio 较大     —— 前 N 大股东，持股占比高
#   3. hold_amount 超大    —— 绝对筹码量大（>某阈值），散户/基金达不到
#
# 排除项（会交易的）：
#   - HKSCC NOMINEES（港股通中央结算名义户）是通道，非单一实控人，剔除（避免过度扣除）
#   其余用"零变动 + 大占比"识别。
def classify_holder_non_tradable(
    holder_name: str,
    holder_type: str,
    hold_ratio: Optional[float],
    hold_change: Optional[float],
) -> bool:
    """判断某股东持股是否"事实不流通"。

    保守口径（避免误伤财务投资）：
    只有"实控人/国资"这类有强"永不出售"约束的筹码才剔除，且必须是
    长期零变动的大股东。财务投资（基金/外资/一般法人）即使占比高也可能
    交易，不剔除。

    关键信号：
      1. 长期零变动（hold_change == 0）——大股东从不交易
      2. 占比高（>= 3%）——筹码有实质锁定意义
    二者同时满足才判不流通；占比 < 3% 的小股东不判（避免扣除过度）。
    """
    name = str(holder_name or "").strip().upper()

    # HKSCC NOMINEES 是港股通中央结算名义户（通道），非单一实控人，剔除排除
    if "HKSCC" in name or "NOMINEES" in name:
        return False

    try:
        ratio = float(hold_ratio or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    try:
        change = float(hold_change or 0.0)
    except (TypeError, ValueError):
        change = 0.0

    # 长期零变动 + 占比 >= 3%（疑似实控人/国资，基金/散户会频繁变动）
    # 注意：这里只认"零变动"这一硬信号，不因占比大就无条件判不流通，
    # 避免把"持股占比大但会交易的财务投资"误判为死筹码。
    if abs(change) < 1e-9 and ratio >= 3.0:
        return True

    return False


# ── 数据库 schema ─────────────────────────────────────────────
def ensure_holder_structure_schema(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS top10_holders (
                ts_code TEXT,
                ann_date TEXT,
                end_date TEXT,
                holder_name TEXT,
                hold_amount REAL,
                hold_ratio REAL,
                hold_float_ratio REAL,
                hold_change REAL,
                holder_type TEXT,
                is_non_tradable INTEGER,
                PRIMARY KEY (ts_code, end_date, holder_name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS share_float (
                ts_code TEXT,
                ann_date TEXT,
                float_date TEXT,
                float_share REAL,
                float_ratio REAL,
                holder_name TEXT,
                share_type TEXT,
                PRIMARY KEY (ts_code, float_date, holder_name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS free_float (
                stock_code TEXT PRIMARY KEY,
                ts_code TEXT,
                total_share REAL,
                float_share REAL,
                locked_share REAL,
                real_free_float_share REAL,
                real_free_float_ratio REAL,
                end_date TEXT,
                updated_at TEXT,
                source TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# ── 数据抓取 ─────────────────────────────────────────────────
def fetch_holder_structure(
    config: SystemConfig,
    stock_codes: Optional[list[str]] = None,
    period: str = "20260331",
    sleep_seconds: float = 0.35,
) -> pd.DataFrame:
    """抓取 top10_holders + share_float，写入 market_cap_db。

    采用 tushare 的 period 批量接口 + offset 分页，一次调用约拿 2000 只
    股票（6000 行封顶），全市场只需约 3-4 次调用，替代逐股调用（8h+ → 秒级）。

    Returns:
        带 is_non_tradable 判定的十大股东 DataFrame。
    """
    if ts is None:
        raise RuntimeError("未安装 tushare，无法抓取股东结构数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN。")

    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    db_path = config.databases.market_cap_db
    ensure_holder_structure_schema(db_path)

    # 目标股票集合（可选过滤）
    want_codes = None
    if stock_codes:
        want_codes = {normalize_ts_code(c) for c in stock_codes}

    # 1. 批量抓 top10_holders（按披露期 + offset 分页）
    all_holders: list[pd.DataFrame] = []
    offset = 0
    while True:
        try:
            holders = pro.top10_holders(period=period, offset=offset, limit=6000)
        except Exception as exc:
            logger.warning("top10_holders period=%s offset=%s 失败: %s", period, offset, exc)
            break
        if holders is None or holders.empty:
            break
        if want_codes is not None:
            holders = holders[holders["ts_code"].isin(want_codes)]
        if not holders.empty:
            all_holders.append(holders)
        # 不足 6000 行说明已到末尾
        if len(holders) < 6000:
            break
        offset += 6000
        time.sleep(sleep_seconds)

    holders_df = pd.concat(all_holders, ignore_index=True) if all_holders else pd.DataFrame()

    if not holders_df.empty:
        # 只保留每只股票最新披露期（同一 period 已是最新，但可能有历史期重叠）
        holders_df = holders_df.drop_duplicates(subset=["ts_code", "holder_name"], keep="first")
        holders_df["is_non_tradable"] = holders_df.apply(
            lambda r: int(
                classify_holder_non_tradable(
                    r.get("holder_name"),
                    r.get("holder_type"),
                    r.get("hold_ratio"),
                    r.get("hold_change"),
                )
            ),
            axis=1,
        )
        with sqlite3.connect(db_path) as conn:
            holders_df.to_sql("top10_holders", conn, if_exists="replace", index=False)

    # 2. 批量抓 share_float（限售解禁，全市场）
    try:
        floats = pro.share_float()
    except Exception as exc:
        logger.warning("share_float 全市场抓取失败: %s", exc)
        floats = pd.DataFrame()
    time.sleep(sleep_seconds)

    if not floats.empty:
        if want_codes is not None:
            floats = floats[floats["ts_code"].isin(want_codes)]
        cols = ["ts_code", "ann_date", "float_date", "float_share",
                "float_ratio", "holder_name", "share_type"]
        for c in cols:
            if c not in floats.columns:
                floats[c] = ""
        with sqlite3.connect(db_path) as conn:
            floats[cols].to_sql("share_float", conn, if_exists="replace", index=False)

    return holders_df


# ── 真实自由流通盘计算 ────────────────────────────────────────
def compute_real_free_float(
    config: SystemConfig,
    trade_date: str | None = None,
) -> pd.DataFrame:
    """用批量数据计算每只股票的真实自由流通股数并落地 free_float 表。

    口径：
      真实自由流通股 = 总股本 - 限售股 - Σ(事实不流通十大股东持股)

    全批量实现（秒级）：daily_basic(trade_date) 一次拿全市场股本，
    事实不流通股东从已抓取的 top10_holders 表用 SQL 聚合。
    """
    if ts is None:
        raise RuntimeError("未安装 tushare，无法计算自由流通盘。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN。")

    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    db_path = config.databases.market_cap_db
    ensure_holder_structure_schema(db_path)

    # 找一个有数据的最新交易日
    if trade_date is None:
        # 用 stock_data 最新日期
        from .config import load_config  # noqa
        conn = sqlite3.connect(config.databases.trade_db)
        try:
            latest = conn.execute(
                "SELECT MAX([日期]) FROM stock_data WHERE [成交量] > 0"
            ).fetchone()[0]
        finally:
            conn.close()
        if latest:
            trade_date = str(latest).replace("-", "")

    # 1. 批量拿全市场股本（daily_basic 单位是「万股」，×10000 转「股」）
    try:
        basic = pro.daily_basic(
            trade_date=trade_date, fields="ts_code,trade_date,total_share,float_share"
        )
    except Exception as exc:
        logger.warning("daily_basic trade_date=%s 失败: %s", trade_date, exc)
        basic = pd.DataFrame()

    if basic.empty:
        logger.warning("daily_basic 无数据，无法计算自由流通盘。")
        return pd.DataFrame()

    basic = basic.copy()
    basic["total_share"] = pd.to_numeric(basic["total_share"], errors="coerce") * 10000.0
    basic["float_share"] = pd.to_numeric(basic["float_share"], errors="coerce") * 10000.0
    basic["locked_share"] = (basic["total_share"] - basic["float_share"]).clip(lower=0.0)

    # 2. 从 top10_holders 聚合事实不流通持股
    conn = sqlite3.connect(db_path)
    try:
        non_trad = pd.read_sql_query(
            "SELECT ts_code, SUM(hold_amount) AS non_tradable_sum "
            "FROM top10_holders WHERE is_non_tradable=1 "
            "GROUP BY ts_code",
            conn,
        )
    finally:
        conn.close()

    # 3. merge 计算真实自由流通
    out = basic.merge(non_trad, on="ts_code", how="left")
    out["non_tradable_sum"] = out["non_tradable_sum"].fillna(0.0)
    out["real_free_float_share"] = (
        out["total_share"] - out["locked_share"] - out["non_tradable_sum"]
    ).clip(lower=0.0)

    # 地板保护：真实自由流通盘不低于名义流通股的 5%。
    # 原因：次新股/高度集中持股时，前十大股东扣减可能过度，导致真实流通盘
    # 被扣到接近 0（换手率爆表）。地板保护保证估计稳健，避免异常值污染模型。
    floor = (out["float_share"] * 0.05).fillna(0.0)
    out["real_free_float_share"] = np.where(
        out["real_free_float_share"] < floor, floor, out["real_free_float_share"]
    )

    out["real_free_float_ratio"] = np.where(
        out["total_share"] > 0,
        out["real_free_float_share"] / out["total_share"],
        np.nan,
    )
    out["stock_code"] = out["ts_code"].str.split(".").str[0].str.zfill(6)
    out["end_date"] = str(trade_date)
    out["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out["source"] = "tushare"

    cols = [
        "stock_code", "ts_code", "total_share", "float_share", "locked_share",
        "real_free_float_share", "real_free_float_ratio", "end_date",
        "updated_at", "source",
    ]
    result = out[cols]
    with sqlite3.connect(db_path) as conn:
        result.to_sql("free_float", conn, if_exists="replace", index=False)

    return result


# ── 真实换手率计算 ─────────────────────────────────────────────
def ensure_real_turnover_schema(trade_db: str) -> None:
    Path(trade_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(trade_db)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS real_turnover_rate (
                stock_code TEXT,
                date TEXT,
                volume REAL,
                real_free_float_share REAL,
                real_turnover_rate REAL,
                PRIMARY KEY (stock_code, date)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def compute_real_turnover_rate(
    config: SystemConfig,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """用「真实自由流通股」修正换手率并落地 real_turnover_rate 表。

    真实换手率 = 当日成交量(股) / 真实自由流通股(股) × 100%

    说明：真实自由流通股数采用 free_float 表的最新时点值（实控人/国资
    持股按季度披露，变动缓慢，用静态近似是合理的第一步）。
    """
    market_db = config.databases.market_cap_db
    trade_db = config.databases.trade_db
    ensure_real_turnover_schema(trade_db)

    # 1. 读取真实自由流通股静态映射
    try:
        ff = pd.read_sql_query(
            "SELECT stock_code, real_free_float_share FROM free_float "
            "WHERE stock_code IS NOT NULL AND real_free_float_share > 0",
            sqlite3.connect(market_db),
        )
    except Exception as exc:
        logger.warning("读取 free_float 失败: %s", exc)
        return pd.DataFrame()

    if ff.empty:
        logger.warning("free_float 表为空，请先抓取股东结构数据。")
        return pd.DataFrame()

    ff["stock_code"] = ff["stock_code"].astype(str).str.zfill(6)
    ff = ff.drop_duplicates(subset=["stock_code"], keep="last")
    ff_map = ff.set_index("stock_code")["real_free_float_share"].to_dict()

    # 2. 读取每日成交量
    params: list[str] = []
    where = "stock_code IS NOT NULL"
    if start_date:
        where += " AND [日期] >= ?"
        params.append(start_date)
    if end_date:
        where += " AND [日期] <= ?"
        params.append(end_date)

    vol = pd.read_sql_query(
        f"SELECT stock_code, [日期] AS date, [成交量] AS volume, [换手率] AS turnover_rate "
        f"FROM stock_data WHERE {where}",
        sqlite3.connect(trade_db),
        params=tuple(params),
    )
    if vol.empty:
        return pd.DataFrame()

    vol["stock_code"] = vol["stock_code"].astype(str).str.zfill(6)
    vol["volume"] = pd.to_numeric(vol["volume"], errors="coerce")
    vol = vol.dropna(subset=["volume"])

    # 3. 计算真实换手率（成交量单位：stock_data 的[成交量]为「手」，×100 转股）
    vol["real_free_float_share"] = vol["stock_code"].map(ff_map)
    vol = vol.dropna(subset=["real_free_float_share"])
    vol["real_turnover_rate"] = vol["volume"].astype(float) * 100.0 / vol["real_free_float_share"].astype(float) * 100.0

    out = vol[["stock_code", "date", "volume", "real_free_float_share", "real_turnover_rate"]]
    if not out.empty:
        with sqlite3.connect(trade_db) as conn:
            out.to_sql("real_turnover_rate", conn, if_exists="replace", index=False)

    return out
