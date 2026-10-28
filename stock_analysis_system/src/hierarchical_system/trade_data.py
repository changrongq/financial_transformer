from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from threading import Lock
import time
from urllib import request as urlrequest
from urllib.error import URLError

import pandas as pd

from .config import SystemConfig
from .logging_utils import setup_logger
from .market_cap import get_all_stock_codes, normalize_ts_code

logger = setup_logger("hierarchical.trade_data")

try:
    import tushare as ts
except Exception:  # pragma: no cover
    ts = None


DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,pe,pe_ttm,pb,ps_ttm,dv_ttm"
STOCK_BASIC_FIELDS = "ts_code,symbol,name,area,industry,market,list_date"
REALTIME_CACHE_TTL_SECONDS = 3.0
CONCEPT_DETAIL_CALLS_PER_RUN = 120
TUSHARE_MIN_CALL_INTERVAL_SECONDS = 0.40
TUSHARE_RATE_LIMIT_WAIT_SECONDS = 35.0
TUSHARE_RATE_LIMIT_MESSAGE = "每分钟最多访问该接口200次"

_REALTIME_CACHE: dict[tuple[str, ...], tuple[float, pd.DataFrame]] = {}
_REALTIME_CACHE_LOCK = Lock()
_TUSHARE_API_LOCK = Lock()
_TUSHARE_NEXT_CALL_AT = 0.0

# ── 腾讯行情 ──
TENCENT_QUOTE_URL = "http://qt.gtimg.cn/q="
TENCENT_MARKET_PREFIX = {"sh": "sh", "sz": "sz", "bj": "bj"}


def _stock_code_to_tencent_format(stock_code: str) -> str:
    """将 stock_code 转为腾讯行情格式前缀。

    600xxx/601xxx/603xxx/605xxx → sh600xxx
    000xxx/001xxx/002xxx/003xxx/300xxx/301xxx → sz000xxx
    8xxxxx/4xxxxx → bj8xxxxx
    """
    code = str(stock_code or "").strip()
    if not code:
        return code
    if code.startswith(("sh", "sz", "bj")) and len(code) > 2:
        return code  # 已经带前缀
    if code.startswith(("60", "68")):
        return "sh" + code
    if code.startswith(("00", "30")):
        return "sz" + code
    if code.startswith(("8", "4")):
        return "bj" + code
    return "sz" + code  # fallback


def fetch_tencent_quotes(
    stock_codes: list[str],
    timeout_seconds: float = 5.0,
) -> pd.DataFrame:
    """从腾讯行情 API 获取实时报价。

    腾讯 qt.gtimg.cn 是免费 HTTP 接口，无需 token。
    返回盘中实时价格（延迟约 3-5 秒）。
    """
    if not stock_codes:
        return pd.DataFrame()

    tencent_codes = [_stock_code_to_tencent_format(c) for c in stock_codes]
    url = TENCENT_QUOTE_URL + ",".join(tencent_codes)

    try:
        req = urlrequest.Request(url)
        with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("gbk", errors="replace")
    except URLError as exc:
        logger.warning("腾讯行情请求失败: %s", exc)
        return pd.DataFrame()

    rows = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        # 格式: v_sh600664="1~平安银行~600664~..."
        try:
            var_name, value = line.split("=", 1)
            value = value.strip().strip('"').strip(";")
        except ValueError:
            continue

        if not value:
            continue

        fields = value.split("~")
        if len(fields) < 40:
            continue

        # 腾讯行情字段映射（精简版，取关键字段）
        # 索引参考: http://qt.gtimg.cn/q=sh600664 返回格式
        try:
            row = {
                "name": fields[1],
                "stock_code": fields[2],
                "price": float(fields[3]) if fields[3] else None,
                "pre_close": float(fields[4]) if fields[4] else None,
                "open": float(fields[5]) if fields[5] else None,
                "volume": float(fields[6]) if fields[6] else None,  # 手
                "high": float(fields[33]) if fields[33] else None,
                "low": float(fields[34]) if fields[34] else None,
                "amount": float(fields[37]) if fields[37] else None,  # 万元
                "pct_change": float(fields[32]) if fields[32] else None,
            }
            # amount 单位是万元，转换为元
            if row["amount"] is not None:
                row["amount"] = row["amount"] * 10000

            row["ts_code"] = normalize_ts_code(row["stock_code"])
            if row["price"] is not None and row["pre_close"] is not None and row["pre_close"] != 0:
                row["change"] = row["price"] - row["pre_close"]
                if row["pct_change"] is None:
                    row["pct_change"] = (row["change"] / row["pre_close"]) * 100.0
            else:
                row["change"] = None

            now = datetime.now()
            row["date"] = now.strftime("%Y%m%d")
            row["time"] = now.strftime("%H%M%S")
            row["timestamp"] = now.strftime("%Y%m%d %H%M%S")

            rows.append(row)
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.attrs["realtime_source"] = "tencent"
    return df


def _normalize_tencent_quote_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """将腾讯行情 DataFrame 转为与 tushare realtime_quote 兼容的格式"""
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = frame.copy()
    # 确保与 fetch_realtime_quotes 输出兼容的列存在
    for col in ["bid", "ask", "b1_v", "b1_p", "a1_v", "a1_p"]:
        if col not in df.columns:
            df[col] = None
    df.attrs["realtime_source"] = "tencent"
    return df


def _ensure_tushare_ready(config: SystemConfig) -> None:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法获取实时行情。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")
    ts.set_token(config.tushare_token)


def _normalize_realtime_source(src: str, allowed: set[str]) -> str:
    source = str(src or "").strip().lower()
    if source not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"src 必须是以下之一: {allowed_text}")
    return source


def _get_cached_frame(cache_key: tuple[str, ...], ttl_seconds: float) -> pd.DataFrame | None:
    if ttl_seconds <= 0:
        return None
    with _REALTIME_CACHE_LOCK:
        cached = _REALTIME_CACHE.get(cache_key)
        if cached is None:
            return None
        expires_at, frame = cached
        if time.monotonic() > expires_at:
            _REALTIME_CACHE.pop(cache_key, None)
            return None
        return frame.copy()


def _store_cached_frame(cache_key: tuple[str, ...], frame: pd.DataFrame, ttl_seconds: float) -> None:
    if ttl_seconds <= 0:
        return
    with _REALTIME_CACHE_LOCK:
        _REALTIME_CACHE[cache_key] = (time.monotonic() + ttl_seconds, frame.copy())


def _throttle_tushare_call(min_interval_seconds: float = TUSHARE_MIN_CALL_INTERVAL_SECONDS) -> None:
    global _TUSHARE_NEXT_CALL_AT
    with _TUSHARE_API_LOCK:
        now = time.monotonic()
        wait_seconds = max(0.0, _TUSHARE_NEXT_CALL_AT - now)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _TUSHARE_NEXT_CALL_AT = time.monotonic() + min_interval_seconds


def _call_tushare_pro_api(
    func,
    *args,
    min_interval_seconds: float = TUSHARE_MIN_CALL_INTERVAL_SECONDS,
    retry_wait_seconds: float = TUSHARE_RATE_LIMIT_WAIT_SECONDS,
    max_attempts: int = 3,
    **kwargs,
):
    for attempt in range(1, max_attempts + 1):
        _throttle_tushare_call(min_interval_seconds=min_interval_seconds)
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            message = str(exc)
            if TUSHARE_RATE_LIMIT_MESSAGE not in message or attempt == max_attempts:
                raise
            logger.warning("Tushare Pro 限流，等待后重试 api=%s attempt=%s", getattr(func, "__name__", str(func)), attempt)
            time.sleep(retry_wait_seconds)
    raise RuntimeError("Tushare Pro 调用重试后仍失败。")


def _normalize_realtime_codes(ts_codes: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(ts_codes, str):
        raw_codes = ts_codes.split(",")
    else:
        raw_codes = list(ts_codes)

    normalized_codes: list[str] = []
    seen: set[str] = set()
    for raw_code in raw_codes:
        text = str(raw_code or "").strip().upper()
        if not text:
            continue
        ts_code = text if "." in text else normalize_ts_code(text)
        if ts_code in seen:
            continue
        seen.add(ts_code)
        normalized_codes.append(ts_code)

    if not normalized_codes:
        raise ValueError("至少提供一个股票代码。")
    return normalized_codes


def _lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(column).strip().lower() for column in renamed.columns]
    return renamed


def _normalize_realtime_quote_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = _lowercase_columns(frame)
    numeric_columns = [
        "open",
        "pre_close",
        "price",
        "high",
        "low",
        "bid",
        "ask",
        "volume",
        "amount",
        "b1_v",
        "b1_p",
        "b2_v",
        "b2_p",
        "b3_v",
        "b3_p",
        "b4_v",
        "b4_p",
        "b5_v",
        "b5_p",
        "a1_v",
        "a1_p",
        "a2_v",
        "a2_p",
        "a3_v",
        "a3_p",
        "a4_v",
        "a4_p",
        "a5_v",
        "a5_p",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.upper()
        df["stock_code"] = df["ts_code"].str.split(".").str[0]

    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)
    if {"date", "time"}.issubset(df.columns):
        df["timestamp"] = (df["date"].str.strip() + " " + df["time"].str.strip()).str.strip()

    if {"price", "pre_close"}.issubset(df.columns):
        df["change"] = df["price"] - df["pre_close"]
        previous_close = df["pre_close"].replace(0, pd.NA)
        df["pct_change"] = (df["change"] / previous_close) * 100.0

    preferred_columns = [
        "stock_code",
        "ts_code",
        "name",
        "date",
        "time",
        "timestamp",
        "price",
        "pct_change",
        "change",
        "open",
        "pre_close",
        "high",
        "low",
        "bid",
        "ask",
        "volume",
        "amount",
        "b1_v",
        "b1_p",
        "b2_v",
        "b2_p",
        "b3_v",
        "b3_p",
        "b4_v",
        "b4_p",
        "b5_v",
        "b5_p",
        "a1_v",
        "a1_p",
        "a2_v",
        "a2_p",
        "a3_v",
        "a3_p",
        "a4_v",
        "a4_p",
        "a5_v",
        "a5_p",
    ]
    ordered_columns = [column for column in preferred_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    return df[ordered_columns + remaining_columns].sort_values(by=[column for column in ["ts_code", "time"] if column in df.columns]).reset_index(drop=True)


def _normalize_realtime_list_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = _lowercase_columns(frame)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.upper()
        df["stock_code"] = df["ts_code"].str.split(".").str[0]

    if "close" in df.columns and "price" not in df.columns:
        df["price"] = df["close"]

    numeric_columns = [
        "price",
        "pct_change",
        "change",
        "volume",
        "amount",
        "swing",
        "high",
        "low",
        "open",
        "close",
        "vol_ratio",
        "turnover_rate",
        "pe",
        "pb",
        "total_mv",
        "float_mv",
        "rise",
        "5min",
        "60day",
        "1year",
        "buy",
        "sale",
        "time",
        "rank",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    preferred_columns = [
        "rank",
        "stock_code",
        "ts_code",
        "name",
        "price",
        "pct_change",
        "change",
        "open",
        "high",
        "low",
        "volume",
        "amount",
        "turnover_rate",
        "vol_ratio",
        "pe",
        "pb",
        "total_mv",
        "float_mv",
        "swing",
        "rise",
        "5min",
        "60day",
        "1year",
        "time",
    ]
    ordered_columns = [column for column in preferred_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    sort_columns = [column for column in ["rank", "pct_change"] if column in df.columns]
    if sort_columns:
        df = df.sort_values(by=sort_columns, ascending=[True] + [False] * (len(sort_columns) - 1))
    return df[ordered_columns + remaining_columns].reset_index(drop=True)


def fetch_realtime_quotes(
    config: SystemConfig,
    ts_codes: list[str] | tuple[str, ...] | str,
    src: str = "auto",
    cache_ttl_seconds: float = REALTIME_CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """获取实时行情。支持 tushare(sina/dc) 和腾讯(tencent) 源。

    src: auto → 优先腾讯(免费实时)，fallback tushare
         tencent → 仅腾讯
         sina / dc → tushare 源
    """
    source = _normalize_realtime_source(src, {"auto", "sina", "dc", "tencent"})
    normalized_codes = _normalize_realtime_codes(ts_codes)
    cache_key = ("realtime_quote", source, ",".join(normalized_codes))
    cached = _get_cached_frame(cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached

    # ── 腾讯行情优先（免费，真正的实时价格）──
    if source in ("auto", "tencent"):
        # 从 ts_code 提取纯股票代码
        pure_codes = [c.split(".")[0] for c in normalized_codes]
        try:
            frame = fetch_tencent_quotes(pure_codes, timeout_seconds=5.0)
            if not frame.empty:
                frame = _normalize_tencent_quote_frame(frame)
                frame.attrs["realtime_source"] = "tencent"
                _store_cached_frame(cache_key, frame, cache_ttl_seconds)
                return frame
        except Exception as exc:
            logger.warning("腾讯行情获取失败，回退 tushare: %s", exc)

        if source == "tencent":
            # 用户明确只要腾讯，但失败了
            return pd.DataFrame()

    # ── Tushare fallback ──
    _ensure_tushare_ready(config)
    sources_to_try = ["sina", "dc"] if source == "auto" else [source]
    last_error: Exception | None = None
    normalized_frame = pd.DataFrame()
    actual_source = sources_to_try[0]
    for candidate_source in sources_to_try:
        try:
            frame = ts.realtime_quote(ts_code=",".join(normalized_codes), src=candidate_source)
            normalized_frame = _normalize_realtime_quote_frame(frame)
            actual_source = candidate_source
            break
        except Exception as exc:
            last_error = exc
            logger.warning("realtime_quote 获取失败 src=%s ts_codes=%s: %s", candidate_source, normalized_codes, exc)
    else:
        assert last_error is not None
        raise last_error

    normalized_frame.attrs["realtime_source"] = actual_source
    _store_cached_frame(cache_key, normalized_frame, cache_ttl_seconds)
    return normalized_frame


def fetch_realtime_market_list(
    config: SystemConfig,
    src: str = "auto",
    page_count: int | None = 1,
    cache_ttl_seconds: float = REALTIME_CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    _ensure_tushare_ready(config)
    source = _normalize_realtime_source(src, {"auto", "sina", "dc"})
    if page_count is not None and page_count <= 0:
        raise ValueError("page_count 必须大于 0。")

    cache_key = ("realtime_list", source, str(page_count or "all"))
    cached = _get_cached_frame(cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached

    sources_to_try = ["dc", "sina"] if source == "auto" else [source]
    last_error: Exception | None = None
    normalized_frame = pd.DataFrame()
    actual_source = sources_to_try[0]
    for candidate_source in sources_to_try:
        try:
            frame = ts.realtime_list(src=candidate_source, page_count=page_count)
            normalized_frame = _normalize_realtime_list_frame(frame)
            actual_source = candidate_source
            break
        except Exception as exc:
            last_error = exc
            logger.warning("realtime_list 获取失败 src=%s page_count=%s: %s", candidate_source, page_count, exc)
    else:
        assert last_error is not None
        raise last_error

    normalized_frame.attrs["realtime_source"] = actual_source
    _store_cached_frame(cache_key, normalized_frame, cache_ttl_seconds)
    return normalized_frame


def ensure_trade_data_schema(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_data (
                stock_code TEXT,
                日期 DATE,
                开盘 REAL,
                最高 REAL,
                最低 REAL,
                收盘 REAL,
                成交量 REAL,
                成交额 REAL,
                换手率 REAL,
                市盈率 REAL,
                市盈率TTM REAL,
                市净率 REAL,
                市销率TTM REAL,
                股息率TTM REAL,
                PRIMARY KEY (stock_code, 日期)
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(stock_data)").fetchall()}
        if "成交额" not in columns:
            conn.execute("ALTER TABLE stock_data ADD COLUMN 成交额 REAL")
        for column_name in ["市盈率", "市盈率TTM", "市净率", "市销率TTM", "股息率TTM"]:
            if column_name not in columns:
                conn.execute(f"ALTER TABLE stock_data ADD COLUMN {column_name} REAL")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_data_code_date
            ON stock_data(stock_code, 日期)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_basic_info (
                stock_code TEXT PRIMARY KEY,
                ts_code TEXT,
                名称 TEXT,
                地域 TEXT,
                行业 TEXT,
                市场 TEXT,
                上市日期 DATE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_basic_info_industry
            ON stock_basic_info(行业)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_concepts (
                concept_id TEXT PRIMARY KEY,
                concept_name TEXT,
                source TEXT,
                sort_index INTEGER
            )
            """
        )
        concept_columns = {row[1] for row in conn.execute("PRAGMA table_info(stock_concepts)").fetchall()}
        if "sort_index" not in concept_columns:
            conn.execute("ALTER TABLE stock_concepts ADD COLUMN sort_index INTEGER")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_concept_members (
                stock_code TEXT,
                concept_id TEXT,
                concept_name TEXT,
                ts_code TEXT,
                stock_name TEXT,
                PRIMARY KEY (stock_code, concept_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_concept_members_stock
            ON stock_concept_members(stock_code)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_concept_members_concept
            ON stock_concept_members(concept_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_concept_refresh_state (
                state_key TEXT PRIMARY KEY,
                state_value TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_moneyflow (
                stock_code TEXT,
                日期 DATE,
                小单买入量 REAL,
                小单买入额 REAL,
                小单卖出量 REAL,
                小单卖出额 REAL,
                中单买入量 REAL,
                中单买入额 REAL,
                中单卖出量 REAL,
                中单卖出额 REAL,
                大单买入量 REAL,
                大单买入额 REAL,
                大单卖出量 REAL,
                大单卖出额 REAL,
                特大单买入量 REAL,
                特大单买入额 REAL,
                特大单卖出量 REAL,
                特大单卖出额 REAL,
                净流入量 REAL,
                净流入额 REAL,
                PRIMARY KEY (stock_code, 日期)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_moneyflow_code_date
            ON stock_moneyflow(stock_code, 日期)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_top_inst (
                stock_code TEXT,
                日期 DATE,
                机构买入额 REAL,
                机构卖出额 REAL,
                机构净买额 REAL,
                机构席位数 REAL,
                PRIMARY KEY (stock_code, 日期)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_top_inst_code_date
            ON stock_top_inst(stock_code, 日期)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_margin_detail (
                stock_code TEXT,
                日期 DATE,
                融资余额 REAL,
                融券余额 REAL,
                融资买入额 REAL,
                融券余量 REAL,
                融资偿还额 REAL,
                融券偿还量 REAL,
                融券卖出量 REAL,
                融资融券余额 REAL,
                PRIMARY KEY (stock_code, 日期)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_margin_detail_code_date
            ON stock_margin_detail(stock_code, 日期)
            """
        )
        conn.commit()
    finally:
        conn.close()


def latest_trade_date(trade_db: str) -> str | None:
    conn = sqlite3.connect(trade_db)
    try:
        value = conn.execute("SELECT MAX(日期) FROM stock_data").fetchone()[0]
        return str(value) if value else None
    finally:
        conn.close()


def latest_moneyflow_date(trade_db: str) -> str | None:
    conn = sqlite3.connect(trade_db)
    try:
        value = conn.execute("SELECT MAX(日期) FROM stock_moneyflow").fetchone()[0]
        return str(value) if value else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def latest_top_inst_date(trade_db: str) -> str | None:
    conn = sqlite3.connect(trade_db)
    try:
        value = conn.execute("SELECT MAX(日期) FROM stock_top_inst").fetchone()[0]
        return str(value) if value else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def latest_margin_detail_date(trade_db: str) -> str | None:
    conn = sqlite3.connect(trade_db)
    try:
        value = conn.execute("SELECT MAX(日期) FROM stock_margin_detail").fetchone()[0]
        return str(value) if value else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def _format_trade_date(date_text: str | None) -> str | None:
    if not date_text:
        return None
    return pd.Timestamp(str(date_text)).strftime("%Y-%m-%d")


def _load_rows_for_date(trade_db: str, table_name: str, date_text: str) -> pd.DataFrame:
    conn = sqlite3.connect(trade_db)
    try:
        return pd.read_sql_query(
            f"SELECT * FROM {table_name} WHERE 日期 = ? ORDER BY stock_code",
            conn,
            params=[date_text],
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def _candidate_trade_dates(days_back: int = 10) -> list[str]:
    today = datetime.now().date()
    return [str(today - timedelta(days=offset)).replace("-", "") for offset in range(days_back + 1)]


def _fetch_trade_date_frames(pro, trade_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_df = _call_tushare_pro_api(pro.daily, trade_date=trade_date, fields="ts_code,trade_date,open,high,low,close,vol")
    basic_df = _call_tushare_pro_api(pro.daily_basic, trade_date=trade_date, fields=DAILY_BASIC_FIELDS)
    return daily_df, basic_df


def _fetch_trade_history_frames(pro, ts_code: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_df = _call_tushare_pro_api(
        pro.daily,
        ts_code=ts_code,
        start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        fields="ts_code,trade_date,open,high,low,close,vol",
    )
    basic_df = _call_tushare_pro_api(
        pro.daily_basic,
        ts_code=ts_code,
        start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        fields=DAILY_BASIC_FIELDS,
    )
    return daily_df, basic_df


def _fetch_stock_basic_frame(pro) -> pd.DataFrame:
    return _call_tushare_pro_api(pro.stock_basic, exchange="", list_status="L", fields=STOCK_BASIC_FIELDS)


def _fetch_moneyflow_frame(pro, trade_date: str) -> pd.DataFrame:
    return _call_tushare_pro_api(pro.moneyflow, trade_date=trade_date)


def _fetch_top_inst_frame(pro, trade_date: str) -> pd.DataFrame:
    return _call_tushare_pro_api(pro.top_inst, trade_date=trade_date)


def _fetch_margin_detail_frame(pro, trade_date: str) -> pd.DataFrame:
    return _call_tushare_pro_api(pro.margin_detail, trade_date=trade_date)


def _fetch_top_inst_frame_with_retry(pro, trade_date: str, retry_wait_seconds: float = TUSHARE_RATE_LIMIT_WAIT_SECONDS, max_attempts: int = 3) -> pd.DataFrame:
    for attempt in range(1, max_attempts + 1):
        try:
            frame = _fetch_top_inst_frame(pro, trade_date)
            return frame
        except Exception as exc:
            message = str(exc)
            if TUSHARE_RATE_LIMIT_MESSAGE not in message or attempt == max_attempts:
                raise
            logger.warning("top_inst 限流，等待后重试 trade_date=%s attempt=%s", trade_date, attempt)
            time.sleep(retry_wait_seconds)
    return pd.DataFrame()


def _fetch_margin_detail_frame_with_retry(pro, trade_date: str, retry_wait_seconds: float = TUSHARE_RATE_LIMIT_WAIT_SECONDS, max_attempts: int = 3) -> pd.DataFrame:
    for attempt in range(1, max_attempts + 1):
        try:
            frame = _fetch_margin_detail_frame(pro, trade_date)
            return frame
        except Exception as exc:
            message = str(exc)
            if TUSHARE_RATE_LIMIT_MESSAGE not in message or attempt == max_attempts:
                raise
            logger.warning("margin_detail 限流，等待后重试 trade_date=%s attempt=%s", trade_date, attempt)
            time.sleep(retry_wait_seconds)
    return pd.DataFrame()


def _resolve_latest_available_trade_date(pro, preferred_date: str | None = None) -> str:
    dates_to_try = [preferred_date] if preferred_date else []
    dates_to_try.extend(date_value for date_value in _candidate_trade_dates() if date_value not in dates_to_try)
    for trade_date in dates_to_try:
        if not trade_date:
            continue
        daily_df, _ = _fetch_trade_date_frames(pro, trade_date)
        if not daily_df.empty:
            return trade_date
    raise RuntimeError("最近 10 天内未从 Tushare 获取到任何日线行情。")


def _normalize_daily_frame(daily_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
    merged = daily_df.merge(basic_df, on=["ts_code", "trade_date"], how="left")
    merged["stock_code"] = merged["ts_code"].astype(str).str.split(".").str[0]
    merged["日期"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    merged["开盘"] = pd.to_numeric(merged["open"], errors="coerce")
    merged["最高"] = pd.to_numeric(merged["high"], errors="coerce")
    merged["最低"] = pd.to_numeric(merged["low"], errors="coerce")
    merged["收盘"] = pd.to_numeric(merged["close"], errors="coerce")
    merged["成交量"] = pd.to_numeric(merged["vol"], errors="coerce")
    merged["成交额"] = merged["收盘"] * merged["成交量"]
    merged["换手率"] = pd.to_numeric(merged["turnover_rate"], errors="coerce")
    merged["市盈率"] = pd.to_numeric(merged.get("pe"), errors="coerce")
    merged["市盈率TTM"] = pd.to_numeric(merged.get("pe_ttm"), errors="coerce")
    merged["市净率"] = pd.to_numeric(merged.get("pb"), errors="coerce")
    merged["市销率TTM"] = pd.to_numeric(merged.get("ps_ttm"), errors="coerce")
    merged["股息率TTM"] = pd.to_numeric(merged.get("dv_ttm"), errors="coerce")
    return merged[["stock_code", "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "市盈率", "市盈率TTM", "市净率", "市销率TTM", "股息率TTM"]].dropna(subset=["stock_code", "日期", "收盘"])


def _normalize_stock_basic_frame(stock_basic_df: pd.DataFrame) -> pd.DataFrame:
    if stock_basic_df.empty:
        return stock_basic_df
    df = stock_basic_df.copy()
    df["stock_code"] = df["symbol"].astype(str).str.zfill(6)
    df["ts_code"] = df["ts_code"].astype(str)
    df["名称"] = df.get("name", "").astype(str)
    df["地域"] = df.get("area", "").astype(str)
    df["行业"] = df.get("industry", "").astype(str)
    df["市场"] = df.get("market", "").astype(str)
    df["上市日期"] = pd.to_datetime(df.get("list_date"), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    return df[["stock_code", "ts_code", "名称", "地域", "行业", "市场", "上市日期"]].drop_duplicates(subset=["stock_code"], keep="last")


def _normalize_concept_frame(concept_df: pd.DataFrame) -> pd.DataFrame:
    if concept_df.empty:
        return concept_df
    df = concept_df.copy()
    df["sort_index"] = list(range(len(df)))
    df["concept_id"] = df.get("code", "").astype(str).str.strip()
    df["concept_name"] = df.get("name", "").astype(str).str.strip()
    df["source"] = df.get("src", "").astype(str).str.strip()
    df = df[(df["concept_id"] != "") & (df["concept_name"] != "")].copy()
    return df[["concept_id", "concept_name", "source", "sort_index"]].drop_duplicates(subset=["concept_id"], keep="last")


def _normalize_concept_detail_frame(concept_detail_df: pd.DataFrame) -> pd.DataFrame:
    if concept_detail_df.empty:
        return concept_detail_df
    df = concept_detail_df.copy()
    df["concept_id"] = df.get("id", "").astype(str).str.strip()
    df["concept_name"] = df.get("concept_name", "").astype(str).str.strip()
    df["ts_code"] = df.get("ts_code", "").astype(str).str.strip().str.upper()
    df["stock_code"] = df["ts_code"].str.split(".").str[0].str.zfill(6)
    df["stock_name"] = df.get("name", "").astype(str).str.strip()
    df = df[(df["concept_id"] != "") & (df["stock_code"] != "")].copy()
    return df[["stock_code", "ts_code", "concept_id", "concept_name", "stock_name"]].drop_duplicates(
        subset=["stock_code", "concept_id"], keep="last"
    )


def _get_concept_refresh_state(trade_db: str, key: str) -> str | None:
    conn = sqlite3.connect(trade_db)
    try:
        row = conn.execute(
            "SELECT state_value FROM stock_concept_refresh_state WHERE state_key = ?",
            (key,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])
    finally:
        conn.close()


def _set_concept_refresh_state(trade_db: str, key: str, value: str) -> None:
    conn = sqlite3.connect(trade_db)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO stock_concept_refresh_state(state_key, state_value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, str(pd.Timestamp.utcnow())),
        )
        conn.commit()
    finally:
        conn.close()


def _load_all_concept_members(trade_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(trade_db)
    try:
        return pd.read_sql_query(
            "SELECT stock_code, concept_id, concept_name, ts_code, stock_name FROM stock_concept_members ORDER BY concept_id, stock_code",
            conn,
        )
    finally:
        conn.close()


def _load_existing_concept_frame(trade_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(trade_db)
    try:
        return pd.read_sql_query(
            "SELECT concept_id, concept_name, source, sort_index FROM stock_concepts ORDER BY COALESCE(sort_index, 2147483647), concept_id",
            conn,
        )
    finally:
        conn.close()


def get_stock_concept_refresh_progress(config: SystemConfig) -> dict[str, object]:
    ensure_trade_data_schema(config.databases.trade_db)
    conn = sqlite3.connect(config.databases.trade_db)
    try:
        concept_ids = [
            str(row[0])
            for row in conn.execute(
                "SELECT concept_id FROM stock_concepts ORDER BY COALESCE(sort_index, 2147483647), concept_id"
            ).fetchall()
            if row and row[0] is not None
        ]
        member_rows = int(conn.execute("SELECT COUNT(*) FROM stock_concept_members").fetchone()[0])
        stock_rows = int(conn.execute("SELECT COUNT(DISTINCT stock_code) FROM stock_concept_members").fetchone()[0])
        covered_concepts = int(conn.execute("SELECT COUNT(DISTINCT concept_id) FROM stock_concept_members").fetchone()[0])
        avg_concepts_per_stock = conn.execute(
            "SELECT AVG(cnt) FROM (SELECT stock_code, COUNT(*) AS cnt FROM stock_concept_members GROUP BY stock_code)"
        ).fetchone()[0]
    finally:
        conn.close()

    total_concepts = len(concept_ids)
    last_concept_id = _get_concept_refresh_state(config.databases.trade_db, "last_concept_id") or ""
    last_cycle_completed_at = _get_concept_refresh_state(config.databases.trade_db, "last_cycle_completed_at")

    if total_concepts == 0:
        return {
            "total_concepts": 0,
            "covered_concepts": 0,
            "covered_concept_ratio": 0.0,
            "member_rows": member_rows,
            "stock_rows": stock_rows,
            "avg_concepts_per_stock": float(avg_concepts_per_stock or 0.0),
            "last_concept_id": "",
            "last_cycle_completed_at": last_cycle_completed_at,
            "processed_concepts": 0,
            "processed_ratio": 0.0,
            "next_concept_id": None,
            "cycle_completed": False,
        }

    if last_concept_id and last_concept_id in concept_ids:
        processed_concepts = concept_ids.index(last_concept_id) + 1
        next_concept_id = concept_ids[processed_concepts] if processed_concepts < total_concepts else None
        cycle_completed = False
    else:
        processed_concepts = total_concepts if last_cycle_completed_at else 0
        next_concept_id = concept_ids[0] if not last_cycle_completed_at else None
        cycle_completed = bool(last_cycle_completed_at)

    return {
        "total_concepts": total_concepts,
        "covered_concepts": covered_concepts,
        "covered_concept_ratio": float(covered_concepts / total_concepts) if total_concepts > 0 else 0.0,
        "member_rows": member_rows,
        "stock_rows": stock_rows,
        "avg_concepts_per_stock": float(avg_concepts_per_stock or 0.0),
        "last_concept_id": last_concept_id,
        "last_cycle_completed_at": last_cycle_completed_at,
        "processed_concepts": processed_concepts,
        "processed_ratio": float(processed_concepts / total_concepts) if total_concepts > 0 else 0.0,
        "next_concept_id": next_concept_id,
        "cycle_completed": cycle_completed,
    }


def _normalize_moneyflow_frame(moneyflow_df: pd.DataFrame) -> pd.DataFrame:
    if moneyflow_df.empty:
        return moneyflow_df
    df = moneyflow_df.copy()
    df["stock_code"] = df["ts_code"].astype(str).str.split(".").str[0]
    df["日期"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    rename_map = {
        "buy_sm_vol": "小单买入量",
        "buy_sm_amount": "小单买入额",
        "sell_sm_vol": "小单卖出量",
        "sell_sm_amount": "小单卖出额",
        "buy_md_vol": "中单买入量",
        "buy_md_amount": "中单买入额",
        "sell_md_vol": "中单卖出量",
        "sell_md_amount": "中单卖出额",
        "buy_lg_vol": "大单买入量",
        "buy_lg_amount": "大单买入额",
        "sell_lg_vol": "大单卖出量",
        "sell_lg_amount": "大单卖出额",
        "buy_elg_vol": "特大单买入量",
        "buy_elg_amount": "特大单买入额",
        "sell_elg_vol": "特大单卖出量",
        "sell_elg_amount": "特大单卖出额",
        "net_mf_vol": "净流入量",
        "net_mf_amount": "净流入额",
    }
    df = df.rename(columns=rename_map)
    keep_cols = ["stock_code", "日期", *rename_map.values()]
    for col in rename_map.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[keep_cols].dropna(subset=["stock_code", "日期"])


def _known_codes_from_trade_db(trade_db: str) -> set[str]:
    conn = sqlite3.connect(trade_db)
    try:
        rows = conn.execute("SELECT DISTINCT stock_code FROM stock_data").fetchall()
        return {str(row[0]) for row in rows if row and row[0] is not None}
    finally:
        conn.close()


def _trade_dates_missing_valuation_fields(trade_db: str, start_date: str, end_date: str) -> list[str]:
    conn = sqlite3.connect(trade_db)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT 日期
            FROM stock_data
            WHERE 日期 >= ?
              AND 日期 <= ?
              AND (
                  换手率 IS NULL
                  OR 市盈率 IS NULL
                  OR 市盈率TTM IS NULL
                  OR 市净率 IS NULL
                  OR 市销率TTM IS NULL
                  OR 股息率TTM IS NULL
              )
            ORDER BY 日期
            """,
            (start_date, end_date),
        ).fetchall()
        return [str(row[0]) for row in rows if row and row[0] is not None]
    finally:
        conn.close()


def _normalize_top_inst_frame(top_inst_df: pd.DataFrame) -> pd.DataFrame:
    if top_inst_df.empty:
        return top_inst_df
    df = top_inst_df.copy()
    df["stock_code"] = df["ts_code"].astype(str).str.split(".").str[0]
    df["日期"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["buy", "sell", "net_buy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["stock_code", "日期", "exalter", "buy", "sell", "net_buy"], keep="first")
    grouped = (
        df.groupby(["stock_code", "日期"], as_index=False)
        .agg(
            机构买入额=("buy", "sum"),
            机构卖出额=("sell", "sum"),
            机构净买额=("net_buy", "sum"),
            机构席位数=("exalter", "nunique"),
        )
    )
    return grouped


def _normalize_margin_detail_frame(margin_df: pd.DataFrame) -> pd.DataFrame:
    if margin_df.empty:
        return margin_df
    df = margin_df.copy()
    df["stock_code"] = df["ts_code"].astype(str).str.split(".").str[0]
    df["日期"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    rename_map = {
        "rzye": "融资余额",
        "rqye": "融券余额",
        "rzmre": "融资买入额",
        "rqyl": "融券余量",
        "rzche": "融资偿还额",
        "rqchl": "融券偿还量",
        "rqmcl": "融券卖出量",
        "rzrqye": "融资融券余额",
    }
    df = df.rename(columns=rename_map)
    keep_cols = ["stock_code", "日期", *rename_map.values()]
    for col in rename_map.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[keep_cols].dropna(subset=["stock_code", "日期"])


def update_latest_trade_data(config: SystemConfig, trade_date: str | None = None, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新交易数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    resolved_trade_date = _resolve_latest_available_trade_date(pro, trade_date)
    resolved_trade_date_text = _format_trade_date(resolved_trade_date)
    existing_trade_date = latest_trade_date(config.databases.trade_db)
    if existing_trade_date == resolved_trade_date_text:
        existing_rows = _load_rows_for_date(config.databases.trade_db, "stock_data", resolved_trade_date_text)
        if not existing_rows.empty:
            logger.info("最新行情已存在，跳过拉取 trade_date=%s rows=%s", resolved_trade_date, len(existing_rows))
            return existing_rows

    daily_df, basic_df = _fetch_trade_date_frames(pro, resolved_trade_date)
    if daily_df.empty:
        raise RuntimeError(f"trade_date={resolved_trade_date} 未获取到行情数据。")

    normalized = _normalize_daily_frame(daily_df, basic_df)
    if only_known_codes:
        known_codes = set(get_all_stock_codes(config.databases.trade_db))
        if known_codes:
            normalized = normalized[normalized["stock_code"].isin(known_codes)].reset_index(drop=True)

    if normalized.empty:
        raise RuntimeError("最新行情为空，无法写入交易库。")

    conn = sqlite3.connect(config.databases.trade_db)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_data(stock_code, 日期, 开盘, 最高, 最低, 收盘, 成交量, 成交额, 换手率, 市盈率, 市盈率TTM, 市净率, 市销率TTM, 股息率TTM)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalized[["stock_code", "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "市盈率", "市盈率TTM", "市净率", "市销率TTM", "股息率TTM"]].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("最新行情更新完成 trade_date=%s rows=%s", resolved_trade_date, len(normalized))
    return normalized


def update_trade_data_for_codes(config: SystemConfig, stock_codes: list[str], trade_date: str | None = None) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新交易数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    resolved_trade_date = _resolve_latest_available_trade_date(pro, trade_date)

    rows: list[dict] = []
    basic_df = _call_tushare_pro_api(pro.daily_basic, trade_date=resolved_trade_date, fields=DAILY_BASIC_FIELDS)
    basic_map = basic_df.set_index("ts_code").to_dict("index") if not basic_df.empty else {}

    for stock_code in stock_codes:
        ts_code = normalize_ts_code(stock_code)
        daily_df = _call_tushare_pro_api(pro.daily, ts_code=ts_code, trade_date=resolved_trade_date, fields="ts_code,trade_date,open,high,low,close,vol")
        if daily_df.empty:
            continue
        latest = daily_df.iloc[-1]
        latest_basic = basic_map.get(ts_code, {})
        rows.append(
            {
                "stock_code": str(stock_code).split(".")[0].zfill(6),
                "日期": datetime.strptime(str(latest["trade_date"]), "%Y%m%d").strftime("%Y-%m-%d"),
                "开盘": float(latest["open"]),
                "最高": float(latest["high"]),
                "最低": float(latest["low"]),
                "收盘": float(latest["close"]),
                "成交量": float(latest["vol"]),
                "成交额": float(latest["close"] * latest["vol"]),
                "换手率": float(latest_basic["turnover_rate"]) if latest_basic.get("turnover_rate") is not None else None,
                "市盈率": float(latest_basic["pe"]) if latest_basic.get("pe") is not None else None,
                "市盈率TTM": float(latest_basic["pe_ttm"]) if latest_basic.get("pe_ttm") is not None else None,
                "市净率": float(latest_basic["pb"]) if latest_basic.get("pb") is not None else None,
                "市销率TTM": float(latest_basic["ps_ttm"]) if latest_basic.get("ps_ttm") is not None else None,
                "股息率TTM": float(latest_basic["dv_ttm"]) if latest_basic.get("dv_ttm") is not None else None,
            }
        )

    if not rows:
        raise RuntimeError("指定股票未获取到任何最新行情。")

    df = pd.DataFrame(rows)
    conn = sqlite3.connect(config.databases.trade_db)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_data(stock_code, 日期, 开盘, 最高, 最低, 收盘, 成交量, 成交额, 换手率, 市盈率, 市盈率TTM, 市净率, 市销率TTM, 股息率TTM)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            df[["stock_code", "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "市盈率", "市盈率TTM", "市净率", "市销率TTM", "股息率TTM"]].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("指定股票行情更新完成 trade_date=%s rows=%s", resolved_trade_date, len(df))
    return df


def update_trade_history(
    config: SystemConfig,
    start_date: str,
    end_date: str,
    only_known_codes: bool = True,
    sleep_seconds: float = 0.12,
) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法回填交易历史数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    if only_known_codes:
        stock_codes = get_all_stock_codes(config.databases.trade_db)
    else:
        stock_codes = sorted(_known_codes_from_trade_db(config.databases.trade_db))

    if not stock_codes:
        raise RuntimeError("交易库中没有股票代码，无法回填历史行情。")

    all_frames: list[pd.DataFrame] = []
    start_text = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    for index, stock_code in enumerate(stock_codes, start=1):
        ts_code = normalize_ts_code(stock_code)
        try:
            daily_df, basic_df = _fetch_trade_history_frames(pro, ts_code, start_text, end_text)
        except Exception as exc:
            logger.warning("历史行情抓取失败 stock_code=%s error=%s", stock_code, exc)
            time.sleep(sleep_seconds)
            continue

        if daily_df.empty:
            time.sleep(sleep_seconds)
            continue

        normalized = _normalize_daily_frame(daily_df, basic_df)
        if normalized.empty:
            time.sleep(sleep_seconds)
            continue

        all_frames.append(normalized)
        conn = sqlite3.connect(config.databases.trade_db)
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_data(stock_code, 日期, 开盘, 最高, 最低, 收盘, 成交量, 成交额, 换手率, 市盈率, 市盈率TTM, 市净率, 市销率TTM, 股息率TTM)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                normalized[["stock_code", "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "市盈率", "市盈率TTM", "市净率", "市销率TTM", "股息率TTM"]].itertuples(index=False, name=None),
            )
            conn.commit()
        finally:
            conn.close()

        if index % 200 == 0:
            logger.info("历史行情回填进度 processed=%s/%s latest_stock=%s rows=%s", index, len(stock_codes), stock_code, len(normalized))
        time.sleep(sleep_seconds)

    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames, ignore_index=True)
    logger.info("交易历史回填完成 rows=%s start=%s end=%s stocks=%s", len(result), start_text, end_text, len(stock_codes))
    return result


def backfill_trade_valuation_fields(
    config: SystemConfig,
    start_date: str,
    end_date: str,
    only_known_codes: bool = True,
    sleep_seconds: float = 0.12,
) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法回填估值字段。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    start_text = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    trade_dates = _trade_dates_missing_valuation_fields(config.databases.trade_db, start_text, end_text)
    if not trade_dates:
        logger.info("估值字段已覆盖，无需回填 start=%s end=%s", start_text, end_text)
        return pd.DataFrame()

    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    known_codes = _known_codes_from_trade_db(config.databases.trade_db) if only_known_codes else set()
    updated_frames: list[pd.DataFrame] = []

    for index, trade_date_text in enumerate(trade_dates, start=1):
        trade_date = pd.Timestamp(trade_date_text).strftime("%Y%m%d")
        try:
            basic_df = _call_tushare_pro_api(pro.daily_basic, trade_date=trade_date, fields=DAILY_BASIC_FIELDS)
        except Exception as exc:
            logger.warning("估值字段抓取失败 trade_date=%s error=%s", trade_date, exc)
            time.sleep(sleep_seconds)
            continue

        if basic_df.empty:
            time.sleep(sleep_seconds)
            continue

        normalized = basic_df.copy()
        normalized["stock_code"] = normalized["ts_code"].astype(str).str.split(".").str[0]
        normalized["日期"] = pd.to_datetime(normalized["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        normalized["换手率"] = pd.to_numeric(normalized.get("turnover_rate"), errors="coerce")
        normalized["市盈率"] = pd.to_numeric(normalized.get("pe"), errors="coerce")
        normalized["市盈率TTM"] = pd.to_numeric(normalized.get("pe_ttm"), errors="coerce")
        normalized["市净率"] = pd.to_numeric(normalized.get("pb"), errors="coerce")
        normalized["市销率TTM"] = pd.to_numeric(normalized.get("ps_ttm"), errors="coerce")
        normalized["股息率TTM"] = pd.to_numeric(normalized.get("dv_ttm"), errors="coerce")
        normalized = normalized[
            ["stock_code", "日期", "换手率", "市盈率", "市盈率TTM", "市净率", "市销率TTM", "股息率TTM"]
        ].dropna(subset=["stock_code", "日期"])

        if known_codes:
            normalized = normalized[normalized["stock_code"].isin(known_codes)].reset_index(drop=True)
        if normalized.empty:
            time.sleep(sleep_seconds)
            continue

        conn = sqlite3.connect(config.databases.trade_db)
        try:
            conn.executemany(
                """
                UPDATE stock_data
                SET 换手率 = ?,
                    市盈率 = ?,
                    市盈率TTM = ?,
                    市净率 = ?,
                    市销率TTM = ?,
                    股息率TTM = ?
                WHERE stock_code = ? AND 日期 = ?
                """,
                normalized[["换手率", "市盈率", "市盈率TTM", "市净率", "市销率TTM", "股息率TTM", "stock_code", "日期"]].itertuples(index=False, name=None),
            )
            conn.commit()
        finally:
            conn.close()

        updated_frames.append(normalized)
        if index % 20 == 0 or index == len(trade_dates):
            logger.info("估值字段回填进度 processed=%s/%s latest_date=%s rows=%s", index, len(trade_dates), trade_date_text, len(normalized))
        time.sleep(sleep_seconds)

    if not updated_frames:
        return pd.DataFrame()
    result = pd.concat(updated_frames, ignore_index=True)
    logger.info("估值字段回填完成 rows=%s start=%s end=%s dates=%s", len(result), start_text, end_text, len(trade_dates))
    return result


def update_stock_basic_info(config: SystemConfig, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新股票静态基本信息。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    stock_basic_df = _fetch_stock_basic_frame(pro)
    normalized = _normalize_stock_basic_frame(stock_basic_df)
    if only_known_codes and not normalized.empty:
        known_codes = _known_codes_from_trade_db(config.databases.trade_db)
        if known_codes:
            normalized = normalized[normalized["stock_code"].isin(known_codes)].reset_index(drop=True)

    conn = sqlite3.connect(config.databases.trade_db)
    try:
        if not normalized.empty:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_basic_info(stock_code, ts_code, 名称, 地域, 行业, 市场, 上市日期)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                normalized[["stock_code", "ts_code", "名称", "地域", "行业", "市场", "上市日期"]].itertuples(index=False, name=None),
            )
            conn.commit()
    finally:
        conn.close()

    logger.info("股票静态基本信息更新完成 rows=%s", len(normalized))
    return normalized


def update_stock_concepts(config: SystemConfig, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新股票概念关系。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    concept_frame = pd.DataFrame()
    try:
        concept_frame = _normalize_concept_frame(_call_tushare_pro_api(pro.concept))
    except Exception as exc:
        logger.warning("概念主表刷新失败，改用本地已保存概念表: %s", exc)
        concept_frame = _load_existing_concept_frame(config.databases.trade_db)

    if concept_frame.empty:
        logger.warning("概念主表为空，跳过概念关系刷新，保留现有数据。")
        return _load_all_concept_members(config.databases.trade_db)

    conn = sqlite3.connect(config.databases.trade_db)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_concepts(concept_id, concept_name, source, sort_index)
            VALUES (?, ?, ?, ?)
            """,
            concept_frame[["concept_id", "concept_name", "source", "sort_index"]].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()

    known_codes = _known_codes_from_trade_db(config.databases.trade_db) if only_known_codes else set()
    concept_ids = concept_frame["concept_id"].astype(str).tolist()
    last_concept_id = _get_concept_refresh_state(config.databases.trade_db, "last_concept_id")
    start_index = 0
    if last_concept_id:
        try:
            start_index = concept_ids.index(last_concept_id) + 1
        except ValueError:
            start_index = 0
    batch_ids = concept_ids[start_index : start_index + CONCEPT_DETAIL_CALLS_PER_RUN]
    if not batch_ids:
        batch_ids = concept_ids[:CONCEPT_DETAIL_CALLS_PER_RUN]
        start_index = 0

    processed_count = 0
    for index, concept_id in enumerate(batch_ids, start=1):
        try:
            detail_frame = _normalize_concept_detail_frame(_call_tushare_pro_api(pro.concept_detail, id=concept_id))
        except Exception as exc:
            message = str(exc)
            if TUSHARE_RATE_LIMIT_MESSAGE in message:
                logger.warning("概念成分拉取命中限流，暂停本轮续刷 concept_id=%s: %s", concept_id, exc)
                break
            logger.warning("概念成分拉取失败 concept_id=%s: %s", concept_id, exc)
            continue
        if detail_frame.empty:
            continue
        if known_codes:
            detail_frame = detail_frame[detail_frame["stock_code"].isin(known_codes)].reset_index(drop=True)
        conn = sqlite3.connect(config.databases.trade_db)
        try:
            conn.execute("DELETE FROM stock_concept_members WHERE concept_id = ?", (concept_id,))
            if not detail_frame.empty:
                detail_frame = detail_frame.merge(
                    concept_frame[["concept_id", "concept_name"]],
                    on="concept_id",
                    how="left",
                    suffixes=("", "_master"),
                )
                detail_frame["concept_name"] = detail_frame["concept_name_master"].fillna(detail_frame["concept_name"])
                detail_frame["concept_name"] = detail_frame["concept_name"].astype(str).str.strip()
                detail_frame = detail_frame.drop(columns=["concept_name_master"])
                detail_frame = detail_frame.drop_duplicates(subset=["stock_code", "concept_id"], keep="last")
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO stock_concept_members(stock_code, concept_id, concept_name, ts_code, stock_name)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    detail_frame[["stock_code", "concept_id", "concept_name", "ts_code", "stock_name"]].itertuples(index=False, name=None),
                )
            conn.commit()
        finally:
            conn.close()
        processed_count += 1
        if index % 50 == 0 or index == len(batch_ids):
            logger.info(
                "概念关系续刷进度 batch=%s/%s global=%s/%s current_concept=%s",
                index,
                len(batch_ids),
                start_index + index,
                len(concept_ids),
                concept_id,
            )
        time.sleep(0.05)

    if processed_count > 0:
        last_processed_concept_id = batch_ids[processed_count - 1]
        if start_index + processed_count >= len(concept_ids):
            _set_concept_refresh_state(config.databases.trade_db, "last_concept_id", "")
            _set_concept_refresh_state(config.databases.trade_db, "last_cycle_completed_at", str(pd.Timestamp.utcnow()))
        else:
            _set_concept_refresh_state(config.databases.trade_db, "last_concept_id", last_processed_concept_id)

    members_frame = _load_all_concept_members(config.databases.trade_db)

    logger.info(
        "股票概念关系续刷完成 processed=%s start_index=%s concepts=%s member_rows=%s unique_stocks=%s",
        processed_count,
        start_index,
        len(concept_frame),
        len(members_frame),
        members_frame["stock_code"].nunique() if not members_frame.empty else 0,
    )
    return members_frame


def update_latest_moneyflow_data(config: SystemConfig, trade_date: str | None = None, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新资金流向数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    resolved_trade_date = _resolve_latest_available_trade_date(pro, trade_date)
    resolved_trade_date_text = _format_trade_date(resolved_trade_date)
    existing_moneyflow_date = latest_moneyflow_date(config.databases.trade_db)
    if existing_moneyflow_date == resolved_trade_date_text:
        existing_rows = _load_rows_for_date(config.databases.trade_db, "stock_moneyflow", resolved_trade_date_text)
        if not existing_rows.empty:
            logger.info("最新资金流已存在，跳过拉取 trade_date=%s rows=%s", resolved_trade_date, len(existing_rows))
            return existing_rows

    moneyflow_df = _fetch_moneyflow_frame(pro, resolved_trade_date)
    if moneyflow_df.empty:
        raise RuntimeError(f"trade_date={resolved_trade_date} 未获取到资金流向数据。")

    normalized = _normalize_moneyflow_frame(moneyflow_df)
    if only_known_codes:
        known_codes = _known_codes_from_trade_db(config.databases.trade_db)
        if known_codes:
            normalized = normalized[normalized["stock_code"].isin(known_codes)].reset_index(drop=True)
    if normalized.empty:
        raise RuntimeError("最新资金流数据为空，无法写入交易库。")

    conn = sqlite3.connect(config.databases.trade_db)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_moneyflow(
                stock_code, 日期, 小单买入量, 小单买入额, 小单卖出量, 小单卖出额,
                中单买入量, 中单买入额, 中单卖出量, 中单卖出额,
                大单买入量, 大单买入额, 大单卖出量, 大单卖出额,
                特大单买入量, 特大单买入额, 特大单卖出量, 特大单卖出额,
                净流入量, 净流入额
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalized[
                [
                    "stock_code", "日期", "小单买入量", "小单买入额", "小单卖出量", "小单卖出额",
                    "中单买入量", "中单买入额", "中单卖出量", "中单卖出额",
                    "大单买入量", "大单买入额", "大单卖出量", "大单卖出额",
                    "特大单买入量", "特大单买入额", "特大单卖出量", "特大单卖出额",
                    "净流入量", "净流入额",
                ]
            ].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("最新资金流更新完成 trade_date=%s rows=%s", resolved_trade_date, len(normalized))
    return normalized


def update_moneyflow_history(config: SystemConfig, start_date: str | None = None, end_date: str | None = None, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新资金流向数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    conn = sqlite3.connect(config.databases.trade_db)
    try:
        query = "SELECT DISTINCT 日期 FROM stock_data"
        params: list[str] = []
        clauses: list[str] = []
        if start_date:
            clauses.append("日期 >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("日期 <= ?")
            params.append(end_date)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY 日期"
        trade_dates = [str(row[0]) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()

    if not trade_dates:
        logger.info("资金流历史回填跳过：区间内没有可用交易日 start=%s end=%s", start_date, end_date)
        return pd.DataFrame()

    latest_existing = latest_moneyflow_date(config.databases.trade_db)
    if latest_existing and not start_date:
        trade_dates = [date_value for date_value in trade_dates if date_value > latest_existing]
    if not trade_dates:
        return pd.DataFrame()

    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    known_codes = _known_codes_from_trade_db(config.databases.trade_db) if only_known_codes else set()
    all_frames: list[pd.DataFrame] = []

    for date_value in trade_dates:
        trade_date = pd.Timestamp(date_value).strftime("%Y%m%d")
        frame = _normalize_moneyflow_frame(_fetch_moneyflow_frame(pro, trade_date))
        if frame.empty:
            continue
        if known_codes:
            frame = frame[frame["stock_code"].isin(known_codes)].reset_index(drop=True)
        if frame.empty:
            continue
        all_frames.append(frame)

        conn = sqlite3.connect(config.databases.trade_db)
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_moneyflow(
                    stock_code, 日期, 小单买入量, 小单买入额, 小单卖出量, 小单卖出额,
                    中单买入量, 中单买入额, 中单卖出量, 中单卖出额,
                    大单买入量, 大单买入额, 大单卖出量, 大单卖出额,
                    特大单买入量, 特大单买入额, 特大单卖出量, 特大单卖出额,
                    净流入量, 净流入额
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                frame[
                    [
                        "stock_code", "日期", "小单买入量", "小单买入额", "小单卖出量", "小单卖出额",
                        "中单买入量", "中单买入额", "中单卖出量", "中单卖出额",
                        "大单买入量", "大单买入额", "大单卖出量", "大单卖出额",
                        "特大单买入量", "特大单买入额", "特大单卖出量", "特大单卖出额",
                        "净流入量", "净流入额",
                    ]
                ].itertuples(index=False, name=None),
            )
            conn.commit()
        finally:
            conn.close()

    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames, ignore_index=True)
    logger.info("资金流历史回填完成 rows=%s start=%s end=%s", len(result), trade_dates[0], trade_dates[-1])
    return result


def update_latest_top_inst_data(config: SystemConfig, trade_date: str | None = None, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新机构龙虎榜数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    resolved_trade_date = _resolve_latest_available_trade_date(pro, trade_date)
    resolved_trade_date_text = _format_trade_date(resolved_trade_date)
    existing_top_inst_date = latest_top_inst_date(config.databases.trade_db)
    if existing_top_inst_date == resolved_trade_date_text:
        existing_rows = _load_rows_for_date(config.databases.trade_db, "stock_top_inst", resolved_trade_date_text)
        if not existing_rows.empty:
            logger.info("最新龙虎榜机构数据已存在，跳过拉取 trade_date=%s rows=%s", resolved_trade_date, len(existing_rows))
            return existing_rows

    top_inst_df = _fetch_top_inst_frame(pro, resolved_trade_date)
    normalized = _normalize_top_inst_frame(top_inst_df)
    if only_known_codes and not normalized.empty:
        known_codes = _known_codes_from_trade_db(config.databases.trade_db)
        if known_codes:
            normalized = normalized[normalized["stock_code"].isin(known_codes)].reset_index(drop=True)

    conn = sqlite3.connect(config.databases.trade_db)
    try:
        if not normalized.empty:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_top_inst(stock_code, 日期, 机构买入额, 机构卖出额, 机构净买额, 机构席位数)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                normalized[["stock_code", "日期", "机构买入额", "机构卖出额", "机构净买额", "机构席位数"]].itertuples(index=False, name=None),
            )
            conn.commit()
    finally:
        conn.close()

    logger.info("最新龙虎榜机构数据更新完成 trade_date=%s rows=%s", resolved_trade_date, len(normalized))
    return normalized


def update_top_inst_history(config: SystemConfig, start_date: str | None = None, end_date: str | None = None, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新机构龙虎榜数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    conn = sqlite3.connect(config.databases.trade_db)
    try:
        query = "SELECT DISTINCT 日期 FROM stock_data"
        params: list[str] = []
        clauses: list[str] = []
        if start_date:
            clauses.append("日期 >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("日期 <= ?")
            params.append(end_date)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY 日期"
        trade_dates = [str(row[0]) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()

    if not trade_dates:
        logger.info("机构龙虎榜历史回填跳过：区间内没有可用交易日 start=%s end=%s", start_date, end_date)
        return pd.DataFrame()

    latest_existing = latest_top_inst_date(config.databases.trade_db)
    if latest_existing and not start_date:
        trade_dates = [date_value for date_value in trade_dates if date_value > latest_existing]
    if not trade_dates:
        return pd.DataFrame()

    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    known_codes = _known_codes_from_trade_db(config.databases.trade_db) if only_known_codes else set()
    all_frames: list[pd.DataFrame] = []

    for date_value in trade_dates:
        trade_date = pd.Timestamp(date_value).strftime("%Y%m%d")
        frame = _normalize_top_inst_frame(_fetch_top_inst_frame_with_retry(pro, trade_date))
        if frame.empty:
            continue
        if known_codes:
            frame = frame[frame["stock_code"].isin(known_codes)].reset_index(drop=True)
        if frame.empty:
            continue
        all_frames.append(frame)

        conn = sqlite3.connect(config.databases.trade_db)
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_top_inst(stock_code, 日期, 机构买入额, 机构卖出额, 机构净买额, 机构席位数)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                frame[["stock_code", "日期", "机构买入额", "机构卖出额", "机构净买额", "机构席位数"]].itertuples(index=False, name=None),
            )
            conn.commit()
        finally:
            conn.close()

    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames, ignore_index=True)
    logger.info("机构龙虎榜历史回填完成 rows=%s start=%s end=%s", len(result), trade_dates[0], trade_dates[-1])
    return result


def update_latest_margin_detail_data(config: SystemConfig, trade_date: str | None = None, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新融资融券明细数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    ts.set_token(config.tushare_token)
    pro = ts.pro_api()

    resolved_trade_date = _resolve_latest_available_trade_date(pro, trade_date)
    resolved_trade_date_text = _format_trade_date(resolved_trade_date)
    existing_margin_date = latest_margin_detail_date(config.databases.trade_db)
    if existing_margin_date == resolved_trade_date_text:
        existing_rows = _load_rows_for_date(config.databases.trade_db, "stock_margin_detail", resolved_trade_date_text)
        if not existing_rows.empty:
            logger.info("最新融资融券明细已存在，跳过拉取 trade_date=%s rows=%s", resolved_trade_date, len(existing_rows))
            return existing_rows

    margin_df = _fetch_margin_detail_frame(pro, resolved_trade_date)
    normalized = _normalize_margin_detail_frame(margin_df)
    if only_known_codes and not normalized.empty:
        known_codes = _known_codes_from_trade_db(config.databases.trade_db)
        if known_codes:
            normalized = normalized[normalized["stock_code"].isin(known_codes)].reset_index(drop=True)

    conn = sqlite3.connect(config.databases.trade_db)
    try:
        if not normalized.empty:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_margin_detail(
                    stock_code, 日期, 融资余额, 融券余额, 融资买入额, 融券余量,
                    融资偿还额, 融券偿还量, 融券卖出量, 融资融券余额
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                normalized[
                    [
                        "stock_code", "日期", "融资余额", "融券余额", "融资买入额", "融券余量",
                        "融资偿还额", "融券偿还量", "融券卖出量", "融资融券余额",
                    ]
                ].itertuples(index=False, name=None),
            )
            conn.commit()
    finally:
        conn.close()

    logger.info("最新融资融券明细更新完成 trade_date=%s rows=%s", resolved_trade_date, len(normalized))
    return normalized


def update_margin_detail_history(config: SystemConfig, start_date: str | None = None, end_date: str | None = None, only_known_codes: bool = True) -> pd.DataFrame:
    if ts is None:
        raise RuntimeError("未安装 tushare，无法更新融资融券明细数据。")
    if not config.tushare_token:
        raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量或 config.yaml 中的 token。")

    ensure_trade_data_schema(config.databases.trade_db)
    conn = sqlite3.connect(config.databases.trade_db)
    try:
        query = "SELECT DISTINCT 日期 FROM stock_data"
        params: list[str] = []
        clauses: list[str] = []
        if start_date:
            clauses.append("日期 >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("日期 <= ?")
            params.append(end_date)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY 日期"
        trade_dates = [str(row[0]) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()

    if not trade_dates:
        logger.info("融资融券明细历史回填跳过：区间内没有可用交易日 start=%s end=%s", start_date, end_date)
        return pd.DataFrame()

    latest_existing = latest_margin_detail_date(config.databases.trade_db)
    if latest_existing and not start_date:
        trade_dates = [date_value for date_value in trade_dates if date_value > latest_existing]
    if not trade_dates:
        return pd.DataFrame()

    ts.set_token(config.tushare_token)
    pro = ts.pro_api()
    known_codes = _known_codes_from_trade_db(config.databases.trade_db) if only_known_codes else set()
    all_frames: list[pd.DataFrame] = []

    for date_value in trade_dates:
        trade_date = pd.Timestamp(date_value).strftime("%Y%m%d")
        frame = _normalize_margin_detail_frame(_fetch_margin_detail_frame_with_retry(pro, trade_date))
        if frame.empty:
            continue
        if known_codes:
            frame = frame[frame["stock_code"].isin(known_codes)].reset_index(drop=True)
        if frame.empty:
            continue
        all_frames.append(frame)

        conn = sqlite3.connect(config.databases.trade_db)
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_margin_detail(
                    stock_code, 日期, 融资余额, 融券余额, 融资买入额, 融券余量,
                    融资偿还额, 融券偿还量, 融券卖出量, 融资融券余额
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                frame[
                    [
                        "stock_code", "日期", "融资余额", "融券余额", "融资买入额", "融券余量",
                        "融资偿还额", "融券偿还量", "融券卖出量", "融资融券余额",
                    ]
                ].itertuples(index=False, name=None),
            )
            conn.commit()
        finally:
            conn.close()

    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames, ignore_index=True)
    logger.info("融资融券明细历史回填完成 rows=%s start=%s end=%s", len(result), trade_dates[0], trade_dates[-1])
    return result