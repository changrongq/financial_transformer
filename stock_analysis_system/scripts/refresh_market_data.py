"""数据刷新模块：把 stock_data / etf_daily / index_daily 更新到最新交易日。

被 daily_regime_timing_signal.py 调用，也可以单独跑。

用法：
  python scripts/refresh_market_data.py                    # 自动检测并补齐
  python scripts/refresh_market_data.py --days 5           # 强制补最近5个交易日
  python scripts/refresh_market_data.py --check-only       # 只检查不拉取

刷新逻辑：
  1. 查数据库最新日期
  2. 从 tushare 拉最新日期的数据（按 trade_date 批量拉全市场，很快）
  3. ETF/指数按单个代码拉
  4. 自动跳过周末/非交易日（tushare 返回空即说明非交易日）
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)

ETF_CODES = ["510050.SH", "510300.SH", "510500.SH", "512100.SH", "159915.SZ", "588000.SH"]
INDEX_CODES = ["000001.SH", "000300.SH", "000905.SH", "000852.SH", "399006.SZ", "000016.SH"]


def resolve_db() -> str:
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


def get_token() -> str | None:
    tok = os.environ.get("TUSHARE_TOKEN")
    if tok:
        return tok.strip()
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def table_latest(conn: sqlite3.Connection, table: str) -> str | None:
    """返回表的最新日期（统一为 YYYYMMDD 格式）。"""
    try:
        if table == "stock_data":
            r = conn.execute("SELECT MAX([日期]) FROM stock_data").fetchone()[0]
        else:
            r = conn.execute(f"SELECT MAX(trade_date) FROM {table}").fetchone()[0]
        if r is None:
            return None
        return str(r).replace("-", "")[:8]
    except Exception:
        return None


def recent_trade_days(n: int) -> list[str]:
    """最近 n 个工作日（YYYYMMDD），从今天往前。"""
    out = []
    d = datetime.now()
    while len(out) < n:
        if d.isoweekday() <= 5:
            out.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
    return out


def upsert_stock_data(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """写入 stock_data（蛇形转中文列名）。

    ⚠️ 重要：stock_data 的「日期」列格式是 'YYYY-MM-DD'（带横杠），
       而 tushare 返回的是 'YYYYMMDD'。必须转换，否则：
         · MAX(日期) 比较会出错（'2026-10-28' > '20261029'）
         · GROUP BY 日期 会当成不同日期
    """
    if df is None or df.empty:
        return 0
    d = df.copy()
    d["ts_code"] = d["ts_code"].astype(str).str[:6]
    # 转成 YYYY-MM-DD 格式，与库内一致
    d["trade_date"] = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d",
                                     errors="coerce").dt.strftime("%Y-%m-%d")

    cols = [r[1] for r in conn.execute("PRAGMA table_info(stock_data)").fetchall()]

    if "日期" in cols:
        out = pd.DataFrame({
            "stock_code": d["ts_code"],
            "日期": d["trade_date"],
            "开盘": d.get("open"),
            "最高": d.get("high"),
            "最低": d.get("low"),
            "收盘": d["close"],
            "成交量": d.get("vol"),
            "换手率": None,
            "成交额": d.get("amount"),
            "市盈率": None,
            "市盈率TTM": None,
            "市净率": None,
            "市销率TTM": None,
            "股息率TTM": None,
        })
        # 先删同日期旧数据，避免重复
        for dt in out["日期"].unique():
            conn.execute("DELETE FROM stock_data WHERE [日期] = ?", (dt,))
        ph = ",".join(["?"] * len(out.columns))
        sql = (f"INSERT OR REPLACE INTO stock_data "
               f"({','.join(out.columns)}) VALUES ({ph})")
        conn.executemany(sql, out.itertuples(index=False, name=None))
    else:
        d = d.rename(columns={"ts_code": "stock_code"})
        for dt in d["trade_date"].unique():
            conn.execute("DELETE FROM stock_data WHERE trade_date = ?", (dt,))
        ph = ",".join(["?"] * len(d.columns))
        sql = (f"INSERT OR REPLACE INTO stock_data "
               f"({','.join(d.columns)}) VALUES ({ph})")
        conn.executemany(sql, d.itertuples(index=False, name=None))
    conn.commit()
    return len(df)


def upsert_etf(conn, df) -> int:
    if df is None or df.empty:
        return 0
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount"]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    d["trade_date"] = d["trade_date"].astype(str)
    ph = ",".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT OR REPLACE INTO etf_daily ({','.join(cols)}) VALUES ({ph})",
        d[cols].itertuples(index=False, name=None))
    conn.commit()
    return len(d)


def _upsert_index_impl(conn, df) -> int:
    if df is None or df.empty:
        return 0
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount"]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    d["trade_date"] = d["trade_date"].astype(str)
    ph = ",".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT OR REPLACE INTO index_daily ({','.join(cols)}) VALUES ({ph})",
        d[cols].itertuples(index=False, name=None))
    conn.commit()
    return len(d)


def refresh(target_date: str | None = None, max_lookback: int = 10, verbose: bool = True) -> dict:
    """刷新数据到最新交易日。

    返回 {"ok": bool, "stock_date": str, "etf_date": str, "index_date": str, "message": str}
    """
    tok = get_token()
    if not tok:
        return {"ok": False, "message": "无 tushare token，跳过刷新"}

    try:
        import tushare as ts
        pro = ts.pro_api(tok)
    except Exception as e:
        return {"ok": False, "message": f"tushare 初始化失败: {e}"}

    db = resolve_db()
    conn = sqlite3.connect(db)

    if verbose:
        print(f"[数据刷新] 数据库: {db}")

    # 当前状态
    cur_stock = table_latest(conn, "stock_data")
    cur_etf = table_latest(conn, "etf_daily")
    cur_index = table_latest(conn, "index_daily")
    if verbose:
        print(f"[数据刷新] 当前: stock_data={cur_stock} etf_daily={cur_etf} index_daily={cur_index}")

    # 候选日期：从今天往前最多 max_lookback 个工作日
    candidates = recent_trade_days(max_lookback)
    if target_date:
        candidates = [target_date] + [c for c in candidates if c != target_date]

    got_stock = got_etf = got_index = 0
    latest_stock = cur_stock
    latest_etf = cur_etf
    latest_index = cur_index

    for td in candidates:
        if latest_stock and td <= latest_stock and latest_etf and td <= latest_etf:
            break  # 已经是最新或更早，不用再往前拉

        # ---- 全市场个股 ----
        if not latest_stock or td > latest_stock:
            try:
                df = pro.daily(trade_date=td)
                if df is not None and not df.empty:
                    n = upsert_stock_data(conn, df)
                    got_stock += n
                    latest_stock = td
                    if verbose:
                        print(f"[数据刷新] stock_data {td}: {n} 只股票 [OK]")
                else:
                    if verbose:
                        print(f"[数据刷新] stock_data {td}: 无数据（非交易日？）")
            except Exception as e:
                if verbose:
                    print(f"[数据刷新] stock_data {td}: {type(e).__name__} {str(e)[:60]}")
            time.sleep(0.5)

        # ---- ETF ----
        if not latest_etf or td > latest_etf:
            for code in ETF_CODES:
                try:
                    df = pro.fund_daily(ts_code=code, start_date=td, end_date=td)
                    if df is not None and not df.empty:
                        n = upsert_etf(conn, df)
                        got_etf += n
                except Exception:
                    pass
                time.sleep(0.15)
            latest_etf = td
            if verbose:
                print(f"[数据刷新] etf_daily {td}: 已更新 {len(ETF_CODES)} 只")

        # ---- 指数 ----
        if not latest_index or td > latest_index:
            for code in INDEX_CODES:
                try:
                    df = pro.index_daily(ts_code=code, start_date=td, end_date=td)
                    if df is not None and not df.empty:
                        n = _upsert_index_impl(conn, df)
                        got_index += n
                except Exception:
                    pass
                time.sleep(0.15)
            latest_index = td
            if verbose:
                print(f"[数据刷新] index_daily {td}: 已更新 {len(INDEX_CODES)} 只")

    # 最终状态
    final_stock = table_latest(conn, "stock_data")
    final_etf = table_latest(conn, "etf_daily")
    final_index = table_latest(conn, "index_daily")
    conn.close()

    if verbose:
        print(f"[数据刷新] 完成: stock_data={final_stock} etf_daily={final_etf} "
              f"index_daily={final_index}")
        print(f"[数据刷新] 本次新增: 个股{got_stock}行 ETF{got_etf}行 指数{got_index}行")

    return {
        "ok": True,
        "stock_date": final_stock,
        "etf_date": final_etf,
        "index_date": final_index,
        "added": {"stock": got_stock, "etf": got_etf, "index": got_index},
        "message": "完成",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="刷新市场数据到最新交易日")
    parser.add_argument("--check-only", action="store_true", help="只显示当前状态，不拉取")
    parser.add_argument("--days", type=int, default=10, help="最多往前检查几个工作日")
    parser.add_argument("--date", default=None, help="指定日期 YYYYMMDD")
    args = parser.parse_args()

    db = resolve_db()
    conn = sqlite3.connect(db)

    if args.check_only:
        print("=" * 60)
        print("数据新鲜度检查")
        print("=" * 60)
        for t in ["stock_data", "etf_daily", "index_daily", "global_daily", "etf_minute"]:
            d = table_latest(conn, t)
            print(f"  {t:<16} 最新: {d if d else '(空)'}")
        conn.close()
        # 判断是否过期
        latest = table_latest(sqlite3.connect(db), "stock_data")
        if latest:
            from datetime import date
            lag = (date.today() - datetime.strptime(latest, "%Y%m%d").date()).days
            print(f"\n  今天 {date.today()}，数据滞后 {lag} 天")
            if lag > 1:
                print(f"  [警告] 数据可能不是最新，建议跑: python scripts/refresh_market_data.py")
        return

    conn.close()
    res = refresh(target_date=args.date, max_lookback=args.days)
    if not res["ok"]:
        print(f"[数据刷新] 失败: {res['message']}")


if __name__ == "__main__":
    main()
