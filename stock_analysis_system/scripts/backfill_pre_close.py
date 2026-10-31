# -*- coding: utf-8 -*-
"""补充 stock_data.pre_close（复权昨收）。

问题诊断（2026-10-31 实测）
--------------------------
`stock_data` 的 `close` 在除权日会跳空，产生假收益：

  000001 平安银行 2025-06-12（除权日）:
    tushare 未复权 close : 11.85 -> 11.68   真实收益 -1.43%
    本地 close           : 11.49 -> 11.68   假收益 +1.65%

  600036 招商银行 2025-07-07 / 07-11:
    本地单日 -3.97% vs 未复权 -5.39%（不一致）

原因：本地 close 在不同日期段用了不同的复权基准
（比率从 1.0313 跳到 1.0000），拼接处产生跳空。
**既不是纯前复权，也不是纯未复权。**

解决方案
--------
用 tushare `daily` 的 `pre_close`（交易所口径、已按除权调整）算真实收益：

    ret = close / pre_close - 1

但本地 close 与 tushare close 差一个恒定比例（复权因子），
所以写入时把 pre_close 换算到本地口径：

    adj_factor    = 本地close / tushare.close
    pre_close_adj = tushare.pre_close × adj_factor

于是 `本地close / pre_close_adj == tushare.close / tushare.pre_close`，
即真实收益，且与本地价格同量纲。

字段
----
  pre_close  : 与本地 close 同口径的复权昨收
  adj_factor : 本地 close / tushare 未复权 close（便于核查）

用法
----
  python scripts/backfill_pre_close.py --limit 10 --reset   # 小样本测试
  python scripts/backfill_pre_close.py                      # 全量
  python scripts/backfill_pre_close.py --verify-only        # 校验
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class RateLimiter:
    """简单的滑动窗口限速器：每分钟最多 max_calls 次。

    ⚠️ tushare `daily` 接口实测限速 300 次/分钟。
       8 并发不限速会触发 300/min 上限，所以必须节流。
    """

    def __init__(self, max_calls: int = 280, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.period]
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                wait = self.period - (now - self.calls[0]) + 0.05
            time.sleep(min(max(wait, 0.05), 5.0))


LIMITER = RateLimiter()


def resolve_db() -> str:
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            with cfg.open("r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            raw = c.get("databases", {}).get("trade_db")
            if raw:
                p = Path(raw)
                if not p.is_absolute():
                    p = ROOT / p
                if p.exists():
                    return str(p)
        except Exception:
            pass
    return str((ROOT.parent / "stock_history.db").resolve())


def load_token() -> str | None:
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return os.environ.get("TUSHARE_TOKEN")


def to_ts_code(code) -> str:
    """6 位代码 -> tushare ts_code。

    ⚠️ 北交所代码前缀复杂，实测本地有 92/83/87/43 开头，
       全部应为 .BJ（此前把 92 误判成 .SH，导致 604 只 no data）。
    """
    code = str(code).strip()
    if not code or code.lower() in ("none", "nan", ""):
        return ""
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) < 6:
        return ""
    digits = digits.zfill(6)[:6]
    p2 = digits[:2]
    # 北交所：43/83/87/88/92
    if p2 in ("43", "83", "87", "88", "92"):
        return digits + ".BJ"
    # 上交所主板 / 科创板
    if digits.startswith(("60", "68", "90")):
        return digits + ".SH"
    # 深交所主板 / 创业板
    if digits.startswith(("00", "30", "20")):
        return digits + ".SZ"
    # 老基金/其他
    if digits.startswith("51"):
        return digits + ".SH"
    if digits.startswith("15", "16", "18"):
        return digits + ".SZ"
    return digits + ".SZ"


def ensure_schema(db: str) -> None:
    con = sqlite3.connect(db)
    cur = con.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(stock_data)")]
    for col in ["pre_close", "adj_factor"]:
        if col not in cols:
            print(f"  新增列 stock_data.{col}")
            cur.execute(f"ALTER TABLE stock_data ADD COLUMN {col} REAL")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_pre_close_progress (
            ts_code TEXT PRIMARY KEY, status TEXT, n_rows INTEGER,
            n_filled INTEGER, msg TEXT, updated_at TEXT)
    """)
    # ★ 兼容旧版表结构（可能缺列），缺则重建
    pcols = [r[1] for r in cur.execute(
        "PRAGMA table_info(stock_pre_close_progress)")]
    need = {"ts_code", "status", "n_rows", "n_filled", "msg", "updated_at"}
    if not need.issubset(set(pcols)):
        print(f"  进度表结构不匹配（{pcols}），重建")
        cur.execute("DROP TABLE stock_pre_close_progress")
        cur.execute("""
            CREATE TABLE stock_pre_close_progress (
                ts_code TEXT PRIMARY KEY, status TEXT, n_rows INTEGER,
                n_filled INTEGER, msg TEXT, updated_at TEXT)
        """)
    con.commit()
    con.close()


def get_stock_list(db: str) -> list:
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT DISTINCT stock_code FROM stock_data ORDER BY stock_code").fetchall()
    con.close()
    return [r[0] for r in rows if r[0] is not None]


def get_done(db: str) -> set:
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute(
            "SELECT ts_code FROM stock_pre_close_progress WHERE status='ok'")}
    finally:
        con.close()


def save_progress(db: str, rec: dict) -> None:
    con = sqlite3.connect(db)
    con.execute("""
        INSERT INTO stock_pre_close_progress
            (ts_code,status,n_rows,n_filled,msg,updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(ts_code) DO UPDATE SET
            status=excluded.status, n_rows=excluded.n_rows,
            n_filled=excluded.n_filled, msg=excluded.msg,
            updated_at=excluded.updated_at
    """, (rec["ts_code"], rec["status"], rec["n_rows"], rec["n_filled"],
          rec["msg"], pd.Timestamp.now().isoformat()))
    con.commit()
    con.close()


def fetch_one(pro, code: str, db: str, start: str, end: str,
              max_retry: int = 5) -> dict:
    ts_code = to_ts_code(code)
    if not ts_code:
        return {"ts_code": str(code), "code": str(code), "status": "invalid",
                "n_rows": 0, "n_filled": 0, "msg": "bad code"}

    for attempt in range(max_retry):
        try:
            LIMITER.acquire()
            raw = pro.daily(ts_code=ts_code, start_date=start, end_date=end,
                            fields="ts_code,trade_date,close,pre_close")
            if raw is None or raw.empty:
                return {"ts_code": ts_code, "code": code, "status": "empty",
                        "n_rows": 0, "n_filled": 0, "msg": "no data"}

            raw["date"] = raw["trade_date"].astype(str).apply(
                lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                if len(s) == 8 and s.isdigit() else s)
            raw = raw[(raw["close"] > 0) & (raw["pre_close"] > 0)]

            con = sqlite3.connect(db)
            loc = pd.read_sql_query(
                "SELECT 日期 AS date, 收盘 AS close_local FROM stock_data "
                "WHERE stock_code=?", con, params=(code,))
            con.close()
            if loc.empty:
                return {"ts_code": ts_code, "code": code, "status": "empty",
                        "n_rows": 0, "n_filled": 0, "msg": "no local rows"}

            m = raw.merge(loc, on="date", how="inner")
            if m.empty:
                return {"ts_code": ts_code, "code": code, "status": "empty",
                        "n_rows": 0, "n_filled": 0, "msg": "no overlap"}

            # ★ 核心：把 pre_close 换算到本地 close 的口径
            m["adj_factor"] = m["close_local"] / m["close"]
            m["pre_close_adj"] = m["pre_close"] * m["adj_factor"]

            con = sqlite3.connect(db)
            try:
                cur = con.cursor()
                cur.executemany(
                    "UPDATE stock_data SET pre_close=?, adj_factor=? "
                    "WHERE stock_code=? AND 日期=?",
                    [(float(r.pre_close_adj), float(r.adj_factor), code, r.date)
                     for r in m.itertuples(index=False)])
                con.commit()
                filled = cur.rowcount
            finally:
                con.close()

            return {"ts_code": ts_code, "code": code, "status": "ok",
                    "n_rows": len(m), "n_filled": filled, "msg": ""}
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            is_rate = ("频率超限" in str(e) or "超限" in str(e)
                       or "limit" in str(e).lower())
            if attempt == max_retry - 1:
                return {"ts_code": ts_code, "code": code, "status": "error",
                        "n_rows": 0, "n_filled": 0, "msg": msg}
            # 限速错误退避更久（5s, 10s, 20s, 40s）
            backoff = (5.0 * (2 ** attempt)) if is_rate else (1.5 ** attempt)
            time.sleep(backoff)
    return {"ts_code": ts_code, "code": code, "status": "error",
            "n_rows": 0, "n_filled": 0, "msg": "retry exhausted"}


def verify(db: str) -> dict:
    con = sqlite3.connect(db)
    r = con.execute("""
        SELECT COUNT(*), SUM(CASE WHEN pre_close IS NULL THEN 1 ELSE 0 END)
        FROM stock_data""").fetchone()
    tot, nulls = r
    prog = pd.read_sql_query(
        "SELECT status, COUNT(*) n FROM stock_pre_close_progress GROUP BY status",
        con)
    df = pd.read_sql_query("""
        SELECT stock_code, 收盘, pre_close FROM stock_data
        WHERE pre_close IS NOT NULL AND pre_close > 0 AND 收盘 > 0
    """, con)
    con.close()

    out = {"total": int(tot), "null": int(nulls or 0),
           "coverage": float(1 - (nulls or 0) / max(tot, 1)),
           "progress": {row["status"]: int(row["n"]) for _, row in prog.iterrows()}}

    if len(df) > 1000:
        df["ret"] = df["收盘"] / df["pre_close"] - 1.0
        for th in [0.11, 0.15, 0.20, 0.30]:
            out[f"below_{int(th*100)}"] = int((df["ret"] < -th).sum())
        out["ret_std"] = float(df["ret"].std())
        out["ret_max"] = float(df["ret"].max())
        out["ret_min"] = float(df["ret"].min())
        out["n_checked"] = int(len(df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", default="20241001")
    ap.add_argument("--end", default="20261031")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    db = args.db or resolve_db()
    print("=" * 88)
    print("补充 stock_data.pre_close（复权昨收）")
    print("=" * 88)

    if args.verify_only:
        v = verify(db)
        print(f"  总行 {v['total']:,}  缺失 {v['null']:,}  "
              f"覆盖率 {v['coverage']*100:.2f}%")
        print(f"  进度 {v['progress']}")
        for k in ["below_11", "below_15", "below_20", "below_30"]:
            if k in v:
                print(f"  单日跌超{k.split('_')[1]}%: {v[k]:,}")
        if "ret_std" in v:
            print(f"  日收益 std {v['ret_std']*100:.3f}%  "
                  f"范围 [{v['ret_min']*100:.1f}%, {v['ret_max']*100:.1f}%]  "
                  f"(n={v['n_checked']:,})")
        return 0

    token = load_token()
    if not token:
        print("未找到 tushare token")
        return 1

    ensure_schema(db)
    if args.reset:
        con = sqlite3.connect(db)
        con.execute("DELETE FROM stock_pre_close_progress")
        con.execute("UPDATE stock_data SET pre_close=NULL, adj_factor=NULL")
        con.commit()
        con.close()
        print("  已重置进度与 pre_close")

    codes = [c for c in get_stock_list(db) if to_ts_code(c)]
    done = get_done(db)
    todo = [c for c in codes if to_ts_code(c) not in done]
    if args.limit > 0:
        todo = todo[:args.limit]

    print(f"  有效股票 {len(codes):,}  已完成 {len(done):,}  待处理 {len(todo):,}")
    if not todo:
        print("\n全部完成")
        v = verify(db)
        print(f"  覆盖率 {v['coverage']*100:.2f}%")
        return 0

    if args.dry_run:
        print(f"[DRY-RUN] 将处理 {len(todo)} 只，"
              f"预计 {len(todo)*0.064/60:.1f} 分钟")
        return 0

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    print(f"\n开始（{args.workers} 并发）...")
    t0 = time.time()
    stats = {"ok": 0, "empty": 0, "error": 0, "invalid": 0}
    filled_total = 0
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, pro, c, db, args.start, args.end): c
                for c in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                c = futs[fut]
                rec = {"ts_code": to_ts_code(c), "code": c, "status": "error",
                       "n_rows": 0, "n_filled": 0, "msg": str(e)}
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
            filled_total += rec["n_filled"]
            n_done += 1
            save_progress(db, rec)
            if n_done % 200 == 0:
                el = time.time() - t0
                left = (len(todo) - n_done) / max(n_done / el, 1e-9)
                print(f"  {n_done}/{len(todo)}  {stats}  已填 {filled_total:,}  "
                      f"用 {el/60:.1f}分  剩约 {left/60:.1f}分")

    el = time.time() - t0
    print(f"\n完成: {stats}  写入 {filled_total:,} 行  耗时 {el/60:.1f} 分钟")

    v = verify(db)
    print(f"\n校验: 覆盖率 {v['coverage']*100:.2f}%")
    for k in ["below_11", "below_20", "below_30"]:
        if k in v:
            print(f"  单日跌超{k.split('_')[1]}%: {v[k]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
