# -*- coding: utf-8 -*-
"""拉取全市场 5 年历史日线，写入 stock_data。

设计要点
--------
1. **含退市股**（避免幸存者偏差）
   用 tushare `stock_basic` 的 L + D 清单（共 5920 只）。
   退市股在退市日之后自然没有数据，不会造成空洞。

2. **只补历史部分，不覆盖现有数据**
   对每只股票：
     · 若 stock_data 已有该股，只补 `日期 < 现有最早日期` 的部分
     · 若没有该股（退市股/本地缺失），全部插入
   这样不会破坏已经验证过的 2024-10 之后的数据（含 pre_close / adj_factor）。

3. **单位与口径**
   沿用 fix_coverage_gap.py 的做法：
     · amount: tushare 千元 → 本地元（逐股按 close 比对校准，或统一 ×1000）
     · vol: 实测与本地一致，×1.0
     · pre_close: × adj_factor（用该股已有的 adj_factor）

   ⚠️ 更稳妥：历史段的 adj_factor 用【该股最新一天的 adj_factor】。
      因为 adj_factor = local_close / tushare_close，在复权基准不变时恒定。

4. **断点续传**
   新表 `stock_history_progress(ts_code, status, n_new, n_skip, msg, updated_at)`

5. **分批提交**，每只股票一次事务，避免大事务锁库

用法
----
  python scripts/backfill_history.py --dry-run
  python scripts/backfill_history.py --years 5 --workers 4
  python scripts/backfill_history.py --limit 20      # 小样本测试
  python scripts/backfill_history.py --verify-only
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
    """tushare daily 限速 300 次/分钟，这里压到 280。"""

    def __init__(self, max_calls: int = 280, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls: list = []
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


def load_token():
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return os.environ.get("TUSHARE_TOKEN")


def ensure_schema(db: str) -> None:
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_history_progress (
            ts_code TEXT PRIMARY KEY,
            status TEXT,
            n_fetched INTEGER,
            n_inserted INTEGER,
            n_skip INTEGER,
            earliest TEXT,
            msg TEXT,
            updated_at TEXT
        )
    """)
    # 加速历史查询
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_stock_date
        ON stock_data(stock_code, 日期)
    """)
    con.commit()
    con.close()


def load_universe(db: str) -> pd.DataFrame:
    """优先用 check_stock_universe.py 导出的清单。"""
    p = ROOT / "results" / "stock_universe.csv"
    if p.exists():
        df = pd.read_csv(p, dtype={"code6": str})
        df = df[df["code6"].notna()]
        df["code6"] = df["code6"].astype(str).str.zfill(6)
        return df.drop_duplicates("code6")
    con = sqlite3.connect(db)
    codes = [r[0] for r in con.execute(
        "SELECT DISTINCT stock_code FROM stock_data").fetchall()
        if r[0] is not None]
    con.close()
    return pd.DataFrame({"code6": codes})


def get_done(db: str) -> set:
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute(
            "SELECT ts_code FROM stock_history_progress WHERE status='ok'")}
    finally:
        con.close()


def save_progress(db: str, rec: dict) -> None:
    con = sqlite3.connect(db)
    con.execute("""
        INSERT INTO stock_history_progress
            (ts_code,status,n_fetched,n_inserted,n_skip,earliest,msg,updated_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(ts_code) DO UPDATE SET
            status=excluded.status, n_fetched=excluded.n_fetched,
            n_inserted=excluded.n_inserted, n_skip=excluded.n_skip,
            earliest=excluded.earliest, msg=excluded.msg,
            updated_at=excluded.updated_at
    """, (rec["ts_code"], rec["status"], rec["n_fetched"],
          rec["n_inserted"], rec["n_skip"], rec.get("earliest", ""),
          rec["msg"], pd.Timestamp.now().isoformat()))
    con.commit()
    con.close()


def fetch_one(pro, db: str, ts_code: str, code6: str,
              start: str, end: str, max_retry: int = 5) -> dict:
    """拉一只股票的历史，插入 stock_data 中缺失的部分。"""
    for attempt in range(max_retry):
        try:
            LIMITER.acquire()
            raw = pro.daily(ts_code=ts_code, start_date=start, end_date=end,
                            fields="ts_code,trade_date,open,high,low,close,"
                                   "pre_close,vol,amount")
            if raw is None or raw.empty:
                return {"ts_code": ts_code, "status": "empty",
                        "n_fetched": 0, "n_inserted": 0, "n_skip": 0,
                        "earliest": "", "msg": "no data"}

            raw["date"] = raw["trade_date"].astype(str).apply(
                lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                if len(s) == 8 and s.isdigit() else s)
            raw = raw[(raw["close"] > 0) & (raw["pre_close"] > 0)]

            con = sqlite3.connect(db, timeout=60)
            try:
                # 该股在库里的现状
                info = con.execute("""
                    SELECT MIN(日期), MAX(日期), COUNT(*) FROM stock_data
                    WHERE stock_code=?
                """, (code6,)).fetchone()
                min_local, max_local, n_local = info

                # 该股的 adj_factor（用于换算历史 pre_close）
                af_row = con.execute("""
                    SELECT adj_factor FROM stock_data
                    WHERE stock_code=? AND adj_factor IS NOT NULL
                    ORDER BY 日期 DESC LIMIT 1
                """, (code6,)).fetchone()
                af = float(af_row[0]) if af_row and af_row[0] else 1.0
                if not np.isfinite(af) or af <= 0:
                    af = 1.0

                # 只用库里没有的日期
                if min_local:
                    raw = raw[raw["date"] < min_local]
                if raw.empty:
                    return {"ts_code": ts_code, "status": "ok",
                            "n_fetched": 0, "n_inserted": 0, "n_skip": 0,
                            "earliest": min_local or "", "msg": "nothing older"}

                existing = set(r[0] for r in con.execute(
                    "SELECT 日期 FROM stock_data WHERE stock_code=?",
                    (code6,)).fetchall())

                # 单位：amount 千元→元（×1000），vol 与本地一致（×1）
                rows = []
                for r in raw.itertuples(index=False):
                    if r.date in existing:
                        continue
                    rows.append((
                        code6, r.date, float(r.open), float(r.high),
                        float(r.low), float(r.close),
                        float(r.vol), float(r.amount) * 1000.0,
                        float(r.pre_close) * af, af,
                    ))
                if not rows:
                    return {"ts_code": ts_code, "status": "ok",
                            "n_fetched": len(raw), "n_inserted": 0,
                            "n_skip": len(raw),
                            "earliest": min_local or "", "msg": "all existed"}

                cur = con.cursor()
                cur.executemany("""
                    INSERT OR IGNORE INTO stock_data
                        (stock_code, 日期, 开盘, 最高, 最低, 收盘,
                         成交量, 成交额, pre_close, adj_factor)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, rows)
                con.commit()
                ins = cur.rowcount
                return {"ts_code": ts_code, "status": "ok",
                        "n_fetched": len(raw), "n_inserted": int(ins),
                        "n_skip": len(raw) - int(ins),
                        "earliest": rows[0][1], "msg": ""}
            finally:
                con.close()

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            is_rate = "频率超限" in str(e) or "超限" in str(e)
            is_lock = "locked" in str(e).lower()
            if attempt == max_retry - 1:
                return {"ts_code": ts_code, "status": "error",
                        "n_fetched": 0, "n_inserted": 0, "n_skip": 0,
                        "earliest": "", "msg": msg}
            backoff = 10.0 * (2 ** attempt) if is_rate else (
                3.0 * (2 ** attempt) if is_lock else 2.0 * (attempt + 1))
            time.sleep(backoff)
    return {"ts_code": ts_code, "status": "error", "n_fetched": 0,
            "n_inserted": 0, "n_skip": 0, "earliest": "", "msg": "exhausted"}


def verify(db: str) -> dict:
    con = sqlite3.connect(db)
    r = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT stock_code),
               MIN(日期), MAX(日期), COUNT(DISTINCT 日期)
        FROM stock_data
    """).fetchone()
    prog = pd.read_sql_query(
        "SELECT status, COUNT(*) n FROM stock_history_progress "
        "GROUP BY status", con)

    # 按年统计
    yr = pd.read_sql_query("""
        SELECT SUBSTR(日期,1,4) y, COUNT(*) n, COUNT(DISTINCT stock_code) s
        FROM stock_data GROUP BY y ORDER BY y
    """, con)
    con.close()
    return {"rows": int(r[0]), "stocks": int(r[1]),
            "min_date": r[2], "max_date": r[3], "days": int(r[4]),
            "progress": {row["status"]: int(row["n"]) for _, row in prog.iterrows()},
            "by_year": yr.to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    db = args.db or resolve_db()
    print("=" * 90)
    print(f"拉取全市场 {args.years} 年历史日线")
    print("=" * 90)

    if args.verify_only:
        v = verify(db)
        print(f"  行数   : {v['rows']:,}")
        print(f"  股票   : {v['stocks']:,}")
        print(f"  日期   : {v['min_date']} ~ {v['max_date']}  ({v['days']} 天)")
        print(f"  进度   : {v['progress']}")
        print(f"\n  按年分布:")
        for r in v["by_year"]:
            print(f"    {r['y']}: {r['n']:>10,} 行  {r['s']:>5} 只")
        return 0

    token = load_token()
    if not token:
        print("未找到 tushare token")
        return 1

    ensure_schema(db)
    if args.reset:
        con = sqlite3.connect(db)
        con.execute("DELETE FROM stock_history_progress")
        con.commit()
        con.close()
        print("  进度表已清空")

    uni = load_universe(db)
    # ★ 按上市日期正序（老股票在前）
    #   清单来自 stock_basic，默认是倒序（新股在前），
    #   新股没有历史可补，正序能让早期进度更能反映真实情况。
    if "list_date" in uni.columns:
        uni["list_date"] = pd.to_numeric(uni["list_date"], errors="coerce")
        uni = uni.sort_values("list_date", na_position="last")
    print(f"\n  清单 {len(uni)} 只（按上市日期正序）")

    # 起始日期
    today = pd.Timestamp.now()
    start = (today - pd.DateOffset(years=args.years)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    print(f"  区间 {start} ~ {end}")

    done = get_done(db)
    todo = []
    for r in uni.itertuples(index=False):
        code6 = r.code6
        ts_code = getattr(r, "ts_code", None)
        if not ts_code or not isinstance(ts_code, str):
            ts_code = _to_ts(code6)
        if not ts_code:
            continue
        if ts_code in done:
            continue
        todo.append((ts_code, code6))

    if args.limit > 0:
        todo = todo[:args.limit]

    print(f"  已完成 {len(done):,}  待处理 {len(todo):,}")

    if args.dry_run:
        print(f"\n[DRY-RUN] 将处理 {len(todo)} 只，"
              f"预计 {len(todo)*0.25/60:.1f} 分钟（4 并发）")
        print(f"  前 10: {[t[0] for t in todo[:10]]}")
        return 0

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    print(f"\n开始（{args.workers} 并发，限速 280/min）...")
    t0 = time.time()
    stats = {"ok": 0, "empty": 0, "error": 0}
    total_ins = 0
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, pro, db, tc, c6, start, end): tc
                for tc, c6 in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                tc = futs[fut]
                rec = {"ts_code": tc, "status": "error", "n_fetched": 0,
                       "n_inserted": 0, "n_skip": 0, "earliest": "",
                       "msg": str(e)}
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
            total_ins += rec["n_inserted"]
            n_done += 1
            try:
                save_progress(db, rec)
            except Exception:
                pass
            if n_done % 300 == 0:
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                left = (len(todo) - n_done) / max(rate, 1e-9)
                print(f"  {n_done}/{len(todo)}  {stats}  "
                      f"新增 {total_ins:,} 行  用 {el/60:.1f}分  "
                      f"剩约 {left/60:.1f}分", flush=True)

    el = time.time() - t0
    print(f"\n完成: {stats}  新增 {total_ins:,} 行  耗时 {el/60:.1f} 分钟")

    v = verify(db)
    print(f"\n校验:")
    print(f"  行数 {v['rows']:,}  股票 {v['stocks']:,}  "
          f"日期 {v['min_date']} ~ {v['max_date']} ({v['days']} 天)")
    for r in v["by_year"]:
        print(f"    {r['y']}: {r['n']:>10,} 行  {r['s']:>5} 只")
    return 0


def _to_ts(code6: str) -> str:
    p2 = code6[:2]
    if p2 in ("43", "83", "87", "88", "92"):
        return code6 + ".BJ"
    if code6.startswith(("60", "68", "90")):
        return code6 + ".SH"
    if code6.startswith(("00", "30", "20")):
        return code6 + ".SZ"
    return code6 + ".SZ"


if __name__ == "__main__":
    raise SystemExit(main())
