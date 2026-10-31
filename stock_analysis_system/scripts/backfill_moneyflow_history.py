# -*- coding: utf-8 -*-
"""补齐 stock_moneyflow 的历史数据（2021-11 ~ 2024-12）。

背景
----
知微复核后续审计发现：
    stock_data     : 2021-11-01 ~ 2026-10-29  (1206 天)
    stock_moneyflow: 2025-01-02 ~ 2026-10-28  (435 天)
    → 771 个交易日（395.7 万行，62.1% 样本）没有资金流数据
    → features.py 把缺失的 7 列 mask 成常数 0.0
    → 训练集 81% 样本的资金流特征是常数，验证/测试是真实值 → 分布漂移

本脚本用 tushare `moneyflow` 按 trade_date 拉全市场补齐。

字段映射（tushare -> 本地）
--------------------------
  buy_sm_vol    -> 小单买入量
  buy_sm_amount -> 小单买入额
  sell_sm_vol   -> 小单卖出量
  sell_sm_amount-> 小单卖出额
  buy_md_*      -> 中单*
  sell_md_*     -> 中单*卖出*
  buy_lg_*      -> 大单*
  sell_lg_*     -> 大单*卖出*
  buy_elg_*     -> 特大单*
  sell_elg_*    -> 特大单*卖出*
  net_mf_vol    -> 净流入量
  net_mf_amount -> 净流入额

⚠️ 单位核对：现有表 2025 年后的数据要与 tushare 对齐，脚本会先做校准。

用法：
  python scripts/backfill_moneyflow_history.py --dry-run
  python scripts/backfill_moneyflow_history.py
  python scripts/backfill_moneyflow_history.py --limit 20   # 测试
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COLMAP = {
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
LOCAL_COLS = list(COLMAP.values())


def resolve_db() -> str:
    return str((ROOT.parent / "stock_history.db").resolve())


def load_token():
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return os.environ.get("TUSHARE_TOKEN")


def to_local_code(ts_code: str) -> str:
    return str(ts_code).split(".")[0]


class RateLimiter:
    def __init__(self, max_calls: int = 280, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def acquire(self):
        while True:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            time.sleep(min(self.period - (now - self.calls[0]) + 0.05, 5.0))


def ensure_schema(db: str):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(stock_moneyflow)")]
    for c in LOCAL_COLS:
        if c not in cols:
            print(f"  新增列 {c}")
            cur.execute(f'ALTER TABLE stock_moneyflow ADD COLUMN "{c}" REAL')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS moneyflow_backfill_progress (
            trade_date TEXT PRIMARY KEY, status TEXT, n_rows INTEGER,
            msg TEXT, updated_at TEXT)
    """)
    con.commit()
    con.close()


def get_done(db: str) -> set:
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute(
            "SELECT trade_date FROM moneyflow_backfill_progress "
            "WHERE status='ok'")}
    finally:
        con.close()


def save_progress(db: str, d: str, status: str, n: int, msg: str = ""):
    con = sqlite3.connect(db)
    con.execute("""
        INSERT INTO moneyflow_backfill_progress
            (trade_date, status, n_rows, msg, updated_at)
        VALUES (?,?,?,?,?)
        ON CONFLICT(trade_date) DO UPDATE SET
            status=excluded.status, n_rows=excluded.n_rows,
            msg=excluded.msg, updated_at=excluded.updated_at
    """, (d, status, n, msg, pd.Timestamp.now().isoformat()))
    con.commit()
    con.close()


def calibrate(pro, db: str) -> dict:
    """用已有的 2025 年数据校准 tushare -> 本地 的单位比。"""
    con = sqlite3.connect(db)
    loc = pd.read_sql_query("""
        SELECT stock_code, 日期, "净流入额" AS loc_net
        FROM stock_moneyflow
        WHERE 日期 = '2025-03-03' AND "净流入额" IS NOT NULL
    """, con)
    con.close()
    if loc.empty:
        print("  校准样本为空，假设 1:1")
        return {"net_mf_amount": 1.0}

    raw = pro.moneyflow(trade_date="20250303")
    if raw is None or raw.empty:
        return {"net_mf_amount": 1.0}
    raw["stock_code"] = raw["ts_code"].apply(to_local_code)
    m = raw.merge(loc, on="stock_code", how="inner")
    if len(m) < 100:
        return {"net_mf_amount": 1.0}

    r = (m["net_mf_amount"] / m["loc_net"].replace(0, np.nan)).median()
    print(f"  校准样本 {len(m)} 只，tushare/local 比值中位 = {r:.6f}")
    return {"net_mf_amount": float(r) if np.isfinite(r) and r != 0 else 1.0}


def fetch_day(pro, d_ts: str, max_retry: int = 5) -> pd.DataFrame:
    for attempt in range(max_retry):
        try:
            df = pro.moneyflow(trade_date=d_ts)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            msg = str(e)
            is_rate = "频率超限" in msg or "超限" in msg
            if attempt == max_retry - 1:
                print(f"  [{d_ts}] 失败: {e}")
                return pd.DataFrame()
            time.sleep(6.0 * (2 ** attempt) if is_rate else 2.0)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--start", default="2021-11-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4,
                    help="并发拉取线程数（写入仍串行，SQLite 单写）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    db = args.db or resolve_db()
    token = load_token()
    if not token:
        print("未找到 tushare token")
        return 1

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    print("=" * 90)
    print("补齐 stock_moneyflow 历史数据")
    print("=" * 90)
    print(f"数据库: {db}")
    print(f"范围: {args.start} ~ {args.end}")

    ensure_schema(db)
    if args.reset:
        con = sqlite3.connect(db)
        con.execute("DELETE FROM moneyflow_backfill_progress")
        con.commit()
        con.close()
        print("  进度表已清空")

    # 需要补的交易日（以 stock_data 为准）
    con = sqlite3.connect(db)
    rows = con.execute("""
        SELECT 日期, COUNT(*) FROM stock_data
        WHERE 日期 >= ? AND 日期 <= ?
        GROUP BY 日期 ORDER BY 日期
    """, (args.start, args.end)).fetchall()
    con.close()
    all_days = [r[0] for r in rows]
    print(f"\n需要覆盖的交易日: {len(all_days)} 天")

    done = get_done(db)
    todo = [d for d in all_days if d.replace("-", "") not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"已完成 {len(done)}  待处理 {len(todo)}")

    if not todo:
        print("\n全部完成")
        return 0

    if args.dry_run:
        print(f"[DRY-RUN] 将处理 {len(todo)} 天")
        print(f"  前 10: {todo[:10]}")
        return 0

    print("\n[0] 单位校准")
    calib = calibrate(pro, db)
    MUL = calib["net_mf_amount"]

    limiter = RateLimiter(280)
    print(f"\n[1] 开始拉取 {len(todo)} 天（{args.workers} 并发拉取，串行写入）...")
    t0 = time.time()
    total_ins = 0
    stats = {"ok": 0, "empty": 0, "error": 0}
    done_n = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(d: str):
        d_ts = d.replace("-", "")
        limiter.acquire()
        return d, fetch_day(pro, d_ts)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, d): d for d in todo}
        for fut in as_completed(futs):
            try:
                d, df = fut.result()
            except Exception as e:
                d = futs[fut]
                save_progress(db, d.replace("-", ""), "error", 0, str(e)[:100])
                stats["error"] += 1
                done_n += 1
                continue

            d_ts = d.replace("-", "")
            done_n += 1

            if df.empty:
                save_progress(db, d_ts, "empty", 0, "no data")
                stats["empty"] += 1
                continue

            df["stock_code"] = df["ts_code"].apply(to_local_code)
            df = df.rename(columns=COLMAP)
            for c in LOCAL_COLS:
                if c not in df.columns:
                    df[c] = np.nan
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if MUL != 1.0:
                df["净流入额"] = df["净流入额"] * MUL

            df["日期"] = d
            out = df[["stock_code", "日期"] + LOCAL_COLS]

            try:
                con = sqlite3.connect(db)
                try:
                    cur = con.cursor()
                    cur.execute("DELETE FROM stock_moneyflow WHERE 日期=?", (d,))
                    placeholders = ",".join("?" * (2 + len(LOCAL_COLS)))
                    cols_sql = ", ".join(f'"{c}"' for c in
                                         ["stock_code", "日期"] + LOCAL_COLS)
                    rows_vals = [
                        tuple([r.stock_code, r.日期] +
                              [None if pd.isna(getattr(r, c))
                               else float(getattr(r, c)) for c in LOCAL_COLS])
                        for r in out.itertuples(index=False)
                    ]
                    cur.executemany(
                        f"INSERT INTO stock_moneyflow ({cols_sql}) "
                        f"VALUES ({placeholders})", rows_vals)
                    con.commit()
                finally:
                    con.close()
                total_ins += len(out)
                stats["ok"] += 1
                save_progress(db, d_ts, "ok", len(out))
            except Exception as e:
                save_progress(db, d_ts, "error", 0, str(e)[:100])
                stats["error"] += 1
                print(f"  [{d}] 写入失败: {e}")

            if done_n % 100 == 0:
                el = time.time() - t0
                left = (len(todo) - done_n) / max(done_n / el, 1e-9)
                print(f"  {done_n}/{len(todo)}  {stats}  写入 {total_ins:,}  "
                      f"用 {el/60:.1f}分  剩约 {left/60:.1f}分", flush=True)

    el = time.time() - t0
    print(f"\n完成: {stats}  写入 {total_ins:,} 行  耗时 {el/60:.1f} 分钟")

    # 验证
    con = sqlite3.connect(db)
    r = con.execute("SELECT MIN(日期), MAX(日期), COUNT(*), "
                    "COUNT(DISTINCT stock_code) FROM stock_moneyflow").fetchone()
    print(f"\n验证: {r[0]} ~ {r[1]}  {r[2]:,} 行  {r[3]:,} 只")
    # 逐年覆盖
    days = pd.read_sql_query("""
        SELECT substr(日期,1,4) y, COUNT(DISTINCT 日期) n
        FROM stock_moneyflow GROUP BY y ORDER BY y
    """, con)
    print(f"\n逐年覆盖:")
    for _, x in days.iterrows():
        print(f"  {x['y']}: {x['n']:>4} 天")
    con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
