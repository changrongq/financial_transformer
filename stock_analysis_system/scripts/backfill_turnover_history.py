# -*- coding: utf-8 -*-
"""补齐 2021-11 ~ 2024-09 的换手率与估值字段。

背景：
  历史数据拉取时，tushare `daily` 接口不含 turnover_rate / pe / pb，
  所以 2021-2023 的换手率全空（2024 只有 24.9%）。

方案：
  用 tushare `daily_basic` 按交易日拉，一次返回当天全市场。
  1206 个交易日里需补约 710 天。

  字段：
    turnover_rate   换手率（%，本地口径一致）
    turnover_rate_f 自由流通换手率
    pe_ttm / pb     估值

  注意：daily_basic 限速比 daily 更严格（实测 200 次/分钟）。
  加滑动窗口限速 + 断点续传。

用法：
  python scripts/backfill_turnover_history.py --dry-run
  python scripts/backfill_turnover_history.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class RateLimiter:
    def __init__(self, max_calls: int = 180, period: float = 60.0):
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


def load_token():
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return os.environ.get("TUSHARE_TOKEN")


def ensure_progress(db: str):
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS turnover_backfill_progress (
            trade_date TEXT PRIMARY KEY, status TEXT,
            n_rows INTEGER, n_updated INTEGER, msg TEXT, updated_at TEXT)
    """)
    con.commit()
    con.close()


def get_done(db: str) -> set:
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute(
            "SELECT trade_date FROM turnover_backfill_progress "
            "WHERE status='ok'")}
    finally:
        con.close()


def save_progress(db: str, d: str, status: str, n_rows: int,
                  n_upd: int, msg: str = ""):
    con = sqlite3.connect(db)
    con.execute("""
        INSERT INTO turnover_backfill_progress
            (trade_date,status,n_rows,n_updated,msg,updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(trade_date) DO UPDATE SET
            status=excluded.status, n_rows=excluded.n_rows,
            n_updated=excluded.n_updated, msg=excluded.msg,
            updated_at=excluded.updated_at
    """, (d, status, n_rows, n_upd, msg,
          pd.Timestamp.now().isoformat()))
    con.commit()
    con.close()


def fetch_basic(pro, d: str, max_retry: int = 5) -> pd.DataFrame:
    tsd = d.replace("-", "")
    for attempt in range(max_retry):
        try:
            LIMITER.acquire()
            df = pro.daily_basic(
                trade_date=tsd,
                fields="ts_code,trade_date,turnover_rate,turnover_rate_f,"
                       "pe_ttm,pb,ps_ttm,dv_ttm")
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            msg = str(e)
            is_rate = "频率超限" in msg or "超限" in msg
            if attempt == max_retry - 1:
                print(f"    [{d}] 失败: {e}")
                return pd.DataFrame()
            time.sleep(8.0 * (2 ** attempt) if is_rate else 2.0)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/turnover_history.json")
    args = ap.parse_args()

    db = args.db or resolve_db()
    token = load_token()
    if not token:
        print("未找到 token")
        return 1

    ensure_progress(db)
    if args.reset:
        con = sqlite3.connect(db)
        con.execute("DELETE FROM turnover_backfill_progress")
        con.commit()
        con.close()

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    con = sqlite3.connect(db)
    # 找出换手率覆盖不足的日期
    rows = con.execute("""
        SELECT 日期, COUNT(*) n,
               SUM(CASE WHEN 换手率 IS NOT NULL THEN 1 ELSE 0 END) ok
        FROM stock_data GROUP BY 日期 ORDER BY 日期
    """).fetchall()
    con.close()

    need = [d for d, n, ok in rows if (ok or 0) / max(n, 1) < 0.9]
    done = get_done(db)
    todo = [d for d in need if d not in done]
    if args.limit > 0:
        todo = todo[:args.limit]

    print("=" * 88)
    print("补齐历史换手率（tushare daily_basic）")
    print("=" * 88)
    print(f"  换手率不足的日期: {len(need)}")
    print(f"  已完成: {len([d for d in need if d in done])}")
    print(f"  待处理: {len(todo)}")
    if todo:
        print(f"  范围: {todo[0]} ~ {todo[-1]}")

    if not todo:
        print("\n全部完成")
        return 0
    if args.dry_run:
        print(f"\n[DRY-RUN] 将调用 {len(todo)} 次 daily_basic，"
              f"预计 {len(todo)*0.35/60:.1f} 分钟")
        return 0

    print(f"\n开始（180 次/分钟限速）...")
    t0 = time.time()
    ok_n = empty_n = err_n = 0
    total_upd = 0

    for i, d in enumerate(todo, 1):
        basic = fetch_basic(pro, d)
        if basic.empty:
            save_progress(db, d, "empty", 0, 0, "no data")
            empty_n += 1
            continue

        basic["stock_code"] = basic["ts_code"].str.split(".").str[0]
        recs = []
        for rr in basic.itertuples(index=False):
            tr = getattr(rr, "turnover_rate", np.nan)
            if not np.isfinite(tr):
                continue
            pe = getattr(rr, "pe_ttm", np.nan)
            pb = getattr(rr, "pb", np.nan)
            recs.append((float(tr),
                         float(pe) if np.isfinite(pe) else None,
                         float(pb) if np.isfinite(pb) else None,
                         rr.stock_code, d))

        if not recs:
            save_progress(db, d, "empty", int(len(basic)), 0, "no turnover")
            empty_n += 1
            continue

        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.executemany("""
                UPDATE stock_data
                SET 换手率 = COALESCE(换手率, ?),
                    市盈率TTM = COALESCE(市盈率TTM, ?),
                    市净率 = COALESCE(市净率, ?)
                WHERE stock_code=? AND 日期=?
            """, recs)
            con.commit()
            n_upd = cur.rowcount
            con.close()
            save_progress(db, d, "ok", int(len(basic)), n_upd)
            ok_n += 1
            total_upd += n_upd
        except Exception as e:
            save_progress(db, d, "error", int(len(basic)), 0, str(e))
            err_n += 1
            if err_n <= 5:
                print(f"    [{d}] 写入失败: {e}")

        if i % 100 == 0:
            el = time.time() - t0
            left = (len(todo) - i) / max(i / el, 1e-9)
            print(f"  {i}/{len(todo)}  ok={ok_n} empty={empty_n} "
                  f"err={err_n}  更新 {total_upd:,} 行  "
                  f"用 {el/60:.1f}分  剩约 {left/60:.1f}分")

    el = time.time() - t0
    print(f"\n完成: ok={ok_n} empty={empty_n} err={err_n}  "
          f"更新 {total_upd:,} 行  耗时 {el/60:.1f} 分钟")

    # 总体验证
    con = sqlite3.connect(db)
    r = con.execute("""
        SELECT COUNT(*), SUM(CASE WHEN 换手率 IS NOT NULL THEN 1 ELSE 0 END)
        FROM stock_data""").fetchone()
    print(f"\n换手率总覆盖: {(r[0]-(r[1] or 0))/r[0]*100:.2f}% "
          f"({r[0]-(r[1] or 0):,}/{r[0]:,})")
    yrs = con.execute("""
        SELECT SUBSTR(日期,1,4), COUNT(*),
               SUM(CASE WHEN 换手率 IS NOT NULL THEN 1 ELSE 0 END)
        FROM stock_data GROUP BY 1 ORDER BY 1
    """).fetchall()
    print(f"\n按年份:")
    for y, n, o in yrs:
        print(f"  {y}: {(o or 0)/n*100:>6.2f}%")
    con.close()

    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump({"ok": ok_n, "empty": empty_n, "err": err_n,
                   "updated": total_upd}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
