# -*- coding: utf-8 -*-
"""补齐新增行的换手率（tushare daily_basic 接口）。

问题：
  补齐覆盖缺口后，新增的约 3200 只/天没有【换手率】，
  导致换手率覆盖只有 41.7%。

tushare `daily` 不含换手率，需要 `daily_basic`：
  字段: ts_code, trade_date, turnover_rate, turnover_rate_f,
        volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm

注意单位：
  tushare turnover_rate 是【百分比】数值（如 0.87 表示 0.87%）
  本地 换手率 也是百分比（实测 000001 是 0.8693）

本脚本：
  · 对每个缺口日期调用 daily_basic
  · 只更新 turnover 为 NULL 的行
  · 顺带补 pe_ttm / pb（如果本地为空）

用法：
  python scripts/backfill_turnover.py
  python scripts/backfill_turnover.py --dry-run
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

GAP_DATES = [
    "20260922", "20260924", "20260928", "20260929", "20260930",
    "20261008", "20261009", "20261012", "20261013", "20261014",
    "20261015", "20261016", "20261019", "20261020", "20261021",
    "20261022", "20261023", "20261026", "20261027",
]


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


def fmt_date(ts: str) -> str:
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"


def fetch_basic(pro, trade_date: str, max_retry: int = 4) -> pd.DataFrame:
    for attempt in range(max_retry):
        try:
            df = pro.daily_basic(
                trade_date=trade_date,
                fields="ts_code,trade_date,turnover_rate,turnover_rate_f,"
                       "volume_ratio,pe_ttm,pb,ps_ttm,dv_ttm")
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            msg = str(e)
            is_rate = "频率超限" in msg or "超限" in msg
            if attempt == max_retry - 1:
                print(f"  [{trade_date}] 失败: {e}")
                return pd.DataFrame()
            time.sleep(6.0 * (2 ** attempt) if is_rate else 2.0)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dates", nargs="+", default=GAP_DATES)
    args = ap.parse_args()

    db = args.db or resolve_db()
    token = load_token()
    if not token:
        print("未找到 tushare token")
        return 1

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    con = sqlite3.connect(db)
    print("=" * 88)
    print("补齐空白的换手率 / 估值字段")
    print("=" * 88)

    total_upd = 0
    report = []

    for tsd in args.dates:
        d = fmt_date(tsd)
        r = con.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN 换手率 IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN 市盈率TTM IS NULL THEN 1 ELSE 0 END)
            FROM stock_data WHERE 日期=?
        """, (d,)).fetchone()
        tot, null_tr, null_pe = r
        print(f"\n  {d}  共 {tot} 只，换手率空 {null_tr or 0}，"
              f"PE空 {null_pe or 0}", end="")

        if null_tr is None or null_tr == 0:
            print("  -> 无需补充")
            continue
        if args.dry_run:
            print("  [DRY-RUN]")
            continue

        basic = fetch_basic(pro, tsd)
        if basic.empty:
            print("  -> daily_basic 无数据")
            report.append({"date": d, "updated": 0, "note": "no data"})
            continue

        basic["stock_code"] = basic["ts_code"].str.split(".").str[0]
        # 本地换手率是百分数（如 0.8693 表示 0.8693%），tushare 也是
        rows = []
        for rr in basic.itertuples(index=False):
            tr = getattr(rr, "turnover_rate", np.nan)
            pe = getattr(rr, "pe_ttm", np.nan)
            pb = getattr(rr, "pb", np.nan)
            if not np.isfinite(tr):
                continue
            rows.append((float(tr),
                         float(pe) if np.isfinite(pe) else None,
                         float(pb) if np.isfinite(pb) else None,
                         rr.stock_code, d))
        try:
            cur = con.cursor()
            cur.executemany("""
                UPDATE stock_data
                SET 换手率 = COALESCE(换手率, ?),
                    市盈率TTM = COALESCE(市盈率TTM, ?),
                    市净率 = COALESCE(市净率, ?)
                WHERE stock_code=? AND 日期=?
            """, rows)
            con.commit()
            print(f"  -> daily_basic {len(basic)} 只，写入 {cur.rowcount} 行")
            total_upd += cur.rowcount
            report.append({"date": d, "fetched": int(len(basic)),
                           "updated": int(cur.rowcount)})
        except Exception as e:
            con.rollback()
            print(f"  -> 失败: {e}")
            report.append({"date": d, "error": str(e)})

    # 总体验证
    print("\n" + "=" * 88)
    r = con.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN 换手率 IS NULL THEN 1 ELSE 0 END)
        FROM stock_data""").fetchone()
    print(f"换手率总覆盖: {(r[0] - (r[1] or 0))/r[0]*100:.2f}% "
          f"({r[0] - (r[1] or 0):,}/{r[0]:,})")
    print(f"本次更新 {total_upd:,} 行")

    # 每日检查
    print("\n每日换手率覆盖（< 4500 只的日期）:")
    rows = con.execute("""
        SELECT 日期, COUNT(*) n,
               SUM(CASE WHEN 换手率 IS NOT NULL THEN 1 ELSE 0 END) ok
        FROM stock_data GROUP BY 日期 ORDER BY 日期
    """).fetchall()
    bad = [(d, n, o or 0) for d, n, o in rows if (o or 0) < 4500]
    for d, n, o in bad:
        print(f"  {d}: {o}/{n} = {o/n*100:.1f}%")
    if not bad:
        print("  无")

    con.close()

    import json
    out = ROOT / "results" / "turnover_backfill.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"updated": total_upd, "detail": report},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
