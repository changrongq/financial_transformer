# -*- coding: utf-8 -*-
"""修复 2026-03-18 ~ 2026-04-10 共 17 天的覆盖缺口。

实测：这 17 天每天只有 1~15 只股票，而前后日期有 5000+ 只。
这是历史数据拉取时遗漏的。

本脚本：
  1. 用 tushare daily(trade_date=...) 按日拉全市场
  2. 单位与当前库对齐（实测该期间 units 与 2024-10 之后一致：
     vol=股? 还是 手? amount=元? —— 用同期已知日的比例校准）
  3. 同时补 pre_close / adj_factor / turnover

用法：
  python scripts/fix_march_gap.py --dry-run
  python scripts/fix_march_gap.py
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

BAD_DATES = [
    "20260318", "20260319", "20260320", "20260323", "20260324",
    "20260325", "20260326", "20260327", "20260330", "20260331",
    "20260401", "20260402", "20260403", "20260407", "20260408",
    "20260409", "20260410",
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


def fetch_daily(pro, trade_date: str, max_retry: int = 4) -> pd.DataFrame:
    for attempt in range(max_retry):
        try:
            df = pro.daily(trade_date=trade_date,
                           fields="ts_code,trade_date,open,high,low,close,"
                                  "pre_close,change,pct_chg,vol,amount")
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            msg = str(e)
            is_rate = "频率超限" in msg or "超限" in msg
            if attempt == max_retry - 1:
                print(f"  [{trade_date}] daily 失败: {e}")
                return pd.DataFrame()
            time.sleep(6.0 * (2 ** attempt) if is_rate else 2.0)
    return pd.DataFrame()


def fetch_basic(pro, trade_date: str, max_retry: int = 4) -> pd.DataFrame:
    for attempt in range(max_retry):
        try:
            df = pro.daily_basic(
                trade_date=trade_date,
                fields="ts_code,turnover_rate,pe_ttm,pb,ps_ttm,dv_ttm")
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            msg = str(e)
            is_rate = "频率超限" in msg or "超限" in msg
            if attempt == max_retry - 1:
                print(f"  [{trade_date}] daily_basic 失败: {e}")
                return pd.DataFrame()
            time.sleep(6.0 * (2 ** attempt) if is_rate else 2.0)
    return pd.DataFrame()


def calibrate(pro, con, ref_ts: str = "20260317") -> dict:
    """用缺口前一天的已知数据校准 tushare -> 本地 的单位比。"""
    d = fmt_date(ref_ts)
    df = fetch_daily(pro, ref_ts)
    if df.empty:
        print("  校准失败：参考日无数据")
        return {"vol": 1.0, "amount": 1.0, "n": 0}
    df["stock_code"] = df["ts_code"].str.split(".").str[0]
    loc = pd.read_sql_query(
        "SELECT stock_code, 成交量 AS v_l, 成交额 AS a_l "
        "FROM stock_data WHERE 日期=?", con, params=(d,))
    m = df.merge(loc, on="stock_code", how="inner")
    m = m[(m["v_l"] > 0) & (m["a_l"] > 0)]
    if len(m) < 100:
        print(f"  校准样本不足 {len(m)}，用 1.0")
        return {"vol": 1.0, "amount": 1.0, "n": len(m)}
    r_vol = float((m["vol"] / m["v_l"]).median())
    r_amt = float((m["amount"] / m["a_l"]).median())
    return {"vol": 1.0 / r_vol, "amount": 1.0 / r_amt,
            "r_vol": r_vol, "r_amt": r_amt, "n": len(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
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
    print("修复 2026-03-18 ~ 2026-04-10 覆盖缺口")
    print("=" * 88)

    print("\n[0] 单位校准（用 2026-03-10 这一天，缺口前最后一个正常日）")
    cal = calibrate(pro, con, "20260310")
    print(f"  vol×{cal['vol']:.6f}  amount×{cal['amount']:.6f}  "
          f"(n={cal['n']}, 原始比 vol {cal.get('r_vol', 0):.4f} "
          f"amt {cal.get('r_amt', 0):.4f})")
    VOL_MUL, AMT_MUL = cal["vol"], cal["amount"]

    total_ins = 0
    total_upd = 0
    report = []

    for tsd in BAD_DATES:
        d = fmt_date(tsd)
        r = con.execute("SELECT COUNT(*) FROM stock_data WHERE 日期=?",
                        (d,)).fetchone()
        existed = r[0]
        print(f"\n  {d}  现有 {existed} 只", end="")

        if args.dry_run:
            print("  [DRY-RUN]")
            continue

        df = fetch_daily(pro, tsd)
        if df.empty:
            print("  -> daily 无数据")
            report.append({"date": d, "existed": existed, "fetched": 0})
            continue
        df["stock_code"] = df["ts_code"].str.split(".").str[0]
        df = df[(df["close"] > 0)]

        basic = fetch_basic(pro, tsd)
        bmap = {}
        if not basic.empty:
            basic["stock_code"] = basic["ts_code"].str.split(".").str[0]
            for rr in basic.itertuples(index=False):
                bmap[rr.stock_code] = {
                    "tr": getattr(rr, "turnover_rate", np.nan),
                    "pe": getattr(rr, "pe_ttm", np.nan),
                    "pb": getattr(rr, "pb", np.nan),
                }

        # 已有 adj_factor（用于 pre_close 换算）
        local_af = pd.read_sql_query(
            "SELECT stock_code, adj_factor FROM stock_data "
            "WHERE adj_factor IS NOT NULL GROUP BY stock_code", con)
        af_map = dict(zip(local_af["stock_code"], local_af["adj_factor"]))

        existed_codes = set(x[0] for x in con.execute(
            "SELECT stock_code FROM stock_data WHERE 日期=?", (d,)).fetchall())

        rows_new, rows_upd = [], []
        for rr in df.itertuples(index=False):
            code = rr.stock_code
            af = af_map.get(code, 1.0)
            if not np.isfinite(af) or af <= 0:
                af = 1.0
            pre_adj = float(rr.pre_close) * af
            v_loc = float(rr.vol) * VOL_MUL
            a_loc = float(rr.amount) * AMT_MUL
            b = bmap.get(code, {})
            tr = b.get("tr", np.nan)
            pe = b.get("pe", np.nan)
            pb = b.get("pb", np.nan)
            tr_v = float(tr) if np.isfinite(tr) else None
            pe_v = float(pe) if np.isfinite(pe) else None
            pb_v = float(pb) if np.isfinite(pb) else None

            if code in existed_codes:
                rows_upd.append((a_loc, v_loc, float(rr.close), pre_adj, af,
                                 tr_v, pe_v, pb_v, code, d))
            else:
                rows_new.append((code, d, float(rr.open), float(rr.high),
                                 float(rr.low), float(rr.close), v_loc, tr_v,
                                 a_loc, pe_v, pb_v, pre_adj, af))

        try:
            cur = con.cursor()
            if rows_upd:
                cur.executemany("""
                    UPDATE stock_data
                    SET 成交额=?, 成交量=?, 收盘=?, pre_close=?, adj_factor=?,
                        换手率=COALESCE(换手率, ?),
                        市盈率TTM=COALESCE(市盈率TTM, ?),
                        市净率=COALESCE(市净率, ?)
                    WHERE stock_code=? AND 日期=?
                """, rows_upd)
            if rows_new:
                cur.executemany("""
                    INSERT INTO stock_data
                        (stock_code, 日期, 开盘, 最高, 最低, 收盘,
                         成交量, 换手率, 成交额, 市盈率TTM, 市净率,
                         pre_close, adj_factor)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, rows_new)
            con.commit()
            ins, upd = len(rows_new), len(rows_upd)
            total_ins += ins
            total_upd += upd
            print(f"  -> tushare {len(df)} 只，插入 {ins}，更新 {upd}")
            report.append({"date": d, "existed": existed,
                           "fetched": int(len(df)),
                           "inserted": ins, "updated": upd})
        except Exception as e:
            con.rollback()
            print(f"  -> 写入失败: {e}")
            report.append({"date": d, "existed": existed, "error": str(e)})

    con.close()
    print("\n" + "=" * 88)
    print(f"完成: 插入 {total_ins:,} 行，更新 {total_upd:,} 行")
    print("=" * 88)

    import json
    out = ROOT / "results" / "march_gap_fix.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"inserted": total_ins, "updated": total_upd,
                   "calibration": cal, "detail": report},
                  f, ensure_ascii=False, indent=2)
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
