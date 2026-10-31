# -*- coding: utf-8 -*-
"""补齐 stock_data 的 19 天覆盖缺口 + 修复 2026-10-29 成交额。

问题
----
1. 覆盖缺口（19 个交易日）
   2026-09-22 ~ 2026-10-27 只拉了深市（00/30），缺：
     · 沪市主板 60xxxx   (约 1686 只)
     · 科创板   68xxxx   (约 602 只)
     · 北交所   92xxxx   (约 297 只)
     · 部分创业板 30xxxx (约 550 只)
   实测 2026-10-15 只有 2320 只，2026-09-18 有 5458 只。

2. 2026-10-29 成交额单位错
   隐含价格 = 成交额/成交量 只有收盘价的 0.1 倍
   （正常应为 1.0 倍，即 amount 单位是元）
   实测 ratio 中位 0.1012 → 成交额单位是「千元」而非常规的「元」，
   或者更可能：amount 被写成「千元」而 volume 是「手」以外的口径。

方案
----
对每个缺失日期：
  · 用 tushare `daily` 按【交易日】拉全市场（trade_date 参数），
    一次调用返回当天所有股票
  · 19 天 × 1 次调用，非常快
  · 只插入本地缺失的 (stock_code, 日期) 行
  · 同时写入 pre_close 与 adj_factor（adj_factor 需要基线，
    这里先用 tushare close 直接比对本地 close 得到）

对 2026-10-29：
  · 用 tushare 重新拉当日全市场，直接覆盖 amount / vol / close / pre_close

用法：
  python scripts/fix_coverage_gap.py --dry-run
  python scripts/fix_coverage_gap.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BAD_DATES = [
    "20260922", "20260924", "20260928", "20260929", "20260930",
    "20261008", "20261009", "20261012", "20261013", "20261014",
    "20261015", "20261016", "20261019", "20261020", "20261021",
    "20261022", "20261023", "20261026", "20261027",
]
AMOUNT_BAD = ["20261029"]


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


# ============================================================
# ⚠️ 单位换算（实测 2026-10-31）
#   tushare daily: vol = 手（100 股），amount = 千元
#   本地 stock_data: 成交量 = 股，成交额 = 元
#   实测交集比值: close 1.0000, vol 1.0000, amount 0.1008
#   → 说明本地 vol 与 tushare vol 单位一致（都是手），
#     但 amount 差 10 倍（tushare 千元 vs 本地元 ... 实际是 0.1008）
#   安全做法：用交集实测比值确定换算系数，而不是硬编码。
# ============================================================
def calibrate_units(pro, con, trade_date_ts: str) -> dict:
    """用已知正常日校准 tushare -> 本地的单位比。"""
    d = fmt_date(trade_date_ts)
    df = fetch_day(pro, trade_date_ts)
    if df.empty:
        return {"close": 1.0, "vol": 1.0, "amount": 1000.0,
                "r_vol_raw": 1.0, "r_amt_raw": 0.001, "n_calib": 0}
    df["stock_code"] = df["ts_code"].str.split(".").str[0]
    loc = pd.read_sql_query(
        "SELECT stock_code, 收盘 AS c_l, 成交量 AS v_l, 成交额 AS a_l "
        "FROM stock_data WHERE 日期=?", con, params=(d,))
    m = df.merge(loc, on="stock_code", how="inner")
    if len(m) < 100:
        return {"close": 1.0, "vol": 1.0, "amount": 1000.0,
                "r_vol_raw": 1.0, "r_amt_raw": 0.001, "n_calib": 0}
    r_vol = float((m["vol"] / m["v_l"].replace(0, np.nan)).median())
    r_amt = float((m["amount"] / m["a_l"].replace(0, np.nan)).median())
    return {
        "close": 1.0,
        "vol": (1.0 / r_vol) if r_vol > 0 else 1.0,
        "amount": (1.0 / r_amt) if r_amt > 0 else 1000.0,
        "r_vol_raw": r_vol, "r_amt_raw": r_amt, "n_calib": int(len(m)),
    }


def calibrate_units_daily(pro, con, trade_date_ts: str, d: str) -> dict:
    """★ 逐日精确校准：用【当天本地已有股票】反推 tushare→本地 的换算系数。

    为什么不用全局系数：
      全局系数（如 10.0288）与当天实际需要的系数（如 10.0039）
      可能差 0.25%，累积到 amount 上就是系统性偏差。
      实测 2026-10-15 的中位比值变成 1.0082（应为 1.0000）。

      正确做法：既然当天本地已有 2300 只股票，直接用它们的
      local_amount / tushare_amount 中位数作为当天系数。
    """
    df = fetch_day(pro, trade_date_ts)
    if df.empty:
        return None
    df["stock_code"] = df["ts_code"].str.split(".").str[0]
    loc = pd.read_sql_query(
        "SELECT stock_code, 收盘 AS c_l, 成交量 AS v_l, 成交额 AS a_l "
        "FROM stock_data WHERE 日期=?", con, params=(d,))
    m = df.merge(loc, on="stock_code", how="inner")
    if len(m) < 100:
        return None
    r_vol = float((m["vol"] / m["v_l"].replace(0, np.nan)).median())
    r_amt = float((m["amount"] / m["a_l"].replace(0, np.nan)).median())
    return {
        "vol": (1.0 / r_vol) if r_vol > 0 else 1.0,
        "amount": (1.0 / r_amt) if r_amt > 0 else 1000.0,
        "n_calib": int(len(m)), "r_amt_raw": r_amt, "r_vol_raw": r_vol,
    }


def to_local_code(ts_code: str) -> str:
    """tushare ts_code -> 本地 6 位代码。"""
    return str(ts_code).split(".")[0]


def fetch_day(pro, trade_date: str, max_retry: int = 4) -> pd.DataFrame:
    """拉某一天的全市场日线。"""
    for attempt in range(max_retry):
        try:
            df = pro.daily(trade_date=trade_date,
                           fields="ts_code,trade_date,open,high,low,close,"
                                  "pre_close,change,pct_chg,vol,amount")
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            msg = str(e)
            is_rate = "频率超限" in msg or "超限" in msg
            if attempt == max_retry - 1:
                print(f"  [{trade_date}] 失败: {e}")
                return pd.DataFrame()
            time.sleep(5.0 * (2 ** attempt) if is_rate else 2.0)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-gap", action="store_true", help="只修 10-29")
    ap.add_argument("--skip-amount", action="store_true", help="只补缺口")
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
    cols = [r[1] for r in con.execute("PRAGMA table_info(stock_data)")]
    print("=" * 90)
    print("修复 stock_data 覆盖缺口与成交额")
    print("=" * 90)
    print(f"数据库: {db}")
    print(f"字段: pre_close={'pre_close' in cols} "
          f"adj_factor={'adj_factor' in cols}")

    total_inserted = 0
    total_updated = 0
    report = []

    # ============================================================
    # 0. 单位校准（全局参考值，实际写入时用逐日精确校准）
    # ============================================================
    print("\n[0] 单位校准（参考值，写入时改用逐日精确校准）")
    print("-" * 90)
    calib = calibrate_units(pro, con, "20260918")
    print(f"  全局参考: vol×{calib['vol']:.4f}  amount×{calib['amount']:.4f}")
    VOL_MUL = calib["vol"]
    AMT_MUL = calib["amount"]

    # ============================================================
    # 1. 补齐覆盖缺口
    # ============================================================
    if not args.skip_gap:
        print(f"\n[1] 补齐 {len(BAD_DATES)} 天覆盖缺口")
        print("-" * 90)

        for tsd in BAD_DATES:
            d = fmt_date(tsd)
            existed = con.execute(
                "SELECT COUNT(*) FROM stock_data WHERE 日期=?", (d,)).fetchone()[0]
            print(f"\n  {d}  本地已有 {existed} 只", end="")

            if args.dry_run:
                print("  [DRY-RUN]")
                continue

            df = fetch_day(pro, tsd)
            if df.empty:
                print("  -> tushare 无数据")
                report.append({"date": d, "existed": existed,
                               "fetched": 0, "inserted": 0})
                continue

            df["stock_code"] = df["ts_code"].apply(to_local_code)
            df = df[(df["close"] > 0)]

            # ★ 逐日精确校准（用当天本地已有股票反推系数）
            dc = calibrate_units_daily(pro, con, tsd, d)
            if dc:
                vol_mul, amt_mul = dc["vol"], dc["amount"]
                print(f"  [校准 n={dc['n_calib']} vol×{vol_mul:.4f} "
                      f"amt×{amt_mul:.4f}]", end="")
            else:
                vol_mul, amt_mul = VOL_MUL, AMT_MUL
                print(f"  [校准失败，用全局]", end="")

            # 本地已有 + 已有 pre_close 的，取 adj_factor 作为换算基准
            local_af = pd.read_sql_query(
                "SELECT stock_code, adj_factor FROM stock_data "
                "WHERE adj_factor IS NOT NULL GROUP BY stock_code", con)
            af_map = dict(zip(local_af["stock_code"], local_af["adj_factor"]))

            existed_codes = set(r[0] for r in con.execute(
                "SELECT stock_code FROM stock_data WHERE 日期=?", (d,)).fetchall())

            rows_new = []
            rows_upd = []
            for r in df.itertuples(index=False):
                code = r.stock_code
                # 本地口径换算：优先用已知 adj_factor，否则用 1.0（未复权）
                af = af_map.get(code, 1.0)
                if not np.isfinite(af) or af <= 0:
                    af = 1.0
                pre_adj = float(r.pre_close) * af
                v_loc = float(r.vol) * vol_mul
                a_loc = float(r.amount) * amt_mul
                if code in existed_codes:
                    rows_upd.append((a_loc, v_loc, float(r.close),
                                     pre_adj, af, code, d))
                else:
                    rows_new.append((code, d, float(r.open), float(r.high),
                                     float(r.low), float(r.close), v_loc,
                                     a_loc, pre_adj, af))

            try:
                cur = con.cursor()
                # 1a. 更新已有行（补 amount/vol/close/pre_close，防单位错）
                if rows_upd:
                    cur.executemany("""
                        UPDATE stock_data
                        SET 成交额=?, 成交量=?, 收盘=?, pre_close=?, adj_factor=?
                        WHERE stock_code=? AND 日期=?
                    """, rows_upd)
                # 1b. 插入缺失行
                if rows_new:
                    cur.executemany("""
                        INSERT INTO stock_data
                            (stock_code, 日期, 开盘, 最高, 最低, 收盘,
                             成交量, 成交额, pre_close, adj_factor)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, rows_new)
                con.commit()
                ins, upd = len(rows_new), len(rows_upd)
                total_inserted += ins
                total_updated += upd
                print(f"  -> tushare {len(df)} 只，插入 {ins}，更新 {upd}")
                report.append({"date": d, "existed": existed,
                               "fetched": int(len(df)),
                               "inserted": ins, "updated": upd})
            except Exception as e:
                con.rollback()
                print(f"  -> 写入失败: {e}")
                report.append({"date": d, "existed": existed,
                               "fetched": int(len(df)), "error": str(e)})

    # ============================================================
    # 2. 修复 2026-10-29 成交额
    # ============================================================
    if not args.skip_amount:
        print(f"\n[2] 修复 {len(AMOUNT_BAD)} 天成交额单位")
        print("-" * 90)

        for tsd in AMOUNT_BAD:
            d = fmt_date(tsd)
            r = con.execute("""
                SELECT COUNT(*), SUM(CASE WHEN pre_close IS NOT NULL THEN 1 ELSE 0 END)
                FROM stock_data WHERE 日期=?
            """, (d,)).fetchone()
            print(f"\n  {d}  本地 {r[0]} 只（pre_close 已覆盖 {r[1] or 0}）")

            if args.dry_run:
                print("  [DRY-RUN]")
                continue

            df = fetch_day(pro, tsd)
            if df.empty:
                print("  -> tushare 无数据")
                continue
            df["stock_code"] = df["ts_code"].apply(to_local_code)
            df = df[(df["close"] > 0)]

            dc = calibrate_units_daily(pro, con, tsd, d)
            # ★ 成交额修复日（如 2026-10-29）不能靠自身反推：
            #   当天数据本身单位就错，用它会算出 1.0 的"错误校准"。
            #   改用前一个正常日的系数。
            if tsd in AMOUNT_BAD or (dc and abs(dc["amount"] - 1.0) < 1e-6):
                prev_calib = calibrate_units(pro, con, "20261028")
                vol_mul, amt_mul = prev_calib["vol"], prev_calib["amount"]
                print(f"  [用前日校准 vol×{vol_mul:.4f} amt×{amt_mul:.4f}]")
            elif dc:
                vol_mul, amt_mul = dc["vol"], dc["amount"]
                print(f"  [校准 n={dc['n_calib']} vol×{vol_mul:.4f} "
                      f"amt×{amt_mul:.4f}]")
            else:
                vol_mul, amt_mul = VOL_MUL, AMT_MUL
                print("  [校准失败，用全局]")

            local_af = pd.read_sql_query(
                "SELECT stock_code, adj_factor FROM stock_data "
                "WHERE adj_factor IS NOT NULL GROUP BY stock_code", con)
            af_map = dict(zip(local_af["stock_code"], local_af["adj_factor"]))

            rows = []
            for rr in df.itertuples(index=False):
                af = af_map.get(rr.stock_code, 1.0)
                if not np.isfinite(af) or af <= 0:
                    af = 1.0
                rows.append((float(rr.amount) * amt_mul,
                             float(rr.vol) * vol_mul,
                             float(rr.close),
                             float(rr.pre_close) * af, af,
                             rr.stock_code, d))
            try:
                cur = con.cursor()
                cur.executemany("""
                    UPDATE stock_data
                    SET 成交额=?, 成交量=?, 收盘=?, pre_close=?, adj_factor=?
                    WHERE stock_code=? AND 日期=?
                """, rows)
                con.commit()
                print(f"  -> 更新 {cur.rowcount} 行")
                total_updated += cur.rowcount
                report.append({"date": d, "updated": cur.rowcount,
                               "mode": "amount_fix"})
            except Exception as e:
                con.rollback()
                print(f"  -> 失败: {e}")

    con.close()

    print("\n" + "=" * 90)
    print(f"完成: 插入 {total_inserted:,} 行，更新 {total_updated:,} 行")
    print("=" * 90)

    import json
    out = ROOT / "results" / "coverage_gap_fix.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"inserted": total_inserted, "updated": total_updated,
                   "detail": report}, f, ensure_ascii=False, indent=2)
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
