# -*- coding: utf-8 -*-
"""补完 pre_close 后的完整验证。

检查项：
  1. 覆盖率（非空率）
  2. 收益分布是否合理（无 -88% 这类怪值）
  3. 极端值是否都是真实涨停/跌停
  4. adj_factor 序列是否只在除权日跳变
  5. 与 tushare 独立抽查比对

用法：
  python scripts/verify_pre_close.py
  python scripts/verify_pre_close.py --sample 30
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--out", default="results/pre_close_verify.json")
    args = ap.parse_args()

    db = resolve_db()
    con = sqlite3.connect(db)

    print("=" * 88)
    print("pre_close 完整性验证")
    print("=" * 88)

    # ---- 1. 覆盖率 ----
    r = con.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN pre_close IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN pre_close <= 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN adj_factor IS NULL THEN 1 ELSE 0 END)
        FROM stock_data""").fetchone()
    tot, nulls, nonpos, no_af = r
    cov = 1 - (nulls or 0) / max(tot, 1)
    print(f"\n[1] 覆盖率")
    print(f"  总行数        : {tot:,}")
    print(f"  pre_close 缺失: {nulls or 0:,}")
    print(f"  pre_close <=0 : {nonpos or 0:,}")
    print(f"  adj_factor 缺失: {no_af or 0:,}")
    print(f"  覆盖率        : {cov*100:.2f}%")

    prog = pd.read_sql_query(
        "SELECT status, COUNT(*) n FROM stock_pre_close_progress GROUP BY status",
        con)
    print(f"  进度: {dict(zip(prog['status'], prog['n']))}")

    # ---- 2. 收益分布 ----
    print(f"\n[2] 用 pre_close 计算的日收益分布")
    df = pd.read_sql_query("""
        SELECT stock_code, 日期, 收盘, pre_close, adj_factor
        FROM stock_data WHERE pre_close IS NOT NULL AND pre_close > 0 AND 收盘 > 0
    """, con)
    df["ret"] = df["收盘"] / df["pre_close"] - 1.0

    print(f"  样本 {len(df):,} 行")
    for q in [0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999]:
        print(f"    p{q*100:>7.2f}%: {df['ret'].quantile(q)*100:>+8.3f}%")

    # ---- 3. 极端值 ----
    print(f"\n[3] 极端值检查（|ret| > 21%，超出所有板块涨跌停）")
    ext = df[df["ret"].abs() > 0.21]
    print(f"  共 {len(ext):,} 行 ({len(ext)/len(df)*100:.4f}%)")
    if len(ext):
        print(f"  {'代码':<8}{'日期':<12}{'收盘':>9}{'pre_close':>11}"
              f"{'收益':>10}{'adj_factor':>12}")
        for _, rr in ext.sort_values("ret").head(10).iterrows():
            print(f"  {rr['stock_code']:<8}{rr['日期']:<12}{rr['收盘']:>9.2f}"
                  f"{rr['pre_close']:>11.4f}{rr['ret']*100:>9.2f}%"
                  f"{rr['adj_factor']:>12.4f}")
        print(f"  ...（共 {len(ext)} 行）")

    # 跌停统计（主板 -10%，创业板/科创 -20%）
    print(f"\n  按板块统计跌停附近的比例:")
    df["board"] = np.where(df["stock_code"].str.startswith(("30", "68")),
                           "创业板/科创", "主板")
    for b in df["board"].unique():
        sub = df[df["board"] == b]
        lim = -0.20 if b == "创业板/科创" else -0.10
        near = sub[(sub["ret"] < lim - 0.005) & (sub["ret"] > lim - 0.03)]
        below = sub[sub["ret"] <= lim - 0.03]
        print(f"    {b:<12} 总 {len(sub):>9,}  "
              f"跌停附近 {len(near):>7,}  严重超跌 {len(below):>6,}")

    # ---- 4. adj_factor 稳定性 ----
    print(f"\n[4] adj_factor 变化频率（应只在除权日跳变）")
    s = df[df["stock_code"] == "600519"].sort_values("日期")
    if len(s) > 10:
        chg = s["adj_factor"].pct_change()
        n_chg = int((chg.abs() > 0.005).sum())
        print(f"  600519 贵州茅台: {len(s)} 天，adj_factor 跳变 {n_chg} 次")
        print(f"    adj_factor 范围: {s['adj_factor'].min():.4f} ~ "
              f"{s['adj_factor'].max():.4f}")

    # ---- 5. 独立抽查比对 ----
    token = load_token()
    if token and args.sample > 0:
        print(f"\n[5] 独立抽查：随机 {args.sample} 只与 tushare 重新比对")
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        rng = np.random.RandomState(0)
        codes = df["stock_code"].unique()
        pick = rng.choice(codes, size=min(args.sample, len(codes)), replace=False)

        ok_n = bad_n = 0
        for code in pick:
            ts_code = code + (".SH" if code.startswith("6") else ".SZ")
            try:
                raw = pro.daily(ts_code=ts_code, start_date="20241001",
                                end_date="20261031",
                                fields="ts_code,trade_date,close,pre_close")
                if raw is None or raw.empty:
                    continue
                raw["date"] = raw["trade_date"].astype(str).apply(
                    lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}")
                raw["ret_t"] = raw["close"] / raw["pre_close"] - 1
                loc = df[df["stock_code"] == code][["日期", "ret"]]
                mm = raw.merge(loc, left_on="date", right_on="日期", how="inner")
                if len(mm) < 20:
                    continue
                diff = np.abs(mm["ret_t"] - mm["ret"]).max()
                if diff < 1e-4:
                    ok_n += 1
                else:
                    bad_n += 1
                    print(f"    MISMATCH {code}: 最大偏差 {diff*100:.4f}%")
            except Exception as e:
                print(f"    ERROR {code}: {e}")
        print(f"  比对结果: 一致 {ok_n}  不一致 {bad_n}")

    con.close()

    out = {"total": int(tot), "coverage": float(cov),
           "n_extreme_21": int(len(ext)),
           "ret_p001": float(df["ret"].quantile(0.001)),
           "ret_p999": float(df["ret"].quantile(0.999))}
    os.makedirs((ROOT / "results"), exist_ok=True)
    import json
    with open(ROOT / "results" / "pre_close_verify.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 results/pre_close_verify.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
