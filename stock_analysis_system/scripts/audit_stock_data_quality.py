# -*- coding: utf-8 -*-
"""stock_data 数据质量审计（横截面选股前置检查）。

发现问题（2026-10-31）：
  · 2026-10-29：成交额单位错（隐含价只有收盘的 1/10），换手率全空
  · 2026-09-22：只有 1975 只股票
  · 2026-09-24 ~ 2026-10-27：只有 2322 只股票
  · 2026-10-28 起才恢复 5460+

本脚本逐日标记"可用 / 可疑 / 不可用"，产出白名单供后续脚本使用。

判定标准：
  可用   : 股票数 >= 4500 且 成交额单位正确率 >= 95% 且 换手率非空率 >= 95%
  可疑   : 股票数 >= 3000 但某项不达标
  不可用 : 其他

用法：
  python scripts/audit_stock_data_quality.py
  python scripts/audit_stock_data_quality.py --out results/stock_data_quality.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


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


def audit(db: str, min_stocks: int = 4500, min_ok_rate: float = 0.95) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql_query("""
        SELECT 日期 AS date, stock_code, 收盘 AS close,
               成交量 AS vol, 成交额 AS amount, 换手率 AS turnover
        FROM stock_data
        WHERE 收盘 > 0
    """, con)
    con.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["close", "vol", "amount", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"])

    rows = []
    for d, g in df.groupby("date", sort=True):
        n = len(g)
        # 单位校验：amount / vol / close 应该 ≈ 1
        ok_mask = (g["vol"] > 0) & (g["amount"] > 0) & (g["close"] > 0)
        if ok_mask.sum() > 0:
            ratio = (g.loc[ok_mask, "amount"] /
                     g.loc[ok_mask, "vol"] / g.loc[ok_mask, "close"]).values
            # 允许 0.9~1.1 或 0.09~0.11（两种单位口径）
            good = np.mean(((ratio > 0.9) & (ratio < 1.1)) |
                           ((ratio > 0.09) & (ratio < 0.11)))
            unit_main = float(np.median(ratio))
        else:
            good, unit_main = 0.0, np.nan

        tr_ok = float(g["turnover"].notna().mean())
        amt_ok = float(g["amount"].notna().mean())

        # 判定
        if n >= min_stocks and good >= min_ok_rate and tr_ok >= min_ok_rate \
                and amt_ok >= min_ok_rate and abs(unit_main - 1.0) < 0.15:
            status = "可用"
        elif n >= 3000:
            status = "可疑"
        else:
            status = "不可用"

        rows.append({
            "date": d, "n_stocks": n,
            "unit_ok_rate": float(good), "unit_median": unit_main,
            "turnover_ok_rate": tr_ok, "amount_ok_rate": amt_ok,
            "status": status,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--min-stocks", type=int, default=4500)
    ap.add_argument("--min-ok-rate", type=float, default=0.95)
    ap.add_argument("--out", default="results/stock_data_quality.json")
    args = ap.parse_args()

    db = args.db or resolve_db()
    print("=" * 88)
    print("stock_data 数据质量审计")
    print("=" * 88)

    q = audit(db, args.min_stocks, args.min_ok_rate)
    print(f"总交易日 {len(q)}")
    print(f"区间 {q['date'].min().date()} ~ {q['date'].max().date()}")

    print("\n状态分布:")
    for s, n in q["status"].value_counts().items():
        print(f"  {s:<8}{n:>5} 天")

    bad = q[q["status"] != "可用"]
    if len(bad):
        print(f"\n非可用日期（{len(bad)} 天）:")
        print(f"  {'日期':<12}{'股票数':>8}{'单位OK率':>10}{'中位比值':>10}"
              f"{'换手率OK':>10}{'成交额OK':>10}  状态")
        for _, r in bad.iterrows():
            print(f"  {str(r['date'].date()):<12}{r['n_stocks']:>8}"
                  f"{r['unit_ok_rate']*100:>9.1f}%{r['unit_median']:>10.4f}"
                  f"{r['turnover_ok_rate']*100:>9.1f}%"
                  f"{r['amount_ok_rate']*100:>9.1f}%  {r['status']}")

    good = q[q["status"] == "可用"]
    if len(good):
        print(f"\n可用区间: {good['date'].min().date()} ~ {good['date'].max().date()}"
              f"  共 {len(good)} 天")
        # 找最长的连续区间
        gaps = good["date"].diff().dt.days
        breaks = np.where(gaps > 10)[0]
        segs = []
        start = 0
        for b in breaks:
            segs.append((good["date"].iloc[start], good["date"].iloc[b - 1], b - start))
            start = b
        segs.append((good["date"].iloc[start], good["date"].iloc[-1], len(good) - start))
        print("\n连续可用区间:")
        for s, e, n in segs:
            print(f"  {s.date()} ~ {e.date()}  ({n} 天)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {
        "total_days": int(len(q)),
        "usable_days": int((q["status"] == "可用").sum()),
        "usable_dates": [str(d.date()) for d in good["date"]],
        "bad_dates": [
            {"date": str(r["date"].date()), "n_stocks": int(r["n_stocks"]),
             "unit_ok_rate": r["unit_ok_rate"], "unit_median": r["unit_median"],
             "turnover_ok_rate": r["turnover_ok_rate"],
             "status": r["status"]}
            for _, r in bad.iterrows()
        ],
        "criteria": {"min_stocks": args.min_stocks, "min_ok_rate": args.min_ok_rate},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
