# -*- coding: utf-8 -*-
"""修复历史数据的成交额单位（2021~2024 年段）。

诊断结论（2026-10-31）：
  抽查 000001 平安银行 2022-11-01：
    tushare amount : 1,459,758.4  （千元）
    本地   amount : 1,459,758,367.0（元）
    比值          : 精确 1000.0000

  即：本地 amount = tushare amount(千元) × 1000 = 元。
  这本身是【正确的】！

  问题在于 2024 年之前的行，amount 被乘了 1000 两次，
  或者写入了「厘」单位。实测 amount/(vol×close) ≈ 100，
  而在 2025 年之后是 1.0。

  → 2021-11 ~ 2024 年间的 amount 需要 ÷ 1000。

判定边界：
  逐日检查 amount/(vol×close) 中位数：
    ≈ 1    正常
    ≈ 100  需 ÷ 1000
  用这个判据自动分段，避免硬编码日期。

用法：
  python scripts/fix_history_amount_unit.py --dry-run
  python scripts/fix_history_amount_unit.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/amount_unit_by_day.csv")
    args = ap.parse_args()

    db = args.db or resolve_db()
    con = sqlite3.connect(db)

    print("=" * 92)
    print("成交额单位分段诊断")
    print("=" * 92)

    # 逐日计算 amount/(vol*close) 中位数
    df = pd.read_sql_query("""
        SELECT 日期, 收盘, 成交量, 成交额 FROM stock_data
        WHERE 收盘>0 AND 成交量>0 AND 成交额>0
    """, con)
    print(f"  载入 {len(df):,} 行")

    df["ratio"] = df["成交额"] / df["成交量"] / df["收盘"]
    daily = df.groupby("日期")["ratio"].agg(["median", "count"]).reset_index()
    daily.columns = ["date", "ratio_median", "n"]
    daily = daily[daily["n"] >= 100].sort_values("date").reset_index(drop=True)

    print(f"  有效交易日 {len(daily)}")

    # 分类
    def classify(r):
        if 0.5 < r < 2.0:
            return "OK"
        if 50 < r < 200:
            return "DIV1000"
        return "UNKNOWN"

    daily["status"] = daily["ratio_median"].apply(classify)

    print(f"\n  状态分布: {daily['status'].value_counts().to_dict()}")

    # 按状态分段
    segs = []
    cur_status = None
    start = None
    for _, row in daily.iterrows():
        if row["status"] != cur_status:
            if cur_status is not None:
                segs.append((cur_status, start, prev))
            cur_status = row["status"]
            start = row["date"]
        prev = row["date"]
    if cur_status is not None:
        segs.append((cur_status, start, prev))

    print(f"\n  日期分段:")
    for st, s, e in segs:
        n = len(daily[(daily["date"] >= s) & (daily["date"] <= e)])
        print(f"    {s} ~ {e}  {st:<10} ({n} 天)")

    # 保存明细
    os.makedirs(ROOT / "results", exist_ok=True)
    daily.to_csv(ROOT / args.out, index=False, encoding="utf-8-sig")
    print(f"\n  已写入 {args.out}")

    need_fix = [s for s in segs if s[0] == "DIV1000"]
    unknown = [s for s in segs if s[0] == "UNKNOWN"]

    if unknown:
        print(f"\n  ⚠️ 有 {len(unknown)} 段状态未知，需人工确认:")
        for st, s, e in unknown:
            sub = daily[(daily["date"] >= s) & (daily["date"] <= e)]
            print(f"    {s} ~ {e}: 中位比值 "
                  f"{sub['ratio_median'].min():.2f} ~ "
                  f"{sub['ratio_median'].max():.2f}")

    if not need_fix:
        print("\n  无需修复")
        con.close()
        return 0

    print(f"\n  → 需要 ÷1000 的分段: {len(need_fix)} 段")

    if args.dry_run:
        for st, s, e in need_fix:
            n = con.execute(
                "SELECT COUNT(*) FROM stock_data WHERE 日期>=? AND 日期<=?",
                (s, e)).fetchone()[0]
            print(f"    [DRY-RUN] {s} ~ {e}: 将修复 {n:,} 行")
        con.close()
        return 0

    # 执行修复
    total = 0
    for st, s, e in need_fix:
        n_before = con.execute(
            "SELECT COUNT(*) FROM stock_data WHERE 日期>=? AND 日期<=?",
            (s, e)).fetchone()[0]
        con.execute("""
            UPDATE stock_data SET 成交额 = 成交额 / 1000.0
            WHERE 日期>=? AND 日期<=?
        """, (s, e))
        con.commit()
        print(f"    {s} ~ {e}: 修复 {n_before:,} 行")
        total += n_before

    print(f"\n  共修复 {total:,} 行")

    # 验证
    print("\n验证:")
    for _, row in daily.iloc[::max(1, len(daily) // 10)].iterrows():
        rr = con.execute("""
            SELECT 成交额/成交量/收盘 FROM stock_data
            WHERE 日期=? AND 收盘>0 AND 成交量>0 AND 成交额>0
            LIMIT 2000
        """, (row["date"],)).fetchall()
        if rr:
            v = np.median([x[0] for x in rr])
            print(f"    {row['date']}: 中位比值 {v:.4f}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
