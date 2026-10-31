"""修复 stock_data 的成交额和成交量单位问题。

====================== 诊断结论 ======================
stock_data 的「成交量」单位有三段不一致：

  段 1：2024-10-08 ~ 2025-04-28   单位 = 手
  段 2：2025-04-29 ~ 2025-10-31   单位 = 股（×100，被污染）
  段 3：2025-11-03 ~ 至今         单位 = 手

证据（000001 平安银行）：
  2025-04-28  成交量      642,018  （手）
  2025-04-29  成交量   64,904,494  （股，×101）
  ...
  2025-11-03  成交量      952,326  （手，恢复）
  中位数比值 110 倍

「成交额」字段：
  · 2024-10-11 ~ 2026-03-31 完全为空（343 个交易日）
  · 2026-04 之后有值，且 = close × vol（此时 vol 单位=股）

====================== 修复方案 ======================
用 tushare `daily` 接口的 `amount` 字段重建：
  · tushare amount 单位 = 千元 → ×1000 得到元
  · tushare vol 单位 = 手    → ×100 得到股
  · 统一：成交量存"股"，成交额存"元"

对每个缺失的交易日，按 trade_date 拉全市场（1 次调用/天），
覆盖写入 stock_data 的 [成交量] 和 [成交额]。

用法：
  python scripts/repair_amount_data.py --dry-run          # 只看要修多少
  python scripts/repair_amount_data.py                    # 执行修复
  python scripts/repair_amount_data.py --start 20241011 --end 20260331
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)


def resolve_db() -> str:
    config = ROOT / "config.yaml"
    if config.exists():
        import yaml
        with config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def get_token():
    t = os.environ.get("TUSHARE_TOKEN")
    if t:
        return t.strip()
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def find_bad_days(conn: sqlite3.Connection) -> list[str]:
    """找出需要修复的交易日（成交额为空，或成交量单位异常）。"""
    q = """
    SELECT [日期] AS d, COUNT(*) AS n,
           SUM(CASE WHEN [成交额] IS NOT NULL AND [成交额] > 0 THEN 1 ELSE 0 END) AS amt_ok,
           AVG([成交量]) AS avg_vol
    FROM stock_data GROUP BY [日期] ORDER BY [日期]
    """
    g = pd.read_sql_query(q, conn)
    g["amt_pct"] = g["amt_ok"] / g["n"]
    # 判据1：成交额缺失
    bad = g[g["amt_pct"] < 0.99].copy()
    # 判据2：成交量量级异常（avg_vol > 300万 手 明显不合理）
    # 正常A股日均成交量约 20-30万手
    vol_bad = g[g["avg_vol"] > 3_000_000].copy()
    out = sorted(set(bad["d"].tolist()) | set(vol_bad["d"].tolist()))
    return out


def repair_day(pro, conn: sqlite3.Connection, day: str) -> tuple[int, str]:
    """修复单个交易日。返回 (影响行数, 状态)。"""
    td = day.replace("-", "")
    try:
        df = pro.daily(trade_date=td)
    except Exception as e:
        return 0, f"API错误: {type(e).__name__}: {str(e)[:60]}"

    if df is None or df.empty:
        return 0, "无数据（非交易日？）"

    d = df.copy()
    d["stock_code"] = d["ts_code"].astype(str).str[:6]
    # tushare: vol 单位=手 -> 股；amount 单位=千元 -> 元
    d["vol_shares"] = pd.to_numeric(d["vol"], errors="coerce") * 100.0
    d["amt_yuan"] = pd.to_numeric(d["amount"], errors="coerce") * 1000.0
    d = d.dropna(subset=["vol_shares", "amt_yuan"])

    rows = [(float(r.vol_shares), float(r.amt_yuan), r.stock_code, day)
            for r in d.itertuples(index=False)]
    conn.executemany(
        "UPDATE stock_data SET [成交量] = ?, [成交额] = ? "
        "WHERE stock_code = ? AND [日期] = ?", rows)
    conn.commit()
    return len(rows), "OK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    parser.add_argument("--limit", type=int, default=0, help="最多修多少天（0=全部）")
    args = parser.parse_args()

    db = resolve_db()
    conn = sqlite3.connect(db)

    bad_days = find_bad_days(conn)
    if args.start:
        s = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:8]}"
        bad_days = [d for d in bad_days if d >= s]
    if args.end:
        e = f"{args.end[:4]}-{args.end[4:6]}-{args.end[6:8]}"
        bad_days = [d for d in bad_days if d <= e]
    if args.limit > 0:
        bad_days = bad_days[:args.limit]

    print(f"待修复交易日: {len(bad_days)}")
    if bad_days:
        print(f"  范围: {bad_days[0]} ~ {bad_days[-1]}")

    if args.dry_run:
        print("\n[dry-run] 只列前 10 个:")
        for d in bad_days[:10]:
            r = pd.read_sql_query(
                "SELECT COUNT(*) n, SUM(CASE WHEN [成交额]>0 THEN 1 ELSE 0 END) ok, "
                "AVG([成交量]) av FROM stock_data WHERE [日期]=?", conn, params=(d,)).iloc[0]
            print(f"  {d}  股票数={int(r['n'])}  有成交额={int(r['ok'] or 0)}  "
                  f"平均成交量={r['av']:,.0f}")
        conn.close()
        return

    tok = get_token()
    if not tok:
        print("无 tushare token，无法修复")
        conn.close()
        return

    import tushare as ts
    pro = ts.pro_api(tok)

    print(f"\n开始修复（{datetime.now():%H:%M:%S}）...")
    total = 0
    ok_days = 0
    fail = 0
    for i, d in enumerate(bad_days, 1):
        n, status = repair_day(pro, conn, d)
        if n > 0:
            total += n
            ok_days += 1
        else:
            fail += 1
        if i % 20 == 0 or i == len(bad_days):
            print(f"  [{i}/{len(bad_days)}] 已修 {ok_days} 天 / {total:,} 行  "
                  f"失败 {fail}  (最新: {d} {status})")
        time.sleep(0.35)

    print(f"\n完成: 修复 {ok_days} 个交易日，{total:,} 行")

    # 验证
    print("\n" + "=" * 70)
    print("修复后验证")
    print("=" * 70)
    q = """
    SELECT [日期] AS d, COUNT(*) n,
           SUM(CASE WHEN [成交额]>0 THEN 1 ELSE 0 END) ok,
           AVG([成交量]) av
    FROM stock_data GROUP BY [日期] ORDER BY [日期]
    """
    g = pd.read_sql_query(q, conn)
    g["pct"] = g["ok"] / g["n"] * 100
    print(f"  仍有成交额缺失的交易日: {int((g['pct'] < 99).sum())}")
    print(f"  成交量量级异常(>300万)的交易日: {int((g['av'] > 3_000_000).sum())}")
    print(f"\n  最近5日:")
    print(g.tail(5)[["d", "n", "ok", "av"]].to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
