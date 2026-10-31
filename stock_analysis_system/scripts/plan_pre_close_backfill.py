# -*- coding: utf-8 -*-
"""估算补充 pre_close 所需的工作量与方案。

pre_close（昨收，复权后）的来源有几种：
  A. tushare `daily` 接口：每只股票一次调用，返回全历史
     限速：500 次/分钟（按积分），5800 只 → 约 12 分钟
  B. tushare `adj_factor` 接口：复权因子，可自己算 pre_close
     需要额外表，但能同时修正历史
  C. 自己算：用 close.shift(1)，但除权日会错
     → 只能检测除权，无法修正

本脚本：
  1. 统计需要覆盖的股票数与日期数
  2. 检测当前数据里"疑似除权"的事件有多频繁
  3. 检查 tushare 是否可用、积分是否够
  4. 输出建议方案

用法：
  python scripts/plan_pre_close_backfill.py
"""

from __future__ import annotations

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


def load_token() -> str | None:
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return os.environ.get("TUSHARE_TOKEN")


def main():
    db = resolve_db()
    con = sqlite3.connect(db)

    print("=" * 84)
    print("补充 pre_close 的工作量估算")
    print("=" * 84)

    # ---- 1. 需要覆盖的范围 ----
    r = con.execute("""
        SELECT COUNT(DISTINCT stock_code), COUNT(DISTINCT 日期),
               MIN(日期), MAX(日期), COUNT(*)
        FROM stock_data
    """).fetchone()
    n_stocks, n_days, d_min, d_max, n_rows = r
    print(f"\n[1] 数据范围")
    print(f"  股票数   : {n_stocks:,}")
    print(f"  交易日   : {n_days:,}")
    print(f"  日期范围 : {d_min} ~ {d_max}")
    print(f"  总行数   : {n_rows:,}")

    # ---- 2. 疑似除权事件统计 ----
    print(f"\n[2] 疑似除权事件统计")
    df = pd.read_sql_query("""
        SELECT stock_code, 日期, 收盘 FROM stock_data
        WHERE 收盘 > 0 ORDER BY stock_code, 日期
    """, con)
    df["ret"] = df.groupby("stock_code")["收盘"].pct_change(fill_method=None)

    for th in [0.11, 0.15, 0.20, 0.30]:
        n = int((df["ret"] < -th).sum())
        print(f"  单日跌超 {th*100:.0f}%: {n:>7,} 次 ({n/len(df)*100:.3f}%)")

    # 按年份看
    df["year"] = df["日期"].str[:4]
    extreme = df[df["ret"] < -0.20]
    if len(extreme):
        print(f"\n  极端下跌（<-20%）按年份:")
        for y, g in extreme.groupby("year"):
            print(f"    {y}: {len(g):>5} 次")

    # ---- 3. tushare 可用性 ----
    print(f"\n[3] tushare 可用性")
    token = load_token()
    if not token:
        print("  未找到 token（config/tushare_token.local 或 TUSHARE_TOKEN）")
        print("  → 无法自动补充，需要先配置")
        con.close()
        return 1
    print(f"  token 已找到（长度 {len(token)}）")

    try:
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        # 测试单次调用
        t0 = pd.Timestamp.now()
        test = pro.daily(ts_code="000001.SZ", start_date="20260101",
                         end_date="20260110",
                         fields="ts_code,trade_date,close,pre_close")
        el = (pd.Timestamp.now() - t0).total_seconds()
        print(f"  测试调用成功，耗时 {el:.2f}s，返回 {len(test)} 行")
        if len(test):
            print(f"  字段: {list(test.columns)}")
            print(f"  样例:\n{test.head(3).to_string(index=False)}")
        est_min = n_stocks * el / 60
        print(f"\n  估算：{n_stocks:,} 只 × {el:.2f}s ≈ {est_min:.1f} 分钟（串行）")
    except Exception as e:
        print(f"  调用失败: {type(e).__name__}: {e}")

    # ---- 4. 检查 etf_daily 已有的 pre_close 格式 ----
    print(f"\n[4] 参考：etf_daily 的 pre_close 覆盖情况")
    r = con.execute("""
        SELECT COUNT(*) tot,
               SUM(CASE WHEN pre_close IS NULL OR pre_close<=0 THEN 1 ELSE 0 END) bad
        FROM etf_daily
    """).fetchone()
    print(f"  总行 {r[0]:,}，缺 pre_close {r[1]:,} ({r[1]/r[0]*100:.2f}%)")

    # ---- 5. 其他表是否已有复权因子 ----
    print(f"\n[5] 现有表检查（是否已有复权相关数据）")
    tables = [t[0] for t in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"  表: {tables}")
    for t in tables:
        cols = [c[1] for c in con.execute(f"PRAGMA table_info({t})")]
        hits = [c for c in cols if any(k in c.lower() for k in
                                       ["pre_close", "adj", "factor", "复权"])]
        if hits:
            print(f"  {t}: {hits}")

    con.close()

    print("\n" + "=" * 84)
    print("建议方案")
    print("=" * 84)
    print("""
  方案 A（推荐）：用 tushare `daily` 接口拉全历史，取 pre_close
    · 每只股票 1 次调用，5800 只 → 约 10-20 分钟
    · 同时能校验 close（发现现有数据错误）
    · 缺点：pre_close 是未复权口径（tushare daily 的 pre_close
      是"昨收"，已按除权调整，实际就是复权后昨收）

  方案 B：额外拉 `adj_factor` 复权因子
    · 可以自己构造任意口径的复权价
    · 多一倍调用量
    · 更适合做长周期研究

  方案 C（最小）：只用 close.shift(1) 填充，除权日标为 NaN
    · 零成本，但除权日收益仍然缺失
    · 只适合"知道哪里有问题"而非"修好"

  建议：先做方案 A，把 pre_close 补进 stock_data，
  然后重跑横截面检验看结论是否改变。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
