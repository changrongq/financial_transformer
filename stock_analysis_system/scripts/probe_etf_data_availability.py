"""测试能否用 tushare 拉取长周期 ETF 数据（含分钟线）。

目的：确认船长要的「10-15 年 ETF 日线 + 14:50 分时价」是否可行。

用法：
  python scripts/probe_etf_data_availability.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def get_token() -> str:
    tok = os.environ.get("TUSHARE_TOKEN")
    if tok:
        return tok.strip()
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("找不到 tushare token")


def main() -> None:
    import tushare as ts

    tok = get_token()
    pro = ts.pro_api(tok)

    print("=" * 72)
    print("1. ETF 日线（fund_daily）—— 沪深300ETF 510300")
    print("=" * 72)
    try:
        df = pro.fund_daily(ts_code="510300.SH", start_date="20100101", end_date="20261028")
        if df is None or df.empty:
            print("  ❌ 返回空")
        else:
            df = df.sort_values("trade_date")
            print(f"  ✅ 拿到 {len(df)} 行")
            print(f"     范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
            print(f"     列: {list(df.columns)}")
            print(f"\n  最近3行:")
            print(df.tail(3).to_string(index=False))
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("2. 各宽基 ETF 的历史深度探测")
    print("=" * 72)
    etfs = [
        ("510300.SH", "沪深300ETF"),
        ("510500.SH", "中证500ETF"),
        ("159915.SZ", "创业板ETF"),
        ("510050.SH", "上证50ETF"),
        ("512100.SH", "中证1000ETF"),
        ("588000.SH", "科创50ETF"),
    ]
    for code, name in etfs:
        try:
            df = pro.fund_daily(ts_code=code, start_date="20050101", end_date="20261028")
            if df is None or df.empty:
                print(f"  {code} {name:<12} ❌ 无数据")
            else:
                df = df.sort_values("trade_date")
                years = len(df) / 244
                print(f"  {code} {name:<12} ✅ {len(df):>5} 行, "
                      f"{df['trade_date'].min()} ~ {df['trade_date'].max()} （约 {years:.1f} 年）")
        except Exception as e:
            print(f"  {code} {name:<12} ❌ {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("3. 分钟线（14:50 价格需要）—— 这是关键限制")
    print("=" * 72)
    for code, name in [("510300.SH", "沪深300ETF")]:
        for freq in ["1min", "5min", "15min", "30min", "60min"]:
            try:
                df = ts.pro_bar(
                    ts_code=code, freq=freq, asset="FD",
                    start_date="20240101", end_date="20261028",
                )
                if df is None or df.empty:
                    print(f"  {freq:<6} ❌ 空")
                else:
                    print(f"  {freq:<6} ✅ {len(df):>6} 行, "
                          f"{df['trade_time'].min() if 'trade_time' in df else df.iloc[-1,1]} ~ "
                          f"{df['trade_time'].max() if 'trade_time' in df else df.iloc[0,1]}")
            except Exception as e:
                msg = str(e)[:100]
                print(f"  {freq:<6} ❌ {type(e).__name__}: {msg}")

    print()
    print("=" * 72)
    print("4. 指数日线（可替代 ETF，历史更长）")
    print("=" * 72)
    for code, name in [("000300.SH", "沪深300"), ("000905.SH", "中证500"),
                       ("000001.SH", "上证综指"), ("399006.SZ", "创业板指")]:
        try:
            df = pro.index_daily(ts_code=code, start_date="20050101", end_date="20261028")
            if df is None or df.empty:
                print(f"  {code} {name:<10} ❌ 无数据")
            else:
                df = df.sort_values("trade_date")
                print(f"  {code} {name:<10} ✅ {len(df):>5} 行, "
                      f"{df['trade_date'].min()} ~ {df['trade_date'].max()} （约 {len(df)/244:.1f} 年）")
        except Exception as e:
            print(f"  {code} {name:<10} ❌ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
