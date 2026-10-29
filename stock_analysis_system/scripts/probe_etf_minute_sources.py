"""探测：还有哪些渠道能拿到 ETF 的长周期日内数据？

背景：
  tushare 分钟线限速 1次/分钟，且只到 2026-09（约 40 天）
  船长问：有没有其他渠道？

本脚本逐个测试可用数据源：
  1. akshare（免费，接口多）
  2. tushare 的不同频率接口（含 stk_mins 的积分要求）
  3. 检查是否有本地已有分钟数据

对每个源，测试：
  · 能否拿到 510300 的分钟线
  · 最早能到什么时候
  · 频率限制

用法：
  python scripts/probe_etf_minute_sources.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)


def test_akshare() -> list[dict]:
    """测试 akshare 的各种分钟线接口。"""
    results = []
    try:
        import akshare as ak
    except ImportError:
        print("  akshare 未安装")
        return results

    # 候选接口
    tests = [
        ("fund_etf_hist_min_em",
         lambda: ak.fund_etf_hist_min_em(symbol="510300", period="1", adjust="")),
        ("fund_etf_hist_min_em_5min",
         lambda: ak.fund_etf_hist_min_em(symbol="510300", period="5", adjust="")),
        ("fund_etf_hist_min_em_60min",
         lambda: ak.fund_etf_hist_min_em(symbol="510300", period="60", adjust="")),
        ("stock_zh_a_hist_min_em",
         lambda: ak.stock_zh_a_hist_min_em(symbol="510300", period="1", adjust="")),
    ]
    for name, fn in tests:
        try:
            t0 = time.time()
            df = fn()
            elapsed = time.time() - t0
            if df is None or df.empty:
                print(f"  {name:<32} [--] 返回空")
                results.append({"source": "akshare", "api": name, "status": "empty"})
                continue
            # 找时间列
            tcol = None
            for c in ["时间", "datetime", "time", "日期"]:
                if c in df.columns:
                    tcol = c
                    break
            if tcol:
                df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
                tmin, tmax = df[tcol].min(), df[tcol].max()
                ndays = df[tcol].dt.date.nunique()
                print(f"  {name:<32} [OK] {len(df):>6} 行  "
                      f"{tmin} ~ {tmax}  ({ndays} 个交易日, {elapsed:.1f}s)")
                results.append({"source": "akshare", "api": name, "status": "ok",
                                "rows": len(df), "start": str(tmin), "end": str(tmax),
                                "trading_days": int(ndays)})
            else:
                print(f"  {name:<32} [OK] {len(df)} 行，但无时间列: {list(df.columns)[:6]}")
                results.append({"source": "akshare", "api": name, "status": "ok_no_time",
                                "rows": len(df)})
        except Exception as e:
            print(f"  {name:<32} [X] {type(e).__name__}: {str(e)[:60]}")
            results.append({"source": "akshare", "api": name, "status": "error",
                            "error": f"{type(e).__name__}: {str(e)[:100]}"})
        time.sleep(1)
    return results


def test_tushare() -> list[dict]:
    """测试 tushare 的分钟接口（注意限速）。"""
    results = []
    tok = os.environ.get("TUSHARE_TOKEN")
    if not tok:
        p = ROOT / "config" / "tushare_token.local"
        if p.exists():
            tok = p.read_text(encoding="utf-8").strip()
    if not tok:
        print("  无 tushare token")
        return results

    try:
        import tushare as ts
        pro = ts.pro_api(tok)
    except Exception as e:
        print(f"  tushare 初始化失败: {e}")
        return results

    # 尝试 pro_bar 拿分钟线（asset='FD' = 基金）
    for freq in ["1min", "5min", "15min", "30min", "60min"]:
        try:
            df = ts.pro_bar(ts_code="510300.SH", freq=freq, asset="FD",
                            start_date="20240101", end_date="20261029")
            if df is None or df.empty:
                print(f"  pro_bar {freq:<8} [--] 空")
                results.append({"source": "tushare", "api": f"pro_bar_{freq}", "status": "empty"})
                continue
            tcol = "trade_time" if "trade_time" in df.columns else df.columns[1]
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
            ndays = df[tcol].dt.date.nunique()
            print(f"  pro_bar {freq:<8} [OK] {len(df):>6} 行  "
                  f"{df[tcol].min()} ~ {df[tcol].max()}  ({ndays} 个交易日)")
            results.append({"source": "tushare", "api": f"pro_bar_{freq}", "status": "ok",
                            "rows": len(df), "start": str(df[tcol].min()),
                            "end": str(df[tcol].max()), "trading_days": int(ndays)})
        except Exception as e:
            print(f"  pro_bar {freq:<8} [X] {type(e).__name__}: {str(e)[:70]}")
            results.append({"source": "tushare", "api": f"pro_bar_{freq}", "status": "error",
                            "error": f"{type(e).__name__}: {str(e)[:100]}"})
        time.sleep(2)
    return results


def main() -> None:
    print("=" * 96)
    print("探测 ETF 长周期日内数据源")
    print("=" * 96)

    all_results = []

    print("\n【1】akshare（免费）")
    print("-" * 96)
    all_results += test_akshare()

    print("\n【2】tushare（注意限速）")
    print("-" * 96)
    all_results += test_tushare()

    # ===== 汇总 =====
    print("\n" + "=" * 96)
    print("汇总：哪些源能拿到 510300 的分钟数据？")
    print("=" * 96)
    ok = [r for r in all_results if r.get("status") == "ok"]
    if not ok:
        print("  没有成功的源")
    else:
        best = max(ok, key=lambda r: r.get("trading_days", 0))
        print(f"  成功 {len(ok)} 个接口")
        print(f"  最大覆盖: {best['source']} / {best['api']}")
        print(f"    {best.get('start')} ~ {best.get('end')}  "
              f"({best.get('trading_days')} 个交易日 ≈ {best.get('trading_days',0)/244:.1f} 年)")

        for r in ok:
            print(f"  - {r['source']:<10} {r['api']:<32} {r.get('trading_days',0):>5} 天  "
                  f"{r.get('start','')[:10]} ~ {r.get('end','')[:10]}")

    print("\n" + "=" * 96)
    print("结论")
    print("=" * 96)
    print("  如果想做「分钟级日内策略」或「14:50 执行」，需要多少数据？")
    print("  · 最少 2-3 年（500-750 天）才能做有意义的回测")
    print("  · 理想的 5-10 年（1200-2400 天）")
    print()
    if ok:
        best_days = max(r.get("trading_days", 0) for r in ok)
        if best_days < 100:
            print(f"  >>> 目前最多只能拿到 {best_days} 天，远不够做日内策略回测")
        elif best_days < 500:
            print(f"  >>> 目前能拿到 {best_days} 天，勉强够初步验证，但样本偏少")
        else:
            print(f"  >>> 能拿到 {best_days} 天，足够做日内策略回测！")


if __name__ == "__main__":
    main()
