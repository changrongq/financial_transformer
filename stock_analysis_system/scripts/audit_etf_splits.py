"""排查所有 ETF 的份额拆分/合并事件（价格不连续点）。

背景：
  510500 中证500ETF 在 2015-04-15 出现 gap=+261% 的异常，
  原因是 ETF 份额拆分未做跨日复权。

影响：
  · 所有"隔夜收益"计算被污染
  · 严格 T+1 的"持仓吃全天收益"在拆分日暴赚
  · 导致 510500 出现 +13834% 的荒谬结果

本脚本：
  1. 扫描所有 ETF，找 |gap| > 20% 的异常点（大概率是拆分）
  2. 对比用"原始价格"vs"用pct_chg复权"算出的收益差异
  3. 给出修正方案

用法：
  python scripts/audit_etf_splits.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_engine as T

CODES = [
    ("510050.SH", "上证50ETF"),
    ("510300.SH", "沪深300ETF"),
    ("510500.SH", "中证500ETF"),
    ("512100.SH", "中证1000ETF"),
    ("159915.SZ", "创业板ETF"),
    ("588000.SH", "科创50ETF"),
]


def main() -> None:
    db = T.resolve_db()

    print("=" * 100)
    print("ETF 份额拆分/合并排查（|gap| > 20% 视为异常）")
    print("=" * 100)

    all_events = []
    for code, name in CODES:
        conn = sqlite3.connect(db)
        raw = pd.read_sql_query(
            "SELECT trade_date, open, high, low, close, pre_close, pct_chg, vol "
            "FROM etf_daily WHERE ts_code=? ORDER BY trade_date", conn, params=(code,))
        conn.close()
        if raw.empty:
            continue
        for c in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce")
        raw = raw.sort_values("date").reset_index(drop=True)

        raw["gap_raw"] = raw["open"] / raw["close"].shift(1) - 1.0
        # 用 pct_chg 推算的"真实"隔夜（pct_chg 是复权后的日收益）
        # 真实隔夜跳空 = open/pre_close - 1（pre_close 是复权后的昨收）
        raw["gap_true"] = raw["open"] / raw["pre_close"] - 1.0

        bad = raw[raw["gap_raw"].abs() > 0.20]
        print(f"\n{code} {name}  （{len(raw)} 天）")
        if bad.empty:
            print("  无异常")
        else:
            print(f"  [警告] 发现 {len(bad)} 个异常点:")
            print(f"  {'日期':<12}{'原始gap':>12}{'真实gap':>12}{'成交量变化':>14}")
            print("  " + "-" * 54)
            for _, r in bad.iterrows():
                idx = r.name
                prev_vol = raw.loc[idx - 1, "vol"] if idx > 0 else np.nan
                vol_ratio = r["vol"] / prev_vol if prev_vol and prev_vol > 0 else np.nan
                print(f"  {str(r['date'].date()):<12}{r['gap_raw']*100:>+11.1f}%"
                      f"{r['gap_true']*100:>+11.1f}%{vol_ratio:>13.1f}x")
                all_events.append({
                    "code": code, "name": name, "date": str(r["date"].date()),
                    "gap_raw": round(float(r["gap_raw"]), 6),
                    "gap_true": round(float(r["gap_true"]), 6),
                })

    # ========== 影响量化 ==========
    print("\n" + "=" * 100)
    print("影响量化：用原始价格 vs 用 pre_close 复权")
    print("=" * 100)
    print(f"{'标的':<16}{'原始价格收益':>16}{'复权后收益':>16}{'差异':>14}{'异常点数':>10}")
    print("-" * 100)

    impact = []
    for code, name in CODES:
        conn = sqlite3.connect(db)
        raw = pd.read_sql_query(
            "SELECT trade_date, open, close, pre_close, pct_chg FROM etf_daily "
            "WHERE ts_code=? ORDER BY trade_date", conn, params=(code,))
        conn.close()
        if raw.empty:
            continue
        for c in ["open", "close", "pre_close", "pct_chg"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.dropna(subset=["close", "pre_close"]).reset_index(drop=True)

        # 方法1：原始价格
        ret_price = raw["close"] / raw["pre_close"] - 1.0
        cum_price = float((1 + ret_price).prod() - 1)

        # 方法2：直接用 pct_chg
        ret_pct = raw["pct_chg"] / 100.0
        cum_pct = float((1 + ret_pct).prod() - 1)

        n_bad = int((raw["close"] / raw["pre_close"] - 1 - raw["pct_chg"]/100).abs().gt(0.001).sum())
        diff = cum_price - cum_pct
        print(f"{name:<16}{cum_price*100:>+15.1f}%{cum_pct*100:>+15.1f}%{diff*100:>+13.1f}pp{n_bad:>10}")
        impact.append({"code": code, "name": name,
                       "cum_by_price": round(cum_price, 6),
                       "cum_by_pct_chg": round(cum_pct, 6),
                       "diff": round(diff, 6)})

    print("-" * 100)
    print("  说明：两个收益应该完全一致（因为 pre_close 是复权后的昨收）")
    print("        如果差异大，说明 pre_close 字段本身有问题")

    # ========== 分隔检查：连续日期 ==========
    print("\n" + "=" * 100)
    print("补充检查：是否有跨日的价格断崖（拆分未标注）")
    print("=" * 100)
    print(f"{'标的':<16}{'close 断崖次数':>16}{'最大断崖':>14}{'最大断崖日期':>16}")
    print("-" * 100)

    for code, name in CODES:
        conn = sqlite3.connect(db)
        raw = pd.read_sql_query(
            "SELECT trade_date, close FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
            conn, params=(code,))
        conn.close()
        if raw.empty:
            continue
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw["date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce")
        raw = raw.dropna(subset=["close"]).reset_index(drop=True)
        raw["r"] = raw["close"].pct_change()
        cliff = raw[raw["r"].abs() > 0.20]
        max_cliff = raw["r"].abs().max()
        max_date = str(raw.loc[raw["r"].abs().idxmax(), "date"].date()) if raw["r"].notna().any() else "-"
        print(f"{name:<16}{len(cliff):>16}{max_cliff*100:>13.1f}%{max_date:>16}")

    # ========== 结论 ==========
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    if all_events:
        print(f"  发现 {len(all_events)} 个可疑的份额拆分/合并事件：")
        for e in all_events:
            print(f"    {e['name']} {e['date']}  原始gap {e['gap_raw']*100:+.1f}%  "
                  f"真实gap {e['gap_true']*100:+.1f}%")
        print()
        print("  >>> 必须修正：用 pre_close（复权后昨收）计算隔夜，")
        print("      而不是用前一天的 close。否则拆分日会产生巨额假收益。")
    else:
        print("  未发现异常")

    import json
    out = Path(__file__).resolve().parents[1] / "results" / "etf_split_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"events": all_events, "impact": impact},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
