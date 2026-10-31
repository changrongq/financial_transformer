"""验证「隔夜负、日内正」是否全球普适。

背景：
  A 股 11/11 标的成立（沪深300指数 19/21 年隔夜为负）。
  如果美股/港股也成立 → 全球性现象（可能是交易机制的普遍结果）
  如果美股/港股不成立 → A 股特有（T+1 制度、散户结构、涨跌停等）

用法：
  python scripts/analyze_global_overnight.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _resolve_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def load_global(db: str, code: str) -> pd.DataFrame | None:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, close, pre_close FROM global_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "close", "pre_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 200:
        return None
    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["pre_close"] - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def ann(x: pd.Series) -> float:
    if len(x) < 50:
        return np.nan
    return float((1 + x).prod() ** (244 / len(x)) - 1)


NAMES = {
    "SPY": "标普500ETF (US)",
    "DIA": "道指ETF (US)",
    "QQQ": "纳斯达克100ETF (US)",
    "IWM": "罗素2000ETF (US小盘)",
    "EEM": "新兴市场ETF (US)",
    "EFA": "发达国家ETF (US)",
    "02800.HK": "盈富基金 (HK恒指)",
    "00700.HK": "腾讯控股 (HK)",
    "00005.HK": "汇丰控股 (HK)",
}


def main() -> None:
    db = _resolve_db()
    conn = sqlite3.connect(db)
    try:
        codes = pd.read_sql_query(
            "SELECT DISTINCT ts_code, market FROM global_daily", conn)
    except Exception:
        codes = pd.DataFrame(columns=["ts_code", "market"])
    conn.close()

    if codes.empty:
        raise SystemExit("global_daily 表为空，先跑 fetch_global_daily.py")

    print("=" * 96)
    print("全球市场「隔夜 / 日内」结构检验")
    print("=" * 96)
    print(f"{'标的':<26}{'年数':>7}{'全天':>11}{'隔夜':>11}{'日内':>11}{'隔夜/日内':>12}")
    print("-" * 96)

    rows = []
    for _, r in codes.iterrows():
        code = r["ts_code"]
        df = load_global(db, code)
        if df is None:
            print(f"{NAMES.get(code, code):<26} 数据不足")
            continue
        a_all, a_gap, a_intra = ann(df["ret"]), ann(df["gap"]), ann(df["intraday"])
        years = len(df) / 244
        ratio = a_intra / a_gap if a_gap not in (0, np.nan) and a_gap != 0 else np.nan
        print(f"{NAMES.get(code, code):<26}{years:>7.1f}{a_all*100:>+10.2f}%"
              f"{a_gap*100:>+10.2f}%{a_intra*100:>+10.2f}%{ratio:>12.2f}")
        rows.append({
            "code": code, "name": NAMES.get(code, code),
            "market": r["market"], "years": round(years, 1),
            "all": round(a_all, 6), "gap": round(a_gap, 6),
            "intraday": round(a_intra, 6), "days": len(df),
        })

    if not rows:
        raise SystemExit("无有效数据")

    print("-" * 96)

    # 分市场汇总
    print("\n" + "=" * 96)
    print("分市场汇总")
    print("=" * 96)
    for mkt in sorted({r["market"] for r in rows}):
        sub = [r for r in rows if r["market"] == mkt]
        n_neg = sum(1 for r in sub if r["gap"] < 0)
        n_pos = sum(1 for r in sub if r["intraday"] > 0)
        print(f"  {mkt}: {len(sub)} 个标的 | 隔夜为负 {n_neg}/{len(sub)} | 日内为正 {n_pos}/{len(sub)}")

    # A 股对照
    print("\n" + "=" * 96)
    print("与 A 股对照")
    print("=" * 96)
    print("  A股（已测11个标的）: 隔夜为负 11/11，日内为正 11/11")
    n_neg_all = sum(1 for r in rows if r["gap"] < 0)
    print(f"  全球（本次测试）:     隔夜为负 {n_neg_all}/{len(rows)}，"
          f"日内为正 {sum(1 for r in rows if r['intraday'] > 0)}/{len(rows)}")

    # 结论
    print()
    if n_neg_all == len(rows):
        verdict = ("全球普适：美股/港股也是隔夜负、日内正 → "
                   "这是市场微结构（开盘定价效率、隔夜风险溢价）的普遍结果，不是A股特有")
    elif n_neg_all >= len(rows) * 0.6:
        verdict = ("多数成立但有例外 → 可能是普遍倾向，但受各市场制度影响")
    else:
        verdict = ("A 股特有：美股/港股不符合 → 说明与 T+1制度/涨跌停/散户结构强相关")

    print(f"  >>> {verdict}")

    # 分年度（每个标的隔夜为负的年份占比）
    print("\n" + "=" * 96)
    print("分年度稳定性（各标的隔夜为负的年份占比）")
    print("=" * 96)
    print(f"{'标的':<26}{'隔夜负年份':>14}{'总年份':>10}{'占比':>10}")
    print("-" * 96)
    yearly_summary = []
    for _, r in codes.iterrows():
        code = r["ts_code"]
        df = load_global(db, code)
        if df is None:
            continue
        df["year"] = df["date"].dt.year
        neg = tot = 0
        for y, sub in df.groupby("year"):
            if len(sub) < 100:
                continue
            tot += 1
            if ann(sub["gap"]) < 0:
                neg += 1
        if tot:
            print(f"{NAMES.get(code, code):<26}{neg:>14}{tot:>10}{neg/tot*100:>9.0f}%")
            yearly_summary.append({"code": code, "neg_years": neg, "total_years": tot,
                                   "ratio": round(neg / tot, 4)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "global_overnight_intraday.json"
    out.write_text(json.dumps({
        "assets": rows, "verdict": verdict,
        "n_gap_negative": n_neg_all, "n_total": len(rows),
        "yearly": yearly_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
