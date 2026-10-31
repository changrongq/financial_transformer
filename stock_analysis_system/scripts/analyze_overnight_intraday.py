"""挖：隔夜/日内收益结构是不是 A 股通性？

背景（重要发现）：
  沪深300ETF(510300) 14年年化：
    隔夜（昨收->今开）：-10.76%
    日内（今开->今收）：+15.92%

疑问：
  这是 A 股通性，还是 510300 的偶然？

检验设计：
  1. 6 个不同风格标的的隔夜/日内结构对比
     - 510050 上证50ETF（大盘蓝筹，20.5年，含2008股灾）
     - 510300 沪深300ETF（大盘，14.4年）
     - 510500 中证500ETF（中盘，13.5年）
     - 159915 创业板ETF（成长，14.8年）
     - 512100 中证1000ETF（小盘，9.9年）
     - 588000 科创50ETF（科技，5.9年）
  2. 用同样 6 个标的的指数日线复核（排除 ETF 特有的问题）
     - 000300 沪深300、000905 中证500、000001 上证综指、399006 创业板指
  3. 分年度稳定性：隔夜为负是否每年都成立？
     —— 如果只是个别年份造成，就不是规律

用法：
  python scripts/analyze_overnight_intraday.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

ETF_LIST = [
    ("510050.SH", "上证50ETF(大盘/20.5年)"),
    ("510300.SH", "沪深300ETF(大盘/14.4年)"),
    ("510500.SH", "中证500ETF(中盘/13.5年)"),
    ("159915.SZ", "创业板ETF(成长/14.8年)"),
    ("512100.SH", "中证1000ETF(小盘/9.9年)"),
    ("588000.SH", "科创50ETF(科技/5.9年)"),
]

IDX_LIST = [
    ("000001.SH", "上证综指(20.8年)"),
    ("000300.SH", "沪深300(20.8年)"),
    ("000905.SH", "中证500(20.8年)"),
    ("000852.SH", "中证1000(20.8年)"),
    ("399006.SZ", "创业板指(16.3年)"),
]


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


def load_prices(db: str, table: str, code: str) -> pd.DataFrame | None:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            f"SELECT trade_date, open, high, low, close, pre_close FROM {table} WHERE ts_code=? ORDER BY trade_date",
            conn, params=(code,))
    except Exception:
        conn.close()
        return None
    conn.close()
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close", "pre_close"]:
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
    if len(x) < 20:
        return np.nan
    return float((1 + x).prod() ** (244 / len(x)) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="overnight_intraday_analysis.json")
    args = parser.parse_args()

    db = _resolve_db()
    print("=" * 100)
    print("【核心问题】隔夜亏钱、日内赚钱 —— 是 A 股通性还是沪深300的偶然？")
    print("=" * 100)

    # ===== 1. ETF 对比 =====
    print("\n" + "=" * 100)
    print("【1】6 个 ETF 的隔夜/日内结构（年化）")
    print("=" * 100)
    print(f"{'标的':<28}{'年数':>7}{'全天':>10}{'隔夜':>11}{'日内':>11}{'隔夜占比':>11}{'日内/隔夜':>12}")
    print("-" * 100)

    etf_rows = []
    for code, name in ETF_LIST:
        df = load_prices(db, "etf_daily", code)
        if df is None:
            print(f"{name:<28} 无数据")
            continue
        a_all, a_gap, a_intra = ann(df["ret"]), ann(df["gap"]), ann(df["intraday"])
        total_abs = abs(a_gap) + abs(a_intra)
        gap_share = a_gap / total_abs if total_abs > 0 else np.nan
        ratio = a_intra / a_gap if a_gap != 0 else np.nan
        years = len(df) / 244
        print(f"{name:<28}{years:>7.1f}{a_all*100:>+9.2f}%{a_gap*100:>+10.2f}%"
              f"{a_intra*100:>+10.2f}%{gap_share*100:>10.1f}%{ratio:>12.2f}")
        etf_rows.append({
            "code": code, "name": name, "years": round(years, 2),
            "all": round(a_all, 6), "gap": round(a_gap, 6), "intraday": round(a_intra, 6),
            "days": len(df),
        })

    # ===== 2. 指数复核 =====
    print("\n" + "=" * 100)
    print("【2】用指数日线复核（排除 ETF 特有因素）")
    print("=" * 100)
    print(f"{'指数':<28}{'年数':>7}{'全天':>10}{'隔夜':>11}{'日内':>11}{'隔夜占比':>11}{'日内/隔夜':>12}")
    print("-" * 100)

    idx_rows = []
    for code, name in IDX_LIST:
        df = load_prices(db, "index_daily", code)
        if df is None:
            print(f"{name:<28} 无数据")
            continue
        a_all, a_gap, a_intra = ann(df["ret"]), ann(df["gap"]), ann(df["intraday"])
        total_abs = abs(a_gap) + abs(a_intra)
        gap_share = a_gap / total_abs if total_abs > 0 else np.nan
        ratio = a_intra / a_gap if a_gap != 0 else np.nan
        years = len(df) / 244
        print(f"{name:<28}{years:>7.1f}{a_all*100:>+9.2f}%{a_gap*100:>+10.2f}%"
              f"{a_intra*100:>+10.2f}%{gap_share*100:>10.1f}%{ratio:>12.2f}")
        idx_rows.append({
            "code": code, "name": name, "years": round(years, 2),
            "all": round(a_all, 6), "gap": round(a_gap, 6), "intraday": round(a_intra, 6),
            "days": len(df),
        })

    # ===== 3. 结论判断 =====
    print("\n" + "=" * 100)
    print("【3】判断：是否所有标的都是「隔夜负 / 日内正」？")
    print("=" * 100)
    all_rows = etf_rows + idx_rows
    n = len(all_rows)
    n_gap_neg = sum(1 for r in all_rows if r["gap"] < 0)
    n_intra_pos = sum(1 for r in all_rows if r["intraday"] > 0)
    print(f"  隔夜为负的标的数: {n_gap_neg}/{n}")
    print(f"  日内为正的标的数: {n_intra_pos}/{n}")

    if n_gap_neg == n and n_intra_pos == n:
        verdict = "强规律：所有标的都是隔夜负、日内正"
    elif n_gap_neg >= n * 0.7 and n_intra_pos >= n * 0.7:
        verdict = "中等规律：大多数标的符合，但有例外"
    else:
        verdict = "不成立：隔夜/日内结构因标的而异"

    print(f"\n  >>> {verdict}")

    # ===== 4. 分年度稳定性（关键检验）=====
    print("\n" + "=" * 100)
    print("【4】分年度稳定性：隔夜为负是否每年都成立？（关键）")
    print("=" * 100)

    yearly_data = {}
    for code, name in [("510300.SH", "沪深300ETF"), ("510050.SH", "上证50ETF"),
                       ("000300.SH", "沪深300指数"), ("000905.SH", "中证500指数")]:
        table = "etf_daily" if code.startswith(("51", "15", "58")) else "index_daily"
        df = load_prices(db, table, code)
        if df is None:
            continue
        df["year"] = df["date"].dt.year
        yearly_data[name] = df

    if "沪深300指数" in yearly_data:
        df = yearly_data["沪深300指数"]
        print(f"\n  沪深300指数（20.8年，最优代表性）：")
        print(f"  {'年份':<8}{'全天':>10}{'隔夜':>11}{'日内':>11}{'隔夜为负':>10}")
        print("  " + "-" * 55)
        neg_years = 0
        tot_years = 0
        yearly_rows = []
        for y in sorted(df["year"].unique()):
            sub = df[df["year"] == y]
            if len(sub) < 100:
                continue
            ag, ai, aa = ann(sub["gap"]), ann(sub["intraday"]), ann(sub["ret"])
            flag = "YES" if ag < 0 else "NO "
            neg_years += int(ag < 0)
            tot_years += 1
            print(f"  {y:<8}{aa*100:>+9.1f}%{ag*100:>+10.1f}%{ai*100:>+10.1f}%{flag:>10}")
            yearly_rows.append({"year": int(y), "all": round(aa, 6),
                                "gap": round(ag, 6), "intraday": round(ai, 6)})
        print("  " + "-" * 55)
        print(f"  隔夜为负的年份: {neg_years}/{tot_years} = {neg_years/tot_years*100:.0f}%")

    # ===== 5. 策略含义 =====
    print("\n" + "=" * 100)
    print("【5】这对策略意味着什么")
    print("=" * 100)
    if etf_rows:
        best = max(etf_rows, key=lambda r: r["intraday"] - r["gap"])
        print(f"  隔夜与日内差距最大的标的: {best['name']}")
        print(f"    隔夜 {best['gap']*100:+.2f}%  日内 {best['intraday']*100:+.2f}%")
        print(f"\n  如果规律成立，'只在日内持仓'（每日开盘买、收盘卖）的理论年化 ≈ 各标的的日内年化")
        print(f"  注意：这没考虑每日交易的超高成本，仅作结构分析")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / args.out
    out.write_text(json.dumps({
        "etf": etf_rows, "index": idx_rows,
        "n_gap_negative": n_gap_neg, "n_total": n,
        "n_intraday_positive": n_intra_pos,
        "verdict": verdict,
        "yearly_hs300_index": yearly_rows if "沪深300指数" in yearly_data else [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
