"""验证 A股隔夜为负是"真规律"还是"数据/机制假象"。

三个关键检验：

  T1【ETF 折溢价污染】
     ETF 价格 = 净值 + 折溢价。开盘时折溢价波动大，
     可能制造虚假的"隔夜负"。
     检验：对比 ETF 本身 vs 对应指数的隔夜收益。
     如果指数也是隔夜负 → 真规律
     如果指数不显著 → ETF 折溢价造成的假象

  T2【2015年前后是否一致】
     T+1 制度 1995 年就有，涨跌停 1996 年就有。
     如果规律是制度造成的，应该全时段稳定。
     检验：分 5 个 5 年区间看。

  T3【隔夜收益分布的真实形态】
     均值负但胜率 50% 意味着"亏多赚少"。
     检验：分位数分布，看是"大量小跌+少数大涨"还是反过来。

用法：
  python scripts/verify_ashare_overnight_mechanism.py
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


def load(db, table, code):
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            f"SELECT trade_date, open, close FROM {table} WHERE ts_code=? ORDER BY trade_date",
            conn, params=(code,))
    except Exception:
        conn.close()
        return None
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)
    if len(df) < 200:
        return None
    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def ann(x):
    return float((1 + x).prod() ** (244 / len(x)) - 1) if len(x) > 50 else np.nan


def main() -> None:
    db = _resolve_db()

    # ===== T1: ETF vs 指数 =====
    print("=" * 100)
    print("【T1】ETF 折溢价污染检验：ETF vs 对应指数")
    print("=" * 100)
    print(f"{'对比组':<34}{'标的':<12}{'年数':>6}{'隔夜':>11}{'日内':>11}{'隔夜均值bp':>13}")
    print("-" * 100)

    pairs = [
        ("沪深300", "510300.SH", "etf_daily", "000300.SH", "index_daily"),
        ("上证50", "510050.SH", "etf_daily", "000016.SH", "index_daily"),
        ("中证500", "510500.SH", "etf_daily", "000905.SH", "index_daily"),
        ("中证1000", "512100.SH", "etf_daily", "000852.SH", "index_daily"),
    ]
    t1 = []
    for label, etf_code, etf_tbl, idx_code, idx_tbl in pairs:
        for tag, tbl, code in [("ETF", etf_tbl, etf_code), ("指数", idx_tbl, idx_code)]:
            df = load(db, tbl, code)
            if df is None:
                print(f"{label:<34}{tag:<12}无数据（需先拉 {code}）")
                continue
            g, i = ann(df["gap"]), ann(df["intraday"])
            gb = df["gap"].mean() * 10000
            print(f"{label:<34}{tag:<12}{len(df)/244:>6.1f}{g*100:>+10.1f}%{i*100:>+10.1f}%{gb:>+12.1f}")
            t1.append({"pair": label, "type": tag, "code": code,
                       "years": round(len(df)/244, 1), "gap": round(g, 6),
                       "intraday": round(i, 6), "gap_bp": round(float(gb), 2)})
        print("-" * 100)

    # ===== T2: 分时段稳定性 =====
    print("\n" + "=" * 100)
    print("【T2】分时段稳定性（沪深300指数，20.8年）")
    print("=" * 100)
    df = load(db, "index_daily", "000300.SH")
    if df is not None:
        df["year"] = df["date"].dt.year
        print(f"{'时段':<16}{'天数':>8}{'隔夜(年化)':>14}{'日内(年化)':>14}{'隔夜均值bp':>13}{'隔夜胜率':>10}")
        print("-" * 100)
        t2 = []
        spans = [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]
        for a, b in spans:
            sub = df[(df["year"] >= a) & (df["year"] <= b)]
            if len(sub) < 100:
                continue
            g, i = ann(sub["gap"]), ann(sub["intraday"])
            gb = sub["gap"].mean() * 10000
            wr = (sub["gap"] > 0).mean()
            print(f"{a}-{b:<11}{len(sub):>8}{g*100:>+13.1f}%{i*100:>+13.1f}%{gb:>+12.1f}{wr*100:>9.1f}%")
            t2.append({"span": f"{a}-{b}", "days": len(sub), "gap": round(g, 6),
                       "intraday": round(i, 6), "gap_bp": round(float(gb), 2),
                       "gap_win": round(float(wr), 4)})
        print("-" * 100)

    # ===== T3: 隔夜分布形态 =====
    print("\n" + "=" * 100)
    print("【T3】隔夜收益分布（分位数，判断是'亏多赚少'还是'少数大跌'）")
    print("=" * 100)
    print(f"{'标的':<16}{'P1':>8}{'P10':>8}{'P25':>8}{'P50':>8}{'P75':>8}{'P90':>8}{'P99':>9}{'均值':>9}")
    print("-" * 100)
    t3 = []
    for name, tbl, code in [("A股 沪深300", "index_daily", "000300.SH"),
                            ("A股 创业板", "index_daily", "399006.SZ"),
                            ("A股 中证1000", "index_daily", "000852.SH"),
                            ("HK 恒指ETF", "global_daily", "02800.HK"),
                            ("US 标普500", "global_daily", "SPY")]:
        df = load(db, tbl, code)
        if df is None:
            continue
        g = df["gap"] * 100
        qs = [g.quantile(q) * 100 for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]]
        print(f"{name:<16}" + "".join(f"{v:>+8.1f}" for v in qs) + f"{g.mean()*100:>+9.1f}")
        t3.append({"name": name, "p1_bp": round(float(qs[0]*100), 1),
                   "p50_bp": round(float(qs[3]*100), 1), "p99_bp": round(float(qs[6]*100), 1),
                   "mean_bp": round(float(g.mean()*100), 1)})
    print("-" * 100)
    print("  单位：bp（1bp = 0.01%）。均值 = 各行分位数的加权结果。")

    # ===== 综合判断 =====
    print("\n" + "=" * 100)
    print("【综合判断】")
    print("=" * 100)

    # ETF vs 指数隔夜差异
    etf_gaps = [x for x in t1 if x["type"] == "ETF"]
    idx_gaps = [x for x in t1 if x["type"] == "指数"]
    if etf_gaps and idx_gaps:
        etf_avg = np.mean([x["gap"] for x in etf_gaps])
        idx_avg = np.mean([x["gap"] for x in idx_gaps])
        print(f"  ETF 平均隔夜年化:   {etf_avg*100:+.1f}%")
        print(f"  指数平均隔夜年化:   {idx_avg*100:+.1f}%")
        diff = (idx_avg - etf_avg) * 100
        print(f"  差异:               {diff:+.1f}pp")
        if abs(diff) < 5:
            print(f"  → 差异小，说明「隔夜负」在指数上同样成立，不是ETF折溢价造成的")
        else:
            print(f"  → 差异较大，ETF 折溢价可能放大/缩小了隔夜效应")

    print()
    print("  T2 分时段：如果各时段隔夜都是负的 → 制度因素（稳定）")
    print("  T3 分布：如果 P50 接近 0 但均值明显为负 → 少数大跌造成")
    print("           如果 P50 也为负 → 系统性偏负（制度性）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ashare_overnight_mechanism.json"
    out.write_text(json.dumps({
        "t1_etf_vs_index": t1, "t2_time_spans": t2, "t3_distribution": t3,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
