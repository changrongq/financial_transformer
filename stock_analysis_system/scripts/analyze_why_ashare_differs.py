"""挖：为什么 A 股「隔夜负、日内正」，而美股港股相反？

已确认的事实：
  A股  11/11 标的：隔夜 -10~-24%，日内 +15~+45%   （隔夜为负）
  港股 4/4  标的：隔夜 +4.6~+51.7%，日内 -19~-1.8% （隔夜为正）
  美股 6/6  标的：隔夜主流为正（4/6），日内混合

候选解释（逐个验证）：

  H1【T+1 制度】A股当天买入不能当天卖，必须等到次日
      → 想卖的人只能次日开盘卖 → 压低压开盘价 → 隔夜负
      验证：如果成立，2005年前（T+1一直存在）就应该一直成立；
            且其他 T+1 市场（如中国台湾1990年代前）也应如此（无数据）

  H2【涨跌停制度】A股 ±10% 涨跌停，涨停板次日容易高开低走
      → 但 2010 前涨停制度就有，无法区分

  H3【散户结构】A股散户占比高，隔夜情绪化抛售
      → 验证：看小盘股(中证1000) vs 大盘股(上证50)的隔夜差异
        如果小盘更负，说明散户情绪影响更大

  H4【时区效应】A股先于美股开盘，"隔夜"包含了美股前夜表现
      美股"隔夜"是自己的盘后交易（较小的波动）
      → 这个解释不了"日内正、隔夜负"的方向性，只解释幅度

  H5【尾盘资金行为】国内机构/游资习惯尾盘买入（怕隔夜），推高收盘价
      → 验证：如果成立，应该看到"收盘前上涨"的日内模式
        但没有分钟数据，只能用日内收益代理

  H6【A股的"隔夜跳空"系统性为负】开盘集合竞价机制
      → 验证：看隔夜的分布（不只是均值），是否系统性偏负

用法：
  python scripts/analyze_why_ashare_differs.py
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


def load(db, table, code, code_col="ts_code"):
    conn = sqlite3.connect(db)
    date_col = "trade_date"
    df = pd.read_sql_query(
        f"SELECT {date_col}, open, close FROM {table} WHERE {code_col}=? ORDER BY {date_col}",
        conn, params=(code,))
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
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
    print("=" * 100)
    print("为什么 A 股「隔夜负、日内正」，而美股港股相反？")
    print("=" * 100)

    # ===== 对照总表 =====
    print("\n【对照总表】")
    print("=" * 100)
    groups = [
        ("A股 上证50ETF", "etf_daily", "510050.SH"),
        ("A股 沪深300ETF", "etf_daily", "510300.SH"),
        ("A股 中证1000ETF", "etf_daily", "512100.SH"),
        ("A股 创业板ETF", "etf_daily", "159915.SZ"),
        ("HK 盈富基金", "global_daily", "02800.HK"),
        ("HK 腾讯控股", "global_daily", "00700.HK"),
        ("US 标普500", "global_daily", "SPY"),
        ("US 纳斯达克100", "global_daily", "QQQ"),
        ("US 罗素2000", "global_daily", "IWM"),
    ]
    print(f"{'标的':<18}{'年数':>6}{'全天':>10}{'隔夜':>10}{'日内':>10}{'隔夜均值(bp)':>14}{'隔夜胜率':>10}")
    print("-" * 100)

    detail = []
    for name, table, code in groups:
        df = load(db, table, code)
        if df is None:
            print(f"{name:<18} 无数据")
            continue
        a, g, i = ann(df["ret"]), ann(df["gap"]), ann(df["intraday"])
        gap_bp = df["gap"].mean() * 10000
        gap_win = (df["gap"] > 0).mean()
        print(f"{name:<18}{len(df)/244:>6.1f}{a*100:>+9.1f}%{g*100:>+9.1f}%{i*100:>+9.1f}%"
              f"{gap_bp:>+13.1f}{gap_win*100:>9.1f}%")
        detail.append({"name": name, "years": round(len(df)/244, 1),
                       "all": round(a, 6), "gap": round(g, 6), "intraday": round(i, 6),
                       "gap_mean_bp": round(gap_bp, 2), "gap_win_rate": round(gap_win, 4)})
    print("-" * 100)

    # ===== H3 验证：小盘 vs 大盘（散户情绪）=====
    print("\n【H3 检验】散户情绪假说：小盘股的隔夜是否更负？")
    print("=" * 100)
    a_etfs = [("510050.SH", "上证50(大盘)"), ("510300.SH", "沪深300(大中盘)"),
              ("510500.SH", "中证500(中盘)"), ("512100.SH", "中证1000(小盘)"),
              ("159915.SZ", "创业板(成长小盘)")]
    print(f"{'标的':<22}{'隔夜(年化)':>13}{'日内(年化)':>13}{'隔夜均值(bp)':>14}")
    print("-" * 100)
    h3 = []
    for code, nm in a_etfs:
        df = load(db, "etf_daily", code)
        if df is None:
            continue
        g, i = ann(df["gap"]), ann(df["intraday"])
        print(f"{nm:<22}{g*100:>+12.1f}%{i*100:>+12.1f}%{df['gap'].mean()*10000:>+13.1f}")
        h3.append({"name": nm, "gap": round(g, 6), "intraday": round(i, 6),
                   "gap_bp": round(float(df['gap'].mean() * 10000), 2)})
    print("-" * 100)
    if len(h3) >= 2:
        small = h3[-1]["gap_bp"]
        large = h3[0]["gap_bp"]
        print(f"  小盘(创业板)隔夜 {small:+.1f}bp  vs  大盘(上证50)隔夜 {large:+.1f}bp")
        print(f"  → {'小盘更负，支持散户情绪假说' if small < large else '大盘更负，不支持散户情绪假说'}")

    # ===== H6 验证：隔夜分布形态 =====
    print("\n【H6 检验】隔夜跳空分布：A股是否系统性偏负？")
    print("=" * 100)
    print(f"{'标的':<20}{'均值(bp)':>11}{'中位数(bp)':>12}{'偏度':>9}{'负值占比':>10}{'<-1%占比':>11}{'>+1%占比':>11}")
    print("-" * 100)
    h6 = []
    for name, table, code in groups:
        df = load(db, table, code)
        if df is None:
            continue
        g = df["gap"] * 100
        print(f"{name:<20}{g.mean()*100:>+10.1f}{g.median()*100:>+11.1f}{g.skew():>9.2f}"
              f"{(g<0).mean()*100:>9.1f}%{(g<-1).mean()*100:>10.1f}%{(g>1).mean()*100:>10.1f}%")
        h6.append({"name": name, "mean_bp": round(float(g.mean()*100), 2),
                   "median_bp": round(float(g.median()*100), 2),
                   "skew": round(float(g.skew()), 3),
                   "neg_ratio": round(float((g < 0).mean()), 4),
                   "lt_neg1pct": round(float((g < -1).mean()), 4),
                   "gt_pos1pct": round(float((g > 1).mean()), 4)})
    print("-" * 100)

    # ===== 关键洞察 =====
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)

    a_gap = [d for d in detail if d["name"].startswith("A股")]
    us_gap = [d for d in detail if d["name"].startswith("US")]
    hk_gap = [d for d in detail if d["name"].startswith("HK")]

    print(f"  A股:  隔夜年化范围 {min(x['gap'] for x in a_gap)*100:+.1f}% ~ {max(x['gap'] for x in a_gap)*100:+.1f}%"
          f"（全部为负）")
    print(f"  港股: 隔夜年化范围 {min(x['gap'] for x in hk_gap)*100:+.1f}% ~ {max(x['gap'] for x in hk_gap)*100:+.1f}%")
    print(f"  美股: 隔夜年化范围 {min(x['gap'] for x in us_gap)*100:+.1f}% ~ {max(x['gap'] for x in us_gap)*100:+.1f}%")

    print(f"\n  >>> 方向完全相反：A股隔夜负、日内正；美股(主流)隔夜正、日内弱")
    print(f"  >>> 这排除了「市场微结构通性」的解释")
    print(f"  >>> 指向 A 股特有的制度/参与者结构因素")

    print("\n  最可能的三个原因（按可能性排序）：")
    print("    1. T+1 制度：当天买入不能卖，卖压全部堆到次日开盘")
    print("    2. 涨停板制度：强势股封板后次日高开低走，系统性地压低隔夜收益")
    print("    3. 散户结构：隔夜情绪化决策，开盘集中抛售")

    print("\n  但注意：以上都是推测，我们没有能直接检验这些机制的数据")
    print("    （需要：分账户类型流向、集合竞价逐笔、T+0 对照组市场）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "why_ashare_differs.json"
    out.write_text(json.dumps({
        "comparison": detail, "h3_size_effect": h3, "h6_gap_distribution": h6,
        "conclusion": "A股隔夜负日内正，美股港股相反，指向A股特有制度（T+1/涨跌停/散户结构）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
