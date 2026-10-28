"""验证豆包理论推出的两个假设。

背景：
  豆包说"大盘股、行业龙头、有基本面支撑的标的，动量的持续性会更强"，
  以及"小票 + 短期 = 反转"。

  如果这两句对，那么在【市值分层】后应该看到：
    - 大盘股组：动量有效（正超额）
    - 小盘股组：动量无效或反转（负超额）

  这正好能解释为什么 277 只行业股（偏中大盘）测出动量、
  而 5700 只全市场（含大量微盘）测出反转。

测法：
  按调仓日的市值把股票分成 N 组，每组内部分别做动量（买最强的 top_pct），
  比较各组的"未来5日超额收益"。

用法：
  python scripts/validate_momentum_by_mktcap.py
  python scripts/validate_momentum_by_mktcap.py --groups 5 --top-pct 0.10
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


def _resolve_trade_db() -> str:
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


def _resolve_cap_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("market_cap_db") or cfg.get("databases", {}).get("cap_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "market_cap_full.db").resolve())


def load_data():
    conn = sqlite3.connect(_resolve_trade_db())
    price = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [收盘] AS close, "
        "[成交额] AS amount, [成交量] AS volume FROM stock_data", conn
    )
    bi = pd.read_sql_query("SELECT stock_code, 名称, 行业 FROM stock_basic_info", conn)
    conn.close()

    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["stock_code"] = price["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    for c in ["close", "amount", "volume"]:
        price[c] = pd.to_numeric(price[c], errors="coerce")
    price = price.dropna(subset=["close"])
    price = price[price["stock_code"].str.len() == 6]
    price = price.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # 成交额兜底（已知 18 个月空洞）
    miss = int(price["amount"].isna().sum())
    if miss:
        price["amount"] = price["amount"].fillna(price["close"] * price["volume"])
        print(f"  [数据修正] 成交额缺失 {miss:,} 行，用 收盘价×成交量 估算")
    price["amount"] = price["amount"].fillna(0.0)

    bi["stock_code"] = bi["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    return price, bi


def load_caps():
    """加载市值数据。返回 DataFrame[stock_code, date, total_mv]（万元）。"""
    db = _resolve_cap_db()
    conn = sqlite3.connect(db)
    cols = pd.read_sql_query("PRAGMA table_info(market_caps)", conn)
    names = list(cols["name"])
    print(f"  market_caps 列: {names}")

    # 猜测列名
    code_col = next((c for c in names if c in ("stock_code", "code", "ts_code")), names[0])
    date_col = next((c for c in names if c in ("date", "trade_date", "日期")), names[1])
    mv_col = next(
        (c for c in names if "total_mv" in c.lower() or "市值" in c or "circ_mv" in c.lower()),
        None,
    )
    if mv_col is None:
        raise SystemExit(f"找不到市值列，现有列: {names}")

    print(f"  使用列: code={code_col}, date={date_col}, mv={mv_col}")
    df = pd.read_sql_query(f"SELECT {code_col} AS stock_code, {date_col} AS date, {mv_col} AS mv FROM market_caps", conn)
    conn.close()

    df["stock_code"] = df["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["mv"] = pd.to_numeric(df["mv"], errors="coerce")
    df = df.dropna(subset=["date", "mv"])
    if df["date"].max().year < 2020:
        # 可能是 YYYYMMDD 整数
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="按市值分层验证动量")
    parser.add_argument("--mom-window", type=int, default=20)
    parser.add_argument("--top-pct", type=float, default=0.10, help="组内选动量最强的比例")
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5, help="按市值分几组")
    parser.add_argument("--min-amount", type=float, default=10000, help="最低成交额（万元）")
    args = parser.parse_args()

    print("加载价格数据...")
    price, bi = load_data()
    print(f"  范围 {price['date'].min().date()} ~ {price['date'].max().date()}, 股票 {price['stock_code'].nunique()}")

    print("加载市值数据...")
    try:
        caps = load_caps()
        print(f"  范围 {caps['date'].min().date()} ~ {caps['date'].max().date()}, 记录 {len(caps):,}")
    except Exception as e:
        raise SystemExit(f"市值数据加载失败: {e}")

    pivot = price.pivot_table(index="date", columns="stock_code", values="close", aggfunc="first").sort_index()
    amt = price.pivot_table(index="date", columns="stock_code", values="amount", aggfunc="first").reindex(index=pivot.index)
    mom = pivot / pivot.shift(args.mom_window) - 1.0

    st = set(bi.loc[bi["名称"].astype(str).str.contains("ST|退", na=False), "stock_code"])
    valid = [c for c in pivot.columns if c not in st]
    dates = pivot.index

    # 市值按日期 pivot（使用最近可得市值向前填充）
    cap_px = caps.pivot_table(index="date", columns="stock_code", values="mv", aggfunc="last")
    cap_px = cap_px.reindex(index=dates.union(cap_px.index)).sort_index().ffill().reindex(index=dates)

    rows = []
    for i in range(args.mom_window + 1, len(dates) - args.hold_days):
        d = dates[i]
        d2 = dates[i + args.hold_days]
        m = mom.loc[d, valid].dropna()
        if len(m) < args.groups * 20:
            continue
        a = amt.loc[d].reindex(m.index)
        m = m[a >= args.min_amount]
        if len(m) < args.groups * 20:
            continue

        cap_today = cap_px.loc[d].reindex(m.index)
        ok = cap_today.notna()
        m, cap_today = m[ok], cap_today[ok]
        if len(m) < args.groups * 20:
            continue

        fwd = (pivot.loc[d2, m.index] / pivot.loc[d, m.index] - 1.0)
        mkt_fwd = (pivot.loc[d2, valid] / pivot.loc[d, valid] - 1.0).mean()

        # 按市值分组（qcut），group 0 = 小市值
        try:
            grp = pd.qcut(cap_today, args.groups, labels=False, duplicates="drop")
        except Exception:
            continue

        n_top = max(int(len(m) * args.top_pct), 3)
        for g in range(args.groups):
            idx = m.index[grp == g]
            if len(idx) < 10:
                continue
            # 组内动量最强的 n_top 只
            n_g = max(int(len(idx) * args.top_pct), 3)
            sel = m[idx].sort_values(ascending=False).head(n_g).index
            f = fwd.reindex(sel).dropna()
            if len(f) == 0:
                continue
            rows.append({
                "date": d,
                "group": g,
                "cap_median": float(cap_today[idx].median()),
                "mom_avg": float(m[sel].mean()),
                "fwd": float(f.mean()),
                "mkt_fwd": float(mkt_fwd),
                "excess": float(f.mean() - mkt_fwd),
                "n": len(sel),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("无有效数据，请检查市值数据覆盖范围")

    print("\n" + "=" * 88)
    print(f"按市值分 {args.groups} 组，组内买动量最强的前 {args.top_pct*100:.0f}%，持有 {args.hold_days} 日")
    print("=" * 88)
    print(f"{'组':<10}{'市值中位数(亿)':>16}{'平均动量':>12}{'未来5日':>11}{'超额':>10}{'跑赢率':>10}{'样本':>8}")

    summary = {}
    for g in range(args.groups):
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        win = float((sub["excess"] > 0).mean())
        label = "小市值" if g == 0 else ("大市值" if g == args.groups - 1 else f"第{g+1}组")
        print(
            f"{label:<10}{sub['cap_median'].median()/1e4:>15.1f} "
            f"{sub['mom_avg'].mean()*100:>+10.1f}%{sub['fwd'].mean()*100:>+10.2f}%"
            f"{sub['excess'].mean()*100:>+9.2f}%{win*100:>9.1f}%{len(sub):>8}"
        )
        summary[f"group{g}_{label}"] = {
            "cap_median_yi": round(float(sub["cap_median"].median()) / 1e4, 2),
            "mom_avg": round(float(sub["mom_avg"].mean()), 4),
            "fwd_avg": round(float(sub["fwd"].mean()), 6),
            "excess": round(float(sub["excess"].mean()), 6),
            "win_rate": round(win, 4),
            "n": int(len(sub)),
        }

    print("=" * 88)
    g0 = summary.get("group0_小市值", {}).get("excess")
    gN = summary.get(f"group{args.groups-1}_大市值", {}).get("excess")
    print("\n豆包的假设检验：")
    if gN is not None:
        verdict_N = "成立" if gN > 0 else "不成立"
        print(f"  「大盘股动量持续性强」→ 大市值组超额 {gN*100:+.2f}% → {verdict_N}")
    if g0 is not None:
        verdict_0 = "成立" if g0 < 0 else "不成立"
        print(f"  「小盘股易反转」     → 小市值组超额 {g0*100:+.2f}% → {verdict_0}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "momentum_by_mktcap.json"
    out.write_text(
        json.dumps({"params": vars(args), "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
