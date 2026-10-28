"""寻找与"择时"正交的第二维度。

问题（豆包提出）：择时（时间维度）已占据主导，那么是否存在
真正与它**正交**的第二维度？（不是像波动率那样在下跌时自动失效）

正交性检验标准：
  1. 信号来源不同（非"大盘涨跌"的派生）
  2. 作用时机不同（在"择时持仓期内"起作用）
  3. 效果不重叠（不在下跌时自动失效）

候选维度（均在择时持仓期内应用）：
  A. 等权基准（择时 + 全行业等权）
  B. 行业动量（只配近期最强 N 个行业）
  C. 行业反转（只配近期最弱 N 个行业）
  D. 个股相对强弱（配强）
  E. 个股相对强弱（配弱）

数据：historical_daily（277只/8行业）+ index_daily_sh（择时信号）

用法：
  python scripts/find_orthogonal_dimension.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0005
REBALANCE_DAYS = 5
TOP_SECTORS = 3  # 行业动量/反转选前几个行业


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


def load_data(db: str):
    conn = sqlite3.connect(db)
    price = pd.read_sql_query("SELECT stock_code, date, close FROM historical_daily", conn)
    index = pd.read_sql_query("SELECT date, pct_chg FROM index_daily_sh", conn)
    bi = pd.read_sql_query("SELECT stock_code, 行业 FROM stock_basic_info", conn)
    conn.close()

    price["date"] = pd.to_datetime(price["date"])
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price["stock_code"] = price["stock_code"].astype(str).str.zfill(6)
    price = price.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    price["ret"] = price.groupby("stock_code")["close"].pct_change()

    index["date"] = pd.to_datetime(index["date"])
    index["ret"] = pd.to_numeric(index["pct_chg"], errors="coerce") / 100.0
    index = index.dropna(subset=["ret"]).reset_index(drop=True)
    index["regime10"] = index["ret"].rolling(10, min_periods=3).mean()

    bi["行业"] = bi["行业"].astype(str)
    sector_map = dict(zip(bi["stock_code"].astype(str).str.zfill(6), bi["行业"]))
    return price, index, sector_map


def run(price, index, sector_map, mode: str, label: str, top_n: int = TOP_SECTORS) -> dict:
    """mode: base / sector_mom / sector_rev / stock_mom / stock_rev"""
    dates = sorted(price["date"].unique())
    by_date = {d: g.set_index("stock_code")["ret"] for d, g in price.groupby("date")}
    pivot = price.pivot_table(index="date", columns="stock_code", values="ret", aggfunc="first").reindex(dates)
    regime = index.set_index("date")["regime10"].to_dict()
    idx_ret = index.set_index("date")["ret"].to_dict()

    # 个股 20 日动量
    mom20 = (pivot + 1.0).rolling(20, min_periods=10).apply(np.prod, raw=True) - 1.0

    holdings = []
    equity = 1.0
    daily_rets = []
    peak = 1.0
    cash_days = 0

    for i, d in enumerate(dates):
        if i < 25:
            continue

        rg = regime.get(d)
        in_market = (rg is None) or pd.isna(rg) or (rg >= 0.0)
        if not in_market:
            if holdings:
                equity *= (1.0 - COST)
                holdings = []
            daily_rets.append(0.0)
            cash_days += 1
            continue

        if i % REBALANCE_DAYS == 0 or not holdings:
            m_today = mom20.loc[d].dropna() if d in mom20.index else pd.Series(dtype=float)
            if len(m_today) < 10:
                daily_rets.append(0.0)
                continue

            if mode == "base":
                keep = m_today.index.tolist()

            elif mode in ("sector_mom", "sector_rev"):
                # 按行业聚合动量（行业等权平均）
                sec_series = pd.Series({c: sector_map.get(c, "NA") for c in m_today.index})
                sec_mom = m_today.groupby(sec_series).mean().dropna()
                if len(sec_mom) == 0:
                    daily_rets.append(0.0)
                    continue
                asc = mode == "sector_rev"  # rev 选最弱
                chosen = sec_mom.sort_values(ascending=asc).head(top_n).index.tolist()
                keep = [c for c in m_today.index if sector_map.get(c, "NA") in chosen]

            elif mode in ("stock_mom", "stock_rev"):
                asc = mode == "stock_rev"  # rev 选最弱
                n = max(int(len(m_today) * 0.3), 5)
                keep = m_today.sort_values(ascending=asc).head(n).index.tolist()

            else:
                keep = m_today.index.tolist()

            if holdings:
                changed = len(set(holdings) - set(keep)) / max(len(keep), 1)
                equity *= (1.0 - changed * COST)
            holdings = keep

        if not holdings:
            daily_rets.append(0.0)
            continue

        day_series = by_date.get(d)
        if day_series is None:
            continue
        rets = day_series.reindex(holdings).dropna()
        if rets.empty:
            continue
        port_ret = float(rets.mean())
        equity *= (1.0 + port_ret)
        peak = max(peak, equity)
        daily_rets.append(port_ret)

    rets = pd.Series(daily_rets)
    cum = np.cumprod(1.0 + rets.values) if len(rets) else np.array([1.0])
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min()) if len(cum) > 1 else 0.0
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0

    return {
        "label": label,
        "mode": mode,
        "total_return": float(equity - 1.0),
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "cash_days": cash_days,
        "calmar": float((equity - 1.0) / abs(mdd)) if mdd != 0 else None,
    }


def main() -> None:
    db = _resolve_db()
    price, index, sector_map = load_data(db)
    print(f"数据: {price['date'].min().date()} ~ {price['date'].max().date()}, "
          f"{price['stock_code'].nunique()} 只, {len(set(sector_map.values()))} 个行业")

    configs = [
        ("base", "择时 + 全行业等权(基准)"),
        ("sector_mom", f"择时 + 行业动量(前{TOP_SECTORS}强行业)"),
        ("sector_rev", f"择时 + 行业反转(前{TOP_SECTORS}弱行业)"),
        ("stock_mom", "择时 + 个股动量(前30%强)"),
        ("stock_rev", "择时 + 个股反转(前30%弱)"),
    ]

    results = [run(price, index, sector_map, m, lbl) for m, lbl in configs]

    print("\n" + "=" * 92)
    print("寻找正交维度：择时持仓期内的配置方式对比（2019-2026）")
    print("=" * 92)
    print(f"{'策略':<32}{'总收益':>12}{'最大回撤':>11}{'夏普':>8}{'Calmar':>9}")
    base = results[0]
    for r in results:
        tag = ""
        if r["mode"] != "base":
            diff = (r["total_return"] - base["total_return"]) * 100
            tag = f"  ({(diff):+.0f}pp)"
        calmar = f"{r['calmar']:.1f}" if r["calmar"] is not None else "N/A"
        print(f"{r['label']:<32}{r['total_return']*100:>11.1f}%{r['max_drawdown']*100:>10.1f}%{r['sharpe']:>8.2f}{calmar:>9}{tag}")

    # 逐年对比（关键）
    print("\n" + "=" * 92)
    print("逐年一致性检验（关键：是否逐年稳定）")
    print("=" * 92)
    years = sorted(price["date"].dt.year.unique())
    yearly = {m: {} for m, _ in configs}
    header = "年份    " + "".join(f"{m[:10]:>12}" for m, _ in configs)
    print(header)
    for y in years:
        p = price[price["date"].dt.year == y]
        if p["date"].nunique() < 60:
            continue
        row = {}
        for m, lbl in configs:
            r = run(p, index, sector_map, m, lbl)
            yearly[m][int(y)] = r["total_return"]
            row[m] = r["total_return"]
        print(f"{y:<8}" + "".join(f"{row[m]*100:>11.1f}%" for m, _ in configs))

    # 逐年胜率
    print("\n" + "=" * 92)
    print("相对基准的逐年胜率（越接近8/8越稳定）")
    print("=" * 92)
    yrs = sorted(yearly["base"].keys())
    for m, lbl in configs:
        if m == "base":
            continue
        wins = sum(1 for y in yrs if yearly[m][y] > yearly["base"][y])
        avg_diff = np.mean([yearly[m][y] - yearly["base"][y] for y in yrs]) * 100
        print(f"  {lbl:<34} 胜 {wins}/{len(yrs)} 年, 平均超额 {avg_diff:+.2f}pp")

    out = ROOT / "results" / "orthogonal_dimension.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "yearly": yearly,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
