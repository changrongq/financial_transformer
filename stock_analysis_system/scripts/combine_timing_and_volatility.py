"""择时 + 波动筛 叠加实验。

两个已验证因子首次合体：
  - 择时（时间维度）：大盘10日均收益 < 0 → 空仓
  - 波动筛（截面维度）：只买波动率低的股票（剔除最高的30%）

四种组合对比：
  1. BASE     全市场等权，一直满仓
  2. VOL      全市场等权，但只买低波动股（剔30%）
  3. TIME     择时（满仓/空仓），不筛股
  4. BOTH     择时 + 波动筛（空仓时清仓，持仓时只买低波动股）

数据：historical_daily（277只，2019-2026）+ index_daily_sh（择时信号）

用法：
  python scripts/combine_timing_and_volatility.py
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
DROP_TOP_PCT = 0.30


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
    conn.close()

    price["date"] = pd.to_datetime(price["date"])
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    price["ret"] = price.groupby("stock_code")["close"].pct_change()

    index["date"] = pd.to_datetime(index["date"])
    index["ret"] = pd.to_numeric(index["pct_chg"], errors="coerce") / 100.0
    index = index.dropna(subset=["ret"]).reset_index(drop=True)
    index["regime10"] = index["ret"].rolling(10, min_periods=3).mean()
    return price, index


def run(
    price: pd.DataFrame,
    index: pd.DataFrame,
    use_timing: bool,
    use_vol_filter: bool,
    drop_pct: float,
    label: str,
) -> dict:
    """通用回测。"""
    dates = sorted(price["date"].unique())
    by_date = {d: g.set_index("stock_code")["ret"] for d, g in price.groupby("date")}
    pivot = price.pivot_table(index="date", columns="stock_code", values="ret", aggfunc="first").reindex(dates)
    vol20 = pivot.rolling(20, min_periods=10).std()
    regime = index.set_index("date")["regime10"].to_dict()
    idx_ret = index.set_index("date")["ret"].to_dict()

    holdings = []
    equity = 1.0
    peak = 1.0
    daily_rets = []
    crash_rets = []
    cash_days = 0

    for i, d in enumerate(dates):
        if i < 25:
            continue

        # 择时判断（当日是否空仓）
        rg = regime.get(d)
        in_market = True
        if use_timing and rg is not None and pd.notna(rg):
            in_market = rg >= 0.0

        if not in_market:
            # 空仓：清掉所有持仓
            if holdings:
                equity *= (1.0 - COST)
                holdings = []
            daily_rets.append(0.0)
            cash_days += 1
            continue

        # 调仓
        if i % REBALANCE_DAYS == 0 or not holdings:
            vol_today = vol20.loc[d].dropna() if d in vol20.index else pd.Series(dtype=float)
            if len(vol_today) >= 10:
                if use_vol_filter and drop_pct > 0:
                    thr = vol_today.quantile(1.0 - drop_pct)
                    keep = vol_today[vol_today <= thr].index.tolist()
                else:
                    keep = vol_today.index.tolist()
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

        mkt = idx_ret.get(d, np.nan)
        if pd.notna(mkt) and mkt < -0.02:
            crash_rets.append(port_ret)

    rets = pd.Series(daily_rets)
    cum = np.cumprod(1.0 + rets.values) if len(rets) else np.array([1.0])
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min()) if len(cum) > 1 else 0.0
    # 夏普（年化，无风险=0）
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0

    return {
        "label": label,
        "total_return": float(equity - 1.0),
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "volatility": float(rets.std()) if len(rets) else 0.0,
        "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
        "cash_days": cash_days,
        "crash_days": len(crash_rets),
        "crash_avg_ret": float(np.mean(crash_rets)) if crash_rets else None,
        "calmar": float((equity - 1.0) / abs(mdd)) if mdd != 0 else None,
    }


def main() -> None:
    db = _resolve_db()
    price, index = load_data(db)
    print(f"数据: {price['date'].min().date()} ~ {price['date'].max().date()}, {price['stock_code'].nunique()} 只")

    configs = [
        (False, False, "BASE 全市场等权"),
        (False, True, "VOL  只买低波动(剔30%)"),
        (True, False, "TIME 择时(不筛股)"),
        (True, True, "BOTH 择时 + 低波动(剔30%)"),
    ]
    results = [run(price, index, t, v, DROP_TOP_PCT, lbl) for t, v, lbl in configs]

    print("\n" + "=" * 96)
    print("择时 + 波动筛 叠加实验（2019-2026，277只，每5日调仓）")
    print("=" * 96)
    print(f"{'策略':<26}{'总收益':>10}{'最大回撤':>10}{'夏普':>8}{'Calmar':>9}{'空仓日':>8}{'暴跌日均值':>12}")
    for r in results:
        crash = f"{r['crash_avg_ret']*100:+.2f}%" if r["crash_avg_ret"] is not None else "N/A"
        calmar = f"{r['calmar']:.2f}" if r["calmar"] is not None else "N/A"
        print(f"{r['label']:<26}{r['total_return']*100:>9.1f}%{r['max_drawdown']*100:>9.1f}%{r['sharpe']:>8.2f}{calmar:>9}{r['cash_days']:>8}{crash:>12}")

    # 逐年
    print("\n" + "=" * 96)
    print("逐年一致性检验")
    print("=" * 96)
    years = sorted(price["date"].dt.year.unique())
    yearly = {c[2].split()[0]: {} for c in configs}
    print(f"{'年份':<8}{'BASE':>10}{'VOL':>10}{'TIME':>10}{'BOTH':>10}")
    for y in years:
        p = price[price["date"].dt.year == y]
        if p["date"].nunique() < 60:
            continue
        vals = {}
        for t, v, lbl in configs:
            key = lbl.split()[0]
            r = run(p, index, t, v, DROP_TOP_PCT, lbl)
            yearly[key][int(y)] = r["total_return"]
            vals[key] = r["total_return"]
        print(f"{y:<8}{vals['BASE']*100:>9.1f}%{vals['VOL']*100:>9.1f}%{vals['TIME']*100:>9.1f}%{vals['BOTH']*100:>9.1f}%")

    # 统计
    print("\n" + "=" * 96)
    print("叠加效果分析")
    print("=" * 96)
    base, vol, time_, both = results
    print(f"  单择时 TIME:        收益 {time_['total_return']*100:+.1f}%, 回撤 {time_['max_drawdown']*100:.1f}%, 夏普 {time_['sharpe']:.2f}")
    print(f"  单波动 VOL:         收益 {vol['total_return']*100:+.1f}%, 回撤 {vol['max_drawdown']*100:.1f}%, 夏普 {vol['sharpe']:.2f}")
    print(f"  叠加 BOTH:          收益 {both['total_return']*100:+.1f}%, 回撤 {both['max_drawdown']*100:.1f}%, 夏普 {both['sharpe']:.2f}")
    print()
    print(f"  叠加 vs 单择时: 收益 {(both['total_return']-time_['total_return'])*100:+.1f}pp, 回撤 {(both['max_drawdown']-time_['max_drawdown'])*100:+.1f}pp")
    print(f"  叠加 vs 单波动: 收益 {(both['total_return']-vol['total_return'])*100:+.1f}pp, 回撤 {(both['max_drawdown']-vol['max_drawdown'])*100:+.1f}pp")

    # 逐年胜负
    if yearly["BOTH"]:
        yrs = sorted(yearly["BOTH"].keys())
        both_win_base = sum(1 for y in yrs if yearly["BOTH"][y] > yearly["BASE"][y])
        both_win_time = sum(1 for y in yrs if yearly["BOTH"][y] > yearly["TIME"][y])
        both_win_vol = sum(1 for y in yrs if yearly["BOTH"][y] > yearly["VOL"][y])
        print(f"\n  BOTH 跑赢 BASE: {both_win_base}/{len(yrs)} 年")
        print(f"  BOTH 跑赢 TIME: {both_win_time}/{len(yrs)} 年")
        print(f"  BOTH 跑赢 VOL:  {both_win_vol}/{len(yrs)} 年")

    out = ROOT / "results" / "combine_timing_volatility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "yearly": yearly,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
