"""波动率风控实验：剔除高波动股是否能"跌得更少"。

来自豆包的建议：
  「既然绝对量化的 MAE 改善很小，没法靠预测波动幅度赚钱，
    那反过来把它当风控工具会不会更合适？
    比如在持仓组合里，把波动排序最高的那批标的剔除掉，
    不求赚得更多，但求跌得更少，刚好和"躲暴跌"的核心逻辑对齐。」

验证方法：
  - 每个调仓日（每 5 个交易日），按过去 20 日波动率排序
  - 对照组：全部股票等权
  - 实验组1：剔除波动率最高的 30%
  - 实验组2：剔除波动率最高的 50%
  - 对比：总收益、最大回撤、暴跌日（市场跌>2%）的表现、胜率

数据：historical_daily（277 只目标行业股，2019-2026）

用法：
  python scripts/volatility_risk_control.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0005  # 单边成本
REBALANCE_DAYS = 5


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


def load_data(db: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return price, index


def run_portfolio(
    price: pd.DataFrame,
    index: pd.DataFrame,
    drop_top_pct: float,
    label: str,
) -> dict:
    """每 REBALANCE_DAYS 调仓，按波动率剔除高波动股票。"""
    dates = sorted(price["date"].unique())
    date_pos = {d: i for i, d in enumerate(dates)}

    # 预处理：按日期分组
    by_date = {d: g.set_index("stock_code")["ret"] for d, g in price.groupby("date")}
    # 波动率：过去20日
    price_pivot = price.pivot_table(index="date", columns="stock_code", values="ret", aggfunc="first")
    price_pivot = price_pivot.reindex(dates)
    vol20 = price_pivot.rolling(20, min_periods=10).std()

    holdings = []
    equity = 1.0
    peak = 1.0
    equity_curve = []
    daily_rets = []
    crash_day_rets = []  # 市场暴跌日的组合表现
    idx_ret = index.set_index("date")["ret"]

    for i, d in enumerate(dates):
        if i < 25:
            continue

        # 调仓
        if i % REBALANCE_DAYS == 0 or not holdings:
            vol_today = vol20.loc[d].dropna()
            if len(vol_today) >= 10:
                if drop_top_pct > 0:
                    threshold = vol_today.quantile(1.0 - drop_top_pct)
                    keep = vol_today[vol_today <= threshold].index.tolist()
                else:
                    keep = vol_today.index.tolist()
                turnover = 1.0 if not holdings else len(set(holdings) - set(keep)) / max(len(keep), 1)
                equity *= (1.0 - abs(turnover) * COST)
                holdings = keep

        if not holdings:
            daily_rets.append(0.0)
            equity_curve.append(equity)
            continue

        # 当日组合收益（等权）
        day_series = by_date.get(d)
        if day_series is None:
            continue
        rets = day_series.reindex(holdings).dropna()
        if rets.empty:
            continue
        port_ret = float(rets.mean())
        equity *= (1.0 + port_ret)
        peak = max(peak, equity)
        equity_curve.append(equity)
        daily_rets.append(port_ret)

        # 记录暴跌日表现
        mkt = idx_ret.get(d, np.nan)
        if pd.notna(mkt) and mkt < -0.02:
            crash_day_rets.append(port_ret)

    rets = pd.Series(daily_rets)
    cum = np.cumprod(1.0 + rets.values)
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min()) if len(cum) else 0.0

    return {
        "label": label,
        "drop_top_pct": drop_top_pct,
        "total_return": float(equity - 1.0),
        "max_drawdown": mdd,
        "days": len(rets),
        "win_rate": float((rets > 0).mean()),
        "avg_daily_ret": float(rets.mean()),
        "volatility": float(rets.std()),
        "crash_days": len(crash_day_rets),
        "crash_avg_ret": float(np.mean(crash_day_rets)) if crash_day_rets else None,
        "crash_total_ret": float(np.sum(crash_day_rets)) if crash_day_rets else None,
    }


def annual_breakdown(price: pd.DataFrame, index: pd.DataFrame, drop_top_pct: float) -> dict:
    """逐年验证。"""
    years = sorted(price["date"].dt.year.unique())
    out = {}
    for y in years:
        p = price[price["date"].dt.year == y]
        if p["date"].nunique() < 60:
            continue
        r = run_portfolio(p, index, drop_top_pct, f"year_{y}")
        out[int(y)] = r["total_return"]
    return out


def main() -> None:
    db = _resolve_db()
    price, index = load_data(db)
    print(f"数据: {price['date'].min().date()} ~ {price['date'].max().date()}, {price['stock_code'].nunique()} 只股票")

    results = []
    for pct, label in [(0.0, "基准: 全部股票等权"), (0.3, "剔除高波动30%"), (0.5, "剔除高波动50%")]:
        r = run_portfolio(price, index, pct, label)
        results.append(r)

    print("\n" + "=" * 88)
    print("波动率风控实验结果（2019-2026，277只股票，每5日调仓）")
    print("=" * 88)
    print(f"{'策略':<22}{'总收益':>10}{'最大回撤':>10}{'日波动':>9}{'胜率':>8}{'暴跌日均值':>12}")
    for r in results:
        crash = f"{r['crash_avg_ret']*100:+.2f}%" if r["crash_avg_ret"] is not None else "N/A"
        print(f"{r['label']:<22}{r['total_return']*100:>9.1f}%{r['max_drawdown']*100:>9.1f}%{r['volatility']*100:>8.2f}%{r['win_rate']*100:>7.1f}%{crash:>12}")

    # 逐年
    print("\n" + "=" * 88)
    print("逐年一致性检验（关键）")
    print("=" * 88)
    yearly = {}
    for pct, label in [(0.0, "全部"), (0.3, "剔30%"), (0.5, "剔50%")]:
        yearly[label] = annual_breakdown(price, index, pct)

    print(f"{'年份':<8}{'全部':>12}{'剔30%':>12}{'剔50%':>12}")
    all_years = sorted(yearly["全部"].keys())
    win30 = win50 = 0
    for y in all_years:
        v0 = yearly["全部"].get(y, 0)
        v3 = yearly["剔30%"].get(y, 0)
        v5 = yearly["剔50%"].get(y, 0)
        if v3 > v0: win30 += 1
        if v5 > v0: win50 += 1
        print(f"{y:<8}{v0*100:>11.1f}%{v3*100:>11.1f}%{v5*100:>11.1f}%")
    print(f"\n  剔除30% 跑赢全部: {win30}/{len(all_years)} 年")
    print(f"  剔除50% 跑赢全部: {win50}/{len(all_years)} 年")

    # 结论判断
    base = results[0]
    r30 = results[1]
    r50 = results[2]
    print("\n" + "=" * 88)
    print("结论")
    print("=" * 88)
    print(f"  剔除30%: 收益 {r30['total_return']*100:+.1f}% vs 基准 {base['total_return']*100:+.1f}% "
          f"(差 {(r30['total_return']-base['total_return'])*100:+.1f}pp)")
    print(f"          回撤 {r30['max_drawdown']*100:.1f}% vs 基准 {base['max_drawdown']*100:.1f}% "
          f"(改善 {(r30['max_drawdown']-base['max_drawdown'])*100:+.1f}pp)")
    print(f"  剔除50%: 收益 {r50['total_return']*100:+.1f}% vs 基准 {base['total_return']*100:+.1f}% "
          f"(差 {(r50['total_return']-base['total_return'])*100:+.1f}pp)")
    print(f"          回撤 {r50['max_drawdown']*100:.1f}% vs 基准 {base['max_drawdown']*100:.1f}% "
          f"(改善 {(r50['max_drawdown']-base['max_drawdown'])*100:+.1f}pp)")

    out = ROOT / "results" / "volatility_risk_control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "yearly": yearly,
        "yearly_win": {"drop30": f"{win30}/{len(all_years)}", "drop50": f"{win50}/{len(all_years)}"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
