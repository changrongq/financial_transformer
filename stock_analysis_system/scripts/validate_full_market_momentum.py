"""验证「全市场动量」是否有效（补上实盘脚本与回测股票池不一致的缺口）。

背景：
  研究报告里的动量回测用的是 historical_daily（277 只行业股，2019-2026），
  但实盘脚本 daily_momentum_signal.py 默认用 stock_data（全市场 5700 只，2024-2026）。
  两者股票池完全不同，必须单独验证全市场版本。

回测设计：
  - 数据：stock_data（全市场，2024-10 ~ 今天）
  - 每 5 个交易日调仓一次
  - 选近 20 日涨幅最强的前 N%（剔除 ST、过滤流动性）
  - 择时：大盘（全市场等权）10 日均收益 < 0 → 空仓
  - 成本：单边 0.15%（双边 0.3%）
  - 涨停过滤：调仓日涨幅 >= 9.5% 的股票不买入（近似"涨停买不进"）

输出：
  - 控制台打印各方案累计收益、年化、最大回撤、夏普
  - results/full_market_momentum_validation.json

用法：
  python scripts/validate_full_market_momentum.py
  python scripts/validate_full_market_momentum.py --top-pct 0.10 --max-picks 10
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
    price = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [收盘] AS close, [成交额] AS amount FROM stock_data",
        conn,
    )
    bi = pd.read_sql_query("SELECT stock_code, 名称 FROM stock_basic_info", conn)
    conn.close()

    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["stock_code"] = price["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price["amount"] = pd.to_numeric(price["amount"], errors="coerce")
    price = price.dropna(subset=["close"])
    # 剔除脏代码（None/空/非6位）
    price = price[price["stock_code"].str.len() == 6]
    price = price.sort_values(["stock_code", "date"]).reset_index(drop=True)

    bi["stock_code"] = bi["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    bi = bi[bi["stock_code"].str.len() == 6]
    return price, bi


def stats(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {}
    ret = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n = len(equity)
    years = n / 244.0
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((equity / equity.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(244)) if ret.std() > 0 else np.nan
    calmar = float(ann / abs(dd)) if dd < 0 else np.nan
    return {
        "total_return": round(total, 6),
        "annual_return": round(ann, 6),
        "max_drawdown": round(dd, 6),
        "sharpe": round(sharpe, 4),
        "calmar": round(calmar, 4),
        "days": n,
    }


def backtest(price, bi, *, mom_window, top_pct, max_picks, hold_days,
             min_amount, cost, limit_filter, use_timing, timing_window):
    # 宽表
    pivot = price.pivot_table(index="date", columns="stock_code", values="close", aggfunc="first")
    amt = price.pivot_table(index="date", columns="stock_code", values="amount", aggfunc="first")
    dates = pivot.index.sort_values()
    rets = pivot.pct_change(fill_method=None)

    # ST 名单（仅保留确实存在于 price 中的代码）
    st_codes = set(
        bi.loc[
            bi["名称"].astype(str).str.contains("ST", na=False)
            | bi["名称"].astype(str).str.contains("退", na=False),
            "stock_code",
        ]
    ) & set(pivot.columns)
    # 只保留同时有 close 和 amount 的股票
    common_cols = set(pivot.columns) & set(amt.columns)
    valid_cols = [c for c in pivot.columns if c not in st_codes and c in common_cols]
    pivot_v = pivot[valid_cols]
    amt_v = amt.reindex(index=dates, columns=valid_cols)
    rets_v = rets[valid_cols]
    pivot_v = pivot_v.reindex(index=dates)

    # 择时序列：全市场等权日收益
    market_ret = rets.mean(axis=1)
    timing_on = pd.Series(True, index=dates)
    if use_timing:
        regime = market_ret.rolling(timing_window, min_periods=3).mean()
        timing_on = regime >= 0
    timing_on = timing_on.fillna(False)

    mom = pivot_v / pivot_v.shift(mom_window) - 1.0

    equity = 1.0
    equity_curve = []
    holdings: list[str] = []
    trades = 0

    for i, d in enumerate(dates):
        if i < mom_window + 1:
            equity_curve.append((d, equity))
            continue

        # 持仓收益
        if holdings:
            r = rets_v.loc[d, holdings].fillna(0.0)
            equity *= float((1 + r).mean())

        # 调仓日
        if (i - mom_window - 1) % hold_days == 0:
            if not bool(timing_on.loc[d]):
                holdings = []
            else:
                m = mom.loc[d].dropna()
                if len(m) > 0:
                    a = amt_v.loc[d]
                    m = m[a.reindex(m.index) >= min_amount]
                    if limit_filter:
                        today_ret = rets_v.loc[d]
                        m = m[today_ret.reindex(m.index).fillna(0) < 0.095]
                    if len(m) > 0:
                        n = max(int(len(m) * top_pct), 3)
                        if max_picks > 0:
                            n = min(n, max_picks)
                        new_h = list(m.sort_values(ascending=False).head(n).index)
                        if set(new_h) != set(holdings):
                            trades += 1
                        # 成本：按换手比例计
                        turnover = 1.0 if not holdings else len(set(new_h) ^ set(holdings)) / max(len(holdings), 1)
                        equity *= (1 - cost * min(turnover, 2.0))
                        holdings = new_h

        equity_curve.append((d, equity))

    eq = pd.Series([e for _, e in equity_curve], index=[d for d, _ in equity_curve])
    out = stats(eq)
    out["trades"] = trades
    return out, eq


def main() -> None:
    parser = argparse.ArgumentParser(description="验证全市场动量有效性")
    parser.add_argument("--mom-window", type=int, default=20)
    parser.add_argument("--top-pct", type=float, default=0.10)
    parser.add_argument("--max-picks", type=int, default=20)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--min-amount", type=float, default=10000)
    parser.add_argument("--cost", type=float, default=0.0015, help="单边成本")
    parser.add_argument("--timing-window", type=int, default=10)
    args = parser.parse_args()

    db = _resolve_db()
    price, bi = load_data(db)
    print(f"数据范围: {price['date'].min().date()} ~ {price['date'].max().date()}")
    print(f"股票数: {price['stock_code'].nunique()}")
    print(f"剔除 ST: {int(bi['名称'].astype(str).str.contains('ST|退', na=False).sum())} 只")

    scenarios = [
        ("买入持有(全市场等权)", dict(use_timing=False, limit_filter=False, cost=0.0, top_pct=1.0, max_picks=0)),
        ("纯动量(无择时)", dict(use_timing=False, limit_filter=False, cost=args.cost)),
        ("纯动量+涨停过滤", dict(use_timing=False, limit_filter=True, cost=args.cost)),
        ("择时+动量", dict(use_timing=True, limit_filter=False, cost=args.cost)),
        ("择时+动量+涨停过滤", dict(use_timing=True, limit_filter=True, cost=args.cost)),
    ]

    common = dict(
        mom_window=args.mom_window, max_picks=args.max_picks, hold_days=args.hold_days,
        min_amount=args.min_amount, timing_window=args.timing_window,
    )

    print("\n" + "=" * 90)
    print(f"{'方案':<24}{'累计收益':>12}{'年化':>10}{'最大回撤':>11}{'夏普':>8}{'Calmar':>9}{'调仓':>6}")
    print("=" * 90)

    results = {}
    curves = {}
    for name, kw in scenarios:
        params = dict(common)
        params.update(kw)
        if "top_pct" not in kw:
            params["top_pct"] = args.top_pct
        s, eq = backtest(price, bi, **params)
        results[name] = s
        curves[name] = eq
        print(
            f"{name:<24}{s['total_return']*100:>11.1f}%{s['annual_return']*100:>9.1f}%"
            f"{s['max_drawdown']*100:>10.1f}%{s['sharpe']:>8.2f}{s['calmar']:>9.2f}{s['trades']:>6}"
        )

    print("=" * 90)

    # 逐年收益
    print("\n逐年收益：")
    print(f"{'方案':<24}" + "".join(f"{y:>10}" for y in range(2024, 2027)))
    all_years = sorted({d.year for d in curves["纯动量(无择时)"].index if d.year >= 2024})
    print(f"{'':<24}" + "".join(f"{y:>10}" for y in all_years))
    for name in ["买入持有(全市场等权)", "纯动量(无择时)", "择时+动量+涨停过滤"]:
        eq = curves[name]
        yearly = eq.groupby(eq.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        print(f"{name:<24}" + "".join(f"{yearly.get(y, np.nan)*100:>9.1f}%" for y in all_years))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "full_market_momentum_validation.json"
    out.write_text(
        json.dumps({"params": vars(args), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写入: {out}")

    # 结论
    pure = results["纯动量(无择时)"]["total_return"]
    bh = results["买入持有(全市场等权)"]["total_return"]
    print("\n" + "=" * 90)
    if pure > bh:
        print(f"结论：全市场动量有效（{pure*100:.1f}% vs 买入持有 {bh*100:.1f}%）")
    else:
        print(f"结论：全市场动量无效（{pure*100:.1f}% vs 买入持有 {bh*100:.1f}%）")
        print("      → 实盘脚本的动量选股在这 2 年样本上没有超额，第二层暂不能用")
    print("=" * 90)


if __name__ == "__main__":
    main()
