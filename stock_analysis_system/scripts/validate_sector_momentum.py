"""验证豆包建议的「行业动量」方案（用行业ETF替代个股）。

豆包的原话：
  "把个股动量降维成行业动量：直接用行业 ETF 替代个股。虽然会丢掉个股阿尔法，
   但彻底解决了涨停、流动性、退市的问题，容量大、滑点低，实盘可执行性会高非常多。
   你们的测试里行业动量虽然不如个股，但叠加择时之后依然有正收益，是很稳妥的折中。"

我们要验证：
  1. 全市场股票按行业分组，做行业动量（买最强行业的成分股等权）是否有效？
  2. 行业动量 + 择时是否为正？
  3. 与个股动量、买入持有对比？

注意：这里用「行业成分股等权」代理行业ETF（因为没有行业ETF的日线数据），
      这比真实ETF乐观（无ETF费率、无跟踪误差），但也不会有涨停买不进问题。

用法：
  python scripts/validate_sector_momentum.py
  python scripts/validate_sector_momentum.py --top-sectors 3 --mom-window 20
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
    bi = pd.read_sql_query("SELECT stock_code, 名称, 行业 FROM stock_basic_info", conn)
    conn.close()

    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["stock_code"] = price["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price["amount"] = pd.to_numeric(price["amount"], errors="coerce")
    price = price.dropna(subset=["close"])
    price = price[price["stock_code"].str.len() == 6]
    price = price.sort_values(["stock_code", "date"]).reset_index(drop=True)

    bi["stock_code"] = bi["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    bi = bi[bi["stock_code"].str.len() == 6]
    return price, bi


def stats(equity: pd.Series):
    if len(equity) < 2:
        return {}
    ret = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = len(equity) / 244.0
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
        "days": len(equity),
    }


def run_sector_momentum(price, bi, *, mom_window, top_sectors, hold_days,
                        cost, use_timing, timing_window, min_stocks_per_sector=5):
    """行业动量回测：每 hold_days 天，买入近 mom_window 日涨幅最强的 top_sectors 个行业。

    行业内按等权持有全部成分股（代理行业ETF）。
    """
    pivot = price.pivot_table(index="date", columns="stock_code", values="close", aggfunc="first")
    dates = pivot.index.sort_values()
    rets = pivot.pct_change(fill_method=None).reindex(index=dates)

    # ST 剔除
    st_codes = set(
        bi.loc[
            bi["名称"].astype(str).str.contains("ST", na=False)
            | bi["名称"].astype(str).str.contains("退", na=False),
            "stock_code",
        ]
    ) & set(pivot.columns)

    # 行业映射
    ind_map = bi.set_index("stock_code")["行业"].to_dict()
    code2ind = {c: ind_map.get(c) for c in pivot.columns if c not in st_codes}
    code2ind = {c: i for c, i in code2ind.items() if isinstance(i, str) and i}

    # 行业 -> 成分股
    from collections import defaultdict
    ind2codes = defaultdict(list)
    for c, i in code2ind.items():
        ind2codes[i].append(c)
    # 只保留成分股 >= min_stocks_per_sector 的行业
    ind2codes = {i: cs for i, cs in ind2codes.items() if len(cs) >= min_stocks_per_sector}

    # 行业指数（等权）
    ind_close = {}
    for i, cs in ind2codes.items():
        sub = pivot[cs]
        # 归一化后等权平均（避免不同价格量级干扰）
        norm = sub / sub.bfill().iloc[0]
        ind_close[i] = norm.mean(axis=1)
    ind_px = pd.DataFrame(ind_close).reindex(index=dates)

    # 择时：全市场等权
    market_ret = rets.mean(axis=1)
    if use_timing:
        timing_on = (market_ret.rolling(timing_window, min_periods=3).mean() >= 0).fillna(False)
    else:
        timing_on = pd.Series(True, index=dates)

    ind_mom = ind_px / ind_px.shift(mom_window) - 1.0

    equity = 1.0
    curve = []
    holdings: list[str] = []
    trades = 0

    for i, d in enumerate(dates):
        if i < mom_window + 1:
            curve.append((d, equity))
            continue

        if holdings:
            r = rets.loc[d, holdings].fillna(0.0)
            equity *= float((1 + r).mean())

        if (i - mom_window - 1) % hold_days == 0:
            if not bool(timing_on.loc[d]):
                holdings = []
            else:
                m = ind_mom.loc[d].dropna()
                if len(m) > 0:
                    top_ind = list(m.sort_values(ascending=False).head(top_sectors).index)
                    new_h = []
                    for ind in top_ind:
                        new_h.extend(ind2codes[ind])
                    if set(new_h) != set(holdings):
                        trades += 1
                    turnover = 1.0 if not holdings else len(set(new_h) ^ set(holdings)) / max(len(holdings), 1)
                    equity *= (1 - cost * min(turnover, 2.0))
                    holdings = new_h

        curve.append((d, equity))

    eq = pd.Series([e for _, e in curve], index=[d for d, _ in curve])
    s = stats(eq)
    s["trades"] = trades
    s["n_industries"] = len(ind2codes)
    s["avg_holdings"] = int(np.mean([len(ind2codes[i]) for i in ind2codes]) * top_sectors) if ind2codes else 0
    return s, eq, ind_px


def main() -> None:
    parser = argparse.ArgumentParser(description="验证行业动量（豆包建议）")
    parser.add_argument("--mom-window", type=int, default=20)
    parser.add_argument("--top-sectors", type=int, default=3, help="买最强的几个行业")
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--cost", type=float, default=0.0015)
    parser.add_argument("--timing-window", type=int, default=10)
    parser.add_argument("--min-stocks", type=int, default=5, help="行业最少几只成分股")
    args = parser.parse_args()

    db = _resolve_db()
    price, bi = load_data(db)
    print(f"数据范围: {price['date'].min().date()} ~ {price['date'].max().date()}")
    print(f"股票数: {price['stock_code'].nunique()}")
    n_ind = bi["行业"].dropna().nunique()
    print(f"行业数: {n_ind}")

    common = dict(
        mom_window=args.mom_window, hold_days=args.hold_days, cost=args.cost,
        timing_window=args.timing_window, min_stocks_per_sector=args.min_stocks,
    )

    print("\n" + "=" * 95)
    print(f"{'方案':<28}{'累计收益':>12}{'年化':>10}{'最大回撤':>11}{'夏普':>8}{'Calmar':>9}{'持仓数':>7}")
    print("=" * 95)

    results = {}
    curves = {}

    # 基准：全市场等权买入持有
    pivot = price.pivot_table(index="date", columns="stock_code", values="close", aggfunc="first")
    rets = pivot.pct_change(fill_method=None)
    bh = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    s = stats(bh)
    results["买入持有(全市场等权)"] = s
    curves["买入持有(全市场等权)"] = bh
    print(f"{'买入持有(全市场等权)':<28}{s['total_return']*100:>11.1f}%{s['annual_return']*100:>9.1f}%"
          f"{s['max_drawdown']*100:>10.1f}%{s['sharpe']:>8.2f}{s['calmar']:>9.2f}{'-':>7}")

    for top in [1, 3, 5]:
        for timing in [False, True]:
            params = dict(common)
            params["top_sectors"] = top
            params["use_timing"] = timing
            name = f"行业动量 top{top}" + ("+择时" if timing else "")
            s, eq, _ = run_sector_momentum(price, bi, **params)
            results[name] = s
            curves[name] = eq
            print(f"{name:<28}{s['total_return']*100:>11.1f}%{s['annual_return']*100:>9.1f}%"
                  f"{s['max_drawdown']*100:>10.1f}%{s['sharpe']:>8.2f}{s['calmar']:>9.2f}"
                  f"{s['avg_holdings']:>7}")

    print("=" * 95)
    print(f"（行业动量按「行业成分股等权」代理行业ETF，成分股>={args.min_stocks}只的行业）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "sector_momentum_validation.json"
    out.write_text(
        json.dumps({"params": vars(args), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写入: {out}")

    # 结论
    print("\n" + "=" * 95)
    best = max(
        [(k, v["total_return"]) for k, v in results.items() if k != "买入持有(全市场等权)"],
        key=lambda x: x[1],
    )
    bh_ret = results["买入持有(全市场等权)"]["total_return"]
    print(f"最佳行业动量方案: {best[0]}  {best[1]*100:.1f}%")
    print(f"买入持有:                    {bh_ret*100:.1f}%")
    if best[1] > bh_ret:
        print(f"→ 行业动量有效（超越买入持有 {((best[1]-bh_ret)*100):.1f}pp）")
    else:
        print(f"→ 行业动量也未超越买入持有（差 {((bh_ret-best[1])*100):.1f}pp）")
    print("=" * 95)


if __name__ == "__main__":
    main()
