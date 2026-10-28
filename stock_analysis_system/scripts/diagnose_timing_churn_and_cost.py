"""诊断择时策略的「换手频率」和「交易成本」影响。

回答三个问题：
  1. 频繁 HOLD→CASH→HOLD 切换是正常现象吗？切换有多频繁？
  2. 回测里有计算交易成本吗？（答案：原回测没有，本脚本补上）
  3. 算上成本之后，择时策略还剩多少收益？

使用方法：
  python scripts/diagnose_timing_churn_and_cost.py
  python scripts/diagnose_timing_churn_and_cost.py --cost 0.0005   # ETF 单边万5
  python scripts/diagnose_timing_churn_and_cost.py --cost 0.001    # 单边千1（股票）
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


def load_market_daily(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    sd = pd.read_sql_query("SELECT stock_code, [日期] AS date, [收盘] AS close FROM stock_data", conn)
    conn.close()

    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    sd["close"] = pd.to_numeric(sd["close"], errors="coerce")
    sd = sd.dropna(subset=["close"])
    sd = sd[sd["stock_code"].str.len() == 6]
    sd = sd.sort_values(["stock_code", "date"])
    sd["ret"] = sd.groupby("stock_code")["close"].pct_change(fill_method=None)
    daily = sd.groupby("date")["ret"].mean().rename("market_ret").reset_index()
    return daily.dropna().sort_values("date").reset_index(drop=True)


def stats(equity: pd.Series):
    ret = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = len(equity) / 244.0
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((equity / equity.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(244)) if ret.std() > 0 else np.nan
    return {
        "total_return": total,
        "annual_return": ann,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "days": len(equity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="择时策略换手与成本诊断")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--cost", type=float, default=0.0005,
                        help="单边交易成本（ETF约0.0005=万5，股票约0.001=千1含印花税）")
    args = parser.parse_args()

    daily = load_market_daily(_resolve_db())
    daily["mean"] = daily["market_ret"].rolling(args.window, min_periods=3).mean()
    daily["held"] = daily["mean"] >= args.threshold
    daily = daily.dropna(subset=["mean"]).reset_index(drop=True)

    d = daily.copy()
    # 信号变化：今天的 held 与昨天不同 = 今天要调仓
    d["prev_held"] = d["held"].shift(1)
    d["switch"] = (d["held"] != d["prev_held"]) & d["prev_held"].notna()

    n_switch = int(d["switch"].sum())
    n_days = len(d)
    n_hold_days = int(d["held"].sum())
    n_cash_days = n_days - n_hold_days

    # 每次切换的成本
    d["gross"] = np.where(d["held"], d["market_ret"], 0.0)
    d["cost"] = np.where(d["switch"], args.cost, 0.0)
    d["net"] = d["gross"] - d["cost"]

    eq_gross = (1 + d["gross"]).cumprod()
    eq_net = (1 + d["net"]).cumprod()
    eq_bh = (1 + d["market_ret"]).cumprod()

    sg, sn, sb = stats(eq_gross), stats(eq_net), stats(eq_bh)

    print("=" * 78)
    print("【问题 1】频繁切换是正常现象吗？")
    print("=" * 78)
    print(f"  回测区间: {d['date'].min().date()} ~ {d['date'].max().date()}（{n_days} 个交易日）")
    print(f"  持仓天数: {n_hold_days} ({n_hold_days/n_days*100:.1f}%)")
    print(f"  空仓天数: {n_cash_days} ({n_cash_days/n_days*100:.1f}%)")
    print()
    print(f"  >>> 信号切换总次数: {n_switch} 次")
    print(f"      平均每 {n_days/max(n_switch,1):.1f} 个交易日切换一次")
    print(f"      约每年 {n_switch/(n_days/244):.1f} 次切换")

    # 连续持仓/空仓的持续时间分布
    runs = []
    cur = d["held"].iloc[0]
    length = 1
    for v in d["held"].iloc[1:]:
        if v == cur:
            length += 1
        else:
            runs.append(("HOLD" if cur else "CASH", length))
            cur, length = v, 1
    runs.append(("HOLD" if cur else "CASH", length))
    runs_df = pd.DataFrame(runs, columns=["state", "len"])

    print()
    for st in ["HOLD", "CASH"]:
        sub = runs_df[runs_df["state"] == st]["len"]
        if len(sub):
            print(f"  {st} 段统计: {len(sub)} 段, "
                  f"中位持续 {sub.median():.0f} 天, 平均 {sub.mean():.1f} 天, 最长 {sub.max()} 天, 最短 {sub.min()} 天")

    # 短命段统计（<=3天就跑的）
    short = runs_df[runs_df["len"] <= 3]
    print(f"\n  持续时间 <= 3 天的短命区间: {len(short)} 段 "
          f"（占全部 {len(runs_df)} 段的 {len(short)/len(runs_df)*100:.1f}%）")
    for st in ["HOLD", "CASH"]:
        s2 = short[short["state"] == st]
        if len(s2):
            print(f"    {st} 只持续 1-3 天就切换: {len(s2)} 次")

    print()
    print("=" * 78)
    print("【问题 2】回测有没有算交易成本？")
    print("=" * 78)
    print("  [否] 原始回测（run_regime_timing_backtest.py）: 没有算")
    print("     代码: strategy_return = np.where(held, market_return, 0.0)")
    print("     （该脚本的 caveat 里也写明了'未计交易成本/滑点'）")
    print()
    print(f"  [是] 本脚本已补上，假设单边成本 {args.cost*100:.3f}%")

    print()
    print("=" * 78)
    print("【问题 3】算上成本之后还剩多少？")
    print("=" * 78)
    print(f"{'方案':<22}{'累计收益':>13}{'年化':>10}{'最大回撤':>12}{'夏普':>9}")
    print("-" * 78)
    for name, s in [("买入持有", sb), ("择时(不含成本)", sg), (f"择时(含成本 {args.cost*1e4:.0f}bp)", sn)]:
        print(f"{name:<22}{s['total_return']*100:>+12.1f}%{s['annual_return']*100:>+9.1f}%"
              f"{s['max_drawdown']*100:>11.1f}%{s['sharpe']:>9.2f}")
    print("-" * 78)
    drag = sg["total_return"] - sn["total_return"]
    print(f"  成本拖累: 累计收益减少 {drag*100:.1f}pp "
          f"（{n_switch} 次切换 × {args.cost*100:.3f}% × 复利效应）")
    print(f"  年化拖累: {(sg['annual_return']-sn['annual_return'])*100:.2f}pp")

    # 敏感测试：不同成本水平
    print("\n  不同成本水平下的择时收益：")
    print(f"  {'单边成本':<12}{'累计收益':>13}{'年化':>10}{'最大回撤':>12}")
    cost_rows = []
    for c in [0.0, 0.0003, 0.0005, 0.001, 0.002]:
        net = d["gross"] - np.where(d["switch"], c, 0.0)
        eq = (1 + net).cumprod()
        s = stats(eq)
        print(f"  {c*100:>6.2f}%     {s['total_return']*100:>+12.1f}%{s['annual_return']*100:>+9.1f}%"
              f"{s['max_drawdown']*100:>11.1f}%")
        cost_rows.append({"cost_one_way": c, **{k: round(v, 6) for k, v in s.items()}})

    # 最近的实际切换记录
    print()
    print("=" * 78)
    print("【附】最近 10 次信号切换记录")
    print("=" * 78)
    sw = d[d["switch"]].tail(10)[["date", "held", "mean"]]
    for _, r in sw.iterrows():
        action = "买入(HOLD)" if r["held"] else "卖出(CASH)"
        print(f"  {r['date'].date()}  {action}  大盘{args.window}日均={r['mean']*100:+.3f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "timing_churn_and_cost.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "n_days": n_days,
        "n_switch": n_switch,
        "switches_per_year": round(n_switch / (n_days / 244), 2),
        "hold_days": n_hold_days,
        "cash_days": n_cash_days,
        "runs_summary": {
            st: {
                "n_runs": int(len(runs_df[runs_df["state"] == st])),
                "median_len": float(runs_df[runs_df["state"] == st]["len"].median()) if len(runs_df[runs_df["state"] == st]) else 0,
                "max_len": int(runs_df[runs_df["state"] == st]["len"].max()) if len(runs_df[runs_df["state"] == st]) else 0,
            }
            for st in ["HOLD", "CASH"]
        },
        "short_runs_le3d": int(len(short)),
        "results": {
            "buy_hold": {k: round(v, 6) for k, v in sb.items()},
            "timing_gross": {k: round(v, 6) for k, v in sg.items()},
            "timing_net": {k: round(v, 6) for k, v in sn.items()},
        },
        "cost_sensitivity": cost_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
