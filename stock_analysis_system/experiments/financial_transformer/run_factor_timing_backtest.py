"""实战回测：单因子选股 + regime 择时空仓（计入交易成本/滑点/资金闲置）。

目标（钳星方向）：
  把已验证的两层——「低换手+低动量反转选股」+「大盘均收益转负则空仓」——
  叠加成一个贴近实战的组合，回答「择时的价值去掉'全市场等权'假想后还剩多少」。

对比三层：
  1. 纯选股：每天按因子选 top-N 持有，一直满仓
  2. 选股 + 择时：下跌市空仓，上涨市选股买入
  3. 纯择时基准：全市场等权 + 择时空仓（对照，之前测出 +152%）

用法：
  python experiments/financial_transformer/run_factor_timing_backtest.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


def _resolve_trade_db() -> str:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return str(p.resolve())
    return str((PROJECT_ROOT.parent / "stock_history.db").resolve())


def _load_price(trade_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(trade_db)
    try:
        sd = pd.read_sql_query("SELECT * FROM stock_data", conn)
    finally:
        conn.close()
    sd.columns = [
        "stock_code", "date", "open", "high", "low", "close", "volume",
        "turnover", "amount", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
    ]
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.zfill(6)
    for c in ["close", "turnover"]:
        sd[c] = pd.to_numeric(sd[c], errors="coerce")
    sd = sd.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    return sd


def _build_signal(sd: pd.DataFrame) -> pd.DataFrame:
    """每只股票每天：低换手 + 低动量（20日反转）打分（越高越值得买）。"""
    g = sd.groupby("stock_code")
    sd["mom_20"] = g["close"].pct_change(20)
    sd["turn_5d"] = g["turnover"].transform(lambda s: s.rolling(5).mean())
    # 反转打分：动量越低、换手越低，分数越高（rank 后取平均）
    sd = sd.dropna(subset=["mom_20", "turn_5d"]).copy()
    sd["mom_rank"] = sd.groupby("date")["mom_20"].rank(pct=True)
    sd["turn_rank"] = sd.groupby("date")["turn_5d"].rank(pct=True)
    # 反转：低动量=高 rank；低换手=高 rank
    sd["factor_score"] = (1 - sd["mom_rank"]) + (1 - sd["turn_rank"])
    return sd


def _market_daily(sd: pd.DataFrame) -> pd.DataFrame:
    sd2 = sd.copy()
    sd2["ret_1d"] = sd2.groupby("stock_code")["close"].pct_change()
    daily = sd2.groupby("date")["ret_1d"].mean().rename("market_return").reset_index()
    daily["regime_mean"] = daily["market_return"].rolling(10, min_periods=3).mean()
    return daily


def _max_drawdown(series: pd.Series) -> float:
    cum = (1.0 + series.fillna(0.0)).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _backtest_factor(
    sd: pd.DataFrame,
    daily: pd.DataFrame,
    top_n: int,
    hold_days: int,
    use_timing: bool,
    timing_window: int,
    timing_threshold: float,
    commission: float,
    slippage: float,
) -> dict:
    """单因子选股 + 可选择时，持有 hold_days 天再换仓的回测。

    持有期内在持仓股票上吃收益；每到换仓日（或择时转空仓）才调仓并计成本。
    """
    dates = sorted(daily["date"].unique())
    regime = daily.set_index("date")["regime_mean"].to_dict()

    # 每日选 top_n（按 factor_score）
    sig = sd.sort_values(["date", "factor_score"], ascending=[True, False])
    top_by_date = {}
    for d, grp in sig.groupby("date"):
        top_by_date[d] = list(grp.head(top_n)["stock_code"].tolist())

    # 每只股票每日收益（供持仓期查收益）
    ret_map = {}
    for d, grp in sd.groupby("date"):
        ret_map[d] = dict(zip(grp["stock_code"], grp["ret_1d"]))

    daily_ret = []
    held_codes = []          # 当前持仓
    rebalance_on = 0          # 距离下次换仓的计数
    round_trip_cost = commission * 2 + slippage * 2  # 一次换仓（卖旧+买新）的双向成本

    for i, d in enumerate(dates):
        is_cash = use_timing and regime.get(d, 0.0) < timing_threshold
        if is_cash:
            daily_ret.append(0.0)
            if held_codes:
                # 清仓：全卖，扣一次卖出成本
                daily_ret[-1] -= (commission + slippage)
            held_codes = []
            rebalance_on = 0
            continue

        if not held_codes or rebalance_on <= 0:
            # 换仓：买入新的 top_n
            old = set(held_codes)
            new = list(top_by_date.get(d, []))[:top_n]
            if not new:
                daily_ret.append(0.0)
                held_codes = []
                continue
            # 换仓成本：换掉的仓位比例 * 双向成本
            overlap = len(old & set(new)) if old else 0
            turnover_ratio = 1.0 - overlap / max(len(new), 1) if old else 1.0
            daily_ret.append(-turnover_ratio * round_trip_cost)
            held_codes = new
            rebalance_on = hold_days
        else:
            # 持有期：吃持仓股票当日等权收益
            day_ret = np.mean([ret_map.get(d, {}).get(c, 0.0) or 0.0 for c in held_codes]) if held_codes else 0.0
            daily_ret.append(float(day_ret) if not pd.isna(day_ret) else 0.0)
        rebalance_on -= 1

    net = pd.Series(daily_ret, index=dates).fillna(0.0)
    buy_hold_market = daily.set_index("date")["market_return"].reindex(dates).fillna(0.0)
    return {
        "strategy_return": float((1.0 + net).cumprod().iloc[-1] - 1.0),
        "strategy_max_drawdown": _max_drawdown(net),
        "buy_hold_market_return": float((1.0 + buy_hold_market).cumprod().iloc[-1] - 1.0),
        "cash_days": int((net == 0.0).sum()),
        "days": int(len(dates)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="单因子选股 + regime 择时 实战回测")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--hold-days", type=int, default=5, help="换仓周期（持有 N 个交易日后重选）")
    parser.add_argument("--timing-window", type=int, default=10)
    parser.add_argument("--timing-threshold", type=float, default=0.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-08-14")
    args = parser.parse_args()

    trade_db = _resolve_trade_db()
    sd = _load_price(trade_db)
    sd = sd[(sd["date"] >= args.start) & (sd["date"] <= args.end)].copy()
    sd = _build_signal(sd)
    sd["ret_1d"] = sd.groupby("stock_code")["close"].pct_change()
    daily = _market_daily(sd)
    daily = daily[(daily["date"] >= args.start) & (daily["date"] <= args.end)]

    base = _backtest_factor(sd, daily, args.top_n, args.hold_days, False, args.timing_window, args.timing_threshold, args.commission, args.slippage)
    timed = _backtest_factor(sd, daily, args.top_n, args.hold_days, True, args.timing_window, args.timing_threshold, args.commission, args.slippage)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": vars(args),
        "pure_factor": base,
        "factor_plus_timing": timed,
    }

    out = ROOT / "results" / "factor_timing_backtest_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("单因子选股 + regime 择时 实战回测（计入成本/滑点）")
    print("=" * 70)
    print(f"区间: {args.start} ~ {args.end}（{base['days']} 天）")
    print(f"选股: 低换手+低动量反转 top{args.top_n}；择时: 大盘{args.timing_window}日均收益<{args.timing_threshold}空仓")
    print(f"\n【纯选股】收益 {base['strategy_return']*100:+.2f}%, 回撤 {base['strategy_max_drawdown']*100:.2f}%")
    print(f"【选股+择时】收益 {timed['strategy_return']*100:+.2f}%, 回撤 {timed['strategy_max_drawdown']*100:.2f}%, 空仓 {timed['cash_days']} 天")
    print(f"【全市场等权】收益 {base['buy_hold_market_return']*100:+.2f}%")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
