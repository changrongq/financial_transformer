"""诊断择时信号的「执行时点」问题（未来函数检验）。

核心问题：
  回测代码 `gross = np.where(held, market_ret, 0.0)` 里，
  held 用当天收盘价算出的 10 日均值判断，却吃当天的收益。
  这等于"当天开盘就知道当天收盘信号"——不现实。

真实可执行的三种方式：
  A) 当日收盘执行（理想但做不到）：信号用当天收盘算，当天收盘价成交
     —— 需要"收盘前预知收盘价"，不现实
  B) 次日开盘执行（最现实）：T日收盘出信号，T+1开盘成交，T+1开始吃收益
  C) 次日收盘执行（保守）：T日收盘出信号，T+1收盘才成交，T+2开始吃收益

本脚本用"次日开盘价"和"次日收盘价"分别回测，给出真实可执行收益。

用法：
  python scripts/diagnose_timing_execution_price.py
  python scripts/diagnose_timing_execution_price.py --cost 0.0005
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


def load_market_ohlc(db: str) -> pd.DataFrame:
    """加载全市场等权日线（含 open/close），用于模拟不同执行价。"""
    conn = sqlite3.connect(db)
    sd = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [开盘] AS open, [收盘] AS close FROM stock_data", conn
    )
    conn.close()

    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    for c in ["open", "close"]:
        sd[c] = pd.to_numeric(sd[c], errors="coerce")
    sd = sd.dropna(subset=["close"])
    sd = sd[sd["stock_code"].str.len() == 6]
    sd = sd.sort_values(["stock_code", "date"])

    # 每只股票的日收益
    sd["ret"] = sd.groupby("stock_code")["close"].pct_change(fill_method=None)
    # 开盘到收盘的日内收益
    sd["intraday"] = sd["close"] / sd["open"] - 1.0
    # 昨收到今开的跳空
    sd["gap"] = sd.groupby("stock_code")["open"].pct_change(fill_method=None)

    agg = sd.groupby("date").agg(
        market_ret=("ret", "mean"),
        market_intraday=("intraday", "mean"),
        market_gap=("gap", "mean"),
    ).reset_index()
    return agg.dropna(subset=["market_ret"]).sort_values("date").reset_index(drop=True)


def stats(equity: pd.Series) -> dict:
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
    parser = argparse.ArgumentParser(description="择时执行时点诊断")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--cost", type=float, default=0.0005, help="单边成本")
    args = parser.parse_args()

    d = load_market_ohlc(_resolve_db())
    d["mean"] = d["market_ret"].rolling(args.window, min_periods=3).mean()
    d["signal"] = d["mean"] >= args.threshold
    d = d.dropna(subset=["mean"]).reset_index(drop=True)

    # 三种执行方式
    # A) 当日收盘执行：当天信号 == 当天持仓（原回测，有未来函数）
    d["heldA"] = d["signal"].astype(bool)

    # B) 次日开盘执行：T日信号决定 T+1 日持仓（T+1 吃 T+1 全天收益）
    d["heldB"] = d["signal"].shift(1).fillna(False).astype(bool)

    # C) 次日收盘执行：T日信号决定 T+2 日持仓
    d["heldC"] = d["signal"].shift(2).fillna(False).astype(bool)

    # 切换次数（按各自持仓序列）
    for k in ["heldA", "heldB", "heldC"]:
        d[f"switch_{k}"] = d[k].ne(d[k].shift(1)) & d[k].shift(1).notna()

    def run(held_col: str, cost: float = 0.0) -> pd.Series:
        gross = np.where(d[held_col], d["market_ret"], 0.0)
        sw = d[f"switch_{held_col}"].values
        net = gross - np.where(sw, cost, 0.0)
        return pd.Series(np.cumprod(1 + net), index=d["date"])

    eq_bh = pd.Series(np.cumprod(1 + d["market_ret"].values), index=d["date"])
    eqA = run("heldA", args.cost)
    eqB = run("heldB", args.cost)
    eqC = run("heldC", args.cost)
    eqA0 = run("heldA", 0.0)
    eqB0 = run("heldB", 0.0)
    eqC0 = run("heldC", 0.0)

    labels = [
        ("买入持有", eq_bh),
        ("A) 当日收盘执行（原回测，有未来函数）", eqA0),
        ("B) 次日开盘执行（最现实）*含成本*", eqB),
        ("C) 次日收盘执行（最保守）*含成本*", eqC),
    ]

    print("=" * 92)
    print("择时信号的「执行时点」对比（区间 %s ~ %s，%d 个交易日）"
          % (d["date"].min().date(), d["date"].max().date(), len(d)))
    print("=" * 92)
    print(f"{'执行方式':<40}{'累计收益':>13}{'年化':>10}{'最大回撤':>12}{'夏普':>9}{'切换':>7}")
    print("-" * 92)
    for name, eq in labels:
        s = stats(eq)
        nsw = int(d["switch_heldA"].sum()) if "原回测" in name else (
            int(d["switch_heldB"].sum()) if "开盘" in name else (
                int(d["switch_heldC"].sum()) if "收盘执行（最保守" in name else 0))
        print(f"{name:<40}{s['total_return']*100:>+12.1f}%{s['annual_return']*100:>+9.1f}%"
              f"{s['max_drawdown']*100:>11.1f}%{s['sharpe']:>9.2f}{nsw:>7}")
    print("-" * 92)

    sA, sB, sC = stats(eqA0), stats(eqB), stats(eqC)
    print("\n未来函数造成的虚高：")
    print(f"  原回测（A，当日收盘执行）:      {sA['total_return']*100:+.1f}%")
    print(f"  真实可执行（B，次日开盘）:      {sB['total_return']*100:+.1f}%")
    print(f"  >>> 差额（虚高部分）:            {(sA['total_return']-sB['total_return'])*100:.1f}pp")
    print(f"  最保守（C，次日收盘）:          {sC['total_return']*100:+.1f}%")

    bh = stats(eq_bh)
    print(f"\n对比买入持有: {bh['total_return']*100:+.1f}%")
    print(f"  真实可执行（B）是否跑赢买入持有: "
          f"{'是，超出 %.1fpp' % ((sB['total_return']-bh['total_return'])*100) if sB['total_return'] > bh['total_return'] else '否'}")

    # 成本影响（B 方案）
    print("\nB 方案（次日开盘）在不同成本下：")
    print(f"  {'单边成本':<12}{'累计收益':>13}{'年化':>10}")
    cost_rows = []
    for c in [0.0, 0.0003, 0.0005, 0.001]:
        eq = run("heldB", c)
        s = stats(eq)
        print(f"  {c*100:>6.2f}%     {s['total_return']*100:>+12.1f}%{s['annual_return']*100:>+9.1f}%")
        cost_rows.append({"cost_one_way": c, **{k: round(v, 6) for k, v in s.items()}})

    # 解释
    print("\n" + "=" * 92)
    print("结论：")
    print("=" * 92)
    print("  * 回测的 +149.92% 假设『当日收盘出信号 + 当日收盘成交』，现实做不到。")
    print("  * 因为 10 日均值要用当天收盘价计算，只有收盘后才知道信号。")
    print("  * 真实做法：T日收盘后看信号 → T+1 开盘执行。这是 B 方案。")
    print(f"  * B 方案的实际收益：{sB['total_return']*100:+.1f}%（年化 {sB['annual_return']*100:+.1f}%）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "timing_execution_price.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "period": {"start": str(d["date"].min().date()), "end": str(d["date"].max().date()), "days": len(d)},
        "results": {
            "buy_hold": {k: round(v, 6) for k, v in stats(eq_bh).items()},
            "A_close_today_lookahead": {k: round(v, 6) for k, v in sA.items()},
            "B_open_next_day": {k: round(v, 6) for k, v in sB.items()},
            "C_close_next_day": {k: round(v, 6) for k, v in sC.items()},
        },
        "lookahead_bias_pp": round((sA["total_return"] - sB["total_return"]) * 100, 2),
        "cost_sensitivity_B": cost_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
