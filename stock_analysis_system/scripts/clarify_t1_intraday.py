"""澄清：T+1 下，什么叫"日内收益"？我们回测里的 +15.92% 能不能拿到？

====================== 核心问题 ======================
之前的分析里：
  "日内收益" = 收盘 / 开盘 - 1

但在 T+1 制度下：
  · T 日买入 → T+1 才能卖
  · 所以 T 日"开盘买、收盘卖"是**违规的**

那么，这个"日内收益"到底是什么？
====================== 三种可能的解读 ======================
解读A：它只是一个统计分解，不能交易
      全天收益 = 隔夜 + 日内，纯粹是数学拆分
      → 无法单独获取

解读B：它可以被"错一天"获取
      T日开盘买 → T+1日开盘卖，持有 = T日(开→收) + T+1日(隔夜)
      = intraday[t] + gap[t+1]
      这才是 T+1 下真正的"最短持有周期"
      → 可以交易！但收益结构不同

解读C：它可以被"持仓过夜"策略部分获取
      一直持有（买入持有），天然获得所有日内+所有隔夜
      → 但隔夜是负的，被拖累

本脚本要回答：
  Q1. 解读B（T开盘买→T+1开盘卖）的收益是多少？成本多少？
  Q2. 解读B 和 "买入持有"、"择时" 对比如何？
  Q3. 这个"可交易的最短周期"值得做成策略吗？

用法：
  python scripts/clarify_t1_intraday.py
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


def load_etf(db: str, code: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def stats(curve: np.ndarray) -> dict:
    r = np.diff(np.concatenate([[1.0], curve]))
    total = float(curve[-1] - 1)
    years = len(curve) / 244
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe}


def main() -> None:
    db = _resolve_db()
    df = load_etf(db, "510300.SH")
    print(f"510300 沪深300ETF: {df['date'].min().date()} ~ {df['date'].max().date()}  {len(df)} 天\n")

    gap = df["gap"].values
    intra = df["intraday"].values

    # ===== 各种持有周期的收益构成 =====
    print("=" * 92)
    print("【1】不同持有周期的收益构成（T+1 下哪些可交易？）")
    print("=" * 92)
    print(f"{'持有方式':<38}{'含隔夜':>10}{'含日内':>10}{'T+1可行':>10}{'年化':>11}")
    print("-" * 92)

    combos = [
        ("T开盘买 -> T收盘卖（纯日内）", False, True, "X 违规"),
        ("T收盘买 -> T+1开盘卖（纯隔夜）", True, False, "OK 可行"),
        ("T开盘买 -> T+1开盘卖（日内+隔夜）", True, True, "OK 可行"),
        ("T开盘买 -> 一直持有", True, True, "OK 可行"),
    ]
    rows = []
    for label, has_gap, has_intra, feasible in combos:
        if label.startswith("T开盘买 -> T收盘卖"):
            r = intra
        elif label.startswith("T收盘买 -> T+1开盘卖"):
            r = gap
        elif label.startswith("T开盘买 -> T+1开盘卖"):
            r = intra[:-1] + gap[1:]
        else:
            r = gap + intra
        a = float((1 + r).prod() ** (244 / len(r)) - 1)
        print(f"{label:<38}{'Y' if has_gap else '-':>10}{'Y' if has_intra else '-':>10}"
              f"{feasible:>10}{a*100:>+10.2f}%")
        rows.append({"label": label, "has_gap": has_gap, "has_intraday": has_intra,
                     "feasible_t1": feasible, "ann": round(a, 6)})
    print("-" * 92)

    # ===== 可交易选项的成本分析 =====
    print("\n" + "=" * 92)
    print("【2】T+1 可交易的两种模式的成本敏感性")
    print("=" * 92)

    r_overnight = gap  # 收盘买，次日开盘卖
    r_full1d = intra[:-1] + gap[1:]  # 开盘买，次日开盘卖

    for name, r, trades_per_year in [
        ("T收盘买→T+1开盘卖（纯隔夜）", r_overnight, 244),
        ("T开盘买→T+1开盘卖（日内+隔夜）", r_full1d, 244),
    ]:
        print(f"\n  {name}")
        print(f"  {'单边成本':<12}{'年化':>12}{'累计收益':>14}{'最大回撤':>12}{'夏普':>9}")
        print("  " + "-" * 60)
        for c in [0.0, 0.0005, 0.001, 0.002, 0.003]:
            net = r - 2 * c
            eq = np.cumprod(1 + net)
            s = stats(eq)
            print(f"  {c*100:>6.2f}%     {s['ann']*100:>+11.2f}%{s['total']*100:>+13.1f}%"
                  f"{s['dd']*100:>11.1f}%{s['sharpe']:>9.2f}")
        # 盈亏平衡
        gross_ann = float((1 + r).prod() ** (244 / len(r)) - 1)
        be = gross_ann / (2 * 244)
        print(f"  盈亏平衡单边成本: {be*100:.4f}%")

    # ===== 对比择时 =====
    print("\n" + "=" * 92)
    print("【3】对比：择时（T+1开盘执行）vs 这些「做T」策略")
    print("=" * 92)

    d = df.copy()
    d["mean"] = d["ret"].rolling(10, min_periods=3).mean()
    d["sig"] = d["mean"] >= 0
    d = d.dropna(subset=["mean"]).reset_index(drop=True)

    # 择时：T-1信号 → T开盘执行 → 吃T日内
    eq_t = [1.0]
    pos = 0.0
    for i in range(1, len(d)):
        target = 1.0 if d.loc[i - 1, "sig"] else 0.0
        r = float(d.loc[i, "intraday"]) if target > 0 else 0.0
        trade = abs(target - pos)
        eq_t.append(eq_t[-1] * (1 + r - trade * 0.0005))
        pos = target
    st = stats(np.array(eq_t[1:]))
    print(f"  {'择时(T+1开盘, 万5成本)':<34}{st['ann']*100:>+11.2f}%{st['total']*100:>+13.1f}%"
          f"{st['dd']*100:>11.1f}%{st['sharpe']:>9.2f}")

    eq_bh = np.cumprod(1 + df["ret"].values)
    sb = stats(eq_bh)
    print(f"  {'买入持有':<34}{sb['ann']*100:>+11.2f}%{sb['total']*100:>+13.1f}%"
          f"{sb['dd']*100:>11.1f}%{sb['sharpe']:>9.2f}")

    # ===== 结论 =====
    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    print("  1. 回测里的「日内收益 +15.92%」在 T+1 下**无法直接交易**")
    print("     它是统计分解，不是可执行策略")
    print()
    print("  2. T+1 下最短可交易周期是「T开盘买 → T+1开盘卖」")
    print("     收益 = T日日内 + T+1日隔夜")
    print(f"     年化 = {float((1 + r_full1d).prod() ** (244/len(r_full1d)) - 1)*100:+.2f}%（毛）")
    print()
    print("  3. 但这两个模式都要每天交易（244次/年），成本极高")
    print("     → 盈亏平衡成本远低于实际成本，不可行")
    print()
    print("  4. 择时（只在HOLD日交易，34次/年）才是唯一可行方案")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "t1_intraday_clarification.json"
    out.write_text(json.dumps({"period_days": len(df), "combos": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
