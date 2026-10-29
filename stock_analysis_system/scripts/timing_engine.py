"""统一口径的择时回测引擎（修正版）。

=================== 为什么需要重写 ===================
之前的回测有口径问题：
  · "只吃日内"（intraday）：假设每天开盘买、收盘卖 → T+1 下违规
  · "next_open" 实现：持仓期间净值不记录，只在平仓时跳变 → 回撤/夏普失真

本模块提供**统一、可验证**的实现，所有策略共用。

=================== 严格 T+1 的正确实现 ===================
交易时序（真实 A 股）：
  T 日 15:00  收盘，计算信号（用 T 日及之前数据）
  T+1 日 09:30 开盘，按信号执行（买入或卖出）
  T+1 日       持仓则吃 T+1 全天收益（含隔夜跳空 + 日内）

因为"T日收盘算信号 → T+1开盘执行"，所以：
  · 持仓状态 held[T+1] = sig[T]
  · T+1 的收益 = close[T+1]/open[T+1] - 1（买入当天只吃日内）
  · T+2 及以后 = close[t]/close[t-1] - 1（含隔夜）

这是唯一真实的实现。不存在"每天只吃日内"的可能。

用法（作为模块）：
  from timing_engine import run_strict_t1, run_intraday_idealized, stats
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve_db() -> str:
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


def load_etf(db: str, code: str) -> pd.DataFrame | None:
    """加载 ETF 日线。

    ⚠️ 关键：必须用 pre_close（复权后昨收）算收益，不能用前一日的 close。
       原因：ETF 有份额拆分/合并（如 510500 在 2015-04-15 拆分 1:3.5），
       原始 close 价格不连续，直接用 close[i]/close[i-1] 会产生 +261% 的假收益。
       pre_close 是交易所复权后的昨收，跨拆分仍然连续。

       同理，隔夜跳空也要用 open/pre_close，不是 open/close[i-1]。
    """
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, pre_close, vol FROM etf_daily "
        "WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close", "pre_close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)

    # 若 pre_close 缺失，用前一日的复权收盘兜底（会有拆分问题，但至少不崩）
    df["pre_close"] = df["pre_close"].replace(0, np.nan)
    if df["pre_close"].isna().any():
        df["pre_close"] = df["pre_close"].fillna(df["close"].shift(1))

    # 用 pre_close 计算，跨拆分连续
    df["ret"] = df["close"] / df["pre_close"] - 1.0
    df["gap"] = df["open"] / df["pre_close"] - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df


def add_signal(df: pd.DataFrame, window: int = 10, threshold: float = 0.0) -> pd.DataFrame:
    d = df.copy()
    d["regime"] = d["ret"].rolling(window, min_periods=3).mean()
    d["sig"] = d["regime"] >= threshold
    return d.dropna(subset=["regime"]).reset_index(drop=True)


def run_strict_t1(df: pd.DataFrame, cost: float = 0.0005, slip: float = 0.0,
                  window: int = 10, threshold: float = 0.0) -> dict:
    """【严格 T+1】T日收盘信号 → T+1开盘执行 → 持仓吃全天收益。

    这是唯一真实可实现的回测口径。
    """
    d = add_signal(df, window, threshold)
    n = len(d)

    eq = [1.0]
    pos = 0.0          # 当前持仓（0 或 1）
    entry_price = None  # 若持仓，记录买入价
    trades = 0

    for i in range(1, n):
        # T-1 日的信号决定 T 日开盘的动作
        target = 1.0 if bool(d.loc[i - 1, "sig"]) else 0.0
        o, c = float(d.loc[i, "open"]), float(d.loc[i, "close"])

        prev_eq = eq[-1]

        if target > pos:      # 空仓 → 买入（T 日开盘）
            entry_price = o * (1 + slip)
            prev_eq *= (1 - cost)
            trades += 1
            # 买入当天只吃日内（开盘到收盘）
            prev_eq *= (c / entry_price) if entry_price > 0 else 1.0
        elif target < pos:    # 持仓 → 卖出（T 日开盘）
            sell = o * (1 - slip)
            prev_eq *= (sell / entry_price) if entry_price else 1.0
            prev_eq *= (1 - cost)
            trades += 1
            entry_price = None
        elif target > 0:      # 继续持有：吃全天（用 pre_close 算，跨拆分连续）
            prev_eq *= (1.0 + float(d.loc[i, "ret"]))

        eq.append(prev_eq)
        pos = target

    curve = np.array(eq[1:])
    return {"curve": curve, "trades": trades, "days": n - 1, "df": d}


def run_intraday_idealized(df: pd.DataFrame, cost: float = 0.0005, slip: float = 0.0,
                           window: int = 10, threshold: float = 0.0) -> dict:
    """【理想化口径】HOLD 期只吃日内收益（开盘买、收盘卖）。

    ⚠️ T+1 制度下不可实现！仅用于对照，展示这种口径会虚高多少。
    """
    d = add_signal(df, window, threshold)
    n = len(d)
    eq = [1.0]
    pos = 0.0
    for i in range(1, n):
        target = 1.0 if bool(d.loc[i - 1, "sig"]) else 0.0
        r = float(d.loc[i, "intraday"]) if target > 0 else 0.0
        trade = abs(target - pos)
        eq.append(eq[-1] * (1 + r - trade * (cost + slip)))
        pos = target
    return {"curve": np.array(eq[1:]), "days": n - 1, "df": d}


def buy_hold_curve(df: pd.DataFrame, start_idx: int = 0) -> np.ndarray:
    """买入持有：从 start_idx+1 开盘买入，一直持有。

    用复权日收益（ret = close/pre_close - 1）累乘，避免拆分造成假收益。
    首日用 open->close 的日内收益。
    """
    d = df.reset_index(drop=True)
    eq = [1.0]
    if start_idx + 1 >= len(d):
        return np.array(eq)
    # 首日：开盘买入，吃当天日内
    first = 1.0 + float(d.loc[start_idx + 1, "intraday"])
    eq.append(first)
    # 之后：吃每天的复权日收益
    for i in range(start_idx + 2, len(d)):
        eq.append(eq[-1] * (1.0 + float(d.loc[i, "ret"])))
    return np.array(eq)


def stats(curve: np.ndarray) -> dict:
    if len(curve) < 2:
        return {"total": 0.0, "ann": np.nan, "dd": 0.0, "sharpe": np.nan, "calmar": np.nan}
    r = np.diff(np.concatenate([[1.0], curve]))
    total = float(curve[-1] - 1)
    years = len(curve) / 244
    ann = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    calmar = float(ann / abs(dd)) if dd < 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe, "calmar": calmar}


if __name__ == "__main__":
    # 自检：同一数据上，三种口径的对比
    db = resolve_db()
    for code, name in [("510300.SH", "沪深300ETF"), ("512100.SH", "中证1000ETF")]:
        df = load_etf(db, code)
        if df is None:
            continue
        strict = run_strict_t1(df)
        ideal = run_intraday_idealized(df)
        off = len(df) - len(strict["curve"])
        bh = buy_hold_curve(df, off)

        # 对齐长度
        m = min(len(strict["curve"]), len(ideal["curve"]), len(bh))
        s_s, s_i, s_b = stats(strict["curve"][:m]), stats(ideal["curve"][:m]), stats(bh[:m])

        print("=" * 78)
        print(f"{name}  ({len(df)} 天 ≈ {len(df)/244:.1f}年)")
        print("=" * 78)
        print(f"{'口径':<28}{'累计收益':>13}{'年化':>10}{'最大回撤':>11}{'夏普':>9}")
        print("-" * 78)
        for label, s in [("买入持有", s_b),
                         ("A 理想化（只吃日内，违规）", s_i),
                         ("B 严格 T+1（真实）", s_s)]:
            print(f"{label:<28}{s['total']*100:>+12.1f}%{s['ann']*100:>+9.1f}%"
                  f"{s['dd']*100:>10.1f}%{s['sharpe']:>9.2f}")
        print(f"\n  严格T+1 vs 买入持有: {(s_s['total']-s_b['total'])*100:+.1f}pp")
        print(f"  理想化虚高: {(s_i['total']-s_s['total'])*100:+.1f}pp")
        print(f"  交易次数: {strict['trades']}")
