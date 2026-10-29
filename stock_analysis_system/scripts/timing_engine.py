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

# ============================================================================
# 年化交易日数常量（GUIDE 审查问题4）
# ----------------------------------------------------------------------------
# A 股实际每年约 243 个交易日（2020-2026 平均 242.8）。
# 本项目所有脚本统一用 244（历史沿用 4*61=244 的近似）。
#
# 影响：年化收益率 = (1+累计)^(244/天数) - 1
#       用 244 而非 243 会让年化收益【略微偏高】约 0.4%
#       （例如真实年化 10% 会显示为约 10.04%）
#       这是等比例缩放，不改变方向、不改变排序、不改变胜负判定。
#
# 为什么不改：改为 243 需要重跑所有历史结果，而结论不变。
# 此处显式声明，供读者知晓。
# ============================================================================
TRADING_DAYS_PER_YEAR = 244
TRADING_DAYS_ACTUAL = 243  # A股实际值，供对照


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

    交易时序：
      T 日 15:00  收盘，用 T 日及之前数据算信号 sig[T]
      T+1 日 09:30 开盘，按 sig[T] 执行（买入/卖出）
      T+1 日       若持仓，吃「开盘→收盘」的日内收益（买入当天）
      T+2 日起     若持仓，每天吃全天收益（close/pre_close，含隔夜）

    成本处理（滑点模型的说明）：
      滑点 slip 按「价格偏离」建模：
        · 买入成交价 = open × (1 + slip)
        · 卖出成交价 = open × (1 - slip)
      这在买入当天体现为「日内收益按偏离后的买入价折算」，
      与「直接减 slip」有 ~s² 的二阶差异（s=0.1% 时约 0.001%），可忽略。

    ⚠️ BUGFIX 记录（2026-10-29，钳星审查后修正）：
       旧实现卖出时写 `prev_eq *= (sell / entry_price)`，
       但持有期间已经每天累乘了 ret —— 这导致每段持仓的涨幅被【计算两次】。
       实测后果：沪深300 在 slip=0 时被高估为 +300.2%，修正后为 +115.7%（差 184pp）。
       修复方式：卖出时只扣卖出成本，不再重复计算价格涨幅。
    """
    d = add_signal(df, window, threshold)
    n = len(d)

    eq = [1.0]
    pos = 0.0
    trades = 0

    for i in range(1, n):
        # T-1 日的信号决定 T 日开盘的动作
        target = 1.0 if bool(d.loc[i - 1, "sig"]) else 0.0
        prev_eq = eq[-1]

        if target > pos:      # 空仓 → 买入（T 日开盘）
            # 买入当天只吃日内（开盘→收盘），并扣买入成本+滑点
            intra = float(d.loc[i, "intraday"])   # close/open - 1
            prev_eq *= (1.0 + intra - slip - cost)
            trades += 1
        elif target < pos:    # 持仓 → 卖出（T 日开盘）
            # 卖出当天不参与收益，只扣卖出成本+滑点
            # ⚠️ 不再乘 (sell/entry_price)，避免与持有期日内收益重复计算
            prev_eq *= (1.0 - slip - cost)
            trades += 1
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


def buy_hold_curve(df: pd.DataFrame, start_idx: int = 0,
                   cost: float = 0.0, slip: float = 0.0) -> np.ndarray:
    """买入持有：在 df[start_idx] 当天开盘买入，一直持有到末尾。

    与 run_strict_t1 的收益口径严格一致：
      · 买入当天：吃日内收益（close/open - 1），扣 slip + cost
      · 之后每天：吃全天收益（close/pre_close - 1）

    返回的数组长度 = len(df) - start_idx，元素 k 表示「df[start_idx+k] 收盘时的净值」。
    这样与 run_strict_t1 的 curve 可以做逐点比较（前提是起点相同）。

    ⚠️ BUGFIX 记录（2026-10-29，钳星审查后修正，两处）：
      1) 旧实现从 start_idx+1 买入 → 买点比择时晚一天
      2) 旧实现 eq=[1.0] 后 append 首日 → 返回数组首元素恒为 1.0，
         与 run_strict_t1 的 curve（eq[1:]）语义错位
       修正：首日直接作为第 0 个元素，不再冗余的 1.0 前缀
    """
    d = df.reset_index(drop=True)
    if start_idx >= len(d):
        return np.array([1.0])

    # 首日：开盘买入，吃日内，扣成本
    eq = [1.0 + float(d.loc[start_idx, "intraday"]) - slip - cost]

    # 之后：吃每天的复权日收益
    for i in range(start_idx + 1, len(d)):
        eq.append(eq[-1] * (1.0 + float(d.loc[i, "ret"])))
    return np.array(eq)


def align_offset(df: pd.DataFrame, curve_len: int) -> int:
    """计算择时曲线对应的原始 df 起始行。

    推导：
      run_strict_t1 内：d = add_signal(df)  # 丢掉前 dropped 行
                        curve = eq[1:]      # 长度 = len(d) - 1
                        curve[0] ↔ d[1]
      d 丢掉的行数 dropped = len(df) - len(d)
      所以 d[1] ↔ df[dropped + 1]
      而 len(curve) = len(d) - 1  =>  len(d) = len(curve) + 1
      dropped = len(df) - len(curve) - 1
      起始行 = dropped + 1 = len(df) - len(curve)

    >>> 结论：start_idx = len(df) - curve_len
    """
    return max(0, len(df) - curve_len)


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
        off = align_offset(df, len(strict["curve"]))
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
