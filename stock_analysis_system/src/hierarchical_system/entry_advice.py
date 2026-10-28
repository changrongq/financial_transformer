from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd


ENTRY_ADVICE_COLUMNS = [
    "entry_plan",
    "entry_now",
    "entry_trigger_price",
    "entry_buy_zone_low",
    "entry_buy_zone_high",
    "entry_support_price",
    "entry_breakout_price",
    "entry_stop_price",
    "entry_reference_price",
    "entry_current_price",
    "entry_advice_text",
    "intraday_entry_plan",
    "intraday_entry_now",
    "intraday_entry_trigger_price",
    "intraday_entry_buy_price",
    "intraday_entry_chase_limit",
    "intraday_entry_text",
]


def _round_price(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def _query_recent_bars(
    conn: sqlite3.Connection,
    stock_code: str,
    signal_date: pd.Timestamp | None,
    lookback_days: int,
) -> pd.DataFrame:
    sql = (
        "SELECT [日期] AS trade_date, [开盘] AS open_price, [最高] AS high_price, [最低] AS low_price, "
        "[收盘] AS close_price, [换手率] AS turnover_rate "
        "FROM stock_data WHERE stock_code = ? "
    )
    params: list[Any] = [stock_code]
    if signal_date is not None and pd.notna(signal_date):
        sql += "AND [日期] <= ? "
        params.append(signal_date.strftime("%Y-%m-%d"))
    sql += "ORDER BY [日期] DESC LIMIT ?"
    params.append(int(lookback_days))
    frame = pd.read_sql_query(sql, conn, params=params)
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in ["open_price", "high_price", "low_price", "close_price", "turnover_rate"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("trade_date").reset_index(drop=True)


def _build_entry_plan(
    bars: pd.DataFrame,
    *,
    signal_close: float | None,
    current_price: float | None,
) -> dict[str, Any]:
    if bars.empty or len(bars) < 3:
        return {
            "entry_plan": "历史数据不足",
            "entry_now": None,
            "entry_trigger_price": None,
            "entry_buy_zone_low": None,
            "entry_buy_zone_high": None,
            "entry_support_price": None,
            "entry_breakout_price": None,
            "entry_stop_price": None,
            "entry_reference_price": None,
            "entry_current_price": _round_price(current_price),
            "entry_advice_text": "近端日线样本不足，先不要机械追单，至少等 3 到 5 根日线后再判断。",
        }

    recent_5 = bars.tail(min(5, len(bars)))
    recent_10 = bars.tail(min(10, len(bars)))
    latest_close = float(recent_10.iloc[-1]["close_price"])
    signal_close_value = float(signal_close) if signal_close is not None and not pd.isna(signal_close) else latest_close
    current_value = float(current_price) if current_price is not None and not pd.isna(current_price) else signal_close_value

    ma5 = float(recent_5["close_price"].mean())
    ma10 = float(recent_10["close_price"].mean())
    support = float(recent_5["low_price"].min())
    breakout = float(recent_5["high_price"].max())
    reclaim = max(signal_close_value, ma5)
    stop_price = support * 0.98

    if current_value < support * 0.995:
        plan = "跌破支撑，先别接"
        now = "等待止跌"
        text = (
            f"当前价 {current_value:.2f} 已跌破近 5 日支撑 {support:.2f}，先不要接飞刀。"
            f" 至少等重新站回 {reclaim:.2f} 再考虑。"
        )
        zone_low = reclaim
        zone_high = reclaim * 1.01
        trigger = reclaim
    elif current_value <= support * 1.015:
        plan = "靠近支撑，小仓试错"
        now = "可试小仓"
        text = (
            f"当前价 {current_value:.2f} 已靠近近 5 日支撑 {support:.2f}。"
            f" 如果盘中止跌，可在 {support:.2f} ~ {support * 1.01:.2f} 小仓分批试，跌破 {stop_price:.2f} 先撤。"
        )
        zone_low = support
        zone_high = support * 1.01
        trigger = None
    elif current_value < reclaim * 0.995:
        plan = "先等站回 5 日线/信号收盘"
        now = "继续等待"
        text = (
            f"当前价 {current_value:.2f} 仍弱于参考位 {reclaim:.2f}（5 日线/信号收盘更高者）。"
            f" 先等重新站回 {reclaim:.2f} 再考虑买入。"
        )
        zone_low = reclaim
        zone_high = reclaim * 1.01
        trigger = reclaim
    elif current_value >= breakout * 0.995:
        plan = "偏强但别追高，等回踩"
        now = "等回踩再买"
        text = (
            f"当前价 {current_value:.2f} 已接近近 5 日高点 {breakout:.2f}，直接追价性价比偏低。"
            f" 更适合等回踩 {reclaim:.2f} ~ {breakout:.2f} 再分批。"
        )
        zone_low = reclaim
        zone_high = breakout
        trigger = reclaim
    else:
        plan = "回踩不破，可分批买"
        now = "可分批参与"
        text = (
            f"当前价 {current_value:.2f} 已站在参考位 {reclaim:.2f} 上方，趋势不算弱。"
            f" 若回踩 {reclaim * 0.995:.2f} ~ {reclaim * 1.01:.2f} 不破，可分两笔参与。"
        )
        zone_low = reclaim * 0.995
        zone_high = reclaim * 1.01
        trigger = reclaim

    reference_price = max(reclaim, ma10)
    return {
        "entry_plan": plan,
        "entry_now": now,
        "entry_trigger_price": _round_price(trigger),
        "entry_buy_zone_low": _round_price(zone_low),
        "entry_buy_zone_high": _round_price(zone_high),
        "entry_support_price": _round_price(support),
        "entry_breakout_price": _round_price(breakout),
        "entry_stop_price": _round_price(stop_price),
        "entry_reference_price": _round_price(reference_price),
        "entry_current_price": _round_price(current_value),
        "entry_advice_text": text,
    }


def _build_intraday_entry_plan(
    daily_plan: dict[str, Any],
    *,
    current_price: float | None,
    open_price: float | None,
    high_price: float | None,
    low_price: float | None,
    pre_close: float | None,
) -> dict[str, Any]:
    if current_price is None or pd.isna(current_price):
        return {
            "intraday_entry_plan": "盘中价格缺失",
            "intraday_entry_now": None,
            "intraday_entry_trigger_price": None,
            "intraday_entry_buy_price": None,
            "intraday_entry_chase_limit": None,
            "intraday_entry_text": "暂无盘中价格，先按日线买入建议执行。",
        }

    current_value = float(current_price)
    reference = float(daily_plan.get("entry_reference_price") or daily_plan.get("entry_trigger_price") or current_value)
    support = float(daily_plan.get("entry_support_price") or reference)
    breakout = float(daily_plan.get("entry_breakout_price") or reference)
    open_value = float(open_price) if open_price is not None and not pd.isna(open_price) else reference
    pre_close_value = float(pre_close) if pre_close is not None and not pd.isna(pre_close) else reference
    low_value = float(low_price) if low_price is not None and not pd.isna(low_price) else current_value
    high_value = float(high_price) if high_price is not None and not pd.isna(high_price) else current_value

    reclaim_trigger = max(reference, open_value, pre_close_value)
    pullback_buy = max(support * 1.002, min(reference, open_value, pre_close_value))
    chase_limit = max(reclaim_trigger * 1.015, breakout * 0.998)

    if current_value < support * 0.995:
        plan = "盘中弱势，先别抄底"
        now = "继续等"
        text = (
            f"现价 {current_value:.2f} 仍在支撑 {support:.2f} 下方，盘中还是弱。"
            f" 先等重新站回 {reclaim_trigger:.2f}，不要边跌边接。"
        )
        trigger = reclaim_trigger
        buy_price = None
    elif current_value < reclaim_trigger * 0.995:
        if low_value <= support * 1.01 and current_value >= low_value * 1.01:
            plan = "有止跌迹象，等站回再上"
            now = "观察回拉"
            text = (
                f"现价 {current_value:.2f} 虽然从盘中低点 {low_value:.2f} 有回拉，但还没站回关键位 {reclaim_trigger:.2f}。"
                f" 更适合等站回后再买第一笔。"
            )
        else:
            plan = "未站回关键位，继续等"
            now = "继续等"
            text = (
                f"现价 {current_value:.2f} 仍低于盘中关键位 {reclaim_trigger:.2f}。"
                f" 先别急，等重新站回再考虑。"
            )
        trigger = reclaim_trigger
        buy_price = None
    elif current_value > chase_limit:
        plan = "盘中拉高，别追"
        now = "等回踩"
        text = (
            f"现价 {current_value:.2f} 已明显高于可接受追价线 {chase_limit:.2f}。"
            f" 宁可等回踩 {pullback_buy:.2f} 附近，也不要盘中硬追。"
        )
        trigger = pullback_buy
        buy_price = pullback_buy
    else:
        if current_value >= open_value and current_value >= pre_close_value:
            plan = "站回关键位，可轻仓首笔"
            now = "可轻仓试"
            text = (
                f"现价 {current_value:.2f} 已站回开盘/昨收关键位，盘中不算弱。"
                f" 可在 {pullback_buy:.2f} 到 {current_value:.2f} 轻仓首笔，剩余仓位等回踩确认。"
            )
        else:
            plan = "临界站回，回踩确认后再买"
            now = "等确认"
            text = (
                f"现价 {current_value:.2f} 已接近关键位 {reclaim_trigger:.2f}，但盘中强度一般。"
                f" 更适合等回踩不破 {pullback_buy:.2f} 再买。"
            )
        trigger = reclaim_trigger
        buy_price = pullback_buy

    return {
        "intraday_entry_plan": plan,
        "intraday_entry_now": now,
        "intraday_entry_trigger_price": _round_price(trigger),
        "intraday_entry_buy_price": _round_price(buy_price),
        "intraday_entry_chase_limit": _round_price(chase_limit),
        "intraday_entry_text": text,
    }


def add_entry_advice(
    df: pd.DataFrame,
    system_config,
    *,
    signal_date_col: str = "date",
    signal_close_col: str = "close",
    current_price_col: str | None = None,
    open_price_col: str | None = None,
    high_price_col: str | None = None,
    low_price_col: str | None = None,
    pre_close_col: str | None = None,
    lookback_days: int = 10,
) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        for column in ENTRY_ADVICE_COLUMNS:
            result[column] = pd.Series(dtype=object)
        return result

    for column in ENTRY_ADVICE_COLUMNS:
        if column not in result.columns:
            result[column] = None

    signal_dates = pd.to_datetime(result.get(signal_date_col), errors="coerce") if signal_date_col in result.columns else pd.Series([pd.NaT] * len(result), index=result.index)
    signal_closes = pd.to_numeric(result.get(signal_close_col), errors="coerce") if signal_close_col in result.columns else pd.Series([pd.NA] * len(result), index=result.index)
    current_prices = pd.to_numeric(result.get(current_price_col), errors="coerce") if current_price_col and current_price_col in result.columns else pd.Series([pd.NA] * len(result), index=result.index)
    open_prices = pd.to_numeric(result.get(open_price_col), errors="coerce") if open_price_col and open_price_col in result.columns else pd.Series([pd.NA] * len(result), index=result.index)
    high_prices = pd.to_numeric(result.get(high_price_col), errors="coerce") if high_price_col and high_price_col in result.columns else pd.Series([pd.NA] * len(result), index=result.index)
    low_prices = pd.to_numeric(result.get(low_price_col), errors="coerce") if low_price_col and low_price_col in result.columns else pd.Series([pd.NA] * len(result), index=result.index)
    pre_closes = pd.to_numeric(result.get(pre_close_col), errors="coerce") if pre_close_col and pre_close_col in result.columns else pd.Series([pd.NA] * len(result), index=result.index)

    conn = sqlite3.connect(system_config.databases.trade_db)
    try:
        for index, row in result.iterrows():
            stock_code = str(row.get("stock_code", "")).zfill(6)
            if not stock_code or stock_code == "000000":
                continue
            bars = _query_recent_bars(conn, stock_code, signal_dates.loc[index], lookback_days=lookback_days)
            plan = _build_entry_plan(
                bars,
                signal_close=signal_closes.loc[index] if index in signal_closes.index else None,
                current_price=current_prices.loc[index] if index in current_prices.index else None,
            )
            for column, value in plan.items():
                result.at[index, column] = value
            intraday_plan = _build_intraday_entry_plan(
                plan,
                current_price=current_prices.loc[index] if index in current_prices.index else None,
                open_price=open_prices.loc[index] if index in open_prices.index else None,
                high_price=high_prices.loc[index] if index in high_prices.index else None,
                low_price=low_prices.loc[index] if index in low_prices.index else None,
                pre_close=pre_closes.loc[index] if index in pre_closes.index else None,
            )
            for column, value in intraday_plan.items():
                result.at[index, column] = value
    finally:
        conn.close()

    return result