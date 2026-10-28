from __future__ import annotations

import pandas as pd

from .config import SystemConfig
from .market_cap import fetch_and_store_market_caps
from .trade_data import (
    latest_margin_detail_date,
    latest_moneyflow_date,
    latest_top_inst_date,
    latest_trade_date,
    update_margin_detail_history,
    update_moneyflow_history,
    update_latest_margin_detail_data,
    update_latest_moneyflow_data,
    update_latest_top_inst_data,
    update_latest_trade_data,
    get_stock_concept_refresh_progress,
    update_stock_basic_info,
    update_stock_concepts,
    update_top_inst_history,
    update_trade_history,
)


def _next_date(date_text: str | None) -> str | None:
    if not date_text:
        return None
    return str((pd.Timestamp(date_text) + pd.Timedelta(days=1)).date())


def _previous_date(date_text: str | None) -> str | None:
    if not date_text:
        return None
    return str((pd.Timestamp(date_text) - pd.Timedelta(days=1)).date())


def refresh_latest_data_bundle(
    config: SystemConfig,
    refresh_market_caps: bool = False,
    refresh_existing_market_caps: bool = False,
) -> dict[str, object]:
    trade_before = latest_trade_date(config.databases.trade_db)
    moneyflow_before = latest_moneyflow_date(config.databases.trade_db)
    top_inst_before = latest_top_inst_date(config.databases.trade_db)
    margin_before = latest_margin_detail_date(config.databases.trade_db)

    trade_df = update_latest_trade_data(config)
    trade_date = str(trade_df["日期"].max()) if not trade_df.empty else latest_trade_date(config.databases.trade_db)
    history_end_date = _previous_date(trade_date)

    trade_gap_start = _next_date(trade_before)
    if trade_gap_start and history_end_date and trade_gap_start <= history_end_date:
        update_trade_history(config, start_date=trade_gap_start, end_date=history_end_date)

    moneyflow_gap_start = _next_date(moneyflow_before)
    if moneyflow_gap_start and history_end_date and moneyflow_gap_start <= history_end_date:
        update_moneyflow_history(config, start_date=moneyflow_gap_start, end_date=history_end_date)
    moneyflow_df = update_latest_moneyflow_data(config)

    top_inst_gap_start = _next_date(top_inst_before)
    if top_inst_gap_start and history_end_date and top_inst_gap_start <= history_end_date:
        update_top_inst_history(config, start_date=top_inst_gap_start, end_date=history_end_date)
    top_inst_df = update_latest_top_inst_data(config)

    margin_gap_start = _next_date(margin_before)
    if margin_gap_start and history_end_date and margin_gap_start <= history_end_date:
        update_margin_detail_history(config, start_date=margin_gap_start, end_date=history_end_date)
    margin_df = update_latest_margin_detail_data(config)

    basic_info_df = update_stock_basic_info(config)
    concept_members_df = update_stock_concepts(config)
    concept_progress = get_stock_concept_refresh_progress(config)

    if refresh_market_caps:
        fetch_and_store_market_caps(
            config,
            refresh_existing=refresh_existing_market_caps,
        )

    return {
        "trade_rows": len(trade_df),
        "trade_date": trade_date,
        "moneyflow_rows": len(moneyflow_df),
        "moneyflow_date": str(moneyflow_df["日期"].max()) if not moneyflow_df.empty else None,
        "top_inst_rows": len(top_inst_df),
        "top_inst_date": str(top_inst_df["日期"].max()) if not top_inst_df.empty else None,
        "margin_rows": len(margin_df),
        "margin_date": str(margin_df["日期"].max()) if not margin_df.empty else None,
        "basic_info_rows": len(basic_info_df),
        "concept_member_rows": len(concept_members_df),
        "concept_stock_rows": int(concept_members_df["stock_code"].nunique()) if not concept_members_df.empty else 0,
        "concept_refresh_progress": concept_progress,
        "market_caps_refreshed": bool(refresh_market_caps),
        "market_caps_refresh_existing": bool(refresh_existing_market_caps),
    }