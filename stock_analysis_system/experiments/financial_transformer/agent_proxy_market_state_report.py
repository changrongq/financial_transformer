from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_TRADE_DB = ROOT.parents[2] / "stock_history.db"


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _read_sql(conn: sqlite3.Connection, query: str, params: tuple | None = None) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params or ())


def _load_stock_data(conn: sqlite3.Connection, start_date: str, end_date: str | None) -> pd.DataFrame:
    query = "SELECT * FROM stock_data WHERE [日期] >= ?"
    params: list[str] = [start_date]
    if end_date is not None:
        query += " AND [日期] <= ?"
        params.append(end_date)
    frame = _read_sql(conn, query, tuple(params))
    renamed = frame.rename(
        columns={
            frame.columns[0]: "stock_code",
            frame.columns[1]: "date",
            frame.columns[2]: "open",
            frame.columns[3]: "high",
            frame.columns[4]: "low",
            frame.columns[5]: "close",
            frame.columns[6]: "volume",
            frame.columns[7]: "turnover_rate",
        }
    )
    keep_cols = ["stock_code", "date", "open", "high", "low", "close", "volume", "turnover_rate"]
    renamed = renamed[keep_cols].copy()
    renamed["date"] = pd.to_datetime(renamed["date"])
    for col in ["open", "high", "low", "close", "volume", "turnover_rate"]:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    renamed = renamed.sort_values(["stock_code", "date"])
    renamed["return_1d"] = renamed.groupby("stock_code")["close"].pct_change().fillna(0.0)
    return renamed


def _load_top_inst(conn: sqlite3.Connection, start_date: str, end_date: str | None) -> pd.DataFrame:
    query = "SELECT * FROM stock_top_inst WHERE [日期] >= ?"
    params: list[str] = [start_date]
    if end_date is not None:
        query += " AND [日期] <= ?"
        params.append(end_date)
    frame = _read_sql(conn, query, tuple(params))
    renamed = frame.rename(
        columns={
            frame.columns[0]: "stock_code",
            frame.columns[1]: "date",
            frame.columns[2]: "inst_buy",
            frame.columns[3]: "inst_sell",
            frame.columns[4]: "inst_net",
            frame.columns[5]: "inst_count",
        }
    )
    renamed["date"] = pd.to_datetime(renamed["date"])
    for col in ["inst_buy", "inst_sell", "inst_net", "inst_count"]:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce").fillna(0.0)
    return renamed


def _load_margin(conn: sqlite3.Connection, start_date: str, end_date: str | None) -> pd.DataFrame:
    query = "SELECT * FROM stock_margin_detail WHERE [日期] >= ?"
    params: list[str] = [start_date]
    if end_date is not None:
        query += " AND [日期] <= ?"
        params.append(end_date)
    frame = _read_sql(conn, query, tuple(params))
    renamed = frame.rename(
        columns={
            frame.columns[0]: "stock_code",
            frame.columns[1]: "date",
            frame.columns[2]: "margin_rzye",
            frame.columns[3]: "margin_rqye",
            frame.columns[4]: "margin_rzmre",
            frame.columns[5]: "margin_rqmcl",
            frame.columns[6]: "margin_rzche",
            frame.columns[7]: "margin_rqchl",
            frame.columns[8]: "margin_rqyl",
            frame.columns[9]: "margin_total",
        }
    )
    renamed["date"] = pd.to_datetime(renamed["date"])
    numeric_cols = [
        "margin_rzye",
        "margin_rqye",
        "margin_rzmre",
        "margin_rqmcl",
        "margin_rzche",
        "margin_rqchl",
        "margin_rqyl",
        "margin_total",
    ]
    for col in numeric_cols:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce").fillna(0.0)
    renamed = renamed.sort_values(["stock_code", "date"])
    renamed["margin_rzye_change_ratio"] = renamed.groupby("stock_code")["margin_rzye"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return renamed


def _load_concept_members(conn: sqlite3.Connection) -> pd.DataFrame:
    frame = _read_sql(conn, "SELECT stock_code, concept_id, concept_name FROM stock_concept_members")
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.drop_duplicates()


def _build_institution_proxy(stock_data: pd.DataFrame, top_inst: pd.DataFrame) -> pd.DataFrame:
    market = stock_data.groupby("date").agg(total_stock_count=("stock_code", "nunique"), total_volume=("volume", "sum")).reset_index()
    daily = top_inst.groupby("date").agg(
        inst_stock_count=("stock_code", "nunique"),
        inst_net_sum=("inst_net", "sum"),
        inst_buy_sum=("inst_buy", "sum"),
        inst_sell_sum=("inst_sell", "sum"),
        inst_seat_sum=("inst_count", "sum"),
    ).reset_index()
    daily = market.merge(daily, on="date", how="left").fillna(0.0)
    daily["institution_active_ratio"] = _safe_ratio(daily["inst_stock_count"], daily["total_stock_count"])
    daily["institution_net_buy_balance"] = _safe_ratio(daily["inst_net_sum"], daily["inst_buy_sum"].abs() + daily["inst_sell_sum"].abs())
    daily["institution_seat_intensity"] = _safe_ratio(daily["inst_seat_sum"], daily["inst_stock_count"])
    daily["institution_activity"] = (
        _zscore(daily["institution_active_ratio"]) * 0.4
        + _zscore(daily["institution_net_buy_balance"].abs()) * 0.3
        + _zscore(daily["institution_seat_intensity"]) * 0.3
    )
    return daily[[
        "date",
        "institution_active_ratio",
        "institution_net_buy_balance",
        "institution_seat_intensity",
        "institution_activity",
    ]]


def _build_margin_proxy(stock_data: pd.DataFrame, margin: pd.DataFrame) -> pd.DataFrame:
    market = stock_data.groupby("date").agg(total_stock_count=("stock_code", "nunique"), total_volume=("volume", "sum")).reset_index()
    daily = margin.groupby("date").agg(
        margin_stock_count=("stock_code", "nunique"),
        margin_rzmre_sum=("margin_rzmre", "sum"),
        margin_rzche_sum=("margin_rzche", "sum"),
        margin_rqmcl_sum=("margin_rqmcl", "sum"),
        margin_rqchl_sum=("margin_rqchl", "sum"),
        margin_rzye_change_mean=("margin_rzye_change_ratio", "mean"),
    ).reset_index()
    daily = market.merge(daily, on="date", how="left").fillna(0.0)
    daily["margin_active_ratio"] = _safe_ratio(daily["margin_stock_count"], daily["total_stock_count"])
    daily["margin_rz_net_buy_balance"] = _safe_ratio(daily["margin_rzmre_sum"] - daily["margin_rzche_sum"], daily["margin_rzmre_sum"].abs() + daily["margin_rzche_sum"].abs())
    daily["margin_rq_net_sell_balance"] = _safe_ratio(daily["margin_rqmcl_sum"] - daily["margin_rqchl_sum"], daily["margin_rqmcl_sum"].abs() + daily["margin_rqchl_sum"].abs())
    daily["margin_trend_activity"] = (
        _zscore(daily["margin_active_ratio"]) * 0.35
        + _zscore(daily["margin_rz_net_buy_balance"]) * 0.4
        + _zscore(daily["margin_rzye_change_mean"]) * 0.25
    )
    return daily[[
        "date",
        "margin_active_ratio",
        "margin_rz_net_buy_balance",
        "margin_rq_net_sell_balance",
        "margin_rzye_change_mean",
        "margin_trend_activity",
    ]]


def _build_crowding_proxy(stock_data: pd.DataFrame, concept_members: pd.DataFrame) -> pd.DataFrame:
    merged = stock_data[["stock_code", "date", "return_1d", "turnover_rate"]].merge(concept_members, on="stock_code", how="inner")
    concept_daily = merged.groupby(["date", "concept_id", "concept_name"]).agg(
        concept_stock_count=("stock_code", "nunique"),
        concept_mean_return=("return_1d", "mean"),
        concept_return_std=("return_1d", "std"),
        concept_turnover_mean=("turnover_rate", "mean"),
    ).reset_index()
    concept_daily["concept_return_std"] = concept_daily["concept_return_std"].fillna(0.0)
    concept_daily = concept_daily[concept_daily["concept_stock_count"] >= 3].copy()
    concept_daily["crowding_score"] = concept_daily["concept_mean_return"] * np.log1p(concept_daily["concept_stock_count"]) + concept_daily["concept_turnover_mean"].fillna(0.0) * 0.01
    top_concept = concept_daily.sort_values(["date", "crowding_score"], ascending=[True, False]).groupby("date", as_index=False).head(1)
    daily_market = stock_data.groupby("date").agg(total_stock_count=("stock_code", "nunique")).reset_index()
    top_concept = daily_market.merge(top_concept, on="date", how="left")
    top_concept["concept_stock_count"] = top_concept["concept_stock_count"].fillna(0.0)
    top_concept["concept_mean_return"] = top_concept["concept_mean_return"].fillna(0.0)
    top_concept["concept_turnover_mean"] = top_concept["concept_turnover_mean"].fillna(0.0)
    top_concept["crowding_breadth_ratio"] = _safe_ratio(top_concept["concept_stock_count"], top_concept["total_stock_count"])
    top_concept["crowding_activity"] = (
        _zscore(top_concept["concept_mean_return"]) * 0.45
        + _zscore(top_concept["concept_turnover_mean"]) * 0.25
        + _zscore(top_concept["crowding_breadth_ratio"]) * 0.30
    )
    return top_concept[[
        "date",
        "concept_id",
        "concept_name",
        "concept_stock_count",
        "concept_mean_return",
        "concept_turnover_mean",
        "crowding_breadth_ratio",
        "crowding_activity",
    ]]


def _period_summary(frame: pd.DataFrame, start_date: str, end_date: str | None, cols: list[str]) -> dict[str, float]:
    period = frame[frame["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        period = period[period["date"] <= pd.Timestamp(end_date)]
    return {col: float(period[col].mean()) for col in cols}


def _apply_baseline_zscores(frame: pd.DataFrame, baseline_start: str, baseline_end: str | None, cols: list[str]) -> pd.DataFrame:
    enriched = frame.copy()
    baseline = enriched[enriched["date"] >= pd.Timestamp(baseline_start)].copy()
    if baseline_end is not None:
        baseline = baseline[baseline["date"] <= pd.Timestamp(baseline_end)]
    if baseline.empty:
        for col in cols:
            enriched[f"{col}_apr_z"] = 0.0
        return enriched

    for col in cols:
        baseline_mean = float(baseline[col].mean())
        baseline_std = float(baseline[col].std(ddof=0))
        if baseline_std == 0.0 or np.isnan(baseline_std):
            enriched[f"{col}_apr_z"] = 0.0
        else:
            enriched[f"{col}_apr_z"] = (enriched[col] - baseline_mean) / baseline_std
    return enriched


def _soft_context_components(
    institution_activity_apr_z: float,
    margin_trend_activity_apr_z: float,
    crowding_activity_apr_z: float,
) -> tuple[float, float, float]:
    institution_weakness = float(np.clip(-float(institution_activity_apr_z or 0.0), 0.0, 3.0) / 3.0)
    margin_support = float(np.clip(float(margin_trend_activity_apr_z or 0.0), 0.0, 3.0) / 3.0)
    crowding_pressure = float(np.clip(float(crowding_activity_apr_z or 0.0), 0.0, 3.0) / 3.0)
    return institution_weakness, margin_support, crowding_pressure


def _build_latest_observation_snapshot(frame: pd.DataFrame) -> dict[str, object]:
    latest_row = frame.iloc[-1]
    institution_weakness, margin_support, crowding_pressure = _soft_context_components(
        institution_activity_apr_z=float(latest_row.get("institution_activity_apr_z", 0.0)),
        margin_trend_activity_apr_z=float(latest_row.get("margin_trend_activity_apr_z", 0.0)),
        crowding_activity_apr_z=float(latest_row.get("crowding_activity_apr_z", 0.0)),
    )
    dominant_pressure = max(institution_weakness, margin_support, crowding_pressure)
    if dominant_pressure < 0.20:
        state_name = "neutral"
        state_reason = "proxy_context_near_april_baseline"
    elif institution_weakness >= max(crowding_pressure, margin_support):
        state_name = "soft_defensive_bias"
        state_reason = "weak_institution_proxy_bias"
    elif crowding_pressure >= max(institution_weakness, margin_support):
        state_name = "soft_crowding_bias"
        state_reason = "crowding_proxy_bias"
    else:
        state_name = "soft_supportive_bias"
        state_reason = "margin_support_proxy_bias"

    return {
        "date": str(latest_row["date"].date()),
        "institution_activity": float(latest_row["institution_activity"]),
        "margin_trend_activity": float(latest_row["margin_trend_activity"]),
        "crowding_activity": float(latest_row["crowding_activity"]),
        "institution_activity_apr_z": float(latest_row.get("institution_activity_apr_z", 0.0)),
        "margin_trend_activity_apr_z": float(latest_row.get("margin_trend_activity_apr_z", 0.0)),
        "crowding_activity_apr_z": float(latest_row.get("crowding_activity_apr_z", 0.0)),
        "observation_state": state_name,
        "observation_reason": state_reason,
        "institution_weakness": institution_weakness,
        "margin_support": margin_support,
        "crowding_pressure": crowding_pressure,
        "dominant_pressure": dominant_pressure,
        "top_concept_name": str(latest_row.get("concept_name", "")),
        "top_concept_mean_return": float(latest_row.get("concept_mean_return", 0.0)),
        "top_concept_turnover_mean": float(latest_row.get("concept_turnover_mean", 0.0)),
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Agent Proxy Market State Report",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Trade DB: {payload['trade_db']}",
        f"- Analysis window: {payload['analysis_window']['start_date']} ~ {payload['analysis_window']['end_date']}",
        f"- Comparison periods: {payload['comparison']['baseline_period']['start_date']} ~ {payload['comparison']['baseline_period']['end_date']} vs {payload['comparison']['target_period']['start_date']} ~ {payload['comparison']['target_period']['end_date']}",
        "",
        "## Proxy Delta",
        "",
        "| Proxy | Baseline Mean | Target Mean | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, row in payload["comparison"]["proxy_delta"].items():
        lines.append(f"| {key} | {row['baseline_mean']:.4f} | {row['target_mean']:.4f} | {row['delta']:.4f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    for item in payload["interpretation"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Latest Daily Snapshot",
        "",
        json.dumps(payload["latest_daily_snapshot"], ensure_ascii=False, indent=2),
        "",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal daily agent proxy market state report from the existing trade DB")
    parser.add_argument("--trade-db", default=str(DEFAULT_TRADE_DB))
    parser.add_argument("--start-date", default="2025-01-02")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--baseline-start", default="2026-04-01")
    parser.add_argument("--baseline-end", default="2026-04-30")
    parser.add_argument("--target-start", default="2026-05-01")
    parser.add_argument("--target-end", default=None)
    parser.add_argument("--output-name", default="agent_proxy_market_state_report")
    args = parser.parse_args()

    trade_db = Path(args.trade_db).resolve()
    with sqlite3.connect(trade_db) as conn:
        stock_data = _load_stock_data(conn, args.start_date, args.end_date)
        top_inst = _load_top_inst(conn, args.start_date, args.end_date)
        margin = _load_margin(conn, args.start_date, args.end_date)
        concept_members = _load_concept_members(conn)

    institution = _build_institution_proxy(stock_data, top_inst)
    margin_proxy = _build_margin_proxy(stock_data, margin)
    crowding = _build_crowding_proxy(stock_data, concept_members)

    daily = institution.merge(margin_proxy, on="date", how="outer").merge(crowding, on="date", how="outer").sort_values("date")
    daily = daily.fillna({
        "institution_active_ratio": 0.0,
        "institution_net_buy_balance": 0.0,
        "institution_seat_intensity": 0.0,
        "institution_activity": 0.0,
        "margin_active_ratio": 0.0,
        "margin_rz_net_buy_balance": 0.0,
        "margin_rq_net_sell_balance": 0.0,
        "margin_rzye_change_mean": 0.0,
        "margin_trend_activity": 0.0,
        "concept_id": "",
        "concept_name": "",
        "concept_stock_count": 0.0,
        "concept_mean_return": 0.0,
        "concept_turnover_mean": 0.0,
        "crowding_breadth_ratio": 0.0,
        "crowding_activity": 0.0,
    })

    proxy_cols = ["institution_activity", "margin_trend_activity", "crowding_activity"]
    daily = _apply_baseline_zscores(daily, args.baseline_start, args.baseline_end, proxy_cols)
    baseline_means = _period_summary(daily, args.baseline_start, args.baseline_end, proxy_cols)
    target_means = _period_summary(daily, args.target_start, args.target_end, proxy_cols)
    proxy_delta = {
        key: {
            "baseline_mean": baseline_means[key],
            "target_mean": target_means[key],
            "delta": float(target_means[key] - baseline_means[key]),
        }
        for key in proxy_cols
    }

    interpretation: list[str] = []
    for key, row in proxy_delta.items():
        if abs(row["delta"]) < 0.15:
            interpretation.append(f"{key} 在 4 月与 5 月之间变化较小，暂未形成明显结构突变。")
        elif row["delta"] > 0:
            interpretation.append(f"{key} 在 5 月明显抬升，提示该类 market proxy 活跃度增强。")
        else:
            interpretation.append(f"{key} 在 5 月明显回落，提示该类 market proxy 活跃度减弱。")

    latest_snapshot = _build_latest_observation_snapshot(daily)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_db": str(trade_db),
        "analysis_window": {
            "start_date": str(daily["date"].min().date()),
            "end_date": str(daily["date"].max().date()),
            "row_count": int(len(daily)),
        },
        "comparison": {
            "baseline_period": {"start_date": args.baseline_start, "end_date": args.baseline_end},
            "target_period": {"start_date": args.target_start, "end_date": args.target_end or str(daily["date"].max().date())},
            "proxy_delta": proxy_delta,
        },
        "latest_daily_snapshot": latest_snapshot,
        "interpretation": interpretation,
    }

    output_dir = RESULTS / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "agent_proxy_daily.csv", index=False, encoding="utf-8-sig")
    (output_dir / "agent_proxy_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "agent_proxy_summary.md").write_text(_render_markdown(payload), encoding="utf-8")
    (output_dir / "latest_snapshot.json").write_text(json.dumps(latest_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "daily": str(output_dir / "agent_proxy_daily.csv"),
        "json": str(output_dir / "agent_proxy_summary.json"),
        "markdown": str(output_dir / "agent_proxy_summary.md"),
        "latest_snapshot": str(output_dir / "latest_snapshot.json"),
        "analysis_window": payload["analysis_window"],
        "proxy_delta": proxy_delta,
        "latest_daily_snapshot": latest_snapshot,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()