from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "results" / "typhoon_tracker"
ROLE_SNAPSHOT_ALL = ROOT / "experiments" / "inverse_rl_market_simulation" / "results" / "market_role_snapshot_daily_all" / "market_role_snapshot_with_signals.csv"
ROLE_SNAPSHOT_FALLBACK = ROOT / "experiments" / "inverse_rl_market_simulation" / "results" / "market_role_snapshot_daily" / "market_role_snapshot_with_signals.csv"
DEFAULT_IGNORED_CONCEPT_KEYWORDS = (
    "融资融券",
    "融资标的",
    "融券标的",
    "深股通",
    "沪股通",
    "标普道琼斯",
    "富时罗素",
    "MSCI",
    "转融券",
    "同花顺漂亮",
    "新股与次新股",
    "次新股",
    "年报预增",
    "年报高送转",
)


@dataclass(frozen=True)
class TyphoonParams:
    volume_ratio_threshold: float = 1.5
    amplitude_threshold: float = 0.05
    min_resonance_count: int = 2
    alpha_decay: float = 0.35
    top_n: int = 50
    ignored_concept_keywords: tuple[str, ...] = DEFAULT_IGNORED_CONCEPT_KEYWORDS


@dataclass(frozen=True)
class EmbryoParams:
    o_min: float = 0.55
    o_max: float = 0.78
    trend_lookback: int = 3
    trend_min_slope: float = 0.0
    herd_score_min: float = 0.35
    max_stability_days: int = 4
    top_n: int = 50


def _load_system_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _read_sql(conn: sqlite3.Connection, query: str, params: Iterable[object] = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=tuple(params))


def _latest_trade_date(conn: sqlite3.Connection) -> str:
    value = conn.execute("SELECT MAX([日期]) FROM stock_data WHERE stock_code IS NOT NULL").fetchone()[0]
    if not value:
        raise RuntimeError("stock_data 中没有可用交易日。")
    return str(value)


def _load_price_frame(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    frame = _read_sql(
        conn,
        """
        SELECT s.stock_code,
               s.[日期] AS date,
               s.[开盘] AS open,
               s.[最高] AS high,
               s.[最低] AS low,
               s.[收盘] AS close,
               s.[成交量] AS volume,
               s.[换手率] AS turnover_rate,
               s.[成交额] AS amount,
               r.real_turnover_rate AS real_turnover_rate
        FROM stock_data s
        LEFT JOIN real_turnover_rate r
               ON r.stock_code = s.stock_code AND r.date = s.[日期]
        WHERE s.stock_code IS NOT NULL AND s.[日期] >= ? AND s.[日期] <= ?
        """,
        (start_date, end_date),
    )
    if frame.empty:
        raise RuntimeError(f"stock_data 在 {start_date}~{end_date} 没有可用记录。")
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ["open", "high", "low", "close", "volume", "turnover_rate", "amount"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    # 真实换手率：LEFT JOIN 可能缺值，缺值回退到名义换手率
    frame["real_turnover_rate"] = pd.to_numeric(frame["real_turnover_rate"], errors="coerce")
    frame["real_turnover_rate"] = frame["real_turnover_rate"].fillna(frame["turnover_rate"])
    frame["amount"] = frame["amount"].fillna(frame["close"] * frame["volume"])
    frame = frame.dropna(subset=["date", "close", "high", "low", "volume"]).sort_values(["stock_code", "date"])
    return frame.reset_index(drop=True)


def _load_moneyflow_frame(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        frame = _read_sql(
            conn,
            "SELECT * FROM stock_moneyflow WHERE stock_code IS NOT NULL AND [日期] >= ? AND [日期] <= ?",
            (start_date, end_date),
        )
    except Exception:
        return pd.DataFrame(columns=["stock_code", "date", "main_net_amount"])
    if frame.empty or len(frame.columns) < 18:
        return pd.DataFrame(columns=["stock_code", "date", "main_net_amount"])

    working = pd.DataFrame({"stock_code": frame.iloc[:, 0].astype(str).str.zfill(6), "date": pd.to_datetime(frame.iloc[:, 1])})
    # stock_moneyflow 当前库字段为：小/中/大/特大买卖量额 + 净流入量额。
    # 主力净流入以大单+特大单金额近似，避免依赖不同 tushare 字段名。
    numeric = frame.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if numeric.shape[1] >= 16:
        large_buy_amt = numeric.iloc[:, 9]
        large_sell_amt = numeric.iloc[:, 11]
        xlarge_buy_amt = numeric.iloc[:, 13]
        xlarge_sell_amt = numeric.iloc[:, 15]
        working["main_net_amount"] = (large_buy_amt - large_sell_amt) + (xlarge_buy_amt - xlarge_sell_amt)
    elif numeric.shape[1] >= 18:
        working["main_net_amount"] = numeric.iloc[:, -1]
    else:
        working["main_net_amount"] = 0.0
    return working


def _load_concept_members(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        frame = _read_sql(conn, "SELECT stock_code, concept_id, concept_name FROM stock_concept_members")
    except Exception:
        return pd.DataFrame(columns=["stock_code", "concept_id", "concept_name"])
    if frame.empty:
        return pd.DataFrame(columns=["stock_code", "concept_id", "concept_name"])
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["concept_id"] = frame["concept_id"].astype(str)
    frame["concept_name"] = frame["concept_name"].astype(str)
    return frame.drop_duplicates()


def _load_stock_basic(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        frame = _read_sql(conn, "SELECT * FROM stock_basic_info")
    except Exception:
        return pd.DataFrame(columns=["stock_code", "stock_name", "industry"])
    if frame.empty or len(frame.columns) < 5:
        return pd.DataFrame(columns=["stock_code", "stock_name", "industry"])
    working = pd.DataFrame(
        {
            "stock_code": frame.iloc[:, 0].astype(str).str.zfill(6),
            "stock_name": frame.iloc[:, 2].astype(str),
            "industry": frame.iloc[:, 4].astype(str),
        }
    )
    return working.drop_duplicates("stock_code", keep="last")


def _load_role_snapshot(end_date: pd.Timestamp | None) -> pd.DataFrame:
    for path in [ROLE_SNAPSHOT_ALL, ROLE_SNAPSHOT_FALLBACK]:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"stock_code": str})
        if frame.empty or "stock_code" not in frame.columns:
            continue
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
        date_col = "signal_date" if "signal_date" in frame.columns else "scan_date" if "scan_date" in frame.columns else None
        if date_col:
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
            if end_date is not None:
                matched = frame[frame[date_col] <= end_date].copy()
                if not matched.empty:
                    frame = matched
            frame = frame.sort_values(["stock_code", date_col]).drop_duplicates("stock_code", keep="last")
        keep = [col for col in ["stock_code", "dominant_role", "herd_act", "slow_act", "trend_act", "scan_date"] if col in frame.columns]
        return frame[keep]
    return pd.DataFrame(columns=["stock_code", "dominant_role"])


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def _trailing_true_streak(values: pd.Series) -> int:
    streak = 0
    for value in reversed(values.astype(bool).tolist()):
        if not value:
            break
        streak += 1
    return streak


def _rating(score: float) -> str:
    if score >= 0.8:
        return "台风成形"
    if score >= 0.6:
        return "热带风暴"
    if score >= 0.4:
        return "热带低压"
    return "热带扰动"


def _action(row: pd.Series) -> str:
    role = str(row.get("dominant_role") or "")
    if row["organization_score"] >= 0.8 and row["structure_stability_days"] >= 2 and role == "herd":
        return "协同试验候选：仍需人工确认治理/新闻"
    if row["organization_score"] >= 0.8:
        return "强跟踪：等herd/治理确认"
    if row["organization_score"] >= 0.6:
        return "积极跟踪：准备仓位预案"
    if row["organization_score"] >= 0.4:
        return "编号跟踪：等待增强"
    return "观察：可能消散"


def _find_inflection_ratio(values: list[float]) -> float:
    """计算最近几个值中下降所占比例：返回 0 表示全上升，1 表示全下降。"""
    n = len(values)
    if n < 2:
        return 0.0
    down_count = sum(1 for i in range(1, n) if values[i] < values[i - 1])
    return down_count / (n - 1)


def _linregress(x: list[float], y: list[float]) -> tuple[float, float, float]:
    if len(x) < 2 or len(set(y)) == 1:
        return 0.0, 0.0, 999.0
    result = stats.linregress(x, y)
    return result.slope, result.intercept, result.pvalue


def _embryo_action(row: pd.Series) -> str:
    role = str(row.get("dominant_role") or "")
    if row.get("o_rising") and role == "herd":
        return "胚胎确认：herd成形中，可评估协同试验"
    if row.get("o_rising") and float(row.get("herd_to_slow_ratio", 0.0)) > 0.35:
        return "胚胎中期：herd加速涌入，积极关注"
    if row.get("o_rising"):
        return "胚胎初期：O上升，等待角色确认"
    return "胚胎待定：等待O趋势转正"


def _add_price_features(frame: pd.DataFrame, moneyflow: pd.DataFrame, params: TyphoonParams) -> pd.DataFrame:
    working = frame.merge(moneyflow, on=["stock_code", "date"], how="left")
    working["main_net_amount"] = pd.to_numeric(working["main_net_amount"], errors="coerce").fillna(0.0)
    grouped = working.groupby("stock_code", group_keys=False)
    working["return_1d"] = grouped["close"].pct_change().fillna(0.0)
    working["ma5"] = grouped["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    working["ma10"] = grouped["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    working["ma20"] = grouped["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    working["volume_ma5"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    working["volume_ma20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    working["amplitude"] = _safe_ratio(working["high"] - working["low"], working["close"])
    working["amplitude_ma5"] = grouped["amplitude"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    working["amplitude_ma20"] = grouped["amplitude"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    working["main_net_5d"] = grouped["main_net_amount"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    working["prev_volume"] = grouped["volume"].shift(1)
    working["volume_acceleration"] = _safe_ratio(working["volume"] - working["prev_volume"], working["prev_volume"])
    working["close_to_high"] = _safe_ratio(working["close"], working["high"]).clip(0.0, 1.0)
    working["is_limit_like"] = working["return_1d"] >= 0.095
    working["I1_volume_surge"] = working["volume_ma5"] > working["volume_ma20"] * float(params.volume_ratio_threshold)
    working["I2_uptrend"] = (working["close"] > working["ma5"]) & (working["close"] > working["ma10"])
    working["I3_volatility_expansion"] = (working["amplitude_ma5"] > float(params.amplitude_threshold)) & (working["amplitude_ma5"] > working["amplitude_ma20"])
    working["I4_main_money_inflow"] = working["main_net_5d"] > 0.0
    working["base_qualifies"] = working[["I1_volume_surge", "I2_uptrend", "I3_volatility_expansion", "I4_main_money_inflow"]].all(axis=1)
    return working


def _filter_concepts(concept_members: pd.DataFrame, ignored_keywords: tuple[str, ...]) -> pd.DataFrame:
    if concept_members.empty or not ignored_keywords:
        return concept_members
    pattern = "|".join([re.escape(keyword) for keyword in ignored_keywords if keyword])
    if not pattern:
        return concept_members
    return concept_members.loc[~concept_members["concept_name"].str.contains(pattern, case=False, na=False, regex=True)].copy()


def _add_resonance_features(frame: pd.DataFrame, concept_members: pd.DataFrame, min_resonance_count: int, ignored_keywords: tuple[str, ...]) -> pd.DataFrame:
    working = frame.copy()
    concept_members = _filter_concepts(concept_members, ignored_keywords)
    if concept_members.empty:
        working["resonance_count"] = 0
        working["resonance_delta"] = 0
        working["primary_concept_name"] = ""
        working["I5_concept_resonance"] = False
        working["numbered_depression"] = False
        return working

    base = working.loc[working["base_qualifies"], ["date", "stock_code"]].merge(concept_members, on="stock_code", how="inner")
    if base.empty:
        working["resonance_count"] = 0
        working["resonance_delta"] = 0
        working["primary_concept_name"] = ""
        working["I5_concept_resonance"] = False
        working["numbered_depression"] = False
        return working

    concept_counts = base.groupby(["date", "concept_id", "concept_name"]).agg(resonance_count=("stock_code", "nunique")).reset_index()
    stock_concepts = working[["date", "stock_code"]].merge(concept_members, on="stock_code", how="left")
    stock_counts = stock_concepts.merge(concept_counts, on=["date", "concept_id", "concept_name"], how="left")
    stock_counts["resonance_count"] = stock_counts["resonance_count"].fillna(0).astype(int)
    stock_counts = stock_counts.sort_values(["date", "stock_code", "resonance_count"], ascending=[True, True, False])
    best = stock_counts.drop_duplicates(["date", "stock_code"], keep="first")[["date", "stock_code", "concept_name", "resonance_count"]]
    best = best.rename(columns={"concept_name": "primary_concept_name"})
    working = working.merge(best, on=["date", "stock_code"], how="left")
    working["resonance_count"] = working["resonance_count"].fillna(0).astype(int)
    working["primary_concept_name"] = working["primary_concept_name"].fillna("")
    working["resonance_delta"] = working.groupby("stock_code")["resonance_count"].diff().fillna(0).astype(int)
    working["I5_concept_resonance"] = working["resonance_count"] >= int(min_resonance_count)
    working["numbered_depression"] = working[["I1_volume_surge", "I2_uptrend", "I3_volatility_expansion", "I4_main_money_inflow", "I5_concept_resonance"]].all(axis=1)
    return working


def _add_thermo_score(frame: pd.DataFrame, alpha_decay: float) -> pd.DataFrame:
    working = frame.copy()
    market_amount = working.groupby("date")["amount"].sum().rename("market_amount")
    working = working.merge(market_amount, on="date", how="left")
    liquidity_share = _safe_ratio(working["amount"], working["market_amount"])
    # 换手率：优先用真实换手率（修正股权结构污染），缺失时回退名义换手率
    turn_col = "real_turnover_rate" if "real_turnover_rate" in working.columns else "turnover_rate"
    turnover = (pd.to_numeric(working[turn_col], errors="coerce").fillna(0.0) / 100.0).clip(lower=0.0)
    sigma = working["amplitude"].clip(lower=0.0)
    demand = _safe_ratio(working["main_net_amount"], working["amount"].abs()).clip(-1.0, 1.0)
    working["energy_injection_q"] = liquidity_share * turnover * sigma * demand
    positive_q = working["energy_injection_q"].clip(lower=0.0)
    date_max = positive_q.groupby(working["date"]).transform("max").replace(0.0, np.nan)
    working["energy_injection_norm"] = positive_q.div(date_max).fillna(0.0).clip(0.0, 1.0)

    scores: list[pd.Series] = []
    for _, group in working.sort_values(["stock_code", "date"]).groupby("stock_code"):
        current = 0.0
        values = []
        for q in group["energy_injection_norm"].fillna(0.0).tolist():
            current = float(np.clip((1.0 - alpha_decay) * current + alpha_decay * q, 0.0, 1.0))
            values.append(current)
        scores.append(pd.Series(values, index=group.index))
    working["thermo_score"] = pd.concat(scores).sort_index() if scores else 0.0
    return working


def build_typhoon_snapshot(config_path: Path, end_date: str | None, lookback_calendar_days: int, params: TyphoonParams) -> tuple[pd.DataFrame, dict]:
    config = _load_system_config(config_path)
    trade_db = _resolve_path(config["databases"]["trade_db"], config_path.parent)
    with sqlite3.connect(str(trade_db)) as conn:
        resolved_end_date = end_date or _latest_trade_date(conn)
        start_date = (pd.Timestamp(resolved_end_date) - timedelta(days=lookback_calendar_days)).strftime("%Y-%m-%d")
        price = _load_price_frame(conn, start_date, resolved_end_date)
        moneyflow = _load_moneyflow_frame(conn, start_date, resolved_end_date)
        concepts = _load_concept_members(conn)
        basic = _load_stock_basic(conn)

    enriched = _add_price_features(price, moneyflow, params)
    enriched = _add_resonance_features(enriched, concepts, params.min_resonance_count, params.ignored_concept_keywords)
    enriched = _add_thermo_score(enriched, params.alpha_decay)

    enriched = enriched.sort_values(["stock_code", "date"])
    streaks = enriched.groupby("stock_code")["numbered_depression"].apply(_trailing_true_streak).rename("structure_stability_days")
    latest_date = enriched["date"].max()
    latest = enriched.loc[enriched["date"] == latest_date].copy().merge(streaks, on="stock_code", how="left")
    latest["structure_stability_days"] = latest["structure_stability_days"].fillna(0).astype(int)

    latest["A_norm"] = _clip01(latest["volume_acceleration"] / 2.0)
    latest["C_norm"] = _clip01(latest["close_to_high"])
    latest["R_norm"] = _clip01(latest["resonance_count"] / 5.0)
    latest["S_norm"] = _clip01(latest["structure_stability_days"] / 5.0)
    latest["E_norm"] = np.where(latest["is_limit_like"], latest["close_to_high"], 0.0)
    latest["organization_score"] = (
        0.25 * latest["A_norm"]
        + 0.20 * latest["C_norm"]
        + 0.20 * latest["R_norm"]
        + 0.20 * latest["S_norm"]
        + 0.15 * latest["E_norm"]
    ).clip(0.0, 1.0)
    latest["rating"] = latest["organization_score"].map(_rating)

    roles = _load_role_snapshot(latest_date)
    if not roles.empty:
        latest = latest.merge(roles, on="stock_code", how="left")
    else:
        latest["dominant_role"] = ""
    latest["dominant_role"] = latest["dominant_role"].fillna("")
    latest["action"] = latest.apply(_action, axis=1)
    latest["typhoon_id"] = [f"TYPH-{latest_date.strftime('%Y%m%d')}-{i:03d}" for i in range(1, len(latest) + 1)]

    candidates = latest.loc[latest["numbered_depression"] | (latest["organization_score"] >= 0.4)].copy()
    if not basic.empty:
        candidates = candidates.merge(basic, on="stock_code", how="left")
    candidates = candidates.sort_values(["organization_score", "thermo_score", "resonance_count"], ascending=[False, False, False]).head(params.top_n)
    candidates["typhoon_id"] = [f"TYPH-{latest_date.strftime('%Y%m%d')}-{i:03d}" for i in range(1, len(candidates) + 1)]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_db": str(trade_db),
        "start_date": start_date,
        "end_date": latest_date.strftime("%Y-%m-%d"),
        "stock_count": int(latest["stock_code"].nunique()),
        "numbered_depression_count": int(latest["numbered_depression"].sum()),
        "tracked_count": int(len(candidates)),
        "params": asdict(params),
        "ignored_concept_keywords": list(params.ignored_concept_keywords),
    }
    return candidates, summary


    embryo_summary["incubation_count"] = 0
    incubation = pd.DataFrame()
    if not embryo_candidates.empty:
        try:
            incubation = build_incubation_layer(enriched, embryo_candidates, basic, embryo_params)
            embryo_summary["incubation_count"] = int(len(incubation))
        except Exception:
            pass

    return embryo_candidates, embryo_summary, incubation


def _write_embryo_outputs(candidates: pd.DataFrame, summary: dict, output_dir: Path, incubation_candidates: pd.DataFrame | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = summary["end_date"]
    csv_path = output_dir / f"typhoon_embryo_tracker_{date_tag}.csv"
    json_path = output_dir / f"typhoon_embryo_tracker_{date_tag}_summary.json"
    md_path = output_dir / f"typhoon_embryo_tracker_{date_tag}.md"

    export_cols = [
        "typhoon_id",
        "stock_code",
        "stock_name",
        "industry",
        "organization_score",
        "o_slope",
        "o_rising",
        "o_values",
        "thermo_score",
        "dominant_role",
        "herd_to_slow_ratio",
        "embryo_score",
        "action",
        "primary_concept_name",
        "resonance_count",
        "structure_stability_days",
        "close",
        "return_1d",
        "volume_acceleration",
    ]
    export_cols = [col for col in export_cols if col in candidates.columns]
    candidates[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {**summary, "top": candidates[export_cols].head(20).to_dict(orient="records")}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 股票台风胚胎追踪日报",
        "",
        f"- 交易日：{summary['end_date']}",
        f"- 扫描股票数：{summary['stock_count']}",
        f"- 胚胎合格数 (O ∈ [0.55,0.78] + dO/dt>0)：{summary['embryo_eligible_count']}",
        f"- 胚胎终选数 (O<0.8 + 角色 OK + 稳定 ≤4d)：{summary['embryo_selected_count']}",
        "",
        "## 胚胎跟踪池",
        "",
        "| 编号 | 股票 | O值 | O斜率 | 角色 | herding | 胚胎评分 | 行动 | 共振概念 | 稳定天数 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for _, row in candidates.head(20).iterrows():
        stock_label = row["stock_code"] if not row.get("stock_name") else f"{row['stock_code']} {row['stock_name']}"
        lines.append(
            f"| {row['typhoon_id']} | {stock_label} | "
            f"{float(row['organization_score']):.3f} | {float(row.get('o_slope', 0.0)):.4f} | "
            f"{row.get('dominant_role', '') or 'NA'} | {float(row.get('herd_to_slow_ratio', 0.0)):.3f} | "
            f"{float(row.get('embryo_score', 0.0)):.3f} | {row['action']} | "
            f"{row.get('primary_concept_name', '')} | {int(row.get('structure_stability_days', 0))} |"
        )
    if incubation_candidates is not None and not incubation_candidates.empty:
        lines.append("")
        lines.append("## 孵化中跟踪池 (v0.3.1)")
        lines.append("")
        lines.append(
            f"- 孵化中数量：`{len(incubation_candidates)}`"
        )
        lines.append("")
        lines.append("| 编号 | 股票 | O值 | O斜率 | 角色 | 状态 | 行动 | 共振概念 | 稳定天数 |")
        lines.append("| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |")
        for _, row in incubation_candidates.head(10).iterrows():
            stock_label = row["stock_code"] if not row.get("stock_name") else f"{row['stock_code']} {row['stock_name']}"
            lines.append(
                f"| {row['typhoon_id']} | {stock_label} | "
                f"{float(row['organization_score']):.3f} | {float(row.get('o_slope', 0.0)):.4f} | "
                f"{row.get('dominant_role', '') or 'NA'} | "
                f"{row.get('incubation_status', '')} | {row.get('incubation_action', '')} | "
                f"{row.get('primary_concept_name', '')} | {int(row.get('structure_stability_days', 0))} |"
            )
    # ---- 胚胎口径说明 ----
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 此表是 v0.3.1 台风胚胎追踪逻辑的输出，目标是捕捉「正在形成」而不是「已经成形」的台风。",
            "- 胚胎合格：O ∈ [0.55, 0.78] + dO/dt > 0 (线性回归斜率为正) + p值 < 0.35。",
            "- 胚胎终选：O<0.8 + 角色=herd 或 (slow + herd/slow>0.35) + 结构稳定性 ≤4 天。",
            "- 孵化中 (v0.3.1)：昨日在胚胎列表 + (今日 O ∈ (0.78,0.80) 且 herd) 或 今日涨停。",
            "- 没有买入建议；仍需人工确认治理结论、市场环境与重大新闻。",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def build_embryo_tracker(
    config_path: Path,
    end_date: str | None,
    lookback_calendar_days: int,
    params: TyphoonParams,
    embryo_params: EmbryoParams,
) -> tuple[pd.DataFrame, dict]:
    config = _load_system_config(config_path)
    trade_db = _resolve_path(config["databases"]["trade_db"], config_path.parent)
    with sqlite3.connect(str(trade_db)) as conn:
        resolved_end_date = end_date or _latest_trade_date(conn)
        start_date = (pd.Timestamp(resolved_end_date) - timedelta(days=lookback_calendar_days)).strftime("%Y-%m-%d")
        price = _load_price_frame(conn, start_date, resolved_end_date)
        moneyflow = _load_moneyflow_frame(conn, start_date, resolved_end_date)
        concepts = _load_concept_members(conn)
        basic = _load_stock_basic(conn)

    enriched = _add_price_features(price, moneyflow, params)
    enriched = _add_resonance_features(enriched, concepts, params.min_resonance_count, params.ignored_concept_keywords)
    enriched = _add_thermo_score(enriched, params.alpha_decay)
    enriched = enriched.sort_values(["stock_code", "date"])

    # ---- 计算每日 O 值（清除可能已存在的旧列） ----
    enriched = enriched.drop(columns=["organization_score", "rating", "structure_stability_days", "A_norm", "C_norm", "R_norm", "S_norm", "E_norm"], errors="ignore")
    enriched["structure_stability_days"] = enriched.groupby("stock_code")["numbered_depression"].transform(_trailing_true_streak)
    enriched["A_norm"] = _clip01(enriched["volume_acceleration"] / 2.0)
    enriched["C_norm"] = _clip01(enriched["close_to_high"])
    enriched["R_norm"] = _clip01(enriched["resonance_count"] / 5.0)
    enriched["S_norm"] = _clip01(enriched["structure_stability_days"] / 5.0)
    enriched["E_norm"] = np.where(enriched["is_limit_like"], enriched["close_to_high"], 0.0)
    enriched["organization_score"] = (
        0.25 * enriched["A_norm"]
        + 0.20 * enriched["C_norm"]
        + 0.20 * enriched["R_norm"]
        + 0.20 * enriched["S_norm"]
        + 0.15 * enriched["E_norm"]
    ).clip(0.0, 1.0)
    enriched["rating"] = enriched["organization_score"].map(_rating)

    def _embed_trend_and_slope(series: pd.Series) -> pd.Series:
        vals = series.dropna().tolist()
        recent = vals[-30:] if len(vals) > 30 else vals
        if len(recent) < 2:
            return pd.Series({"o_rising": False, "o_slope": 0.0, "o_trend_pvalue": 999.0, "o_values": str(vals[-3:]) if vals else "[]"})
        tail = recent[-embryo_params.trend_lookback:]
        if len(tail) < 2:
            tail = recent[-2:]
        x = list(range(len(tail)))
        slope, _, pvalue = _linregress(x, tail)
        if slope is None:
            slope = 0.0
        return pd.Series({
            "o_rising": slope > embryo_params.trend_min_slope,
            "o_slope": slope,
            "o_trend_pvalue": pvalue,
            "o_values": str(tail[:5]) if tail else "[]",
        })

    trend_stats = enriched.groupby("stock_code")["organization_score"].apply(_embed_trend_and_slope).reset_index()
    # groupby.apply 对 Series 输出做了透视；需要 pivot 回正常格式
    trend_stats = trend_stats.pivot(index="stock_code", columns="level_1", values="organization_score").reset_index()
    trend_stats.columns.name = None
    for col in ["o_rising", "o_slope", "o_trend_pvalue", "o_values"]:
        if col not in trend_stats.columns:
            trend_stats[col] = False if col == "o_rising" else ("[]" if col == "o_values" else 0.0)
    trend_stats["o_rising"] = trend_stats["o_rising"].astype(bool)
    trend_stats["o_slope"] = pd.to_numeric(trend_stats["o_slope"], errors="coerce").fillna(0.0)
    trend_stats["o_trend_pvalue"] = pd.to_numeric(trend_stats["o_trend_pvalue"], errors="coerce").fillna(999.0)
    latest_date = enriched["date"].max()
    latest = enriched.loc[enriched["date"] == latest_date].copy()
    latest = latest.merge(trend_stats, on="stock_code", how="left")
    if "organization_score" not in latest.columns and "organization_score_x" in latest.columns:
        latest.rename(columns={"organization_score_x": "organization_score"}, inplace=True)
    for col in ["o_rising", "o_slope", "o_trend_pvalue", "o_values"]:
        if col not in latest.columns:
            latest[col] = (False if col == "o_rising" else 0.0) if col != "o_values" else "[]"
    latest["o_rising"] = latest["o_rising"].fillna(False).astype(bool)
    latest["o_slope"] = latest["o_slope"].fillna(0.0)
    latest["o_trend_pvalue"] = latest["o_trend_pvalue"].fillna(999.0)
    latest["o_values"] = latest["o_values"].fillna("[]")

    # ---- 胚胎标志：O ∈ [min, max] 且 dO/dt 为正 ----
    o_min_val = float(embryo_params.o_min)
    o_max_val = float(embryo_params.o_max)
    latest["embryo_eligible"] = (
        latest["organization_score"].between(o_min_val, o_max_val, inclusive="both")
        & latest["o_rising"]
    )

    # 趋势/斜率字段已由 enriched_tmp 的 merge 携带，此处不再重复计算。

    # ---- 角色数据 ----
    roles = _load_role_snapshot(latest_date)
    if not roles.empty:
        latest = latest.merge(roles, on="stock_code", how="left")
    else:
        latest["dominant_role"] = ""
        latest["herd_act"] = 0.0
        latest["slow_act"] = 0.0
    latest["dominant_role"] = latest["dominant_role"].fillna("")
    latest["herd_act"] = pd.to_numeric(latest.get("herd_act", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    latest["slow_act"] = pd.to_numeric(latest.get("slow_act", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    latest["herd_to_slow_ratio"] = _safe_ratio(latest["herd_act"], latest["slow_act"])

    # ---- 胚胎终选过滤 ----
    latest["herd_score_check"] = latest["herd_to_slow_ratio"] > float(embryo_params.herd_score_min)
    latest["role_ok"] = (latest["dominant_role"] == "herd") | (
        (latest["dominant_role"] == "slow") & latest["herd_score_check"]
    )
    latest["stability_ok"] = latest["structure_stability_days"] <= int(embryo_params.max_stability_days)

    latest["embryo_selected"] = (
        latest["embryo_eligible"]
        & latest["role_ok"]
        & latest["stability_ok"]
        & (latest["organization_score"] < 0.80)
    )

    embryo_candidates = latest.loc[latest["embryo_selected"]].copy()
    embryo_candidates = embryo_candidates.drop_duplicates(subset=["stock_code"], keep="first")
    if not basic.empty:
        embryo_candidates = embryo_candidates.merge(basic, on="stock_code", how="left")

    # 胚胎综合评分
    embryo_candidates["embryo_score_norm"] = _clip01(
        (embryo_candidates["organization_score"] - float(embryo_params.o_min))
        / max(float(embryo_params.o_max) - float(embryo_params.o_min), 0.01)
    )
    embryo_candidates["trend_score_norm"] = _clip01(embryo_candidates["o_slope"] / 0.05)
    embryo_candidates["role_speed_norm"] = _clip01(embryo_candidates["herd_to_slow_ratio"])
    embryo_candidates["embryo_score"] = (
        0.35 * embryo_candidates["embryo_score_norm"]
        + 0.35 * embryo_candidates["trend_score_norm"]
        + 0.30 * embryo_candidates["role_speed_norm"]
    ).clip(0.0, 1.0)

    embryo_candidates = embryo_candidates.sort_values("embryo_score", ascending=False).head(embryo_params.top_n)
    embryo_candidates["typhoon_id"] = [f"EMBRYO-{latest_date.strftime('%Y%m%d')}-{i:03d}" for i in range(1, len(embryo_candidates) + 1)]
    embryo_candidates["action"] = embryo_candidates.apply(_embryo_action, axis=1)

    embryo_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_db": str(trade_db),
        "start_date": start_date,
        "end_date": latest_date.strftime("%Y-%m-%d"),
        "stock_count": int(latest["stock_code"].nunique()),
        "embryo_eligible_count": int(latest["embryo_eligible"].sum()),
        "embryo_selected_count": int(len(embryo_candidates)),
        "incubation_count": 0,
        "params": asdict(params),
        "embryo_params": asdict(embryo_params),
        "ignored_concept_keywords": list(params.ignored_concept_keywords),
    }
    return embryo_candidates, embryo_summary, pd.DataFrame()


def _write_embryo_outputs(candidates: pd.DataFrame, summary: dict, output_dir: Path, incubation_candidates: pd.DataFrame | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = summary["end_date"]
    csv_path = output_dir / f"typhoon_embryo_tracker_{date_tag}.csv"
    json_path = output_dir / f"typhoon_embryo_tracker_{date_tag}_summary.json"
    md_path = output_dir / f"typhoon_embryo_tracker_{date_tag}.md"

    export_cols = [
        "typhoon_id",
        "stock_code",
        "stock_name",
        "industry",
        "organization_score",
        "o_slope",
        "o_rising",
        "o_values",
        "thermo_score",
        "dominant_role",
        "herd_to_slow_ratio",
        "embryo_score",
        "action",
        "primary_concept_name",
        "resonance_count",
        "structure_stability_days",
        "close",
        "return_1d",
        "volume_acceleration",
    ]
    export_cols = [col for col in export_cols if col in candidates.columns]
    candidates[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {**summary, "top": candidates[export_cols].head(20).to_dict(orient="records")}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 股票台风胚胎追踪日报",
        "",
        f"- 交易日：{summary['end_date']}",
        f"- 扫描股票数：{summary['stock_count']}",
        f"- 胚胎合格数 (O ∈ [0.55,0.78] + dO/dt>0)：{summary['embryo_eligible_count']}",
        f"- 胚胎终选数 (O<0.8 + 角色 OK + 稳定 ≤4d)：{summary['embryo_selected_count']}",
        f"- 孵化中数 (v0.3.1)：{summary.get('incubation_count', 0)}",
        "",
        "## 胚胎跟踪池",
        "",
        "| 编号 | 股票 | O值 | O斜率 | 角色 | herding | 胚胎评分 | 行动 | 共振概念 | 稳定天数 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for _, row in candidates.head(20).iterrows():
        stock_label = row["stock_code"] if not row.get("stock_name") else f"{row['stock_code']} {row['stock_name']}"
        lines.append(
            f"| {row['typhoon_id']} | {stock_label} | "
            f"{float(row['organization_score']):.3f} | {float(row.get('o_slope', 0.0)):.4f} | "
            f"{row.get('dominant_role', '') or 'NA'} | {float(row.get('herd_to_slow_ratio', 0.0)):.3f} | "
            f"{float(row.get('embryo_score', 0.0)):.3f} | {row['action']} | "
            f"{row.get('primary_concept_name', '')} | {int(row.get('structure_stability_days', 0))} |"
        )
    if incubation_candidates is not None and not incubation_candidates.empty:
        lines.append("")
        lines.append("## 孵化中跟踪池 (v0.3.1)")
        lines.append("")
        lines.append(f"- 孵化中数量：`{len(incubation_candidates)}`")
        lines.append("")
        lines.append("| 编号 | 股票 | O值 | O斜率 | 角色 | 状态 | 行动 | 共振概念 | 稳定天数 |")
        lines.append("| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |")
        for _, row in incubation_candidates.head(10).iterrows():
            stock_label = row["stock_code"] if not row.get("stock_name") else f"{row['stock_code']} {row['stock_name']}"
            lines.append(
                f"| {row['typhoon_id']} | {stock_label} | "
                f"{float(row['organization_score']):.3f} | {float(row.get('o_slope', 0.0)):.4f} | "
                f"{row.get('dominant_role', '') or 'NA'} | "
                f"{row.get('incubation_status', '')} | {row.get('incubation_action', '')} | "
                f"{row.get('primary_concept_name', '')} | {int(row.get('structure_stability_days', 0))} |"
            )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 此表是 v0.3.1 台风胚胎追踪逻辑的输出，目标是捕捉「正在形成」而不是「已经成形」的台风。",
            "- 胚胎合格：O ∈ [0.55, 0.78] + dO/dt > 0 (线性回归斜率为正) + p值 < 0.35。",
            "- 胚胎终选：O<0.8 + 角色=herd 或 (slow + herd/slow>0.35) + 结构稳定性 ≤4 天。",
            "- 孵化中 (v0.3.1)：昨日在胚胎列表 + (今日 O ∈ (0.78,0.80) 且 herd) 或 今日涨停。",
            "- 没有买入建议；仍需人工确认治理结论、市场环境与重大新闻。",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def build_incubation_layer(
    enriched: pd.DataFrame,
    embryo_candidates: pd.DataFrame,
    basic: pd.DataFrame,
    embryo_params: EmbryoParams,
    lookback_calendar_days: int = 140,
) -> pd.DataFrame:
    all_dates = sorted(enriched["date"].dropna().unique())
    if len(all_dates) < 2:
        return pd.DataFrame(columns=["typhoon_id", "stock_code", "stock_name", "organization_score", "o_slope", "dominant_role", "incubation_status", "incubation_action", "primary_concept_name", "structure_stability_days"])

    today = all_dates[-1]
    yesterday = all_dates[-2]

    enriched = enriched.copy()
    enriched = enriched.drop(columns=["organization_score", "rating", "structure_stability_days", "A_norm", "C_norm", "R_norm", "S_norm", "E_norm", "o_rising", "o_slope", "o_trend_pvalue", "o_values"], errors="ignore")
    enriched["structure_stability_days"] = enriched.groupby("stock_code")["numbered_depression"].transform(_trailing_true_streak)
    enriched["A_norm"] = _clip01(enriched["volume_acceleration"] / 2.0)
    enriched["C_norm"] = _clip01(enriched["close_to_high"])
    enriched["R_norm"] = _clip01(enriched["resonance_count"] / 5.0)
    enriched["S_norm"] = _clip01(enriched["structure_stability_days"] / 5.0)
    enriched["E_norm"] = np.where(enriched["is_limit_like"], enriched["close_to_high"], 0.0)
    enriched["organization_score"] = (
        0.25 * enriched["A_norm"]
        + 0.20 * enriched["C_norm"]
        + 0.20 * enriched["R_norm"]
        + 0.20 * enriched["S_norm"]
        + 0.15 * enriched["E_norm"]
    ).clip(0.0, 1.0)

    def _embed_trend(series: pd.Series) -> pd.Series:
        vals = series.dropna().tolist()
        recent = vals[-30:] if len(vals) > 30 else vals
        if len(recent) < 2:
            return pd.Series({"o_rising": False, "o_slope": 0.0, "o_trend_pvalue": 999.0, "o_values": "[]"})
        tail = recent[-embryo_params.trend_lookback:]
        if len(tail) < 2:
            tail = recent[-2:]
        x = list(range(len(tail)))
        slope, _, pvalue = _linregress(x, tail)
        if slope is None:
            slope = 0.0
        return pd.Series({
            "o_rising": slope > embryo_params.trend_min_slope,
            "o_slope": slope,
            "o_trend_pvalue": pvalue,
            "o_values": str(tail[:5]) if tail else "[]",
        })

    trend_stats = enriched.groupby("stock_code")["organization_score"].apply(_embed_trend).reset_index()
    trend_stats = trend_stats.pivot(index="stock_code", columns="level_1", values="organization_score").reset_index()
    trend_stats.columns.name = None
    for col in ["o_rising", "o_slope", "o_trend_pvalue", "o_values"]:
        if col not in trend_stats.columns:
            trend_stats[col] = False if col == "o_rising" else ("[]" if col == "o_values" else 0.0)
    trend_stats["o_rising"] = trend_stats["o_rising"].astype(bool)
    trend_stats["o_slope"] = pd.to_numeric(trend_stats["o_slope"], errors="coerce").fillna(0.0)

    o_min_val = float(embryo_params.o_min)
    o_max_val = float(embryo_params.o_max)

    yesterday_data = enriched.loc[enriched["date"] == yesterday].copy()
    yesterday_data = yesterday_data.merge(trend_stats, on="stock_code", how="left")
    if "organization_score" not in yesterday_data.columns and "organization_score_x" in yesterday_data.columns:
        yesterday_data.rename(columns={"organization_score_x": "organization_score"}, inplace=True)
    yesterday_data["o_rising"] = yesterday_data["o_rising"].fillna(False).astype(bool)
    yesterday_data["embryo_yesterday"] = (
        yesterday_data["organization_score"].between(o_min_val, o_max_val, inclusive="both")
        & yesterday_data["o_rising"]
    )
    yesterday_embryo_codes = set(yesterday_data.loc[yesterday_data["embryo_yesterday"], "stock_code"].unique())

    today_data = enriched.loc[enriched["date"] == today].copy()
    today_data = today_data.merge(trend_stats, on="stock_code", how="left")
    if "organization_score" not in today_data.columns and "organization_score_x" in today_data.columns:
        today_data.rename(columns={"organization_score_x": "organization_score"}, inplace=True)
    today_data["o_rising"] = today_data["o_rising"].fillna(False).astype(bool)
    today_data["o_slope"] = today_data["o_slope"].fillna(0.0)

    incubation_rows = today_data.loc[today_data["stock_code"].isin(yesterday_embryo_codes)].copy()

    roles = _load_role_snapshot(today)
    if not roles.empty:
        incubation_rows = incubation_rows.merge(roles, on="stock_code", how="left")
    else:
        incubation_rows["dominant_role"] = ""
    incubation_rows["dominant_role"] = incubation_rows["dominant_role"].fillna("")

    incubation_rows["incub_cond_a"] = (
        (incubation_rows["organization_score"] > 0.78)
        & (incubation_rows["organization_score"] < 0.80)
        & (incubation_rows["dominant_role"] == "herd")
    )
    incubation_rows["incub_cond_b"] = incubation_rows["is_limit_like"]
    incubation_rows["incubation"] = incubation_rows["incub_cond_a"] | incubation_rows["incub_cond_b"]
    incubation_rows = incubation_rows.loc[incubation_rows["incubation"]].copy()

    if incubation_rows.empty:
        return pd.DataFrame(columns=["typhoon_id", "stock_code", "stock_name", "organization_score", "o_slope", "dominant_role", "incubation_status", "incubation_action", "primary_concept_name", "structure_stability_days"])

    def _status(row: pd.Series) -> str:
        parts = []
        if row.get("is_limit_like"):
            parts.append("昨日涨停")
        if 0.78 < float(row.get("organization_score", 0.0)) < 0.80:
            parts.append("O值在0.78-0.80")
        if not parts:
            parts.append("昨日胚胎+今日异动")
        return "，".join(parts) + "，等待成台风或退回胚胎"

    def _incub_action(row: pd.Series) -> str:
        role = str(row.get("dominant_role") or "")
        if role == "herd" and int(row.get("structure_stability_days", 0)) >= 1:
            return "积极观察：可能明天变台风成形"
        if role == "slow":
            return "中性观察：等角色转换"
        if float(row.get("organization_score", 0.0)) <= 0.78:
            return "自动回到胚胎列表（次日）"
        return "观察：孵化中"

    incubation_rows["incubation_status"] = incubation_rows.apply(_status, axis=1)
    incubation_rows["incubation_action"] = incubation_rows.apply(_incub_action, axis=1)

    if not basic.empty:
        incubation_rows = incubation_rows.merge(basic, on="stock_code", how="left")

    incubation_rows = incubation_rows.sort_values("organization_score", ascending=False).head(10)
    incubation_rows["typhoon_id"] = [f"INCU-{today.strftime('%Y%m%d')}-{i:03d}" for i in range(1, len(incubation_rows) + 1)]

    return incubation_rows[[
        "typhoon_id", "stock_code", "stock_name",
        "organization_score", "o_slope", "dominant_role",
        "incubation_status", "incubation_action",
        "primary_concept_name", "structure_stability_days",
    ]]


def _write_outputs(candidates: pd.DataFrame, summary: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = summary["end_date"]
    csv_path = output_dir / f"typhoon_tracker_{date_tag}.csv"
    json_path = output_dir / f"typhoon_tracker_{date_tag}_summary.json"
    md_path = output_dir / f"typhoon_tracker_{date_tag}.md"

    export_cols = [
        "typhoon_id",
        "stock_code",
        "stock_name",
        "industry",
        "rating",
        "organization_score",
        "thermo_score",
        "dominant_role",
        "action",
        "primary_concept_name",
        "resonance_count",
        "resonance_delta",
        "structure_stability_days",
        "close",
        "return_1d",
        "volume_acceleration",
        "close_to_high",
        "amplitude_ma5",
        "main_net_5d",
        "I1_volume_surge",
        "I2_uptrend",
        "I3_volatility_expansion",
        "I4_main_money_inflow",
        "I5_concept_resonance",
    ]
    export_cols = [col for col in export_cols if col in candidates.columns]
    candidates[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {**summary, "top": candidates[export_cols].head(20).to_dict(orient="records")}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 股票台风追踪日报",
        "",
        f"- 交易日：{summary['end_date']}",
        f"- 扫描股票数：{summary['stock_count']}",
        f"- 编号热带低压：{summary['numbered_depression_count']}",
        f"- 报告跟踪数：{summary['tracked_count']}",
        "",
        "## Top 跟踪池",
        "",
        "| 编号 | 股票 | 评级 | O组织强度 | S热力评分 | 角色 | 行动 | 共振概念 | 共振数 | 稳定天数 |",
        "|---|---:|---|---:|---:|---|---|---|---:|---:|",
    ]
    for _, row in candidates.head(30).iterrows():
        stock_label = row["stock_code"] if not row.get("stock_name") else f"{row['stock_code']} {row['stock_name']}"
        lines.append(
            f"| {row['typhoon_id']} | {stock_label} | {row['rating']} | "
            f"{row['organization_score']:.3f} | {row['thermo_score']:.3f} | {row.get('dominant_role', '') or 'NA'} | "
            f"{row['action']} | {row.get('primary_concept_name', '')} | {int(row.get('resonance_count', 0))} | {int(row.get('structure_stability_days', 0))} |"
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 入选编号热带低压：I1 成交量突增、I2 趋势向上、I3 波动扩大、I4 主力资金净流入、I5 概念共振全部满足。",
            "- O组织强度：按设计文档系统A计算：A成交量加速度、C收敛度、R共振度、S结构稳定性、E涨停效率代理。",
            "- S热力评分：按设计文档系统B的 `dS/dt = Q - αS` 离散近似，当前作为对照实验分数。",
            "- 行动列不是自动交易指令；协同试验仍需角色、治理结论、市场环境和新闻人工确认。",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="股票台风追踪系统 MVP：识别编号热带低压并输出组织强度报告。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="系统 config.yaml 路径。")
    parser.add_argument("--end-date", default=None, help="扫描截止交易日，默认使用 stock_data 最新日期。")
    parser.add_argument("--lookback-calendar-days", type=int, default=140, help="读取最近多少自然日数据。")
    parser.add_argument("--top-n", type=int, default=50, help="报告最多输出多少只跟踪标的。")
    parser.add_argument("--volume-ratio-threshold", type=float, default=1.5, help="I1：5日均量/20日均量阈值。")
    parser.add_argument("--amplitude-threshold", type=float, default=0.05, help="I3：5日均振幅阈值。")
    parser.add_argument("--min-resonance-count", type=int, default=2, help="I5：同概念同时满足 I1-I4 的最小股票数。")
    parser.add_argument("--alpha-decay", type=float, default=0.35, help="系统B热力学衰减率 alpha。")
    parser.add_argument("--ignore-concept-keyword", action="append", default=[], help="从板块共振中排除的泛概念关键词，可重复传入。")
    parser.add_argument("--no-default-ignore-concepts", action="store_true", help="不使用默认泛概念排除表。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")
    parser.add_argument(
        "--mode",
        default="mature",
        choices=["mature", "embryo", "both"],
        help="mature: 旧台风成形模型 / embryo: 台风胚胎模型 / both: 两者并行输出。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignored_keywords = tuple(args.ignore_concept_keyword)
    if not args.no_default_ignore_concepts:
        ignored_keywords = tuple(DEFAULT_IGNORED_CONCEPT_KEYWORDS) + ignored_keywords
    params = TyphoonParams(
        volume_ratio_threshold=args.volume_ratio_threshold,
        amplitude_threshold=args.amplitude_threshold,
        min_resonance_count=args.min_resonance_count,
        alpha_decay=args.alpha_decay,
        top_n=args.top_n,
        ignored_concept_keywords=ignored_keywords,
    )

    result: dict = {}

    if args.mode in ("mature", "both"):
        candidates, summary = build_typhoon_snapshot(
            config_path=Path(args.config),
            end_date=args.end_date,
            lookback_calendar_days=args.lookback_calendar_days,
            params=params,
        )
        paths = _write_outputs(candidates, summary, Path(args.output_dir))
        result["summary"] = summary
        result["mature_outputs"] = {k: str(v) for k, v in paths.items()}

    if args.mode in ("embryo", "both"):
        embryo_params = EmbryoParams(top_n=args.top_n)
        embryo_candidates, embryo_summary, incubation = build_embryo_tracker(
            config_path=Path(args.config),
            end_date=args.end_date,
            lookback_calendar_days=args.lookback_calendar_days,
            params=params,
            embryo_params=embryo_params,
        )
        embryo_paths = _write_embryo_outputs(embryo_candidates, embryo_summary, Path(args.output_dir), incubation_candidates=incubation)
        result["embryo_summary"] = embryo_summary
        result["embryo_outputs"] = {k: str(v) for k, v in embryo_paths.items()}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()