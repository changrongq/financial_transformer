from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
import sys
import threading
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel

from .backtest import Backtester, resolve_prediction_file
from .config import load_config
from .entry_advice import add_entry_advice
from .inference import ProductionInferenceEngine
from .features import layer_from_market_cap_yi
from .paper_trading import (
    get_account_snapshot,
    get_position_watch,
    get_trade_history,
    initialize_account,
    place_batch_buy_orders,
    place_batch_sell_orders,
    place_buy_order,
    place_sell_order,
)
from .trade_data import fetch_realtime_market_list, fetch_realtime_quotes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "financial_transformer"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(EXPERIMENT_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SRC))

from financial_transformer_experiment.config import load_config as load_transformer_config
from financial_transformer_experiment.scoring import (
    annotate_confidence_tiers,
    summarize_confidence_distribution,
    summarize_market_environment,
)
from compare_latest_moneyflow_vs_rerank_variants import run_latest_profile_comparison
from predict_latest_broader_history_rerank import run_latest_broader_history_rerank
from financial_transformer_experiment.two_stage_execution import predict_latest_for_execution
from financial_transformer_experiment.two_stage_execution import resolve_two_stage_execution_spec


DEFAULT_SIGNAL_CONFIG_PATH = EXPERIMENT_ROOT / "config_two_stage_rerank_leader_tiered_v1.yaml"
DEFAULT_TRIAL_CAPITAL_FRACTION = 0.25
DEFAULT_MAX_SINGLE_POSITION_FRACTION = 0.07
DEFAULT_SIGNAL_PROFILE = "micro"
SIGNAL_PROFILE_CONFIGS = {
    "micro": EXPERIMENT_ROOT / "config_two_stage_rerank_leader_tiered_v1.yaml",
    "large": EXPERIMENT_ROOT / "config_moneyflow_v1_large_v1.yaml",
    "large_rerank": EXPERIMENT_ROOT / "config_two_stage_rerank_large_v1.yaml",
}
BROADER_HISTORY_RERANK_PROFILE_VARIANTS = {
    "broader_history_rerank_default_exec": "default_exec",
    "broader_history_rerank_transfer_exec": "transfer_exec",
}
SIGNAL_PROFILE_DISPLAY_NAMES = {
    "micro": "micro",
    "large": "large",
    "large_rerank": "large_rerank",
    "broader_history_rerank_default_exec": "broader_history_rerank_default_exec",
    "broader_history_rerank_transfer_exec": "broader_history_rerank_transfer_exec",
}
SINGLE_STAGE_PROFILE_DEFAULTS = {
    "large": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "min_score_ratio_to_leader": 0.0,
        "execution_hold_days": 45,
        "execution_take_profit": 0.14,
        "execution_stop_loss": -0.06,
        "leader_max_day_rank": None,
        "leader_hold_days": None,
        "leader_take_profit": None,
        "leader_stop_loss": None,
    },
}


class PredictRequest(BaseModel):
    top_n: int = 20


class BacktestRequest(BaseModel):
    prediction_file: Optional[str] = None
    hold_days: int = 5
    top_n: int = 10
    initial_capital: float = 100000.0


class InitialPosition(BaseModel):
    stock_code: str
    quantity: int
    avg_cost: Optional[float] = None


class PaperAccountInitRequest(BaseModel):
    initial_cash: float
    positions: list[InitialPosition] = []
    reset: bool = True
    quote_source: str = "sina"


class PaperBuyRequest(BaseModel):
    stock_code: str
    quantity: Optional[int] = None
    cash_amount: Optional[float] = None
    position_fraction: Optional[float] = None
    limit_price: Optional[float] = None
    src: str = "sina"
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    board_lot: int = 100
    note: Optional[str] = None


class PaperSellRequest(BaseModel):
    stock_code: str
    quantity: Optional[int] = None
    position_fraction: Optional[float] = None
    limit_price: Optional[float] = None
    src: str = "sina"
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.001
    note: Optional[str] = None


class PaperBatchOrderItem(BaseModel):
    stock_code: str
    quantity: Optional[int] = None
    cash_amount: Optional[float] = None
    position_fraction: Optional[float] = None
    limit_price: Optional[float] = None
    src: str = "sina"
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    board_lot: int = 100
    stamp_duty_rate: float = 0.001
    note: Optional[str] = None


class PaperBatchOrderRequest(BaseModel):
    orders: list[PaperBatchOrderItem]


def _parse_ts_codes(ts_codes: str) -> list[str]:
    parsed = [item.strip() for item in str(ts_codes or "").split(",") if item.strip()]
    if not parsed:
        raise ValueError("ts_codes 不能为空，使用逗号分隔多个股票代码。")
    return parsed


def _normalize_signal_profile(signal_profile: str | None) -> str:
    profile = str(signal_profile or DEFAULT_SIGNAL_PROFILE).strip().lower()
    if profile not in SIGNAL_PROFILE_CONFIGS and profile not in BROADER_HISTORY_RERANK_PROFILE_VARIANTS and profile != "all_split":
        supported = ", ".join(sorted([*SIGNAL_PROFILE_CONFIGS.keys(), *BROADER_HISTORY_RERANK_PROFILE_VARIANTS.keys(), "all_split"]))
        raise ValueError(f"不支持的 signal_profile: {profile}。可选值: {supported}")
    return profile


def _is_broader_history_rerank_profile(signal_profile: str) -> bool:
    return signal_profile in BROADER_HISTORY_RERANK_PROFILE_VARIANTS


def _load_broader_history_rerank_payload(signal_profile: str, refresh: bool, use_calibration: bool) -> dict:
    variant = BROADER_HISTORY_RERANK_PROFILE_VARIANTS[signal_profile]
    output_dir = EXPERIMENT_ROOT / "results" / "broader_history_rerank_variants_v1" / variant / "latest"
    summary_path = output_dir / "latest_summary.json"
    if refresh or not summary_path.exists():
        return run_latest_broader_history_rerank(
            variant=variant,
            use_calibration=use_calibration,
        )
    import json

    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_signal_config_path(signal_profile: str | None) -> Path:
    profile = _normalize_signal_profile(signal_profile)
    if profile == "all_split":
        raise ValueError("all_split 不是单一信号配置，请在聚合接口中使用。")
    if _is_broader_history_rerank_profile(profile):
        payload = _load_broader_history_rerank_payload(profile, refresh=False, use_calibration=False)
        return Path(payload["eval_config"]).resolve()
    return SIGNAL_PROFILE_CONFIGS[profile].resolve()


def _resolve_execution_defaults(signal_profile: str, spec) -> dict:
    if spec is not None:
        return {
            "short_up_threshold": spec.default_short_up_threshold,
            "take_profit_threshold": spec.default_take_profit_threshold,
            "stop_loss_threshold": spec.default_stop_loss_threshold,
            "top_k_per_day": spec.default_top_k_per_day,
            "min_score_ratio_to_leader": spec.default_min_score_ratio_to_leader,
            "execution_hold_days": spec.default_execution_hold_days,
            "execution_take_profit": spec.default_execution_take_profit,
            "execution_stop_loss": spec.default_execution_stop_loss,
            "leader_max_day_rank": spec.default_leader_max_day_rank,
            "leader_hold_days": spec.default_leader_hold_days,
            "leader_take_profit": spec.default_leader_take_profit,
            "leader_stop_loss": spec.default_leader_stop_loss,
        }
    defaults = SINGLE_STAGE_PROFILE_DEFAULTS.get(signal_profile)
    if defaults is None:
        return {}
    return dict(defaults)


def _load_mainline_prediction_health() -> dict | None:
    path = EXPERIMENT_ROOT / "results" / "mainline_prediction_health.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_prediction_health_execution_guard(signal_profile: str) -> dict:
    profile = _normalize_signal_profile(signal_profile)
    if profile != "micro":
        return {"blocked": False, "status": None, "reason": None, "mode": "normal", "capital_fraction_multiplier": 1.0}
    payload = _load_mainline_prediction_health() or {}
    health_status = payload.get("health_status") or {}
    execution_plan = payload.get("execution_plan") or {}
    status = str(health_status.get("status") or "").strip().lower()
    if status != "pause":
        return {
            "blocked": False,
            "status": health_status.get("status"),
            "reason": health_status.get("recommendation"),
            "mode": execution_plan.get("mode") or ("reduced_risk" if status == "caution" else "normal"),
            "capital_fraction_multiplier": float(execution_plan.get("capital_fraction_multiplier", 0.5 if status == "caution" else 1.0)),
        }
    return {
        "blocked": True,
        "status": health_status.get("status"),
        "reason": health_status.get("recommendation") or "recent incremental prediction health is weak",
        "mode": execution_plan.get("mode") or "no_new_trades",
        "capital_fraction_multiplier": float(execution_plan.get("capital_fraction_multiplier", 0.0)),
    }


def _apply_execution_filters(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    if not defaults or df.empty:
        return df.copy()

    filtered = df[
        (df["pred_short_up"] > defaults["short_up_threshold"])
        & (df["pred_take_profit"] > defaults["take_profit_threshold"])
        & (df["pred_stop_loss"] < defaults["stop_loss_threshold"])
    ].copy()
    filtered = filtered.sort_values(["signal_score", "pred_take_profit", "pred_short_up"], ascending=[False, False, False]).reset_index(drop=True)

    min_ratio = float(defaults.get("min_score_ratio_to_leader", 0.0) or 0.0)
    if min_ratio > 0.0 and not filtered.empty:
        leader_score = float(filtered["signal_score"].max())
        if abs(leader_score) > 1e-9:
            filtered["score_ratio_to_leader"] = filtered["signal_score"] / leader_score
            filtered = filtered[filtered["score_ratio_to_leader"] >= min_ratio].copy()
            filtered = filtered.sort_values(["signal_score", "pred_take_profit", "pred_short_up"], ascending=[False, False, False]).reset_index(drop=True)

    top_k = int(defaults.get("top_k_per_day", 0) or 0)
    if top_k > 0 and not filtered.empty:
        filtered = filtered.head(top_k).reset_index(drop=True)

    filtered = annotate_confidence_tiers(filtered)
    return _resolve_transformer_exit_policy(
        filtered,
        base_hold_days=int(defaults["execution_hold_days"]),
        base_take_profit=float(defaults["execution_take_profit"]),
        base_stop_loss=float(defaults["execution_stop_loss"]),
        leader_max_day_rank=defaults.get("leader_max_day_rank"),
        leader_hold_days=defaults.get("leader_hold_days"),
        leader_take_profit=defaults.get("leader_take_profit"),
        leader_stop_loss=defaults.get("leader_stop_loss"),
    )


def _records(df: pd.DataFrame) -> list[dict]:
    safe_df = df.astype(object).where(pd.notna(df), None)
    return safe_df.to_dict(orient="records")


def _paper_snapshot_response(snapshot: dict) -> dict:
    return {
        "source": snapshot.get("source"),
        "summary": snapshot.get("summary", {}),
        "order": snapshot.get("order"),
        "batch_orders": snapshot.get("batch_orders"),
        "positions": _records(snapshot.get("positions", pd.DataFrame())),
        "trades": _records(snapshot.get("trades", pd.DataFrame())),
    }


def _allocate_signal_positions(
    df: pd.DataFrame,
    total_capital_fraction: float,
    max_single_fraction: float,
) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result["suggested_total_capital_pct"] = pd.Series(dtype=float)
        result["suggested_strategy_bucket_pct"] = pd.Series(dtype=float)
        return result

    scores = pd.to_numeric(df["signal_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
    score_sum = float(scores.sum())
    if score_sum <= 0:
        raw_weights = pd.Series([1.0 / len(df)] * len(df), index=df.index)
    else:
        raw_weights = scores / score_sum

    remaining_total = float(total_capital_fraction)
    remaining_index = list(df.index)
    allocated = pd.Series(0.0, index=df.index, dtype=float)

    while remaining_index and remaining_total > 1e-9:
        base = raw_weights.loc[remaining_index]
        base_sum = float(base.sum())
        if base_sum <= 0:
            equal_weight = remaining_total / len(remaining_index)
            for idx in remaining_index:
                allocated.loc[idx] += min(equal_weight, max_single_fraction - allocated.loc[idx])
            break

        capped_any = False
        for idx in list(remaining_index):
            proposed = remaining_total * float(base.loc[idx] / base_sum)
            capacity = max_single_fraction - float(allocated.loc[idx])
            if capacity <= 1e-9:
                remaining_index.remove(idx)
                continue
            if proposed >= capacity:
                allocated.loc[idx] += capacity
                remaining_total -= capacity
                remaining_index.remove(idx)
                capped_any = True
        if not capped_any:
            for idx in remaining_index:
                allocated.loc[idx] += remaining_total * float(base.loc[idx] / base_sum)
            remaining_total = 0.0

    result = df.copy()
    result["suggested_total_capital_pct"] = allocated * 100.0
    bucket_base = max(total_capital_fraction, 1e-9)
    result["suggested_strategy_bucket_pct"] = allocated / bucket_base * 100.0
    return result


def _load_market_cap_frame(config, stock_codes: list[str]) -> pd.DataFrame:
    if not stock_codes:
        return pd.DataFrame(columns=["stock_code", "market_cap_yi", "size_bucket"])
    normalized = sorted({str(code).zfill(6) for code in stock_codes if str(code).strip()})
    placeholders = ", ".join(["?"] * len(normalized))
    conn = sqlite3.connect(config.databases.market_cap_db)
    try:
        frame = pd.read_sql_query(
            f"SELECT stock_code, total_mv FROM market_caps WHERE stock_code IN ({placeholders})",
            conn,
            params=normalized,
        )
    finally:
        conn.close()

    if frame.empty:
        return pd.DataFrame(columns=["stock_code", "market_cap_yi", "size_bucket"])
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["market_cap_yi"] = pd.to_numeric(frame["total_mv"], errors="coerce") / 10000.0
    frame["size_bucket"] = frame["market_cap_yi"].apply(
        lambda value: layer_from_market_cap_yi(float(value)) if pd.notna(value) else "unknown"
    )
    return frame[["stock_code", "market_cap_yi", "size_bucket"]]


def _weight_by_bucket(frame: pd.DataFrame, weight_col: str) -> dict[str, float]:
    if frame.empty or weight_col not in frame.columns:
        return {}
    working = frame.copy()
    working[weight_col] = pd.to_numeric(working[weight_col], errors="coerce").fillna(0.0)
    if "size_bucket" not in working.columns:
        working["size_bucket"] = "unknown"
    grouped = working.groupby("size_bucket", as_index=False)[weight_col].sum()
    return {
        str(row["size_bucket"]): float(row[weight_col])
        for _, row in grouped.iterrows()
        if float(row[weight_col]) > 1e-9
    }


def _format_market_environment_label(label: object) -> str:
    mapping = {
        "high_risk": "高风险",
        "trend_up": "上行趋势",
        "trend_down": "下行趋势",
        "range": "震荡区间",
        "unknown": "未知",
    }
    return mapping.get(str(label), str(label))


def _format_market_environment_style(style: object) -> str:
    mapping = {
        "aggressive": "偏积极",
        "defensive": "偏防御",
        "neutral": "中性",
    }
    return mapping.get(str(style), str(style))


def _format_confidence_label(label: object) -> str:
    mapping = {
        "leader_concentrated": "头部集中",
        "balanced": "分布均衡",
        "mixed": "轻度集中",
        "weak_confidence": "弱信号日",
        "single_candidate": "单票日",
        "unknown": "未知",
    }
    return mapping.get(str(label), str(label))


def _format_confidence_tier(tier: object) -> str:
    mapping = {
        "high": "高",
        "medium": "中",
        "low": "低",
        "unknown": "未知",
    }
    return mapping.get(str(tier), str(tier))


def _format_confidence_counts(confidence_summary: dict) -> str:
    counts = confidence_summary.get("counts", {}) if confidence_summary else {}
    ordered = [("high", "高"), ("medium", "中"), ("low", "低")]
    return " / ".join(f"{label}{int(counts.get(tier, 0))}" for tier, label in ordered)


def _enrich_market_environment_display(market_environment: dict) -> dict:
    enriched = dict(market_environment or {})
    enriched["display"] = {
        "label": _format_market_environment_label(enriched.get("label", "unknown")),
        "style": _format_market_environment_style(enriched.get("style", "neutral")),
    }
    return enriched


def _enrich_confidence_summary_display(confidence_summary: dict) -> dict:
    enriched = dict(confidence_summary or {})
    enriched["display"] = {
        "label": _format_confidence_label(enriched.get("label", "unknown")),
        "counts": _format_confidence_counts(enriched),
    }
    return enriched


def _enrich_pick_display_fields(
    df: pd.DataFrame,
    *,
    system_config=None,
    signal_date_col: str = "date",
    current_price_col: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    if "signal_confidence_tier" in result.columns:
        result["signal_confidence_tier_display"] = result["signal_confidence_tier"].apply(_format_confidence_tier)
    if system_config is not None:
        result = add_entry_advice(
            result,
            system_config,
            signal_date_col=signal_date_col,
            signal_close_col="signal_close" if "signal_close" in result.columns else "close",
            current_price_col=current_price_col,
            open_price_col="open" if "open" in result.columns else None,
            high_price_col="high" if "high" in result.columns else None,
            low_price_col="low" if "low" in result.columns else None,
            pre_close_col="pre_close" if "pre_close" in result.columns else None,
        )
    return result


def _build_portfolio_impact_summary(config, picks: pd.DataFrame) -> dict:
    try:
        snapshot = get_account_snapshot(config, src="auto", trades_limit=1)
    except RuntimeError:
        return {
            "available": False,
            "message": "模拟盘账户尚未初始化，暂无法计算组合影响。",
        }
    except Exception as exc:
        return {
            "available": False,
            "message": f"组合影响计算失败：{exc}",
        }

    positions = snapshot.get("positions", pd.DataFrame()).copy()
    if positions.empty and picks.empty:
        return {
            "available": True,
            "message": "当前无持仓，也无新增候选。",
            "existing_bucket_weights": {},
            "post_trade_bucket_weights": {},
            "overlap_stock_count": 0,
            "overlap_stock_codes": [],
        }

    holding_codes = positions.get("stock_code", pd.Series(dtype=str)).astype(str).str.zfill(6).tolist() if not positions.empty else []
    pick_codes = picks.get("stock_code", pd.Series(dtype=str)).astype(str).str.zfill(6).tolist() if not picks.empty else []
    cap_frame = _load_market_cap_frame(config, holding_codes + pick_codes)

    if not positions.empty:
        positions["stock_code"] = positions["stock_code"].astype(str).str.zfill(6)
        positions = positions.merge(cap_frame, on="stock_code", how="left")
    else:
        positions = pd.DataFrame(columns=["stock_code", "position_weight", "size_bucket"])

    new_allocations = picks[[column for column in ["stock_code", "suggested_total_capital_pct"] if column in picks.columns]].copy()
    if not new_allocations.empty:
        new_allocations["stock_code"] = new_allocations["stock_code"].astype(str).str.zfill(6)
        new_allocations = new_allocations.merge(cap_frame[["stock_code", "size_bucket"]], on="stock_code", how="left")
        new_allocations["new_weight"] = pd.to_numeric(new_allocations["suggested_total_capital_pct"], errors="coerce").fillna(0.0) / 100.0
    else:
        new_allocations = pd.DataFrame(columns=["stock_code", "size_bucket", "new_weight"])

    existing_bucket_weights = _weight_by_bucket(positions, "position_weight")
    post_frame = pd.concat(
        [
            positions[[column for column in ["stock_code", "size_bucket", "position_weight"] if column in positions.columns]].rename(columns={"position_weight": "weight"}),
            new_allocations[[column for column in ["stock_code", "size_bucket", "new_weight"] if column in new_allocations.columns]].rename(columns={"new_weight": "weight"}),
        ],
        ignore_index=True,
    ) if (not positions.empty or not new_allocations.empty) else pd.DataFrame(columns=["stock_code", "size_bucket", "weight"])
    post_trade_bucket_weights = _weight_by_bucket(post_frame, "weight")
    overlap_codes = sorted(set(holding_codes) & set(pick_codes))
    return {
        "available": True,
        "message": None,
        "existing_bucket_weights": existing_bucket_weights,
        "post_trade_bucket_weights": post_trade_bucket_weights,
        "overlap_stock_count": int(len(overlap_codes)),
        "overlap_stock_codes": overlap_codes,
        "positions_count": int(len(positions)),
        "new_pick_count": int(len(new_allocations)),
    }


def _resolve_transformer_exit_policy(
    df: pd.DataFrame,
    *,
    base_hold_days: int,
    base_take_profit: float,
    base_stop_loss: float,
    leader_max_day_rank: int | None,
    leader_hold_days: int | None,
    leader_take_profit: float | None,
    leader_stop_loss: float | None,
) -> pd.DataFrame:
    resolved = df.copy()
    resolved["execution_hold_days"] = int(base_hold_days)
    resolved["execution_take_profit"] = float(base_take_profit)
    resolved["execution_stop_loss"] = float(base_stop_loss)
    if leader_max_day_rank is None or leader_max_day_rank <= 0 or resolved.empty:
        return resolved
    leader_mask = resolved.index < int(leader_max_day_rank)
    if leader_hold_days is not None:
        resolved.loc[leader_mask, "execution_hold_days"] = int(leader_hold_days)
    if leader_take_profit is not None:
        resolved.loc[leader_mask, "execution_take_profit"] = float(leader_take_profit)
    if leader_stop_loss is not None:
        resolved.loc[leader_mask, "execution_stop_loss"] = float(leader_stop_loss)
    return resolved


def _load_transformer_latest_signals(top_n: int, refresh: bool, use_calibration: bool = False, signal_profile: str = DEFAULT_SIGNAL_PROFILE) -> pd.DataFrame:
    signal_profile = _normalize_signal_profile(signal_profile)
    if _is_broader_history_rerank_profile(signal_profile):
        payload = _load_broader_history_rerank_payload(signal_profile, refresh=refresh, use_calibration=use_calibration)
        latest_selected_path = Path(payload["outputs"]["latest_selected"])
        if not latest_selected_path.exists():
            raise FileNotFoundError(f"未找到 broader-history rerank 最新信号文件: {latest_selected_path}")
        latest_df = pd.read_csv(latest_selected_path, dtype={"stock_code": str})
        latest_df["stock_code"] = latest_df["stock_code"].astype(str).str.zfill(6)
        if "date" in latest_df.columns:
            latest_df["date"] = pd.to_datetime(latest_df["date"], errors="coerce")
        latest_df = annotate_confidence_tiers(latest_df)
        return latest_df.head(top_n).reset_index(drop=True)

    signal_config_path = _resolve_signal_config_path(signal_profile)
    transformer_config = load_transformer_config(signal_config_path)
    spec = resolve_two_stage_execution_spec(signal_config_path)
    if refresh:
        predict_latest_for_execution(signal_config_path, use_calibration=use_calibration)

    file_name = "latest_predictions_calibrated.csv" if use_calibration else "latest_predictions.csv"
    latest_path = transformer_config.results_dir / file_name
    if not latest_path.exists():
        raise FileNotFoundError(f"未找到最新执行信号文件: {latest_path}，请先运行实验信号生成或使用 refresh=true。")

    latest_df = pd.read_csv(latest_path, dtype={"stock_code": str})
    latest_df["stock_code"] = latest_df["stock_code"].astype(str).str.zfill(6)
    latest_df["date"] = pd.to_datetime(latest_df["date"], errors="coerce")

    execution_defaults = _resolve_execution_defaults(signal_profile, spec)
    if execution_defaults:
        latest_df = _apply_execution_filters(latest_df, execution_defaults)
    else:
        latest_df = annotate_confidence_tiers(latest_df)

    execution_guard = _resolve_prediction_health_execution_guard(signal_profile)
    if execution_guard.get("blocked"):
        latest_df = latest_df.iloc[0:0].copy()

    return latest_df.head(top_n).reset_index(drop=True)


def _load_latest_picks(config_path: str | Path, results_dir: Path, top_n: int, refresh: bool, signal_profile: str = DEFAULT_SIGNAL_PROFILE) -> pd.DataFrame:
    return _load_transformer_latest_signals(top_n=top_n, refresh=refresh, use_calibration=False, signal_profile=signal_profile)


def _load_transformer_market_environment(refresh: bool, use_calibration: bool = False, signal_profile: str = DEFAULT_SIGNAL_PROFILE) -> dict:
    signal_profile = _normalize_signal_profile(signal_profile)
    if _is_broader_history_rerank_profile(signal_profile):
        payload = _load_broader_history_rerank_payload(signal_profile, refresh=refresh, use_calibration=use_calibration)
        eval_config_path = Path(payload["eval_config"]).resolve()
        transformer_config = load_transformer_config(eval_config_path)
        file_name = "latest_predictions_calibrated.csv" if use_calibration else "latest_predictions.csv"
        latest_path = transformer_config.results_dir / file_name
        if not latest_path.exists():
            raise FileNotFoundError(f"未找到最新执行信号文件: {latest_path}，请先运行实验信号生成或使用 refresh=true。")
        latest_df = pd.read_csv(latest_path, dtype={"stock_code": str})
        latest_df["stock_code"] = latest_df["stock_code"].astype(str).str.zfill(6)
        if "date" in latest_df.columns and payload.get("latest_available_date"):
            latest_df["date"] = pd.to_datetime(latest_df["date"], errors="coerce")
            latest_df = latest_df[
                latest_df["date"] == pd.to_datetime(payload["latest_available_date"], errors="coerce")
            ].copy()
        return summarize_market_environment(latest_df, filtered_count=int(payload.get("selected_count", 0)))

    signal_config_path = _resolve_signal_config_path(signal_profile)
    transformer_config = load_transformer_config(signal_config_path)
    spec = resolve_two_stage_execution_spec(signal_config_path)
    if refresh:
        predict_latest_for_execution(signal_config_path, use_calibration=use_calibration)

    file_name = "latest_predictions_calibrated.csv" if use_calibration else "latest_predictions.csv"
    latest_path = transformer_config.results_dir / file_name
    if not latest_path.exists():
        raise FileNotFoundError(f"未找到最新执行信号文件: {latest_path}，请先运行实验信号生成或使用 refresh=true。")

    latest_df = pd.read_csv(latest_path, dtype={"stock_code": str})
    filtered_count = None
    execution_defaults = _resolve_execution_defaults(signal_profile, spec)
    if execution_defaults:
        filtered = _apply_execution_filters(latest_df, execution_defaults)
        filtered_count = int(len(filtered))
    return summarize_market_environment(latest_df, filtered_count=filtered_count)


def _build_latest_pick_payload(
    config,
    config_path: str | Path,
    *,
    signal_profile: str,
    top_n: int,
    refresh: bool,
    src: str,
    trial_capital_fraction: float,
    max_single_position_fraction: float,
) -> dict:
    profile = _normalize_signal_profile(signal_profile)
    execution_guard = _resolve_prediction_health_execution_guard(profile)
    effective_trial_capital_fraction = trial_capital_fraction * float(execution_guard.get("capital_fraction_multiplier", 1.0))
    picks = _load_latest_picks(config_path, config.results_dir, top_n=top_n, refresh=refresh, signal_profile=profile)
    allocated_picks = _allocate_signal_positions(
        picks.copy(),
        total_capital_fraction=effective_trial_capital_fraction,
        max_single_fraction=max_single_position_fraction,
    )
    market_environment = _enrich_market_environment_display(
        _load_transformer_market_environment(refresh=refresh, use_calibration=False, signal_profile=profile)
    )
    confidence_summary = _enrich_confidence_summary_display(
        summarize_confidence_distribution(allocated_picks)
    )
    portfolio_impact = _build_portfolio_impact_summary(config, allocated_picks)
    merged, source = _build_realtime_watchlist(config, allocated_picks, src=src)
    merged = _enrich_pick_display_fields(
        merged,
        system_config=config,
        signal_date_col="signal_date",
        current_price_col="price",
    )
    signal_date = None
    signal_age_days = None
    signal_is_stale = None
    if "signal_date" in merged.columns and not merged.empty:
        signal_date = str(merged["signal_date"].iloc[0])
        signal_ts = pd.to_datetime(signal_date, errors="coerce")
        if pd.notna(signal_ts):
            signal_age_days = int((pd.Timestamp.today().normalize() - signal_ts.normalize()).days)
            signal_is_stale = signal_age_days > 0
    signal_config_path = _resolve_signal_config_path(profile)
    return {
        "signal_profile": profile,
        "signal_profile_display": SIGNAL_PROFILE_DISPLAY_NAMES.get(profile, profile),
        "signal_config": str(signal_config_path),
        "count": int(len(merged)),
        "source": source,
        "refresh": refresh,
        "trial_capital_fraction": trial_capital_fraction,
        "effective_trial_capital_fraction": effective_trial_capital_fraction,
        "max_single_position_fraction": max_single_position_fraction,
        "signal_date": signal_date,
        "signal_age_days": signal_age_days,
        "signal_is_stale": signal_is_stale,
        "execution_guard": execution_guard,
        "market_environment": market_environment,
        "confidence_summary": confidence_summary,
        "portfolio_impact": portfolio_impact,
        "items": _records(merged),
    }


def _build_realtime_watchlist(config, picks: pd.DataFrame, src: str) -> tuple[pd.DataFrame, str]:
    if picks.empty:
        return picks.copy(), str(src).lower()

    quote_frame = fetch_realtime_quotes(config, picks["stock_code"].astype(str).tolist(), src=src)
    quote_cols = [
        "stock_code",
        "ts_code",
        "name",
        "timestamp",
        "date",
        "time",
        "price",
        "pct_change",
        "change",
        "open",
        "pre_close",
        "high",
        "low",
        "bid",
        "ask",
        "volume",
        "amount",
    ]
    available_quote_cols = [column for column in quote_cols if column in quote_frame.columns]
    merged = picks.copy()
    merged["stock_code"] = merged["stock_code"].astype(str).str.zfill(6)
    if "date" in merged.columns:
        merged = merged.rename(columns={"date": "signal_date"})
    if "close" in merged.columns:
        merged = merged.rename(columns={"close": "signal_close"})
    if "amount" in merged.columns:
        merged = merged.rename(columns={"amount": "signal_amount"})

    merged = merged.merge(
        quote_frame[available_quote_cols].rename(columns={"amount": "realtime_amount", "date": "quote_date", "time": "quote_time"}),
        on="stock_code",
        how="left",
    )

    if {"price", "signal_close"}.issubset(merged.columns):
        previous_signal_close = pd.to_numeric(merged["signal_close"], errors="coerce").replace(0, pd.NA)
        merged["intraday_return_vs_signal_close"] = ((pd.to_numeric(merged["price"], errors="coerce") - previous_signal_close) / previous_signal_close) * 100.0

    preferred_columns = [
        "stock_code",
        "ts_code",
        "name",
        "layer",
        "signal_date",
        "signal_score",
        "timestamp",
        "quote_date",
        "quote_time",
        "prediction_prob",
        "primary_signal_score",
        "secondary_signal_score",
        "signal_close",
        "price",
        "intraday_return_vs_signal_close",
        "pct_change",
        "change",
        "turnover_rate",
        "signal_amount",
        "realtime_amount",
        "market_cap_yi",
        "execution_hold_days",
        "execution_take_profit",
        "execution_stop_loss",
        "trend_strength",
        "open",
        "high",
        "low",
        "bid",
        "ask",
        "volume",
    ]
    ordered_columns = [column for column in preferred_columns if column in merged.columns]
    remaining_columns = [column for column in merged.columns if column not in ordered_columns]
    merged = merged[ordered_columns + remaining_columns]
    return merged, quote_frame.attrs.get("realtime_source", str(src).lower())


def create_app(config_path: str | Path) -> FastAPI:
    config = load_config(config_path)
    app = FastAPI(title="Hierarchical Stock Analysis API", version="1.0.0")
    comparison_jobs: dict[str, dict] = {}
    comparison_jobs_lock = threading.Lock()

    def _snapshot_comparison_job(job_id: str) -> dict | None:
        with comparison_jobs_lock:
            job = comparison_jobs.get(job_id)
            if job is None:
                return None
            snapshot = dict(job)
            result = snapshot.get("result")
            if isinstance(result, dict) and result.get("qianxing_message"):
                snapshot["qianxing_message"] = result.get("qianxing_message")
            return snapshot

    def _summarize_comparison_job(job: dict, *, include_result: bool) -> dict:
        summary = dict(job)
        if not include_result:
            summary.pop("result", None)
        return summary

    def _list_comparison_jobs(*, limit: int, status: Optional[str] = None) -> list[dict]:
        with comparison_jobs_lock:
            jobs = [dict(job) for job in comparison_jobs.values()]
        if status:
            jobs = [job for job in jobs if str(job.get("status")) == status]
        jobs.sort(key=lambda job: str(job.get("created_at") or ""), reverse=True)
        trimmed = jobs[:limit]
        for job in trimmed:
            result = job.get("result")
            if isinstance(result, dict) and result.get("qianxing_message"):
                job["qianxing_message"] = result.get("qianxing_message")
        return trimmed

    def _run_comparison_job(job_id: str, *, parsed_variants: list[str], refresh: bool, write_outputs: bool) -> None:
        started_at = datetime.now().isoformat(sep=" ")
        with comparison_jobs_lock:
            comparison_jobs[job_id] = {
                "job_id": job_id,
                "job_type": "latest_profile_comparison_refresh",
                "status": "running",
                "variants": list(parsed_variants),
                "refresh": bool(refresh),
                "write_outputs": bool(write_outputs),
                "created_at": started_at,
                "started_at": started_at,
                "completed_at": None,
                "error": None,
                "result": None,
            }
        try:
            result = run_latest_profile_comparison(
                variants=parsed_variants,
                refresh=refresh,
                write_outputs=write_outputs,
            )
            with comparison_jobs_lock:
                comparison_jobs[job_id].update(
                    {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(sep=" "),
                        "result": result,
                        "qianxing_message": result.get("qianxing_message") if isinstance(result, dict) else None,
                    }
                )
        except Exception as exc:
            with comparison_jobs_lock:
                comparison_jobs[job_id].update(
                    {
                        "status": "failed",
                        "completed_at": datetime.now().isoformat(sep=" "),
                        "error": str(exc),
                        "qianxing_message": None,
                    }
                )

    @app.get("/health")
    def health():
        return {"status": "ok", "models_dir": str(config.models_dir), "results_dir": str(config.results_dir)}

    @app.post("/predict/latest")
    def predict_latest(
        req: PredictRequest,
        signal_profile: str = Query(
            DEFAULT_SIGNAL_PROFILE,
            description="信号配置档位: micro、large、large_rerank、broader_history_rerank_default_exec 或 broader_history_rerank_transfer_exec",
        ),
    ):
        try:
            profile = _normalize_signal_profile(signal_profile)
            if profile == "all_split":
                raise ValueError("/predict/latest 不支持 all_split，请改用 /market/realtime/latest-picks。")
            df = _load_transformer_latest_signals(top_n=req.top_n, refresh=True, use_calibration=False, signal_profile=profile)
            execution_guard = _resolve_prediction_health_execution_guard(profile)
            return {
                "count": int(len(df)),
                "signal_profile": profile,
                "signal_config": str(_resolve_signal_config_path(profile)),
                "execution_guard": execution_guard,
                "items": _records(df),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/backtest/run")
    def run_backtest(req: BacktestRequest):
        try:
            prediction_file = req.prediction_file or str(resolve_prediction_file(config.results_dir))
            result = Backtester(config_path).run(
                prediction_file=prediction_file,
                hold_days=req.hold_days,
                top_n=req.top_n,
                initial_capital=req.initial_capital,
            )
            return {
                "summary": result.summary,
                "trades": _records(result.trades.head(50)),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/market/realtime/quote")
    def get_realtime_quote(
        ts_codes: str = Query(..., description="逗号分隔的 ts_code 或 6 位股票代码"),
        src: str = Query("sina", description="实时行情数据源: sina、dc 或 auto"),
    ):
        try:
            codes = _parse_ts_codes(ts_codes)
            frame = fetch_realtime_quotes(config, codes, src=src)
            return {
                "count": int(len(frame)),
                "source": frame.attrs.get("realtime_source", str(src).lower()),
                "ts_codes": frame["ts_code"].tolist() if "ts_code" in frame.columns else codes,
                "items": _records(frame),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/market/realtime/list")
    def get_realtime_list(
        src: str = Query("auto", description="实时行情列表数据源: auto、dc 或 sina"),
        page_count: Optional[int] = Query(1, ge=1, description="抓取页数，默认 1 页"),
    ):
        try:
            frame = fetch_realtime_market_list(config, src=src, page_count=page_count)
            return {
                "count": int(len(frame)),
                "source": frame.attrs.get("realtime_source", str(src).lower()),
                "page_count": page_count,
                "items": _records(frame),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/market/realtime/latest-picks")
    def get_realtime_latest_picks(
        top_n: int = Query(20, ge=1, le=200, description="返回最新推荐中的前 N 只股票"),
        src: str = Query("sina", description="实时行情数据源: sina、dc 或 auto"),
        refresh: bool = Query(False, description="是否先实时重跑最新推理，再合并盘中行情"),
        signal_profile: str = Query(
            DEFAULT_SIGNAL_PROFILE,
            description="信号配置档位: micro、large、large_rerank、broader_history_rerank_default_exec、broader_history_rerank_transfer_exec 或 all_split",
        ),
        trial_capital_fraction: float = Query(DEFAULT_TRIAL_CAPITAL_FRACTION, ge=0.01, le=1.0, description="试运行总仓位占总资金比例"),
        max_single_position_fraction: float = Query(DEFAULT_MAX_SINGLE_POSITION_FRACTION, ge=0.01, le=1.0, description="单只股票总资金仓位上限"),
    ):
        try:
            profile = _normalize_signal_profile(signal_profile)
            if profile == "all_split":
                sections = [
                    _build_latest_pick_payload(
                        config,
                        config_path,
                        signal_profile=section_profile,
                        top_n=top_n,
                        refresh=refresh,
                        src=src,
                        trial_capital_fraction=trial_capital_fraction,
                        max_single_position_fraction=max_single_position_fraction,
                    )
                    for section_profile in ["micro", "large"]
                ]
                return {
                    "signal_profile": profile,
                    "count": int(sum(section["count"] for section in sections)),
                    "refresh": refresh,
                    "trial_capital_fraction": trial_capital_fraction,
                    "max_single_position_fraction": max_single_position_fraction,
                    "sections": sections,
                }

            return _build_latest_pick_payload(
                config,
                config_path,
                signal_profile=profile,
                top_n=top_n,
                refresh=refresh,
                src=src,
                trial_capital_fraction=trial_capital_fraction,
                max_single_position_fraction=max_single_position_fraction,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/market/realtime/latest-profile-comparison")
    def get_realtime_latest_profile_comparison(
        variants: str = Query("default_exec,transfer_exec", description="逗号分隔的 rerank variant 列表"),
        refresh: bool = Query(False, description="是否先重算单模型 latest 与 rerank latest，再返回对比结果"),
        write_outputs: bool = Query(True, description="是否写出 comparison json/markdown artifact"),
    ):
        try:
            parsed_variants = [item.strip() for item in str(variants).split(",") if item.strip()]
            if not parsed_variants:
                raise ValueError("variants 不能为空，至少提供一个 rerank variant。")
            return run_latest_profile_comparison(
                variants=parsed_variants,
                refresh=refresh,
                write_outputs=write_outputs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/market/realtime/latest-profile-comparison/refresh")
    def start_realtime_latest_profile_comparison_refresh(
        variants: str = Query("default_exec,transfer_exec", description="逗号分隔的 rerank variant 列表"),
        refresh: bool = Query(True, description="是否先重算 latest 再生成 comparison；false 时只复用现有 artifact"),
        write_outputs: bool = Query(True, description="是否写出 comparison json/markdown artifact"),
    ):
        try:
            parsed_variants = [item.strip() for item in str(variants).split(",") if item.strip()]
            if not parsed_variants:
                raise ValueError("variants 不能为空，至少提供一个 rerank variant。")
            job_id = str(uuid4())
            created_at = datetime.now().isoformat(sep=" ")
            with comparison_jobs_lock:
                comparison_jobs[job_id] = {
                    "job_id": job_id,
                    "job_type": "latest_profile_comparison_refresh",
                    "status": "queued",
                    "variants": list(parsed_variants),
                    "refresh": bool(refresh),
                    "write_outputs": bool(write_outputs),
                    "created_at": created_at,
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                    "result": None,
                    "qianxing_message": None,
                }
            worker = threading.Thread(
                target=_run_comparison_job,
                kwargs={
                    "job_id": job_id,
                    "parsed_variants": parsed_variants,
                    "refresh": refresh,
                    "write_outputs": write_outputs,
                },
                daemon=True,
            )
            worker.start()
            return {
                "job_id": job_id,
                "status": "queued",
                "variants": parsed_variants,
                "refresh": bool(refresh),
                "write_outputs": bool(write_outputs),
                "status_url": f"/market/realtime/latest-profile-comparison/jobs/{job_id}",
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/market/realtime/latest-profile-comparison/jobs/{job_id}")
    def get_realtime_latest_profile_comparison_refresh_job(
        job_id: str,
        include_result: bool = Query(True, description="是否返回完整 result 大对象；false 时只返回任务摘要"),
    ):
        job = _snapshot_comparison_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"comparison refresh job not found: {job_id}")
        return _summarize_comparison_job(job, include_result=include_result)

    @app.get("/market/realtime/latest-profile-comparison/jobs")
    def list_realtime_latest_profile_comparison_refresh_jobs(
        limit: int = Query(10, ge=1, le=100, description="返回最近任务条数"),
        status: Optional[str] = Query(None, description="可选状态过滤: queued、running、completed、failed"),
        include_result: bool = Query(False, description="是否返回完整 result 大对象；默认只返回任务摘要"),
    ):
        if status is not None and status not in {"queued", "running", "completed", "failed"}:
            raise HTTPException(status_code=400, detail=f"unsupported status filter: {status}")
        jobs = _list_comparison_jobs(limit=limit, status=status)
        return {
            "count": len(jobs),
            "limit": limit,
            "status_filter": status,
            "include_result": bool(include_result),
            "items": [_summarize_comparison_job(job, include_result=include_result) for job in jobs],
        }

    @app.post("/paper-trading/account/init")
    def init_paper_account(req: PaperAccountInitRequest):
        try:
            snapshot = initialize_account(
                config,
                initial_cash=req.initial_cash,
                positions=[position.model_dump() for position in req.positions],
                reset=req.reset,
                quote_source=req.quote_source,
            )
            return _paper_snapshot_response(snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/paper-trading/account")
    def get_paper_account(
        src: str = Query("auto", description="持仓估值使用的实时行情数据源"),
        trades_limit: int = Query(50, ge=1, le=500, description="返回最近交易记录条数"),
    ):
        try:
            snapshot = get_account_snapshot(config, src=src, trades_limit=trades_limit)
            return _paper_snapshot_response(snapshot)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/paper-trading/positions")
    def get_paper_positions(
        src: str = Query("auto", description="持仓估值使用的实时行情数据源"),
    ):
        try:
            snapshot = get_account_snapshot(config, src=src, trades_limit=1)
            return {
                "source": snapshot.get("source"),
                "summary": snapshot.get("summary", {}),
                "positions": _records(snapshot.get("positions", pd.DataFrame())),
            }
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/paper-trading/trades")
    def get_paper_trades(
        limit: int = Query(100, ge=1, le=1000, description="返回最近交易记录条数"),
    ):
        try:
            trades = get_trade_history(config, limit=limit)
            return {
                "count": int(len(trades)),
                "items": _records(trades),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/paper-trading/orders/buy")
    def buy_paper_stock(req: PaperBuyRequest):
        try:
            snapshot = place_buy_order(
                config,
                stock_code=req.stock_code,
                quantity=req.quantity,
                cash_amount=req.cash_amount,
                position_fraction=req.position_fraction,
                limit_price=req.limit_price,
                src=req.src,
                commission_rate=req.commission_rate,
                min_commission=req.min_commission,
                board_lot=req.board_lot,
                note=req.note,
            )
            return _paper_snapshot_response(snapshot)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/paper-trading/orders/sell")
    def sell_paper_stock(req: PaperSellRequest):
        try:
            snapshot = place_sell_order(
                config,
                stock_code=req.stock_code,
                quantity=req.quantity,
                position_fraction=req.position_fraction,
                limit_price=req.limit_price,
                src=req.src,
                commission_rate=req.commission_rate,
                min_commission=req.min_commission,
                stamp_duty_rate=req.stamp_duty_rate,
                note=req.note,
            )
            return _paper_snapshot_response(snapshot)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/paper-trading/orders/batch-buy")
    def batch_buy_paper_stocks(req: PaperBatchOrderRequest):
        try:
            snapshot = place_batch_buy_orders(
                config,
                orders=[item.model_dump() for item in req.orders],
            )
            return _paper_snapshot_response(snapshot)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/paper-trading/orders/batch-sell")
    def batch_sell_paper_stocks(req: PaperBatchOrderRequest):
        try:
            snapshot = place_batch_sell_orders(
                config,
                orders=[item.model_dump() for item in req.orders],
            )
            return _paper_snapshot_response(snapshot)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/paper-trading/quote")
    def get_paper_quote(
        stock_codes: Optional[str] = Query(None, description="逗号分隔的股票代码或 ts_code"),
        stock_code: Optional[str] = Query(None, description="单个股票代码，兼容旧调用方式"),
        src: str = Query("auto", description="实时行情数据源"),
    ):
        try:
            requested_codes = stock_codes or stock_code
            if not requested_codes:
                raise ValueError("stock_codes 或 stock_code 至少提供一个。")
            payload = get_position_watch(config, stock_codes=requested_codes, src=src)
            return {
                "source": payload.get("source"),
                "count": int(len(payload.get("quote", pd.DataFrame()))),
                "quote": _records(payload.get("quote", pd.DataFrame())),
                "holding": _records(payload.get("holding", pd.DataFrame())),
            }
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── 持仓价格刷新（钳星需求）──

    @app.post("/paper-trading/positions/refresh")
    def refresh_positions():
        """强制刷新持仓实时价格（腾讯行情）。"""
        try:
            snapshot = get_account_snapshot(config, src="tencent", trades_limit=1)
            return {
                "source": snapshot.get("source"),
                "summary": snapshot.get("summary", {}),
                "positions": _records(snapshot.get("positions", pd.DataFrame())),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/paper-trading/health")
    def paper_trading_health():
        """Paper trading 健康检查 + 持仓摘要。"""
        try:
            snapshot = get_account_snapshot(config, src="tencent", trades_limit=1)
            summary = snapshot.get("summary", {})
            positions = _records(snapshot.get("positions", pd.DataFrame()))
            return {
                "status": "ok",
                "source": snapshot.get("source"),
                "total_assets": summary.get("total_assets"),
                "unrealized_pnl": summary.get("unrealized_pnl"),
                "positions_count": len(positions),
                "positions": positions,
            }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    # ── 后台定时刷新持仓价格 ──
    _last_positions_refresh: float = 0.0

    @app.on_event("startup")
    async def start_position_refresh():
        import asyncio
        async def _refresh_loop():
            nonlocal _last_positions_refresh
            while True:
                await asyncio.sleep(300)  # 5 分钟
                try:
                    import time
                    snapshot = get_account_snapshot(config, src="tencent", trades_limit=1)
                    _last_positions_refresh = time.monotonic()
                except Exception:
                    pass
        asyncio.create_task(_refresh_loop())

    # ── 钳星快捷接口 ──

    @app.post("/positions")
    def post_positions_view():
        """POST 版持仓查询（用腾讯行情刷新），方便脚本调用。"""
        try:
            snapshot = get_account_snapshot(config, src="tencent", trades_limit=1)
            return {
                "source": snapshot.get("source"),
                "summary": snapshot.get("summary", {}),
                "positions": _records(snapshot.get("positions", pd.DataFrame())),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/trade")
    def post_trade(req: PaperBuyRequest):
        """快捷交易接口：stock_code + quantity/cash_amount，自动判断买卖。"""
        try:
            src = req.src or "tencent"
            # 先查当前持仓，判断是否已有该股票
            snapshot = get_account_snapshot(config, src=src, trades_limit=1)
            positions = snapshot.get("positions", pd.DataFrame())
            holding = pd.DataFrame() if positions.empty or "stock_code" not in positions.columns else positions[positions["stock_code"].str.zfill(6) == req.stock_code.zfill(6)]

            if not holding.empty and int(holding["quantity"].iloc[0]) > 0:
                # 有持仓 → 卖出
                qty = req.quantity if req.quantity else None
                frac = req.position_fraction if req.position_fraction else None
                snapshot = place_sell_order(
                    config,
                    stock_code=req.stock_code,
                    quantity=qty,
                    position_fraction=frac,
                    limit_price=req.limit_price,
                    src=src,
                    commission_rate=req.commission_rate,
                    min_commission=req.min_commission,
                    note=req.note,
                )
            else:
                # 无持仓 → 买入
                snapshot = place_buy_order(
                    config,
                    stock_code=req.stock_code,
                    quantity=req.quantity,
                    cash_amount=req.cash_amount,
                    position_fraction=req.position_fraction,
                    limit_price=req.limit_price,
                    src=src,
                    commission_rate=req.commission_rate,
                    min_commission=req.min_commission,
                    board_lot=req.board_lot,
                    note=req.note,
                )
            return _paper_snapshot_response(snapshot)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── 自定义重启端点（保持对外）──

    @app.post("/__restart__")
    async def restart_server():
        import os, sys, asyncio
        async def _shutdown():
            await asyncio.sleep(1)
            python = sys.executable
            os.execv(python, [python, *sys.argv])
        asyncio.create_task(_shutdown())
        return {"status": "restarting"}

    return app


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
app = create_app(DEFAULT_CONFIG_PATH)
