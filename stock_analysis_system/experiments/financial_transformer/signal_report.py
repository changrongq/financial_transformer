from __future__ import annotations

import argparse
from pathlib import Path
import json
import sqlite3
import subprocess
import sys
import warnings

import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"
IRL_SRC = PROJECT_ROOT / "experiments" / "inverse_rl_market_simulation" / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC, IRL_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.scoring import (
    annotate_confidence_tiers,
    summarize_confidence_distribution,
    summarize_market_environment,
)
from compare_latest_moneyflow_vs_rerank_variants import build_qianxing_comparison_message
from compare_latest_moneyflow_vs_rerank_variants import run_latest_profile_comparison
from predict_latest_broader_history_rerank import VARIANT_RULES
from predict_latest_broader_history_rerank import run_latest_broader_history_rerank
from financial_transformer_experiment.two_stage_execution import predict_latest_for_execution
from financial_transformer_experiment.two_stage_execution import resolve_two_stage_execution_spec
from financial_transformer_experiment.two_stage_execution import resolve_single_stage_execution_defaults
from hierarchical_system.config import load_config as load_system_config
from hierarchical_system.data_refresh import refresh_latest_data_bundle
from hierarchical_system.entry_advice import add_entry_advice
from hierarchical_system.features import layer_from_market_cap_yi
from hierarchical_system.paper_trading import get_account_snapshot


DEFAULT_EXECUTION_HOLD_DAYS = 45
DEFAULT_EXECUTION_TAKE_PROFIT = 0.14
DEFAULT_EXECUTION_STOP_LOSS = -0.06
DEFAULT_SIGNAL_PROFILE = "all_split"
DISABLED_DEFAULT_SIGNAL_PROFILES = {
    "large": (
        "默认 large 档位当前仍指向已被失败重训覆盖的 checkpoint，已临时停止对外默认输出。"
        "如需继续验证 large，请改用 --signal-profile large_experimental，"
        "或显式传 --config experiments/financial_transformer/config_moneyflow_selective_large_v1.yaml。"
    ),
}
SIGNAL_PROFILE_CONFIGS = {
    "micro": EXPERIMENT_ROOT / "config_two_stage_rerank_leader_tiered_v1.yaml",
    "micro_manual": EXPERIMENT_ROOT / "config_moneyflow_v1_concept_sentiment_manual_micro_v1.yaml",
    "large": EXPERIMENT_ROOT / "config_moneyflow_v1_large_v1.yaml",
    "large_experimental": EXPERIMENT_ROOT / "config_moneyflow_selective_large_v1.yaml",
    "large_rerank": EXPERIMENT_ROOT / "config_two_stage_rerank_large_v1.yaml",
    "filtered_v5": EXPERIMENT_ROOT / "config_moneyflow_v1_flowstructure_seed33_v5_eval.yaml",
}
SIGNAL_PROFILE_DISPLAY_NAMES = {
    "micro": "微盘",
    "micro_manual": "微盘单模型手工概念",
    "large": "大盘",
    "large_experimental": "大盘实验线",
    "large_rerank": "大盘 Rerank",
    "filtered_v5": "Filter试验线 (seed33, 54票)",
    "custom": "自定义",
}
BROADER_HISTORY_RERANK_DISPLAY_NAMES = {
    "default_exec": "Broader History Rerank 默认执行",
    "transfer_exec": "Broader History Rerank 迁移执行",
}
SINGLE_STAGE_PROFILE_DEFAULTS = {
    "micro_manual": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "min_score_ratio_to_leader": 0.0,
        "execution_hold_days": 30,
        "execution_take_profit": 0.16,
        "execution_stop_loss": -0.06,
        "leader_max_day_rank": None,
        "leader_hold_days": None,
        "leader_take_profit": None,
        "leader_stop_loss": None,
    },
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
    "large_experimental": {
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
    "filtered_v5": {
        "short_up_threshold": 0.254,
        "take_profit_threshold": 0.572,
        "stop_loss_threshold": 0.50,
        "top_k_per_day": 2,
        "min_score_ratio_to_leader": 0.0,
        "execution_hold_days": 35,
        "execution_take_profit": 0.14,
        "execution_stop_loss": -0.06,
        "leader_max_day_rank": None,
        "leader_hold_days": None,
        "leader_take_profit": None,
        "leader_stop_loss": None,
    },
}
GOVERNANCE_PROFILE_ARTIFACTS = {
    "micro": {
        "scorecard_json": "mainline_acceptance_scorecard.json",
        "health_json": "mainline_prediction_health.json",
        "candidate_label": "two_stage_leader_tiered_mainline",
        "latest_incremental_label": "two_stage_mainline",
    },
    "large": {
        "scorecard_json": "large_acceptance_scorecard.json",
        "candidate_label": "plain_large",
        "latest_incremental_label": "plain_large",
    },
    "large_experimental": {
        "scorecard_json": "large_acceptance_scorecard.json",
        "candidate_label": "large_experimental",
        "latest_incremental_label": "large_experimental",
    },
}
FILTERED_V5_DEFAULTS = {
    "short_up_threshold": 0.254,
    "take_profit_threshold": 0.572,
    "stop_loss_threshold": 0.50,
    "top_k_per_day": 2,
    "min_score_ratio_to_leader": 0.0,
    "execution_hold_days": 35,
    "execution_take_profit": 0.14,
    "execution_stop_loss": -0.06,
    "leader_max_day_rank": None,
    "leader_hold_days": None,
    "leader_take_profit": None,
    "leader_stop_loss": None,
}
ROLE_SNAPSHOT_WITH_SIGNALS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "inverse_rl_market_simulation"
    / "results"
    / "market_role_snapshot_daily_all"
    / "market_role_snapshot_with_signals.csv"
)
ROLE_SNAPSHOT_FALLBACK_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "inverse_rl_market_simulation"
    / "results"
    / "market_role_snapshot_daily"
    / "market_role_snapshot_with_signals.csv"
)
ROLE_SNAPSHOT_DAILY_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "inverse_rl_market_simulation"
    / "results"
    / "market_role_snapshot_daily"
)
ROLE_SNAPSHOT_ALL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "inverse_rl_market_simulation"
    / "results"
    / "market_role_snapshot_daily_all"
)
ROLE_SNAPSHOT_SCAN_SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "inverse_rl_market_simulation"
    / "scan_market_role_snapshot.py"
)
ROLE_ANALYSIS_COLUMNS = [
    "signal_date",
    "scan_date",
    "dominant_role",
    "tags",
    "herd_risk_score",
    "slow_build_score",
    "trend_watch_score",
    "review_priority_score",
]


def _load_market_cap_frame(system_config, stock_codes: list[str]) -> pd.DataFrame:
    if not stock_codes:
        return pd.DataFrame(columns=["stock_code", "market_cap_yi", "size_bucket"])
    normalized = sorted({str(code).zfill(6) for code in stock_codes if str(code).strip()})
    placeholders = ", ".join(["?"] * len(normalized))
    conn = sqlite3.connect(system_config.databases.market_cap_db)
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


def _format_bucket_weights(weights: dict[str, float]) -> str:
    if not weights:
        return "无"
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return " / ".join(f"{bucket}:{weight * 100:.1f}%" for bucket, weight in ordered)


def _format_confidence_label(confidence_summary: dict) -> str:
    label = str((confidence_summary or {}).get("label", "unknown"))
    mapping = {
        "leader_concentrated": "头部集中",
        "balanced": "分布均衡",
        "mixed": "轻度集中",
        "weak_confidence": "弱信号日",
        "single_candidate": "单票日",
        "unknown": "未知",
    }
    return mapping.get(label, label)


def _format_market_environment_label(market_environment: dict) -> str:
    label = str((market_environment or {}).get("label", "unknown"))
    mapping = {
        "high_risk": "高风险",
        "trend_up": "上行趋势",
        "trend_down": "下行趋势",
        "range": "震荡区间",
        "unknown": "未知",
    }
    return mapping.get(label, label)


def _format_market_environment_style(market_environment: dict) -> str:
    style = str((market_environment or {}).get("style", "neutral"))
    mapping = {
        "aggressive": "偏积极",
        "defensive": "偏防御",
        "neutral": "中性",
    }
    return mapping.get(style, style)


def _format_confidence_counts(confidence_summary: dict) -> str:
    counts = confidence_summary.get("counts", {}) if confidence_summary else {}
    return " / ".join(
        f"{label}{int(counts.get(tier, 0))}"
        for tier, label in [("high", "高"), ("medium", "中"), ("low", "低")]
    )


def _format_confidence_tier(tier: object) -> str:
    mapping = {
        "high": "高",
        "medium": "中",
        "low": "低",
        "unknown": "未知",
    }
    return mapping.get(str(tier), str(tier))


def _format_optional_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "无"
    return f"{float(value) * 100:.1f}%"


def _records_from_frame(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    available = [column for column in columns if column in df.columns]
    if not available:
        return []
    return df[available].to_dict(orient="records")


def _refresh_role_snapshot(
    candidate_file: Path,
    *,
    stock_col: str = "stock_code",
    score_col: str = "signal_score",
    date_col: str = "date",
    top_n: int = 20,
    max_trials: int = 20,
) -> dict:
    candidate_path = candidate_file.resolve()
    if not candidate_path.exists():
        return {
            "ran": False,
            "ok": False,
            "reason": f"candidate file not found: {candidate_path}",
            "candidate_file": str(candidate_path),
            "output_dir": str(ROLE_SNAPSHOT_DAILY_OUTPUT_DIR),
        }
    if not ROLE_SNAPSHOT_SCAN_SCRIPT_PATH.exists():
        return {
            "ran": False,
            "ok": False,
            "reason": f"scan script not found: {ROLE_SNAPSHOT_SCAN_SCRIPT_PATH}",
            "candidate_file": str(candidate_path),
            "output_dir": str(ROLE_SNAPSHOT_DAILY_OUTPUT_DIR),
        }

    command = [
        str(Path(sys.executable).resolve()),
        str(ROLE_SNAPSHOT_SCAN_SCRIPT_PATH.resolve()),
        "--candidate-file",
        str(candidate_path),
        "--candidate-stock-col",
        stock_col,
        "--candidate-score-col",
        score_col,
        "--candidate-date-col",
        date_col,
        "--candidate-top-n",
        str(max(int(top_n), 1)),
        "--max-trials",
        str(max(int(max_trials), 1)),
        "--output-dir",
        str(ROLE_SNAPSHOT_DAILY_OUTPUT_DIR.resolve()),
    ]

    process = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout_tail = "\n".join((process.stdout or "").splitlines()[-10:])
    stderr_tail = "\n".join((process.stderr or "").splitlines()[-10:])
    return {
        "ran": True,
        "ok": process.returncode == 0,
        "return_code": int(process.returncode),
        "candidate_file": str(candidate_path),
        "output_dir": str(ROLE_SNAPSHOT_DAILY_OUTPUT_DIR.resolve()),
        "candidate_top_n": int(max(int(top_n), 1)),
        "max_trials": int(max(int(max_trials), 1)),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(result):
        return default
    return result


def _build_role_comment(row: pd.Series) -> str:
    dominant_role = str(row.get("dominant_role") or "unknown")
    tags = str(row.get("role_tags") or "")
    herd_risk = _safe_float(row.get("herd_risk_score"), default=0.0)
    slow_build = _safe_float(row.get("slow_build_score"), default=0.0)
    trend_watch = _safe_float(row.get("trend_watch_score"), default=0.0)

    if "crowded_herd" in tags or herd_risk >= 0.55:
        return "拥挤交易偏强，谨慎追高"
    if dominant_role == "slow" and slow_build >= 0.60:
        return "慢资金主导，可分批跟踪"
    if dominant_role == "trend" or trend_watch >= 0.30:
        return "趋势动量活跃，关注放量确认"
    if dominant_role == "herd":
        return "短线情绪主导，注意波动"
    return "角色中性，按纪律执行"


def _compute_roles_live(stock_codes: list[str], end_date: str | None = None, max_trials: int = 3) -> pd.DataFrame:
    """对给定股票列表现场跑一次角色快照（不走缓存文件），返回 role_columns DataFrame。"""
    from dataclasses import replace
    from inverse_rl_market_simulation.config import load_config as load_irl_config
    from inverse_rl_market_simulation.data import load_stock_window
    from inverse_rl_market_simulation.optimize import run_calibration
    from inverse_rl_market_simulation.diagnostics import build_role_activity_vectors

    template_path = IRL_SRC.parent / "configs" / "config_single_stock_soft_switch_v1_300124.yaml"
    template_config = load_irl_config(str(template_path))
    if end_date is None:
        from datetime import date, timedelta
        d = date.today()
        end_date = str(d - timedelta(days=1) if d.weekday() < 5 else d - timedelta(days=max(1, d.weekday() - 4)))

    rows = []
    for stock_code in stock_codes:
        try:
            config = replace(
                template_config,
                experiment=replace(template_config.experiment, stock_code=str(stock_code), end_date=end_date),
                optimization=replace(template_config.optimization, n_trials=max_trials),
            )
            real_frame = load_stock_window(config)
            calibration = run_calibration(config, real_frame)
            best_sim = calibration["best_simulation"]
            role_vecs = build_role_activity_vectors(best_sim, str(stock_code))
            latest = role_vecs.iloc[-1]
            latest_sim = best_sim.iloc[-1]

            latest_close = float(latest_sim.get("real_close", 0) or 0)
            ma20 = float(real_frame["close"].tail(20).mean()) if "close" in real_frame.columns and len(real_frame) >= 20 else latest_close
            close_vs_ma20 = 0.0 if ma20 == 0 else latest_close / ma20 - 1.0
            above_cost = 1.0 if latest_close >= ma20 else 0.0

            role_5d = role_vecs[["herd_act", "slow_act", "trend_act"]].tail(5).mean()
            ha = float(latest["herd_act"])
            sa = float(latest["slow_act"])
            ta = float(latest["trend_act"])
            ha5 = float(role_5d["herd_act"])
            sa5 = float(role_5d["slow_act"])
            ta5 = float(role_5d["trend_act"])
            h2s = float(latest.get("herd_to_slow_weight", 0) or 0)
            s2t = float(latest.get("slow_to_trend_weight", 0) or 0)
            total_act = float(latest.get("total_activity", 0) or 0)
            dominant = str(latest.get("dominant_role", "none") or "none")

            herd_risk = 0.7 * ha + 0.3 * ha5 - 0.25 * h2s
            slow_build = 0.6 * sa + 0.4 * sa5 + 0.35 * above_cost + 0.35 * s2t
            trend_watch = 0.5 * ta + 0.3 * ta5 + 0.45 * s2t - 0.15 * ha
            review_priority = slow_build + 0.8 * trend_watch - 0.6 * herd_risk

            tags = []
            if herd_risk >= 0.45:
                tags.append("crowded_herd")
            if slow_build >= 0.65:
                tags.append("slow_accumulation")
            if trend_watch >= 0.35:
                tags.append("trend_watch")
            if close_vs_ma20 > 0.03:
                tags.append("above_cost_proxy")

            rows.append({
                "stock_code": str(stock_code),
                "role_scan_date": str(pd.Timestamp(latest["date"]).date()),
                "role_tags": "|".join(tags) if tags else "",
                "dominant_role": dominant,
                "herd_risk_score": round(herd_risk, 6),
                "slow_build_score": round(slow_build, 6),
                "trend_watch_score": round(trend_watch, 6),
                "review_priority_score": round(review_priority, 6),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "stock_code", "role_scan_date", "role_tags", "dominant_role",
            "herd_risk_score", "slow_build_score", "trend_watch_score", "review_priority_score",
        ])
    return pd.DataFrame(rows)


def _attach_role_analysis(df: pd.DataFrame, report_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    result = df.copy()
    role_columns = [
        "role_signal_date",
        "role_scan_date",
        "dominant_role",
        "role_tags",
        "herd_risk_score",
        "slow_build_score",
        "trend_watch_score",
        "review_priority_score",
        "role_comment",
    ]

    if result.empty:
        for column_name in role_columns:
            result[column_name] = pd.NA
        result["dominant_role"] = "unknown"
        result["role_tags"] = ""
        result["role_comment"] = "暂无角色快照"
        return result, {
            "available": False,
            "source": "live",
            "matched_count": 0,
            "coverage": 0.0,
            "report_date_filter": report_date,
            "message": "无候选股票，未执行角色匹配。",
        }

    # ======== 优先：现场跑角色（最可靠） ========
    codes = sorted(set(result["stock_code"].astype(str).str.zfill(6).tolist()))
    live_df = _compute_roles_live(codes, end_date=report_date, max_trials=3)
    if not live_df.empty:
        live_df["stock_code"] = live_df["stock_code"].astype(str).str.zfill(6)
        result = result.drop(columns=role_columns, errors="ignore")
        result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)
        result = result.merge(live_df, on="stock_code", how="left")
        result["role_signal_date"] = result.get("role_scan_date", pd.NA)
        result["dominant_role"] = result["dominant_role"].fillna("unknown")
        result["role_tags"] = result["role_tags"].fillna("")
        result["role_comment"] = result.apply(_build_role_comment, axis=1)
        matched = int(result["dominant_role"].notna().sum())
        return result, {
            "available": True,
            "source": "live",
            "matched_count": matched,
            "coverage": float(matched / max(len(result), 1)),
            "report_date_filter": report_date,
            "message": "现场计算角色向量。",
        }

    # ======== 回退：读缓存文件 ========

    if not snapshot_path.exists():
        for column_name in role_columns:
            result[column_name] = pd.NA
        result["dominant_role"] = "unknown"
        result["role_tags"] = ""
        result["role_comment"] = "暂无角色快照"
        return result, {
            "available": False,
            "source": str(snapshot_path),
            "matched_count": 0,
            "coverage": 0.0,
            "report_date_filter": report_date,
            "message": "角色快照文件不存在。",
        }

    role_df = pd.read_csv(snapshot_path, dtype={"stock_code": str})
    if role_df.empty or "stock_code" not in role_df.columns:
        for column_name in role_columns:
            result[column_name] = pd.NA
        result["dominant_role"] = "unknown"
        result["role_tags"] = ""
        result["role_comment"] = "暂无角色快照"
        return result, {
            "available": True,
            "source": str(snapshot_path),
            "matched_count": 0,
            "coverage": 0.0,
            "report_date_filter": report_date,
            "message": "角色快照为空或缺少 stock_code。",
        }

    role_df = role_df.copy()
    role_df["stock_code"] = role_df["stock_code"].astype(str).str.zfill(6)
    if "signal_date" in role_df.columns:
        role_df["signal_date"] = pd.to_datetime(role_df["signal_date"], errors="coerce")
    if "scan_date" in role_df.columns:
        role_df["scan_date"] = pd.to_datetime(role_df["scan_date"], errors="coerce")

    for column_name in ROLE_ANALYSIS_COLUMNS:
        if column_name not in role_df.columns:
            role_df[column_name] = pd.NA

    matched_pool = role_df
    parsed_report_date = pd.to_datetime(report_date, errors="coerce") if report_date else pd.NaT
    if pd.notna(parsed_report_date):
        date_col = "signal_date" if "signal_date" in role_df.columns else "scan_date"
        if date_col in role_df.columns:
            same_date = role_df[role_df[date_col] == parsed_report_date].copy()
            if not same_date.empty:
                matched_pool = same_date

    sort_cols = [c for c in ["signal_date", "scan_date"] if c in role_df.columns]
    if sort_cols:
        matched_pool = matched_pool.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    matched_pool = matched_pool.drop_duplicates(subset=["stock_code"], keep="first")
    matched_pool = matched_pool[
        ["stock_code", *ROLE_ANALYSIS_COLUMNS]
    ].rename(
        columns={
            "signal_date": "role_signal_date",
            "scan_date": "role_scan_date",
            "tags": "role_tags",
        }
    )

    result = result.drop(columns=role_columns, errors="ignore")
    result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)
    result = result.merge(matched_pool, on="stock_code", how="left")

    matched_count = int(result["dominant_role"].notna().sum())
    result["dominant_role"] = result["dominant_role"].fillna("unknown")
    result["role_tags"] = result["role_tags"].fillna("")
    result["role_comment"] = result.apply(_build_role_comment, axis=1)

    if "role_signal_date" in result.columns:
        result["role_signal_date"] = pd.to_datetime(result["role_signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "role_scan_date" in result.columns:
        result["role_scan_date"] = pd.to_datetime(result["role_scan_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return result, {
        "available": True,
        "source": str(snapshot_path),
        "matched_count": matched_count,
        "coverage": float(matched_count / max(len(result), 1)),
        "report_date_filter": report_date,
        "message": None,
    }


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_governance_summary(signal_profile: str) -> dict | None:
    artifact_spec = GOVERNANCE_PROFILE_ARTIFACTS.get(signal_profile)
    if not artifact_spec:
        return None

    results_dir = EXPERIMENT_ROOT / "results"
    scorecard_path = results_dir / artifact_spec["scorecard_json"]
    health_path = results_dir / artifact_spec.get("health_json", "") if artifact_spec.get("health_json") else None
    latest_incremental_path = results_dir / "latest_incremental_validation_summary.json"
    scorecard_payload = _load_json_if_exists(scorecard_path)
    health_payload = _load_json_if_exists(health_path) if health_path else None
    latest_incremental_payload = _load_json_if_exists(latest_incremental_path)

    candidate = None
    if scorecard_payload:
        if "candidate" in scorecard_payload:
            raw_candidate = scorecard_payload.get("candidate") or {}
            if raw_candidate.get("label") == artifact_spec["candidate_label"]:
                candidate = raw_candidate
        else:
            candidate = next(
                (
                    item
                    for item in scorecard_payload.get("candidates", [])
                    if item.get("label") == artifact_spec["candidate_label"]
                ),
                None,
            )

    latest_incremental_entry = None
    if latest_incremental_payload:
        latest_incremental_entry = next(
            (
                entry
                for entry in latest_incremental_payload.get("entries", [])
                if entry.get("label") == artifact_spec["latest_incremental_label"]
            ),
            None,
        )

    return {
        "available": bool(candidate or latest_incremental_entry),
        "profile": signal_profile,
        "scorecard": {
            "path": str(scorecard_path),
            "generated_at": scorecard_payload.get("generated_at") if scorecard_payload else None,
            "summary": scorecard_payload.get("summary") if scorecard_payload else None,
            "candidate": candidate,
        },
        "prediction_health": {
            "path": str(health_path) if health_path else None,
            "generated_at": health_payload.get("generated_at") if health_payload else None,
            "status": ((health_payload or {}).get("health_status") or {}).get("status"),
            "recommendation": ((health_payload or {}).get("health_status") or {}).get("recommendation"),
            "reasons": ((health_payload or {}).get("health_status") or {}).get("reasons") or [],
            "execution_plan": (health_payload or {}).get("execution_plan") or {},
        },
        "latest_incremental": {
            "path": str(latest_incremental_path),
            "generated_at": latest_incremental_payload.get("generated_at") if latest_incremental_payload else None,
            "interpretation": latest_incremental_payload.get("interpretation") if latest_incremental_payload else None,
            "entry": latest_incremental_entry,
        },
    }


def _format_governance_gate_status(status: object) -> str:
    mapping = {
        "pass": "通过",
        "fail": "失败",
        "partial": "部分通过",
        "unknown": "未知",
        "n/a": "不适用",
    }
    return mapping.get(str(status), str(status))


def _format_governance_decision(decision: object) -> str:
    mapping = {
        "keep_default": "保留默认",
        "not_promotable": "不可晋升",
        "experimental_only": "仅实验线",
        "research_only": "仅研究线",
    }
    return mapping.get(str(decision), str(decision))


def _render_governance_summary_lines(governance_summary: dict | None) -> list[str]:
    if not governance_summary:
        return []

    lines = ["", "## 治理与验证摘要", ""]
    scorecard = governance_summary.get("scorecard", {})
    candidate = scorecard.get("candidate") or {}
    prediction_health = governance_summary.get("prediction_health") or {}
    latest_incremental = governance_summary.get("latest_incremental", {})
    latest_entry = latest_incremental.get("entry") or {}

    if not governance_summary.get("available"):
        lines.append("- 当前档位暂无已接入的 acceptance / latest-incremental 治理产物。")
        return lines

    if scorecard.get("summary"):
        lines.append(f"- 治理总判断：`{scorecard['summary']}`")
    if candidate:
        lines.append(
            f"- 当前档位决策：`{_format_governance_decision(candidate.get('decision'))}`"
            f"（candidate=`{candidate.get('label', 'unknown')}`）。"
        )
        gates = candidate.get("gates") or []
        if gates:
            gate_text = " / ".join(
                f"{gate.get('id')}:{_format_governance_gate_status(gate.get('status'))}"
                for gate in gates
            )
            lines.append(f"- Gate 摘要：{gate_text}")
            first_fail_gate = next(
                (gate for gate in gates if str(gate.get("status")) not in {"pass", "n/a"}),
                None,
            )
            if first_fail_gate:
                lines.append(
                    f"- 当前主要约束：`{first_fail_gate.get('id')}`，{first_fail_gate.get('evidence', '无补充证据')}"
                )
    if scorecard.get("generated_at"):
        lines.append(f"- Acceptance scorecard 生成时间：`{scorecard['generated_at']}`")

    if prediction_health.get("status"):
        lines.extend(
            [
                "",
                "### 预测健康度",
                "",
                f"- 当前状态：`{prediction_health.get('status')}`",
                f"- 建议：{prediction_health.get('recommendation', '无')}",
                f"- 执行模式：`{(prediction_health.get('execution_plan') or {}).get('mode', 'normal')}` / 建议仓位倍率：`x{float((prediction_health.get('execution_plan') or {}).get('capital_fraction_multiplier', 1.0)):.2f}`",
            ]
        )
        reasons = prediction_health.get("reasons") or []
        for reason in reasons[:3]:
            lines.append(f"- {reason}")

    if latest_entry:
        latest_window = latest_entry.get("window") or {}
        latest_metrics = latest_entry.get("metrics") or {}
        lines.extend(
            [
                "",
                "### 最新增量验证",
                "",
                (
                    f"- 窗口：`{latest_window.get('start_date', 'unknown')} ~ {latest_window.get('end_date', 'unknown')}`"
                    if latest_window
                    else "- 窗口：未知"
                ),
                f"- 结果：total_return=`{latest_metrics.get('total_return', 0.0):.4f}` / sharpe=`{latest_metrics.get('sharpe_ratio', 0.0):.2f}` / signals=`{int(latest_metrics.get('signal_count', 0))}` / trades=`{int(latest_metrics.get('trade_count', 0))}`",
                f"- 解释：{latest_entry.get('notes', latest_incremental.get('interpretation') or '无')}",
                "- 口径提醒：latest incremental 只是最新小窗口 sanity check，不替代更长 recent-OOS / rolling 验证。",
            ]
        )

    return lines


def _extract_governance_brief(governance_summary: dict | None) -> str:
    if not governance_summary or not governance_summary.get("available"):
        return "暂无治理结论"
    scorecard = governance_summary.get("scorecard", {})
    candidate = scorecard.get("candidate") or {}
    decision = candidate.get("decision")
    summary = scorecard.get("summary")
    if decision and summary:
        return f"{_format_governance_decision(decision)} / {summary}"
    if decision:
        return _format_governance_decision(decision)
    return str(summary or "暂无治理结论")


def _resolve_prediction_health_execution_guard(signal_profile: str, governance_summary: dict | None) -> dict:
    if signal_profile != "micro":
        return {"blocked": False, "status": None, "reason": None, "mode": "normal", "capital_fraction_multiplier": 1.0}
    prediction_health = (governance_summary or {}).get("prediction_health") or {}
    status = str(prediction_health.get("status") or "").strip().lower()
    execution_plan = prediction_health.get("execution_plan") or {}
    mode = execution_plan.get("mode") or ("no_new_trades" if status == "pause" else "normal")
    capital_fraction_multiplier = float(execution_plan.get("capital_fraction_multiplier", 0.0 if status == "pause" else 1.0))
    if status != "pause":
        return {
            "blocked": False,
            "status": prediction_health.get("status"),
            "reason": prediction_health.get("recommendation"),
            "mode": mode,
            "capital_fraction_multiplier": capital_fraction_multiplier,
        }
    return {
        "blocked": True,
        "status": prediction_health.get("status"),
        "reason": prediction_health.get("recommendation") or "recent incremental prediction health is weak",
        "mode": mode,
        "capital_fraction_multiplier": capital_fraction_multiplier,
    }


def _resolve_aggregate_report_date(summaries: dict[str, dict]) -> str:
    dates = sorted(
        {
            str(summary.get("date"))
            for summary in summaries.values()
            if summary.get("date")
        }
    )
    if not dates:
        return "undated"
    return dates[-1]


def _load_results_artifact_json(filename: str) -> dict | None:
    return _load_json_if_exists(EXPERIMENT_ROOT / "results" / filename)


def _load_latest_profile_comparison_summary() -> dict | None:
    return _load_json_if_exists(
        EXPERIMENT_ROOT
        / "results"
        / "broader_history_rerank_variants_v1"
        / "latest_profile_comparison_signal_report_summary.json"
    )


def _build_typhoon_tracker_artifact(report_date: str, *, top_n: int = 30) -> dict:
    if not report_date or report_date == "undated":
        return {"available": False, "message": "signal_report 没有可用日期，跳过台风追踪。"}
    try:
        from typhoon_tracker import DEFAULT_CONFIG_PATH
        from typhoon_tracker import DEFAULT_OUTPUT_DIR
        from typhoon_tracker import TyphoonParams
        from typhoon_tracker import _write_outputs
        from typhoon_tracker import build_typhoon_snapshot

        candidates, typhoon_summary = build_typhoon_snapshot(
            config_path=DEFAULT_CONFIG_PATH,
            end_date=report_date,
            lookback_calendar_days=140,
            params=TyphoonParams(top_n=top_n),
        )
        paths = _write_outputs(candidates, typhoon_summary, DEFAULT_OUTPUT_DIR)
        export_cols = [
            "typhoon_id",
            "stock_code",
            "stock_name",
            "rating",
            "organization_score",
            "thermo_score",
            "dominant_role",
            "action",
            "primary_concept_name",
            "resonance_count",
            "structure_stability_days",
        ]
        export_cols = [col for col in export_cols if col in candidates.columns]
        return {
            "available": True,
            "summary": typhoon_summary,
            "output_files": {key: str(value) for key, value in paths.items()},
            "top": candidates[export_cols].head(top_n).to_dict(orient="records"),
        }
    except Exception as exc:
        return {"available": False, "message": f"台风追踪生成失败：{exc}"}


def _build_typhoon_embryo_artifact(report_date: str, *, top_n: int = 30) -> dict:
    if not report_date or report_date == "undated":
        return {"available": False, "message": "signal_report 没有可用日期，跳过台风胚胎追踪。"}
    try:
        from typhoon_tracker import DEFAULT_CONFIG_PATH
        from typhoon_tracker import DEFAULT_OUTPUT_DIR
        from typhoon_tracker import TyphoonParams, EmbryoParams
        from typhoon_tracker import _write_embryo_outputs
        from typhoon_tracker import build_embryo_tracker

        embryo_candidates, embryo_summary, incubation = build_embryo_tracker(
            config_path=DEFAULT_CONFIG_PATH,
            end_date=report_date,
            lookback_calendar_days=140,
            params=TyphoonParams(top_n=top_n),
            embryo_params=EmbryoParams(top_n=top_n),
        )
        paths = _write_embryo_outputs(embryo_candidates, embryo_summary, DEFAULT_OUTPUT_DIR, incubation_candidates=incubation if not incubation.empty else None)
        export_cols = [
            "typhoon_id",
            "stock_code",
            "stock_name",
            "organization_score",
            "o_slope",
            "o_rising",
            "dominant_role",
            "herd_to_slow_ratio",
            "embryo_score",
            "action",
            "primary_concept_name",
            "structure_stability_days",
        ]
        export_cols = [col for col in export_cols if col in embryo_candidates.columns]
        incubation_top = []
        if not incubation.empty:
            incub_export = [
                "typhoon_id", "stock_code", "stock_name",
                "organization_score", "o_slope", "dominant_role",
                "incubation_status", "incubation_action",
                "primary_concept_name", "structure_stability_days",
            ]
            incub_export = [c for c in incub_export if c in incubation.columns]
            incubation_top = incubation[incub_export].head(10).to_dict(orient="records")
        return {
            "available": True,
            "summary": embryo_summary,
            "output_files": {key: str(value) for key, value in paths.items()},
            "top": embryo_candidates[export_cols].head(top_n).to_dict(orient="records"),
            "incubation_top": incubation_top,
        }
    except Exception as exc:
        return {"available": False, "message": f"台风胚胎追踪生成失败：{exc}"}


def _render_typhoon_tracker_lines(typhoon_artifact: dict) -> list[str]:
    lines = ["", "## 股票台风追踪系统", ""]
    if not typhoon_artifact.get("available"):
        lines.append(f"- 状态：不可用，{typhoon_artifact.get('message', '未知原因')}")
        return lines

    summary = typhoon_artifact.get("summary") or {}
    outputs = typhoon_artifact.get("output_files") or {}
    top_rows = typhoon_artifact.get("top") or []
    lines.extend(
        [
            f"- 台风扫描日：`{summary.get('end_date', '无')}`",
            f"- 扫描股票数：`{int(summary.get('stock_count', 0))}`",
            f"- 编号热带低压：`{int(summary.get('numbered_depression_count', 0))}`",
            f"- 跟踪池输出：`{int(summary.get('tracked_count', 0))}`",
            f"- 台风明细报告：`{outputs.get('markdown', '无')}`",
            "",
            "| 编号 | 股票 | 评级 | O组织强度 | S热力评分 | 角色 | 行动 | 共振概念 | 共振数 | 稳定天数 |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in top_rows[:15]:
        stock_label = row.get("stock_code", "无")
        if row.get("stock_name"):
            stock_label = f"{stock_label} {row.get('stock_name')}"
        lines.append(
            "| {typhoon_id} | {stock} | {rating} | {organization_score:.3f} | {thermo_score:.3f} | {role} | {action} | {concept} | {resonance} | {stability} |".format(
                typhoon_id=row.get("typhoon_id", "无"),
                stock=stock_label,
                rating=row.get("rating", "无"),
                organization_score=float(row.get("organization_score", 0.0) or 0.0),
                thermo_score=float(row.get("thermo_score", 0.0) or 0.0),
                role=row.get("dominant_role", "NA") or "NA",
                action=row.get("action", "无"),
                concept=row.get("primary_concept_name", ""),
                resonance=int(row.get("resonance_count", 0) or 0),
                stability=int(row.get("structure_stability_days", 0) or 0),
            )
        )
    lines.extend(
        [
            "",
            "- 说明：台风系统是进攻天气模型，只做识别和跟踪；协同试验仍需角色、治理、市场环境和新闻人工确认。",
        ]
    )
    return lines


def _render_typhoon_embryo_lines(embryo_artifact: dict) -> list[str]:
    lines = ["", "## 股票台风胚胎追踪 (v0.3)", ""]
    if not embryo_artifact.get("available"):
        lines.append(f"- 状态：不可用，{embryo_artifact.get('message', '未知原因')}")
        return lines

    summary = embryo_artifact.get("summary") or {}
    outputs = embryo_artifact.get("output_files") or {}
    top_rows = embryo_artifact.get("top") or []
    lines.extend(
        [
            f"- 胚胎扫描日：`{summary.get('end_date', '无')}`",
            f"- 胚胎合格数 (O ∈ [0.55,0.78] + dO/dt>0)：`{int(summary.get('embryo_eligible_count', 0))}`",
            f"- 胚胎终选数：`{int(summary.get('embryo_selected_count', 0))}`",
            f"- 胚胎明细报告：`{outputs.get('markdown', '无')}`",
            "",
            "| 编号 | 股票 | O值 | O斜率 | 角色 | herding | 胚胎评分 | 行动 | 共振概念 | 稳定天数 |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for row in top_rows[:15]:
        stock_label = row.get("stock_code", "无")
        if row.get("stock_name"):
            stock_label = f"{stock_label} {row.get('stock_name')}"
        lines.append(
            "| {id} | {stock} | {o:.3f} | {slope:.4f} | {role} | {ratio:.3f} | {score:.3f} | {action} | {concept} | {stability} |".format(
                id=row.get("typhoon_id", "无"),
                stock=stock_label,
                o=float(row.get("organization_score", 0.0) or 0.0),
                slope=float(row.get("o_slope", 0.0) or 0.0),
                role=row.get("dominant_role", "NA") or "NA",
                ratio=float(row.get("herd_to_slow_ratio", 0.0) or 0.0),
                score=float(row.get("embryo_score", 0.0) or 0.0),
                action=row.get("action", "无"),
                concept=row.get("primary_concept_name", ""),
                stability=int(row.get("structure_stability_days", 0) or 0),
            )
        )
    lines.extend(
        [
            "",
            "- 说明：v0.3.1 胚胎逻辑捕捉「正在形成」而非「已经成形」的台风；O ∈ [0.55,0.78] + dO/dt>0 + 角色 OK + 稳定 ≤4d。",
            "- 孵化中：昨日在胚胎列表 + (今日 O ∈ (0.78,0.80) 且 herd) 或 今日涨停。",
        ]
    )
    incubation_top = embryo_artifact.get("incubation_top") or []
    if incubation_top:
        lines.extend(
            [
                "",
                "### 孵化中跟踪池 (v0.3.1)",
                "",
                f"- 孵化中数量：`{len(incubation_top)}`",
                "",
                "| 编号 | 股票 | O值 | O斜率 | 角色 | 状态 | 行动 | 共振概念 | 稳定天数 |",
                "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |",
            ]
        )
        for row in incubation_top[:10]:
            stock_label = row.get("stock_code", "无")
            if row.get("stock_name"):
                stock_label = f"{stock_label} {row.get('stock_name')}"
            lines.append(
                "| {id} | {stock} | {o:.3f} | {slope:.4f} | {role} | {status} | {action} | {concept} | {stability} |".format(
                    id=row.get("typhoon_id", "无"),
                    stock=stock_label,
                    o=float(row.get("organization_score", 0.0) or 0.0),
                    slope=float(row.get("o_slope", 0.0) or 0.0),
                    role=row.get("dominant_role", "NA") or "NA",
                    status=row.get("incubation_status", ""),
                    action=row.get("incubation_action", ""),
                    concept=row.get("primary_concept_name", ""),
                    stability=int(row.get("structure_stability_days", 0) or 0),
                )
            )
    return lines


def _build_typhoon_persistence_artifact(report_date: str) -> dict:
    """加载持续性预测模型对最新胚胎的输出（JSON格式）。"""
    if not report_date or report_date == "undated":
        return {"available": False, "message": "没有可用日期"}
    try:
        from typhoon_persistence_model import DEFAULT_OUTPUT_DIR
        predictor_dir = Path(str(DEFAULT_OUTPUT_DIR))
        json_path = predictor_dir / f"persistence_predictions_{report_date}.json"
        if not json_path.exists():
            if not (predictor_dir / "persistence_next_day.joblib").exists():
                return {"available": False, "message": "持续性预测模型尚未训练，请先运行 typhoon_persistence_model.py --train"}
            return {"available": False, "message": f"预测JSON不存在: {json_path}"}

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        predictions = payload.get("predictions", [])
        sorted_preds = sorted(predictions, key=lambda r: r.get("persistence_prob") or 0.0, reverse=True)
        return {
            "available": True,
            "source": str(json_path),
            "date": payload.get("date"),
            "count": payload.get("count", 0),
            "top": sorted_preds[:10],
        }
    except Exception as exc:
        return {"available": False, "message": f"持续性预测加载失败: {exc}"}


def _render_typhoon_persistence_lines(persistence_artifact: dict) -> list[str]:
    lines = ["", "## 台风胚胎持续性预测 (ML v1.0) — 观察池信号", ""]
    if not persistence_artifact.get("available"):
        lines.append(f"- 状态：不可用，{persistence_artifact.get('message', '未知原因')}")
        return lines

    lines.extend([
        "⚠️ 以下为模型观察池候选——非买入指令——需人工确认：",
        "  1. 基本面（业绩真实性、概念相关性）",
        "  2. 资金结构（主力净流入vs大单出货）",
        "  3. 买入时机（等回踩——不追涨停）",
        "  4. 仓位控制（单只≤3-5%，板块总仓≤10-15%）",
        "",
    ])

    top_rows = persistence_artifact.get("top") or []
    lines.extend([
        f"- 来源：`{persistence_artifact.get('source', '无')}`",
        "",
        "| 股票 | O值 | persistence | risk | 信号 | 真实换手 | 放大 | 归因 | 风控提示 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ])
    for row in top_rows[:10]:
        stock_label = row.get("stock_code", "无")
        if row.get("stock_name"):
            stock_label = f"{stock_label} {row.get('stock_name')}"

        star = row.get("star_rating", "N/A")
        p = row.get("persistence_prob")
        r = row.get("risk_prob")

        real_turn = row.get("real_turnover_rate")
        amplify = row.get("turnover_amplify_ratio")
        real_turn_text = f"{float(real_turn):.2f}%" if real_turn is not None else "N/A"
        amplify_text = f"{float(amplify):.1f}x" if amplify is not None else "N/A"

        attr_items = row.get("persistence_attribution") or []
        attr_text = " / ".join(
            f"{a.get('display', a.get('feature','?'))}({a.get('value',0):.2f})"
            for a in attr_items[:3]
        ) if attr_items else "-"

        warns = row.get("warnings") or []
        warn_text = " / ".join(warns[:2]) if warns else "-"

        lines.append(
            "| {stock} | {o:.3f} | {p} | {r} | {star} | {rt} | {amp} | {attr} | {warn} |".format(
                stock=stock_label,
                o=float(row.get("organization_score", 0.0) or 0.0),
                p=f"{float(p):.1%}" if p is not None else "N/A",
                r=f"{float(r):.1%}" if r is not None else "N/A",
                star=star,
                rt=real_turn_text,
                amp=amplify_text,
                attr=attr_text,
                warn=warn_text,
            )
        )
    lines.extend([
        "",
        "- 信号等级: `=== 强信号` (>85% + 低风险) / `== 中等信号` (70-85%) / `= 弱信号` (<70%)",
        "- persistence_prob：次日涨幅 > 3% 的概率；risk_prob：次日跌幅 > 3% 的概率",
        "- 真实换手：修正股权结构（剔除国资/实控人长锁筹码）后的真实换手率",
        "- 放大 = 真实换手率 / 名义换手率，>1 表示羊群效应被名义换手率低估",
        "- 风控提示：资金过热、机构反手、散户割肉等自动标注",
        "- 以上概率仅供辅助决策，不构成自动交易指令。",
    ])
    return lines


def _build_regime_timing_artifact() -> dict:
    """计算最新大盘择时信号（路线 C：ETF 择时）。

    规则：大盘（全市场等权）近 10 日均收益 < 0 → CASH（空仓）；否则 HOLD（持有指数 ETF）。
    每次调用都实时基于 stock_data 重新计算（不再读历史 JSON），
    并在计算后更新落盘 JSON，保证报告里的 ETF 信号始终是最新的。
    """
    try:
        system_config = load_system_config(PROJECT_ROOT / "config.yaml")
        trade_db = system_config.databases.trade_db
    except Exception:
        return {"available": False, "message": "无法加载 config.yaml"}

    try:
        conn = sqlite3.connect(trade_db)
        sd = pd.read_sql_query("SELECT * FROM stock_data", conn)
        conn.close()
    except Exception as exc:
        return {"available": False, "message": f"读取 stock_data 失败: {exc}"}

    sd.columns = [
        "stock_code", "date", "open", "high", "low", "close", "volume",
        "turnover", "amount", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
    ]
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.zfill(6)
    sd["close"] = pd.to_numeric(sd["close"], errors="coerce")
    sd = sd.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)

    sd["ret_1d"] = sd.groupby("stock_code")["close"].pct_change()
    daily = sd.groupby("date")["ret_1d"].mean().rename("market_return").reset_index()
    daily["regime_mean"] = daily["market_return"].rolling(10, min_periods=3).mean()

    latest = daily.dropna(subset=["regime_mean"]).iloc[-1]
    latest_date = latest["date"]
    regime_mean = float(latest["regime_mean"])
    market_return = float(latest["market_return"])

    action = "CASH" if regime_mean < 0.0 else "HOLD"
    advice = (
        "大盘近10日均收益转负，建议空仓/不持有ETF，等待转正"
        if action == "CASH"
        else "大盘近10日均收益为正，建议持有沪深300ETF(510300)"
    )

    recent = daily.dropna(subset=["regime_mean"]).tail(10)
    trend = [round(float(v), 6) for v in recent["regime_mean"].tolist()]

    artifact = {
        "available": True,
        "date": str(pd.Timestamp(latest_date).date()),
        "regime_mean_10d": regime_mean,
        "market_return": market_return,
        "action": action,
        "advice": advice,
        "regime_mean_trend": trend,
        "rule": "大盘10日均收益 < 0 → CASH；否则 HOLD 沪深300ETF",
        "source": "realtime",
    }

    # 实时计算结果顺手落盘，保持每日信号 JSON 与历史 CSV 同步更新
    try:
        results_dir = EXPERIMENT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        daily_json = results_dir / "regime_timing_daily_signal.json"
        daily_json.write_text(json.dumps({
            "date": artifact["date"],
            "raw_signal": action,
            "confirmed_signal": action,
            "regime_mean": round(regime_mean, 6),
            "market_return": round(market_return, 6),
            "window": 10,
            "threshold": 0.0,
            "regime_mean_trend": trend,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return artifact


def _render_regime_timing_lines(artifact: dict) -> list[str]:
    lines = ["", "## 大盘择时信号 (路线 C · ETF)", ""]
    if not artifact.get("available"):
        lines.append(f"- 状态：不可用，{artifact.get('message', '未知原因')}")
        return lines

    action = artifact.get("action", "未知")
    regime_mean = artifact.get("regime_mean_10d", 0.0)
    advice = artifact.get("advice", "")
    trend = artifact.get("regime_mean_trend", [])
    trend_text = " → ".join(f"{v*100:.2f}%" for v in trend) if trend else "-"

    if action == "CASH":
        action_text = "CASH（空仓，不持有 ETF）"
    elif action == "HOLD":
        action_text = "HOLD（持有沪深300ETF 510300）"
    else:
        action_text = action

    lines.extend([
        f"- 日期：`{artifact.get('date')}`",
        f"- 大盘近 10 日均收益：**{regime_mean*100:+.3f}%**",
        f"- 操作建议：**{action_text}**",
        f"- {advice}",
        f"- 近 10 日走势：{trend_text}",
        "",
        "> 说明：这是「择时 + 指数 ETF」策略（不选个股，标的用沪深300ETF 510300）。",
        "> 规则：大盘10日均收益 < 0 → CASH（空仓），否则 HOLD（持有 ETF）。",
        "",
    ])
    return lines


def _format_gate_status_brief(gates: list[dict]) -> str:
    if not gates:
        return "无"
    return " / ".join(
        f"{gate.get('id')}:{_format_governance_gate_status(gate.get('status'))}"
        for gate in gates
    )


def _generate_all_split_aggregate_report(
    summaries: dict[str, dict],
    *,
    skipped_profiles: dict[str, str],
) -> dict:
    output_dir = EXPERIMENT_ROOT / "results"
    report_date = _resolve_aggregate_report_date(summaries)
    summary_json_path = output_dir / f"latest_signal_report_all_split_{report_date}_summary.json"
    markdown_path = output_dir / f"latest_signal_report_all_split_{report_date}.md"
    latest_profile_comparison = _load_latest_profile_comparison_summary()
    if latest_profile_comparison:
        comparison_date = str(latest_profile_comparison.get("date") or "无")
        if comparison_date != str(report_date):
            latest_profile_comparison = None
    mainline_scorecard = _load_results_artifact_json("mainline_acceptance_scorecard.json")
    mainline_prediction_health = _load_results_artifact_json("mainline_prediction_health.json")
    large_scorecard = _load_results_artifact_json("large_acceptance_scorecard.json")
    typhoon_tracker = _build_typhoon_tracker_artifact(report_date, top_n=30)
    typhoon_embryo = _build_typhoon_embryo_artifact(report_date, top_n=30)
    typhoon_persistence = _build_typhoon_persistence_artifact(report_date)
    regime_timing = _build_regime_timing_artifact()

    profile_entries = []
    for profile, summary in summaries.items():
        thresholds = summary.get("thresholds", {})
        execution_policy = summary.get("execution_policy", {})
        market_environment = summary.get("market_environment", {})
        confidence_summary = summary.get("confidence_summary", {})
        latest_incremental = (summary.get("governance_summary") or {}).get("latest_incremental", {})
        latest_entry = latest_incremental.get("entry") or {}
        latest_metrics = latest_entry.get("metrics") or {}
        role_analysis = summary.get("role_analysis") or {}
        role_snapshot_refresh = summary.get("role_snapshot_refresh") or {}
        output_files = summary.get("output_files", {})
        profile_entries.append(
            {
                "signal_profile": profile,
                "signal_profile_display_name": summary.get("signal_profile_display_name", profile),
                "date": summary.get("date"),
                "prediction_count": int(summary.get("prediction_count", 0)),
                "filtered_count": int(summary.get("filtered_count", 0)),
                "top_pick": (summary.get("top_picks") or [None])[0],
                "thresholds": thresholds,
                "execution_policy": execution_policy,
                "market_environment": {
                    "label": _format_market_environment_label(market_environment),
                    "style": _format_market_environment_style(market_environment),
                },
                "role_snapshot_refreshed": bool(role_snapshot_refresh.get("ok", False)),
                "role_snapshot_top_n": int(role_snapshot_refresh.get("candidate_top_n", 0)),
                "role_matched_count": int(role_analysis.get("matched_count", 0)),
                "role_coverage": float(role_analysis.get("coverage", 0.0)),
                "confidence_summary": {
                    "label": _format_confidence_label(confidence_summary),
                    "counts": _format_confidence_counts(confidence_summary),
                },
                "governance_brief": _extract_governance_brief(summary.get("governance_summary")),
                "latest_incremental": {
                    "total_return": latest_metrics.get("total_return"),
                    "sharpe_ratio": latest_metrics.get("sharpe_ratio"),
                    "signal_count": latest_metrics.get("signal_count"),
                    "trade_count": latest_metrics.get("trade_count"),
                    "window": latest_entry.get("window"),
                },
                "output_files": {
                    "markdown_report": output_files.get("markdown_report"),
                    "summary_json": output_files.get("summary_json"),
                },
            }
        )

    active_snapshot = ROLE_SNAPSHOT_WITH_SIGNALS_PATH if ROLE_SNAPSHOT_WITH_SIGNALS_PATH.exists() else ROLE_SNAPSHOT_FALLBACK_PATH

    aggregate_summary = {
        "signal_profile": "all_split",
        "date": None if report_date == "undated" else report_date,
        "profiles": profile_entries,
        "role_snapshot_source": str(active_snapshot),
        "artifacts": {
            "latest_profile_comparison": latest_profile_comparison,
            "mainline_acceptance_scorecard": mainline_scorecard,
            "mainline_prediction_health": mainline_prediction_health,
            "large_acceptance_scorecard": large_scorecard,
            "typhoon_tracker": typhoon_tracker,
            "typhoon_embryo": typhoon_embryo,
            "typhoon_persistence": typhoon_persistence,
            "regime_timing": regime_timing,
        },
        "skipped_profiles": skipped_profiles,
        "output_files": {
            "summary_json": str(summary_json_path),
            "markdown_report": str(markdown_path),
        },
    }
    summary_json_path.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Transformer 多档位最新信号总报告",
        "",
        f"- 报告日期：`{report_date}`" if report_date != "undated" else "- 报告日期：无统一可用日期",
        f"- 汇总档位数：`{len(profile_entries)}`",
        f"- 跳过档位数：`{len(skipped_profiles)}`",
        (f"- 角色快照来源：`{active_snapshot}`" if active_snapshot.exists() else "- 角色快照来源：文件不存在"),
        "- 这份总报告用于把 micro 与 large 线的最新信号和治理结论放在一页内查看。",
        "",
        "## 汇总看板",
        "",
        "| 档位 | 日期 | 候选/预测 | Top1 | 市场环境 | 治理结论 | 最新增量验证 | 角色 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in profile_entries:
        top_pick = entry.get("top_pick") or {}
        latest_incremental = entry.get("latest_incremental") or {}
        if latest_incremental.get("total_return") is None:
            latest_text = "无"
        else:
            latest_text = (
                f"ret={float(latest_incremental['total_return']):.4f}, "
                f"sharpe={float(latest_incremental.get('sharpe_ratio', 0.0)):.2f}"
            )
        role_text_parts = []
        if entry.get("role_snapshot_refreshed"):
            role_text_parts.append(f"刷新ok({entry.get('role_snapshot_top_n', 0)}只)")
        matched = int(entry.get("role_matched_count", 0))
        coverage = float(entry.get("role_coverage", 0.0))
        role_text_parts.append(f"匹配{matched}/{int(entry.get('filtered_count', 0))}({coverage * 100:.0f}%)")
        role_text = " | ".join(role_text_parts)

        report_lines.append(
            "| {display} | {date} | {filtered}/{predicted} | {top1} | {market} | {governance} | {latest} | {role} |".format(
                display=entry.get("signal_profile_display_name", entry["signal_profile"]),
                date=entry.get("date") or "无",
                filtered=entry.get("filtered_count", 0),
                predicted=entry.get("prediction_count", 0),
                top1=top_pick.get("stock_code", "无"),
                market=entry.get("market_environment", {}).get("label", "未知"),
                governance=entry.get("governance_brief", "暂无治理结论"),
                latest=latest_text,
                role=role_text,
            )
        )

    report_lines.extend(_render_regime_timing_lines(regime_timing))
    report_lines.extend(_render_typhoon_tracker_lines(typhoon_tracker))
    report_lines.extend(_render_typhoon_embryo_lines(typhoon_embryo))
    report_lines.extend(_render_typhoon_persistence_lines(typhoon_persistence))

    for entry in profile_entries:
        top_pick = entry.get("top_pick") or {}
        thresholds = entry.get("thresholds") or {}
        execution_policy = entry.get("execution_policy") or {}
        market_environment = entry.get("market_environment") or {}
        confidence_summary = entry.get("confidence_summary") or {}
        matched = int(entry.get("role_matched_count", 0))
        coverage = float(entry.get("role_coverage", 0.0))
        refresh_ok = bool(entry.get("role_snapshot_refreshed", False))
        latest_incremental = entry.get("latest_incremental") or {}
        report_lines.extend(
            [
                "",
                f"## {entry.get('signal_profile_display_name', entry['signal_profile'])}",
                "",
                f"- signal_profile：`{entry['signal_profile']}`",
                f"- 预测日期：`{entry.get('date') or '无'}`",
                f"- 通过阈值筛选数：`{entry.get('filtered_count', 0)}` / 全部预测数：`{entry.get('prediction_count', 0)}`",
                (
                    f"- Top1：`{top_pick.get('stock_code', '无')}` / signal_score=`{float(top_pick.get('signal_score', 0.0)):.4f}`"
                    if top_pick
                    else "- Top1：无"
                ),
                (
                    f"- Top1 买入建议：{top_pick.get('entry_advice_text', '暂无')}"
                    if top_pick and top_pick.get('entry_advice_text')
                    else None
                ),
                (
                    f"- Top1 盘中建议：{top_pick.get('intraday_entry_text', '暂无')}"
                    if top_pick and top_pick.get('intraday_entry_text')
                    else None
                ),
                f"- 市场环境：`{market_environment.get('label', '未知')}` / 风格：`{market_environment.get('style', '未知')}`",
                f"- 置信度结构：`{confidence_summary.get('label', '未知')}`，高/中/低 = `{confidence_summary.get('counts', '无')}`",
                f"- 阈值：short_up > `{float(thresholds.get('short_up_gt', 0.0)):.3f}` / take_profit > `{float(thresholds.get('take_profit_gt', 0.0)):.3f}` / stop_loss < `{float(thresholds.get('stop_loss_lt', 0.0)):.2f}`",
                f"- 执行：top_k=`{int(execution_policy.get('top_k_per_day', 0))}` / hold=`{int(execution_policy.get('hold_days', 0))}` / TP=`{float(execution_policy.get('take_profit', 0.0)) * 100:.1f}%` / SL=`{float(execution_policy.get('stop_loss', 0.0)) * 100:.1f}%`",
                f"- 治理摘要：{entry.get('governance_brief', '暂无治理结论')}",
                f"- 角色快照刷新：`{'成功' if refresh_ok else '未刷新/失败'}`",
                f"- 角色匹配：`{matched}/{int(entry.get('filtered_count', 0))}`，覆盖率 `{coverage * 100:.1f}%`",
                (f"- 角色归属：`{(top_pick or {}).get('dominant_role', 'unknown')}` / `{(top_pick or {}).get('role_comment', '暂无')}`" if top_pick and top_pick.get('dominant_role') else "- 角色归属：暂无"),
                (
                    f"- 最新增量验证：ret=`{float(latest_incremental['total_return']):.4f}` / sharpe=`{float(latest_incremental.get('sharpe_ratio', 0.0)):.2f}` / signals=`{int(latest_incremental.get('signal_count', 0))}` / trades=`{int(latest_incremental.get('trade_count', 0))}`"
                    if latest_incremental.get("total_return") is not None
                    else "- 最新增量验证：无"
                ),
                f"- 明细报告：`{entry.get('output_files', {}).get('markdown_report', '无')}`",
            ]
        )

    if latest_profile_comparison:
        comparison_message = latest_profile_comparison.get("comparison_message") or {}
        base = latest_profile_comparison.get("base") or {}
        variants = latest_profile_comparison.get("variants") or []
        alignment = latest_profile_comparison.get("alignment") or {}
        output_files = latest_profile_comparison.get("output_files") or {}
        comparison_date = str(latest_profile_comparison.get("date") or "无")
        if comparison_date == str(report_date):
            report_lines.extend(
                [
                    "",
                    "## Latest Profile Comparison",
                    "",
                    f"- 对比日期：`{comparison_date}`",
                    "- 状态：`当日对比`，可作为当前路径分歧参考。",
                    f"- base 路线：`{base.get('display_name', base.get('label', 'unknown'))}` / top_stock=`{base.get('top_stock', '无')}` / selected=`{int(base.get('selected_count', 0))}`",
                    f"- rerank variants 数：`{len(variants)}` / 顶部是否完全一致：`{'是' if alignment.get('all_same_top_stock') else '否'}`",
                    f"- 当前分歧摘要：`{comparison_message.get('has_divergence', False)}` / base_top=`{comparison_message.get('base_top_stock', '无')}` / rerank_top=`{comparison_message.get('rerank_top_stock', '无')}`",
                    f"- 结论：{comparison_message.get('lines', ['无'])[0] if comparison_message.get('lines') else '无'}",
                    f"- 对比明细：`{output_files.get('markdown_report', output_files.get('markdown', '无'))}`",
                ]
            )
            if comparison_message.get("lines"):
                report_lines.extend(["", "### 对比解读", ""])
                for line in comparison_message.get("lines", []):
                    report_lines.append(f"- {line}")

    if mainline_scorecard:
        candidate = mainline_scorecard.get("candidate") or {}
        gates = candidate.get("gates") or []
        report_lines.extend(
            [
                "",
                "## Mainline Acceptance Scorecard",
                "",
                f"- 生成时间：`{mainline_scorecard.get('generated_at', '无')}`",
                f"- 总结：`{mainline_scorecard.get('summary', '无')}`",
                f"- 决策：`{_format_governance_decision(candidate.get('decision'))}` / candidate=`{candidate.get('label', 'unknown')}`",
                f"- Gate 摘要：{_format_gate_status_brief(gates)}",
            ]
        )
        if gates:
            report_lines.extend(["", "### Mainline Gates", ""])
            for gate in gates:
                report_lines.append(
                    f"- `{gate.get('id')}` `{_format_governance_gate_status(gate.get('status'))}`: {gate.get('evidence', '无')}"
                )

    if large_scorecard:
        candidates = large_scorecard.get("candidates") or []
        report_lines.extend(
            [
                "",
                "## Large Acceptance Scorecard",
                "",
                f"- 生成时间：`{large_scorecard.get('generated_at', '无')}`",
                f"- 总结：`{large_scorecard.get('summary', '无')}`",
            ]
        )
        for candidate in candidates:
            gates = candidate.get("gates") or []
            report_lines.extend(
                [
                    "",
                    f"### {candidate.get('label', 'unknown')}",
                    "",
                    f"- 决策：`{_format_governance_decision(candidate.get('decision'))}`",
                    f"- Gate 摘要：{_format_gate_status_brief(gates)}",
                ]
            )
            first_key_constraint = next(
                (gate for gate in gates if str(gate.get("status")) not in {"pass", "n/a"}),
                None,
            )
            if first_key_constraint:
                report_lines.append(
                    f"- 主要约束：`{first_key_constraint.get('id')}`，{first_key_constraint.get('evidence', '无')}"
                )

    if skipped_profiles:
        report_lines.extend(["", "## 跳过档位", ""])
        for profile, reason in skipped_profiles.items():
            report_lines.append(f"- `{profile}`: {reason}")

    report_lines = [line for line in report_lines if line is not None]
    markdown_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return aggregate_summary


def _validate_broader_history_rerank_args(args: argparse.Namespace) -> None:
    unsupported_flags = []
    if args.config:
        unsupported_flags.append("--config")
    if args.signal_profile:
        unsupported_flags.append("--signal-profile")
    if args.short_up_threshold is not None:
        unsupported_flags.append("--short-up-threshold")
    if args.take_profit_threshold is not None:
        unsupported_flags.append("--take-profit-threshold")
    if args.stop_loss_threshold is not None:
        unsupported_flags.append("--stop-loss-threshold")
    if args.top_k_per_day is not None:
        unsupported_flags.append("--top-k-per-day")
    if args.min_score_ratio_to_leader is not None:
        unsupported_flags.append("--min-score-ratio-to-leader")
    if args.execution_hold_days is not None:
        unsupported_flags.append("--execution-hold-days")
    if args.execution_take_profit is not None:
        unsupported_flags.append("--execution-take-profit")
    if args.execution_stop_loss is not None:
        unsupported_flags.append("--execution-stop-loss")
    if args.leader_max_day_rank is not None:
        unsupported_flags.append("--leader-max-day-rank")
    if args.leader_hold_days is not None:
        unsupported_flags.append("--leader-hold-days")
    if args.leader_take_profit is not None:
        unsupported_flags.append("--leader-take-profit")
    if args.leader_stop_loss is not None:
        unsupported_flags.append("--leader-stop-loss")
    if unsupported_flags:
        raise ValueError(
            "使用 --broader-history-rerank-variant 时不支持以下参数: "
            + ", ".join(unsupported_flags)
        )


def _generate_latest_profile_comparison_report(
    args: argparse.Namespace,
    *,
    refresh_summary,
) -> dict:
    payload = run_latest_profile_comparison(
        refresh=not args.skip_predict_latest,
        write_outputs=True,
    )
    output_files = dict(payload.get("output_files", {}))
    output_dir = Path(output_files["json"]).resolve().parent if output_files.get("json") else (EXPERIMENT_ROOT / "results" / "broader_history_rerank_variants_v1")
    summary_json_path = output_dir / "latest_profile_comparison_signal_report_summary.json"
    markdown_path = output_dir / "latest_profile_comparison_signal_report.md"
    qianxing_message_path = output_dir / "latest_profile_comparison_qianxing_message.md"

    base = payload["base"]
    variants = payload["variants"]
    alignment = payload["alignment"]

    summary = {
        "report_type": "latest_profile_comparison",
        "date": base.get("latest_available_date"),
        "generated_at": payload.get("generated_at"),
        "refresh": bool(payload.get("refresh", False)),
        "data_refresh": refresh_summary,
        "base": base,
        "variants": variants,
        "alignment": alignment,
        "comparison_message": payload.get("comparison_message") or build_qianxing_comparison_message(payload),
        "qianxing_message": (payload.get("comparison_message") or build_qianxing_comparison_message(payload))["text"],
        "output_files": {
            **output_files,
            "summary_json": str(summary_json_path),
            "markdown_report": str(markdown_path),
            "qianxing_message_markdown": str(qianxing_message_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Transformer Latest Profile Comparison 日报",
        "",
        f"- 对比日期：`{base.get('latest_available_date') or '无'}`",
        f"- 本次是否重算 latest：`{bool(payload.get('refresh', False))}`",
        f"- 生成时间：`{payload.get('generated_at')}`",
        f"- 顶部候选是否一致：`{alignment.get('all_same_top_stock')}`",
        f"- 顶部候选集合：`{', '.join(alignment.get('unique_top_stocks', [])) if alignment.get('unique_top_stocks') else '无'}`",
        "",
        "## 顶部候选对比",
        "",
        "| Profile | Latest Date | Candidates | Selected | Top Stock | Score |",
        "| --- | --- | ---: | ---: | --- | ---: |",
        f"| {base['label']} | {base['latest_available_date']} | {base['candidate_count']} | {base['selected_count']} | {base.get('top_stock') or 'none'} | {base.get('top_signal_score') if base.get('top_signal_score') is not None else 'n/a'} |",
    ]
    for variant in variants:
        report_lines.append(
            f"| {variant['label']} | {variant['latest_available_date']} | {variant['candidate_count']} | {variant['selected_count']} | {variant.get('top_stock') or 'none'} | {variant.get('top_rerank_score') if variant.get('top_rerank_score') is not None else 'n/a'} |"
        )

    report_lines.extend([
        "",
        "## 当前结论",
        "",
    ])
    if alignment.get("all_same_top_stock"):
        report_lines.append(f"- 当前单模型 base 与 rerank variants 顶部候选一致，统一为 `{alignment.get('unique_top_stocks', ['无'])[0] if alignment.get('unique_top_stocks') else '无'}`。")
    else:
        report_lines.append(f"- 当前单模型 base 与 rerank variants 顶部候选不一致：base 为 `{base.get('top_stock')}`，rerank 侧为 `{variants[0].get('top_stock') if variants else '无'}`。")
    report_lines.append("- 这个对比只代表最新可用日期的实时 operational difference，不替代历史 walkforward 结论。")
    report_lines.append("")
    report_lines.append("## Top Picks")
    report_lines.append("")
    report_lines.append(f"- `{base['label']}`: {json.dumps(base.get('top_picks', []), ensure_ascii=False)}")
    for variant in variants:
        report_lines.append(f"- `{variant['label']}`: {json.dumps(variant.get('top_picks', []), ensure_ascii=False)}")
    markdown_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    message_lines = list(summary["comparison_message"]["lines"])
    qianxing_message_path.write_text("\n".join(message_lines) + "\n", encoding="utf-8")
    return summary


def _build_portfolio_impact_summary(system_config, picks: pd.DataFrame) -> dict | None:
    try:
        snapshot = get_account_snapshot(system_config, src="auto", trades_limit=1)
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
    cap_frame = _load_market_cap_frame(system_config, holding_codes + pick_codes)

    if not positions.empty:
        positions["stock_code"] = positions["stock_code"].astype(str).str.zfill(6)
        positions = positions.merge(cap_frame, on="stock_code", how="left")
    else:
        positions = pd.DataFrame(columns=["stock_code", "position_weight", "size_bucket"])

    new_allocations = picks[[column for column in ["stock_code", "market_cap_yi", "suggested_total_capital_pct"] if column in picks.columns]].copy()
    if not new_allocations.empty:
        new_allocations["stock_code"] = new_allocations["stock_code"].astype(str).str.zfill(6)
        new_allocations = new_allocations.merge(
            cap_frame[["stock_code", "size_bucket"]],
            on="stock_code",
            how="left",
        )
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


def _normalize_signal_profile(signal_profile: str | None) -> str:
    profile = str(signal_profile or DEFAULT_SIGNAL_PROFILE).strip().lower()
    if profile not in SIGNAL_PROFILE_CONFIGS and profile != "all_split":
        supported = ", ".join(sorted([*SIGNAL_PROFILE_CONFIGS.keys(), "all_split"]))
        raise ValueError(f"不支持的 signal_profile: {profile}。可选值: {supported}")
    return profile


def _resolve_signal_config_path(signal_profile: str | None) -> Path:
    profile = _normalize_signal_profile(signal_profile)
    if profile == "all_split":
        raise ValueError("all_split 不是单一信号配置，请改用聚合模式。")
    disabled_reason = DISABLED_DEFAULT_SIGNAL_PROFILES.get(profile)
    if disabled_reason is not None:
        raise ValueError(disabled_reason)
    return SIGNAL_PROFILE_CONFIGS[profile].resolve()


def _default_signal_profiles_for_all_split() -> tuple[str, ...]:
    return tuple(
        profile
        for profile in ("micro", "large_experimental", "filtered_v5")
        if profile not in DISABLED_DEFAULT_SIGNAL_PROFILES
    )


def _infer_signal_profile_from_config_path(config_path: Path) -> str:
    resolved = config_path.resolve()
    for profile, candidate_path in SIGNAL_PROFILE_CONFIGS.items():
        if resolved == candidate_path.resolve():
            return profile
    return "custom"


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
        return {
            "short_up_threshold": 0.254,
            "take_profit_threshold": 0.572,
            "stop_loss_threshold": 0.50,
            "top_k_per_day": 2,
            "min_score_ratio_to_leader": 0.0,
            "execution_hold_days": DEFAULT_EXECUTION_HOLD_DAYS,
            "execution_take_profit": DEFAULT_EXECUTION_TAKE_PROFIT,
            "execution_stop_loss": DEFAULT_EXECUTION_STOP_LOSS,
            "leader_max_day_rank": None,
            "leader_hold_days": None,
            "leader_take_profit": None,
            "leader_stop_loss": None,
        }
    return dict(defaults)


def _resolve_use_calibration(args: argparse.Namespace, signal_profile: str) -> bool:
    if args.use_calibration is not None:
        return bool(args.use_calibration)
    return signal_profile == "micro"


def _prediction_count_label(signal_profile: str, spec) -> str:
    if spec is not None:
        return "两阶段候选池股票数"
    return "全部预测股票数"


def _prediction_count_explainer(signal_profile: str, spec) -> str | None:
    if spec is None:
        return None
    return (
        f"- 这里的候选池数不是全市场总股票数，而是 two-stage 执行口径下，"
        f"primary 侧先截到 `candidate_pool={spec.candidate_pool}` 后进入二阶段融合的股票数。"
    )


def _resolve_exit_policy(
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


def _format_table(df: pd.DataFrame) -> str:
    header = "| 排名 | 股票 | signal_score | 置信度 | short_up | take_profit | stop_loss | max_gain | max_drawdown | 收盘价 | 市值(亿) | 角色 | 角色备注 |"
    divider = "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
    rows = [header, divider]
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        tier = _format_confidence_tier(row.get("signal_confidence_tier"))
        row_dict = row.to_dict()
        row_dict.setdefault("dominant_role", "unknown")
        row_dict.setdefault("role_comment", "角色中性，按纪律执行")
        rows.append(
            "| {rank} | {stock_code} | {signal_score:.4f} | {tier} | {pred_short_up:.4f} | {pred_take_profit:.4f} | {pred_stop_loss:.4f} | {pred_max_gain:.4f} | {pred_max_drawdown:.4f} | {close:.2f} | {market_cap_yi:.2f} | {dominant_role} | {role_comment} |".format(
                rank=index,
                tier=tier,
                **row_dict,
            )
        )
    return "\n".join(rows)


def _allocate_positions(
    df: pd.DataFrame,
    total_capital_fraction: float,
    max_single_fraction: float,
) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result["suggested_total_capital_pct"] = []
        result["suggested_strategy_bucket_pct"] = []
        return result

    scores = df["signal_score"].clip(lower=0.0)
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


def _format_qianxing_table(df: pd.DataFrame) -> str:
    header = "| 排名 | 股票 | signal_score | 置信度 | 建议总仓位 | 试运行子仓位占比 | short_up | take_profit | stop_loss | 收盘价 | 角色 |"
    divider = "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    rows = [header, divider]
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        tier = _format_confidence_tier(row.get("signal_confidence_tier"))
        row_dict = row.to_dict()
        row_dict.setdefault("dominant_role", "unknown")
        rows.append(
            "| {rank} | {stock_code} | {signal_score:.4f} | {tier} | {suggested_total_capital_pct:.2f}% | {suggested_strategy_bucket_pct:.2f}% | {pred_short_up:.4f} | {pred_take_profit:.4f} | {pred_stop_loss:.4f} | {close:.2f} | {dominant_role} |".format(
                rank=index,
                tier=tier,
                **row_dict,
            )
        )
    return "\n".join(rows)


def _render_entry_advice_lines(df: pd.DataFrame) -> list[str]:
    if df.empty or "entry_advice_text" not in df.columns:
        return []
    lines = ["", "## 买入建议", ""]
    for _, row in df.iterrows():
        stock_code = str(row.get("stock_code", "未知"))
        plan = str(row.get("entry_plan") or "观察")
        text = str(row.get("entry_advice_text") or "暂无建议")
        lines.append(f"- `{stock_code}` `{plan}`: {text}")
        intraday_plan = row.get("intraday_entry_plan")
        intraday_text = row.get("intraday_entry_text")
        if intraday_plan or intraday_text:
            lines.append(
                f"- `{stock_code}` 盘中 `{str(intraday_plan or '观察')}`: {str(intraday_text or '暂无盘中建议')}"
            )
    return lines


def _generate_single_profile_report(
    args: argparse.Namespace,
    *,
    system_config,
    refresh_summary,
    config_path: Path,
    signal_profile: str,
) -> dict:
    config = load_config(config_path)
    spec = resolve_two_stage_execution_spec(config_path)
    use_calibration = _resolve_use_calibration(args, signal_profile)
    defaults = _resolve_execution_defaults(signal_profile, spec)
    short_up_threshold = args.short_up_threshold if args.short_up_threshold is not None else defaults["short_up_threshold"]
    take_profit_threshold = args.take_profit_threshold if args.take_profit_threshold is not None else defaults["take_profit_threshold"]
    stop_loss_threshold = args.stop_loss_threshold if args.stop_loss_threshold is not None else defaults["stop_loss_threshold"]
    top_k_per_day = args.top_k_per_day if args.top_k_per_day is not None else defaults["top_k_per_day"]
    min_score_ratio_to_leader = (
        args.min_score_ratio_to_leader
        if args.min_score_ratio_to_leader is not None
        else defaults["min_score_ratio_to_leader"]
    )
    execution_hold_days = args.execution_hold_days if args.execution_hold_days is not None else defaults["execution_hold_days"]
    execution_take_profit = args.execution_take_profit if args.execution_take_profit is not None else defaults["execution_take_profit"]
    execution_stop_loss = args.execution_stop_loss if args.execution_stop_loss is not None else defaults["execution_stop_loss"]
    leader_max_day_rank = args.leader_max_day_rank if args.leader_max_day_rank is not None else defaults["leader_max_day_rank"]
    leader_hold_days = args.leader_hold_days if args.leader_hold_days is not None else defaults["leader_hold_days"]
    leader_take_profit = args.leader_take_profit if args.leader_take_profit is not None else defaults["leader_take_profit"]
    leader_stop_loss = args.leader_stop_loss if args.leader_stop_loss is not None else defaults["leader_stop_loss"]
    prediction_horizon = int(config.experiment.prediction_horizon)

    if not args.skip_predict_latest:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            warnings.filterwarnings(
                "ignore",
                message=r"enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer\.norm_first was True",
                category=UserWarning,
            )
            predict_latest_for_execution(str(config_path), use_calibration=use_calibration)

    file_suffix = "_calibrated" if use_calibration else ""
    latest_path = config.results_dir / f"latest_predictions{file_suffix}.csv"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"{signal_profile} 档位缺少最新预测文件: {latest_path}。"
            "如果这是第一次生成该档位报告，请移除 --skip-predict-latest 后重试。"
        )
    latest_df = pd.read_csv(latest_path, dtype={"stock_code": str})
    latest_df["stock_code"] = latest_df["stock_code"].astype(str).str.zfill(6)
    if args.skip_refresh_role_snapshot:
        role_snapshot_refresh = {
            "ran": False,
            "ok": False,
            "reason": "skipped by --skip-refresh-role-snapshot",
            "candidate_file": str(latest_path),
            "output_dir": str(ROLE_SNAPSHOT_DAILY_OUTPUT_DIR),
        }
    else:
        role_snapshot_refresh = _refresh_role_snapshot(
            latest_path,
            score_col="signal_score",
            date_col="date",
            top_n=args.role_snapshot_top_n,
            max_trials=args.role_snapshot_max_trials,
        )

    filtered = latest_df[
        (latest_df["pred_short_up"] > short_up_threshold)
        & (latest_df["pred_take_profit"] > take_profit_threshold)
        & (latest_df["pred_stop_loss"] < stop_loss_threshold)
    ].copy()
    filtered = filtered.sort_values(["signal_score", "pred_take_profit", "pred_short_up"], ascending=[False, False, False]).reset_index(drop=True)
    if min_score_ratio_to_leader > 0.0 and not filtered.empty:
        leader_score = float(filtered["signal_score"].max())
        if abs(leader_score) > 1e-9:
            filtered["score_ratio_to_leader"] = filtered["signal_score"] / leader_score
            filtered = filtered[filtered["score_ratio_to_leader"] >= min_score_ratio_to_leader].copy()
            filtered = filtered.sort_values(["signal_score", "pred_take_profit", "pred_short_up"], ascending=[False, False, False]).reset_index(drop=True)
    if top_k_per_day and top_k_per_day > 0 and not filtered.empty:
        filtered = filtered.head(top_k_per_day).reset_index(drop=True)
    filtered = annotate_confidence_tiers(filtered)
    market_environment = summarize_market_environment(latest_df, filtered_count=len(filtered))
    confidence_summary = summarize_confidence_distribution(filtered)
    filtered = _resolve_exit_policy(
        filtered,
        base_hold_days=execution_hold_days,
        base_take_profit=execution_take_profit,
        base_stop_loss=execution_stop_loss,
        leader_max_day_rank=leader_max_day_rank,
        leader_hold_days=leader_hold_days,
        leader_take_profit=leader_take_profit,
        leader_stop_loss=leader_stop_loss,
    )

    report_date = (
        str(filtered["date"].values[0]) if not filtered.empty
        else (str(latest_df["date"].values[0]) if "date" in latest_df.columns and not latest_df.empty else None)
    )
    filtered, role_analysis = _attach_role_analysis(filtered, report_date=report_date)

    top_df = filtered.head(args.top_n).copy()
    top_df = add_entry_advice(top_df, system_config, signal_date_col="date")
    default_output_tag = spec.default_output_tag if spec and spec.default_output_tag else signal_profile
    output_tag = args.output_tag.strip() or default_output_tag

    calibrated_tag = f"{output_tag}_calibrated" if use_calibration else output_tag
    filtered_csv_path = config.results_dir / f"latest_predictions_filtered_{calibrated_tag}.csv"
    summary_json_path = config.results_dir / f"latest_predictions_filtered_{calibrated_tag}_summary.json"
    markdown_path = config.results_dir / f"latest_signal_report_{calibrated_tag}.md"
    qianxing_json_path = config.results_dir / f"latest_qianxing_signal_brief_{calibrated_tag}.json"
    qianxing_markdown_path = config.results_dir / f"latest_qianxing_signal_brief_{calibrated_tag}.md"
    qianxing_message_path = config.results_dir / f"latest_qianxing_signal_message_{calibrated_tag}.md"

    profile_display_name = SIGNAL_PROFILE_DISPLAY_NAMES.get(signal_profile, signal_profile)
    governance_summary = _resolve_governance_summary(signal_profile)
    execution_guard = _resolve_prediction_health_execution_guard(signal_profile, governance_summary)
    qianxing_df = _allocate_positions(
        filtered.head(args.qianxing_top_n).copy(),
        total_capital_fraction=args.trial_capital_fraction * float(execution_guard.get("capital_fraction_multiplier", 1.0)),
        max_single_fraction=args.max_single_position_fraction,
    )
    qianxing_df = add_entry_advice(qianxing_df, system_config, signal_date_col="date")
    portfolio_impact = _build_portfolio_impact_summary(system_config, qianxing_df)
    if execution_guard.get("blocked"):
        filtered = filtered.iloc[0:0].copy()
        top_df = top_df.iloc[0:0].copy()
        qianxing_df = qianxing_df.iloc[0:0].copy()
        role_analysis = {
            **role_analysis,
            "matched_count": 0,
            "coverage": 0.0,
            "message": "execution blocked by governance; effective candidate list is empty",
        }
        portfolio_impact = {
            "available": False,
            "message": f"当前因预测健康度状态 `{execution_guard.get('status')}` 暂停新的默认交易信号。",
        }
    filtered.to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")
    prediction_count_label = _prediction_count_label(signal_profile, spec)
    prediction_count_explainer = _prediction_count_explainer(signal_profile, spec)

    summary = {
        "signal_profile": signal_profile,
        "signal_profile_display_name": profile_display_name,
        "config_path": str(config_path),
        "date": report_date,
        "thresholds": {
            "short_up_gt": short_up_threshold,
            "take_profit_gt": take_profit_threshold,
            "stop_loss_lt": stop_loss_threshold,
        },
        "execution_policy": {
            "top_k_per_day": int(top_k_per_day),
            "min_score_ratio_to_leader": float(min_score_ratio_to_leader),
            "use_calibration": use_calibration,
            "auto_refresh_data": not args.skip_refresh_data,
            "auto_predict_latest": not args.skip_predict_latest,
            "refresh_market_caps": bool(args.refresh_market_caps),
            "hold_days": int(execution_hold_days),
            "take_profit": float(execution_take_profit),
            "stop_loss": float(execution_stop_loss),
            "leader_max_day_rank": int(leader_max_day_rank) if leader_max_day_rank is not None else None,
            "leader_hold_days": int(leader_hold_days) if leader_hold_days is not None else None,
            "leader_take_profit": float(leader_take_profit) if leader_take_profit is not None else None,
            "leader_stop_loss": float(leader_stop_loss) if leader_stop_loss is not None else None,
        },
        "data_refresh": refresh_summary,
        "prediction_count": int(len(latest_df)),
        "filtered_count": int(len(filtered)),
        "market_environment": market_environment,
        "confidence_summary": confidence_summary,
        "top_n": int(args.top_n),
        "qianxing_top_n": int(args.qianxing_top_n),
        "portfolio_impact": portfolio_impact,
        "role_snapshot_refresh": role_snapshot_refresh,
        "role_analysis": role_analysis,
        "governance_summary": governance_summary,
        "execution_guard": execution_guard,
        "effective_trial_capital_fraction": args.trial_capital_fraction * float(execution_guard.get("capital_fraction_multiplier", 1.0)),
        "top_picks": top_df[[
            "stock_code",
            "signal_score",
            "signal_confidence_pct",
            "signal_confidence_tier",
            "pred_short_up",
            "pred_take_profit",
            "pred_stop_loss",
            "pred_max_gain",
            "pred_max_drawdown",
            "close",
            "market_cap_yi",
            "entry_plan",
            "entry_trigger_price",
            "entry_buy_zone_low",
            "entry_buy_zone_high",
            "entry_support_price",
            "entry_stop_price",
            "entry_advice_text",
            "role_signal_date",
            "role_scan_date",
            "dominant_role",
            "role_tags",
            "herd_risk_score",
            "slow_build_score",
            "trend_watch_score",
            "review_priority_score",
            "role_comment",
        ]].to_dict(orient="records"),
        "qianxing_top_picks": qianxing_df[[
            "stock_code",
            "signal_score",
            "signal_confidence_pct",
            "signal_confidence_tier",
            "suggested_total_capital_pct",
            "suggested_strategy_bucket_pct",
            "pred_short_up",
            "pred_take_profit",
            "pred_stop_loss",
            "pred_max_gain",
            "pred_max_drawdown",
            "close",
            "market_cap_yi",
            "entry_plan",
            "entry_trigger_price",
            "entry_buy_zone_low",
            "entry_buy_zone_high",
            "entry_support_price",
            "entry_stop_price",
            "entry_advice_text",
            "role_signal_date",
            "role_scan_date",
            "dominant_role",
            "role_tags",
            "herd_risk_score",
            "slow_build_score",
            "trend_watch_score",
            "review_priority_score",
            "role_comment",
        ]].to_dict(orient="records"),
        "output_files": {
            "filtered_csv": str(filtered_csv_path),
            "summary_json": str(summary_json_path),
            "markdown_report": str(markdown_path),
            "qianxing_brief_json": str(qianxing_json_path),
            "qianxing_brief_markdown": str(qianxing_markdown_path),
            "qianxing_message_markdown": str(qianxing_message_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    qianxing_json_path.write_text(
        json.dumps(
            {
                "signal_profile": signal_profile,
                "signal_profile_display_name": profile_display_name,
                "date": report_date,
                "trial_capital_fraction": args.trial_capital_fraction * float(execution_guard.get("capital_fraction_multiplier", 1.0)),
                "requested_trial_capital_fraction": args.trial_capital_fraction,
                "max_single_position_fraction": args.max_single_position_fraction,
                "portfolio_impact": portfolio_impact,
                "top_picks": summary["qianxing_top_picks"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        f"# Transformer {profile_display_name} 最新信号日报",
        "",
        f"- 信号档位：`{signal_profile}` / `{profile_display_name}`",
        f"- 预测日期：`{report_date}`" if report_date else "- 预测日期：无可用结果",
        f"- {prediction_count_label}：`{len(latest_df)}`",
        f"- 通过阈值筛选数：`{len(filtered)}`",
        f"- 市场环境标签：`{_format_market_environment_label(market_environment)}` / 建议风格：`{_format_market_environment_style(market_environment)}`",
        f"- 候选置信度结构：`{_format_confidence_label(confidence_summary)}`，分布 `高/中/低 = {_format_confidence_counts(confidence_summary)}`",
        (
            f"- 角色快照刷新：`成功`，top_n=`{int(role_snapshot_refresh.get('candidate_top_n', 0))}`，max_trials=`{int(role_snapshot_refresh.get('max_trials', 0))}`"
            if role_snapshot_refresh.get("ran") and role_snapshot_refresh.get("ok")
            else f"- 角色快照刷新：`未成功`（{role_snapshot_refresh.get('reason') or role_snapshot_refresh.get('stderr_tail') or 'unknown'}）"
        ),
        f"- 角色快照匹配：`{int(role_analysis.get('matched_count', 0))}/{len(filtered)}`，覆盖率 `{float(role_analysis.get('coverage', 0.0)) * 100:.1f}%`",
        f"- short_up 阈值：`> {short_up_threshold:.2f}`",
        f"- take_profit 阈值：`> {take_profit_threshold:.2f}`",
        f"- stop_loss 阈值：`< {stop_loss_threshold:.2f}`",
        f"- 每日最多执行标的数：`{top_k_per_day}`",
        f"- 候选相对 leader 分数底线：`>= {min_score_ratio_to_leader:.2f}`" if min_score_ratio_to_leader > 0 else "- 候选相对 leader 分数底线：`未启用`",
        (
            f"- leader 分层退出：第 1 名用 `+{leader_take_profit * 100:.1f}% / {leader_stop_loss * 100:.1f}% / {leader_hold_days}`，其余保持基础退出"
            if leader_max_day_rank == 1 and leader_take_profit is not None and leader_stop_loss is not None and leader_hold_days is not None
            else "- leader 分层退出：`未启用`"
        ),
        (
            f"- 治理执行状态：`{execution_guard.get('status')}`，当前默认不输出新的可执行候选"
            if execution_guard.get("blocked")
            else f"- 治理执行状态：`{execution_guard.get('status') or 'normal'}` / 执行模式：`{execution_guard.get('mode') or 'normal'}` / 建议仓位倍率：`x{float(execution_guard.get('capital_fraction_multiplier', 1.0)):.2f}`"
        ),
        "",
        f"## 今日 {profile_display_name} Top 候选",
        "",
    ]
    if top_df.empty:
        if execution_guard.get("blocked"):
            report_lines.append(
                f"当前{profile_display_name}档位因预测健康度状态 `{execution_guard.get('status')}` 被治理层暂停，默认不输出新的可执行候选。"
            )
            if execution_guard.get("reason"):
                report_lines.append("")
                report_lines.append(f"治理原因：{execution_guard.get('reason')}")
        else:
            report_lines.append(f"当前没有{profile_display_name}股票通过阈值筛选。")
    else:
        report_lines.append(_format_table(top_df))
        report_lines.extend(_render_entry_advice_lines(top_df))
        report_lines.extend(
            [
                "",
                "## 市场环境",
                "",
                f"- {market_environment['summary']}",
                "",
                "## 置信度结构",
                "",
                f"- {confidence_summary['summary']}",
                f"- 分层统计：`高/中/低 = {_format_confidence_counts(confidence_summary)}`。",
                f"- 头部信号分数占比：Top1 `{_format_optional_pct(confidence_summary.get('score_concentration', {}).get('top1_share'))}`，Top2 `{_format_optional_pct(confidence_summary.get('score_concentration', {}).get('top2_share'))}`，Top3 `{_format_optional_pct(confidence_summary.get('score_concentration', {}).get('top3_share'))}`。",
                "",
                "## 执行提醒",
                "",
                f"- 这是一份{profile_display_name}试运行候选清单，默认执行当日综合分数最高的 `{top_k_per_day}` 只股票。",
                f"- 预测时间范围：模型先看未来 `{prediction_horizon}` 个交易日的方向/触发概率；这不是无限期看多。",
                prediction_count_explainer,
                f"- 当前默认阈值：`short_up > {short_up_threshold:.3f}`、`take_profit > {take_profit_threshold:.3f}`、`stop_loss < {stop_loss_threshold:.2f}`。",
                "- 置信度分级：按当日候选 `signal_score` 横截面分位数划分，前 20% 为 `高`，后 20% 为 `低`，其余为 `中`。",
                f"- 基础退出纪律：`+{execution_take_profit * 100:.1f}%` 止盈、`{execution_stop_loss * 100:.1f}%` 止损、`{execution_hold_days}` 个交易日上限。",
                (
                    f"- leader 分层：当日第 1 名单独使用 `+{leader_take_profit * 100:.1f}%` 止盈、`{leader_stop_loss * 100:.1f}%` 止损、`{leader_hold_days}` 个交易日上限。"
                    if leader_max_day_rank == 1 and leader_take_profit is not None and leader_stop_loss is not None and leader_hold_days is not None
                    else None
                ),
                "- 单只股票仓位建议仍控制在总资金的 `10%` 到 `15%`。",
            ]
        )
        if portfolio_impact and portfolio_impact.get("available"):
            report_lines.extend(
                [
                    "",
                    "## 组合影响提示",
                    "",
                    f"- 当前持仓市值层分布：{_format_bucket_weights(portfolio_impact.get('existing_bucket_weights', {}))}",
                    f"- 买入建议后市值层分布：{_format_bucket_weights(portfolio_impact.get('post_trade_bucket_weights', {}))}",
                    f"- 与现有持仓重叠股票数：`{portfolio_impact.get('overlap_stock_count', 0)}`",
                ]
            )
        elif portfolio_impact:
            report_lines.extend(["", "## 组合影响提示", "", f"- {portfolio_impact.get('message', '暂不可用')}"])
        report_lines.extend(_render_governance_summary_lines(governance_summary))
        report_lines = [line for line in report_lines if line is not None]
    markdown_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    qianxing_lines = [
        f"# 钳星 Transformer {profile_display_name} 今日简版候选",
        "",
        f"- 信号档位：`{signal_profile}` / `{profile_display_name}`",
        f"- 日期：`{report_date}`" if report_date else "- 日期：无可用结果",
        f"- 试运行总仓位建议：`{args.trial_capital_fraction * float(execution_guard.get('capital_fraction_multiplier', 1.0)) * 100:.0f}%` 的总资金",
        f"- 单只股票上限：`{args.max_single_position_fraction * 100:.0f}%` 的总资金",
        f"- 当前候选数：`{len(filtered)}`，默认执行当日第 `1-{top_k_per_day}` 名" if len(filtered) else f"- 当前没有{profile_display_name}候选股通过筛选",
        (
            f"- 治理执行状态：`{execution_guard.get('status')}`，当前默认不建议新增交易"
            if execution_guard.get("blocked")
            else f"- 治理执行状态：`{execution_guard.get('status') or 'normal'}` / 执行模式：`{execution_guard.get('mode') or 'normal'}` / 建议仓位倍率：`x{float(execution_guard.get('capital_fraction_multiplier', 1.0)):.2f}`"
        ),
        "",
        "## 今日建议关注",
        "",
    ]
    if qianxing_df.empty:
        if execution_guard.get("blocked"):
            qianxing_lines.append(
                f"今天{profile_display_name}档位因预测健康度状态 `{execution_guard.get('status')}` 暂停新增默认交易，仅保留观察。"
            )
        else:
            qianxing_lines.append(f"今天没有{profile_display_name}股票通过 Transformer 阈值筛选。")
    else:
        qianxing_lines.append(_format_qianxing_table(qianxing_df))
        qianxing_lines.extend(_render_entry_advice_lines(qianxing_df))
        qianxing_lines.extend(
            [
                "",
                "## 市场环境",
                "",
                f"- 当前标签：`{_format_market_environment_label(market_environment)}`，建议风格：`{_format_market_environment_style(market_environment)}`",
                f"- {market_environment['summary']}",
                "",
                "## 置信度结构",
                "",
                f"- {confidence_summary['summary']}",
                f"- 分层统计：`高/中/低 = {_format_confidence_counts(confidence_summary)}`。",
                f"- 头部信号占比：Top1 `{_format_optional_pct(confidence_summary.get('score_concentration', {}).get('top1_share'))}`，Top2 `{_format_optional_pct(confidence_summary.get('score_concentration', {}).get('top2_share'))}`。",
                "",
                "## 执行口径",
                "",
                f"- 默认只做上表前 `{top_k_per_day}` 名，仓位按 signal_score 分配，不建议继续向后扩散。",
                "- 买入时点：下一交易日开盘。",
                f"- 预测时间范围：这份信号主要评估未来 `{prediction_horizon}` 个交易日的短线方向与触发概率。",
                f"- 基础卖出纪律：`+{execution_take_profit * 100:.1f}%` 止盈、`{execution_stop_loss * 100:.1f}%` 止损、或持有满 `{execution_hold_days}` 个交易日。",
                (
                    f"- leader 分层：当日第 1 名改用 `+{leader_take_profit * 100:.1f}% / {leader_stop_loss * 100:.1f}% / {leader_hold_days}`。"
                    if leader_max_day_rank == 1 and leader_take_profit is not None and leader_stop_loss is not None and leader_hold_days is not None
                    else None
                ),
                "- 如果连续两笔止损，暂停新增仓位一周。",
            ]
        )
        if portfolio_impact and portfolio_impact.get("available"):
            qianxing_lines.extend(
                [
                    "",
                    "## 组合影响提示",
                    "",
                    f"- 当前持仓市值层分布：{_format_bucket_weights(portfolio_impact.get('existing_bucket_weights', {}))}",
                    f"- 买入建议后市值层分布：{_format_bucket_weights(portfolio_impact.get('post_trade_bucket_weights', {}))}",
                    f"- 与现有持仓重叠股票数：`{portfolio_impact.get('overlap_stock_count', 0)}`",
                ]
            )
        elif portfolio_impact:
            qianxing_lines.extend(["", "## 组合影响提示", "", f"- {portfolio_impact.get('message', '暂不可用')}"])
        qianxing_lines = [line for line in qianxing_lines if line is not None]
    qianxing_markdown_path.write_text("\n".join(qianxing_lines) + "\n", encoding="utf-8")

    if qianxing_df.empty:
        message_lines = [
            f"今天 Transformer 没有筛出符合阈值的{profile_display_name}候选，先不新增仓位，继续等下一批信号。",
        ]
    else:
        pick_text = "、".join(
            f"{row.stock_code}({row.suggested_total_capital_pct:.1f}%)"
            for _, row in qianxing_df.iterrows()
        )
        message_lines = [
            f"Transformer {profile_display_name}试运行信号已更新，{report_date} 共筛出 {len(filtered)} 只候选。",
            f"按当前默认口径，执行前 1 到 {top_k_per_day} 名，候选名单为：{pick_text}。",
            f"当前第 1 名候选置信度为{_format_confidence_tier(qianxing_df.iloc[0]['signal_confidence_tier'])}。",
            f"买入建议：{qianxing_df.iloc[0].get('entry_advice_text', '暂无')}。",
            f"盘中建议：{qianxing_df.iloc[0].get('intraday_entry_text', '暂无')}。",
            f"今日置信度结构为{_format_confidence_label(confidence_summary)}，高/中/低 = {_format_confidence_counts(confidence_summary)}。",
            f"这份信号先看未来 {prediction_horizon} 个交易日的短线方向，不是无限期持有判断。",
            (
                f"执行口径：下一交易日开盘买入；第 1 名用 +{leader_take_profit * 100:.1f}% / {leader_stop_loss * 100:.1f}% / {leader_hold_days}，其余用 +{execution_take_profit * 100:.1f}% / {execution_stop_loss * 100:.1f}% / {execution_hold_days}。"
                if leader_max_day_rank == 1 and leader_take_profit is not None and leader_stop_loss is not None and leader_hold_days is not None
                else f"执行口径不变：下一交易日开盘买入，+{execution_take_profit * 100:.1f}% 止盈，{execution_stop_loss * 100:.1f}% 止损，最多持有 {execution_hold_days} 个交易日。"
            ),
            f"总试运行仓位先控制在 {args.trial_capital_fraction * float(execution_guard.get('capital_fraction_multiplier', 1.0)) * 100:.0f}% 左右。",
        ]
    qianxing_message_path.write_text("\n".join(message_lines) + "\n", encoding="utf-8")

    return summary


def _generate_broader_history_rerank_report(
    args: argparse.Namespace,
    *,
    system_config,
    refresh_summary,
    variant: str,
) -> dict:
    output_dir = EXPERIMENT_ROOT / "results" / "broader_history_rerank_variants_v1" / variant / "latest"
    summary_path = output_dir / "latest_summary.json"

    if not args.skip_predict_latest:
        payload = run_latest_broader_history_rerank(
            variant=variant,
            use_calibration=args.use_calibration,
            top_n=args.top_n,
        )
    else:
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{variant} 缺少 latest_summary.json: {summary_path}。"
                "如果这是第一次生成该档位报告，请移除 --skip-predict-latest 后重试。"
            )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))

    latest_selected_path = Path(payload["outputs"]["latest_selected"])
    latest_candidate_pool_path = Path(payload["outputs"]["latest_candidate_pool"])
    if not latest_selected_path.exists() or not latest_candidate_pool_path.exists():
        raise FileNotFoundError("broader-history rerank latest outputs are incomplete")

    selected = pd.read_csv(latest_selected_path, dtype={"stock_code": str})
    candidates = pd.read_csv(latest_candidate_pool_path, dtype={"stock_code": str})
    for frame in (selected, candidates):
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)

    if args.skip_refresh_role_snapshot:
        role_snapshot_refresh = {
            "ran": False,
            "ok": False,
            "reason": "skipped by --skip-refresh-role-snapshot",
            "candidate_file": str(latest_candidate_pool_path),
            "output_dir": str(ROLE_SNAPSHOT_DAILY_OUTPUT_DIR),
        }
    else:
        role_snapshot_refresh = _refresh_role_snapshot(
            latest_candidate_pool_path,
            score_col="signal_score",
            date_col="date",
            top_n=args.role_snapshot_top_n,
            max_trials=args.role_snapshot_max_trials,
        )

    eval_config = load_config(payload["eval_config"])
    suffix = "_calibrated" if args.use_calibration else ""
    latest_path = eval_config.results_dir / f"latest_predictions{suffix}.csv"
    if latest_path.exists():
        latest_df = pd.read_csv(latest_path, dtype={"stock_code": str})
        latest_df["stock_code"] = latest_df["stock_code"].astype(str).str.zfill(6)
        if "date" in latest_df.columns and payload.get("latest_available_date"):
            latest_df["date"] = pd.to_datetime(latest_df["date"], errors="coerce")
            latest_df = latest_df[
                latest_df["date"] == pd.to_datetime(payload["latest_available_date"], errors="coerce")
            ].copy()
    else:
        latest_df = candidates.copy()

    selected, role_analysis = _attach_role_analysis(selected, report_date=payload.get("latest_available_date"))
    selected = annotate_confidence_tiers(selected)
    confidence_summary = summarize_confidence_distribution(selected)
    market_environment = summarize_market_environment(latest_df, filtered_count=len(selected))
    top_df = selected.head(args.top_n).copy()
    top_df = add_entry_advice(top_df, system_config, signal_date_col="date")
    qianxing_df = _allocate_positions(
        selected.head(args.qianxing_top_n).copy(),
        total_capital_fraction=args.trial_capital_fraction,
        max_single_fraction=args.max_single_position_fraction,
    )
    qianxing_df = add_entry_advice(qianxing_df, system_config, signal_date_col="date")
    portfolio_impact = _build_portfolio_impact_summary(system_config, qianxing_df)

    report_date = payload.get("latest_available_date")
    fair = dict(payload.get("fair", VARIANT_RULES[variant]))
    profile_name = f"broader_history_rerank_{variant}"
    profile_display_name = BROADER_HISTORY_RERANK_DISPLAY_NAMES.get(variant, profile_name)
    output_tag = args.output_tag.strip() or profile_name
    calibrated_tag = f"{output_tag}_calibrated" if args.use_calibration else output_tag

    filtered_csv_path = output_dir / f"latest_predictions_filtered_{calibrated_tag}.csv"
    summary_json_path = output_dir / f"latest_predictions_filtered_{calibrated_tag}_summary.json"
    markdown_path = output_dir / f"latest_signal_report_{calibrated_tag}.md"
    qianxing_json_path = output_dir / f"latest_qianxing_signal_brief_{calibrated_tag}.json"
    qianxing_markdown_path = output_dir / f"latest_qianxing_signal_brief_{calibrated_tag}.md"
    qianxing_message_path = output_dir / f"latest_qianxing_signal_message_{calibrated_tag}.md"

    selected.to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "signal_profile": profile_name,
        "signal_profile_display_name": profile_display_name,
        "config_path": payload["eval_config"],
        "date": report_date,
        "thresholds": {
            "short_up_gt": fair["short_up_threshold"],
            "take_profit_gt": fair["take_profit_threshold"],
            "stop_loss_lt": fair["stop_loss_threshold"],
        },
        "execution_policy": {
            "top_k_per_day": int(fair["top_k_per_day"]),
            "min_score_ratio_to_leader": 0.0,
            "use_calibration": bool(args.use_calibration),
            "auto_refresh_data": not args.skip_refresh_data,
            "auto_predict_latest": not args.skip_predict_latest,
            "refresh_market_caps": bool(args.refresh_market_caps),
            "hold_days": int(fair["hold_days"]),
            "take_profit": float(fair["take_profit"]),
            "stop_loss": float(fair["stop_loss"]),
            "leader_max_day_rank": None,
            "leader_hold_days": None,
            "leader_take_profit": None,
            "leader_stop_loss": None,
        },
        "data_refresh": refresh_summary,
        "prediction_count": int(len(latest_df)),
        "filtered_count": int(len(selected)),
        "candidate_count": int(len(candidates)),
        "market_environment": market_environment,
        "confidence_summary": confidence_summary,
        "top_n": int(args.top_n),
        "qianxing_top_n": int(args.qianxing_top_n),
        "portfolio_impact": portfolio_impact,
        "role_snapshot_refresh": role_snapshot_refresh,
        "role_analysis": role_analysis,
        "top_picks": _records_from_frame(
            top_df,
            [
                "stock_code",
                "rerank_score",
                "signal_score",
                "signal_confidence_pct",
                "signal_confidence_tier",
                "pred_short_up",
                "pred_take_profit",
                "pred_stop_loss",
                "pred_max_gain",
                "pred_max_drawdown",
                "close",
                "market_cap_yi",
                "entry_plan",
                "entry_trigger_price",
                "entry_buy_zone_low",
                "entry_buy_zone_high",
                "entry_support_price",
                "entry_stop_price",
                "entry_advice_text",
                "role_signal_date",
                "role_scan_date",
                "dominant_role",
                "role_tags",
                "herd_risk_score",
                "slow_build_score",
                "trend_watch_score",
                "review_priority_score",
                "role_comment",
            ],
        ),
        "qianxing_top_picks": _records_from_frame(
            qianxing_df,
            [
                "stock_code",
                "rerank_score",
                "signal_score",
                "signal_confidence_pct",
                "signal_confidence_tier",
                "suggested_total_capital_pct",
                "suggested_strategy_bucket_pct",
                "pred_short_up",
                "pred_take_profit",
                "pred_stop_loss",
                "pred_max_gain",
                "pred_max_drawdown",
                "close",
                "market_cap_yi",
                "entry_plan",
                "entry_trigger_price",
                "entry_buy_zone_low",
                "entry_buy_zone_high",
                "entry_support_price",
                "entry_stop_price",
                "entry_advice_text",
                "role_signal_date",
                "role_scan_date",
                "dominant_role",
                "role_tags",
                "herd_risk_score",
                "slow_build_score",
                "trend_watch_score",
                "review_priority_score",
                "role_comment",
            ],
        ),
        "output_files": {
            "filtered_csv": str(filtered_csv_path),
            "summary_json": str(summary_json_path),
            "markdown_report": str(markdown_path),
            "qianxing_brief_json": str(qianxing_json_path),
            "qianxing_brief_markdown": str(qianxing_markdown_path),
            "qianxing_message_markdown": str(qianxing_message_path),
            "latest_candidate_pool": str(latest_candidate_pool_path),
            "latest_selected": str(latest_selected_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    qianxing_json_path.write_text(
        json.dumps(
            {
                "signal_profile": profile_name,
                "signal_profile_display_name": profile_display_name,
                "date": report_date,
                "trial_capital_fraction": args.trial_capital_fraction,
                "max_single_position_fraction": args.max_single_position_fraction,
                "portfolio_impact": portfolio_impact,
                "top_picks": summary["qianxing_top_picks"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        f"# Transformer {profile_display_name} 最新信号日报",
        "",
        f"- 最新交易日: {report_date or '无'}",
        f"- latest 候选池数量: {len(candidates)}",
        f"- 最终入选数量: {len(selected)}",
        f"- rerank 规则: top_k={fair['top_k_per_day']} / hold={fair['hold_days']} / TP=+{fair['take_profit'] * 100:.1f}% / SL={fair['stop_loss'] * 100:.1f}%",
        f"- 信号阈值: short_up>{fair['short_up_threshold']:.2f}, take_profit>{fair['take_profit_threshold']:.2f}, stop_loss<{fair['stop_loss_threshold']:.2f}",
        (
            f"- 角色快照刷新: 成功 (top_n={int(role_snapshot_refresh.get('candidate_top_n', 0))}, max_trials={int(role_snapshot_refresh.get('max_trials', 0))})"
            if role_snapshot_refresh.get("ran") and role_snapshot_refresh.get("ok")
            else f"- 角色快照刷新: 未成功 ({role_snapshot_refresh.get('reason') or role_snapshot_refresh.get('stderr_tail') or 'unknown'})"
        ),
        f"- 角色快照匹配: {int(role_analysis.get('matched_count', 0))}/{len(selected)} (覆盖率 {float(role_analysis.get('coverage', 0.0)) * 100:.1f}%)",
        f"- 市场环境: {_format_market_environment_label(market_environment)} / {_format_market_environment_style(market_environment)}",
        f"- 置信度结构: {_format_confidence_label(confidence_summary)}，高/中/低 = {_format_confidence_counts(confidence_summary)}",
        "",
        "## Top Picks",
        "",
        _format_table(top_df) if not top_df.empty else "暂无符合条件的 rerank 候选。",
        *_render_entry_advice_lines(top_df),
        "",
        "## 千行简报",
        "",
        _format_qianxing_table(qianxing_df) if not qianxing_df.empty else "暂无可分配仓位的候选。",
        *_render_entry_advice_lines(qianxing_df),
    ]
    markdown_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    qianxing_markdown_lines = [
        f"# {profile_display_name} 最新千行简报",
        "",
        f"- 最新交易日: {report_date or '无'}",
        f"- 试运行总仓位: {args.trial_capital_fraction * 100:.1f}%",
        f"- 单票上限: {args.max_single_position_fraction * 100:.1f}%",
        "",
        _format_qianxing_table(qianxing_df) if not qianxing_df.empty else "暂无可分配仓位的候选。",
        *_render_entry_advice_lines(qianxing_df),
    ]
    qianxing_markdown_path.write_text("\n".join(qianxing_markdown_lines) + "\n", encoding="utf-8")

    if not qianxing_df.empty:
        first_row = qianxing_df.iloc[0]
        message_lines = [
            f"{profile_display_name} 最新信号日: {report_date}。",
            f"当前第 1 名候选是 {first_row['stock_code']}，rerank_score={float(first_row.get('rerank_score', 0.0)):.4f}，signal_score={float(first_row.get('signal_score', 0.0)):.4f}。",
            f"买入建议：{first_row.get('entry_advice_text', '暂无')}。",
            f"盘中建议：{first_row.get('intraday_entry_text', '暂无')}。",
            f"执行口径：下一交易日开盘买入，+{fair['take_profit'] * 100:.1f}% 止盈，{fair['stop_loss'] * 100:.1f}% 止损，最多持有 {fair['hold_days']} 个交易日。",
            f"总试运行仓位先控制在 {args.trial_capital_fraction * 100:.0f}% 左右。",
        ]
    else:
        message_lines = [
            f"{profile_display_name} 最新信号日: {report_date or '无'}。",
            "当前没有满足 broader-history rerank 条件的候选。",
        ]
    qianxing_message_path.write_text("\n".join(message_lines) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate latest multitask transformer signal report. "
            "Default all_split currently emits micro only; large should use large_experimental."
        )
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--signal-profile",
        help=(
            "Signal profile name. Supported: all_split, micro, micro_manual, large_experimental, large_rerank, filtered_v5. "
            "The legacy large profile is temporarily disabled because its default checkpoint was overwritten by a failed retrain."
        ),
    )
    parser.add_argument("--broader-history-rerank-variant", choices=sorted(VARIANT_RULES))
    parser.add_argument("--latest-profile-comparison", action="store_true")
    parser.add_argument("--skip-refresh-data", action="store_true")
    parser.add_argument("--skip-predict-latest", action="store_true")
    parser.add_argument("--refresh-market-caps", action="store_true")
    parser.add_argument("--short-up-threshold", type=float)
    parser.add_argument("--take-profit-threshold", type=float)
    parser.add_argument("--stop-loss-threshold", type=float)
    parser.add_argument("--top-k-per-day", type=int)
    parser.add_argument("--min-score-ratio-to-leader", type=float)
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument("--use-calibration", dest="use_calibration", action="store_true")
    calibration_group.add_argument("--no-calibration", dest="use_calibration", action="store_false")
    parser.set_defaults(use_calibration=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--qianxing-top-n", type=int, default=5)
    parser.add_argument("--trial-capital-fraction", type=float, default=0.25)
    parser.add_argument("--max-single-position-fraction", type=float, default=0.07)
    parser.add_argument("--execution-hold-days", type=int)
    parser.add_argument("--execution-take-profit", type=float)
    parser.add_argument("--execution-stop-loss", type=float)
    parser.add_argument("--leader-max-day-rank", type=int)
    parser.add_argument("--leader-hold-days", type=int)
    parser.add_argument("--leader-take-profit", type=float)
    parser.add_argument("--leader-stop-loss", type=float)
    parser.add_argument("--skip-refresh-role-snapshot", action="store_true")
    parser.add_argument("--role-snapshot-top-n", type=int, default=20)
    parser.add_argument("--role-snapshot-max-trials", type=int, default=20)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    signal_profile = _normalize_signal_profile(args.signal_profile or DEFAULT_SIGNAL_PROFILE)
    if args.config and args.signal_profile is not None and signal_profile == "all_split":
        raise ValueError("使用 --config 时不支持 --signal-profile all_split，请改为单一 profile 或移除 --config。")

    system_config = load_system_config(PROJECT_ROOT / "config.yaml")
    refresh_summary = None
    if not args.skip_refresh_data:
        refresh_summary = refresh_latest_data_bundle(
            system_config,
            refresh_market_caps=args.refresh_market_caps,
            refresh_existing_market_caps=args.refresh_market_caps,
        )

    if args.latest_profile_comparison:
        if args.config or args.signal_profile or args.broader_history_rerank_variant:
            raise ValueError("使用 --latest-profile-comparison 时，不应再同时传 --config、--signal-profile 或 --broader-history-rerank-variant。")
        summary = _generate_latest_profile_comparison_report(
            args,
            refresh_summary=refresh_summary,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.broader_history_rerank_variant:
        _validate_broader_history_rerank_args(args)
        summary = _generate_broader_history_rerank_report(
            args,
            system_config=system_config,
            refresh_summary=refresh_summary,
            variant=args.broader_history_rerank_variant,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.config:
        config_path = Path(args.config).resolve()
        summary = _generate_single_profile_report(
            args,
            system_config=system_config,
            refresh_summary=refresh_summary,
            config_path=config_path,
            signal_profile=signal_profile if args.signal_profile is not None else _infer_signal_profile_from_config_path(config_path),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if signal_profile == "all_split":
        summaries = {
            profile: _generate_single_profile_report(
                args,
                system_config=system_config,
                refresh_summary=refresh_summary,
                config_path=_resolve_signal_config_path(profile),
                signal_profile=profile,
            )
            for profile in _default_signal_profiles_for_all_split()
        }
        skipped_profiles = {
            profile: reason
            for profile, reason in DISABLED_DEFAULT_SIGNAL_PROFILES.items()
            if profile not in summaries
        }
        aggregate_summary = _generate_all_split_aggregate_report(
            summaries,
            skipped_profiles=skipped_profiles,
        )
        print(
            json.dumps(
                {
                    "signal_profile": signal_profile,
                    "profiles": summaries,
                    "skipped_profiles": skipped_profiles,
                    "aggregate_report": aggregate_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    summary = _generate_single_profile_report(
        args,
        system_config=system_config,
        refresh_summary=refresh_summary,
        config_path=_resolve_signal_config_path(signal_profile),
        signal_profile=signal_profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()