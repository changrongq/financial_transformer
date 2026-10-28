from __future__ import annotations

import pandas as pd


def annotate_confidence_tiers(
    df: pd.DataFrame,
    *,
    score_col: str = "signal_score",
    percentile_col: str = "signal_confidence_pct",
    tier_col: str = "signal_confidence_tier",
) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        result[percentile_col] = pd.Series(dtype=float)
        result[tier_col] = pd.Series(dtype=object)
        return result

    if score_col not in result.columns:
        result[percentile_col] = 0.0
        result[tier_col] = "unknown"
        return result

    scores = pd.to_numeric(result[score_col], errors="coerce")
    valid = scores.notna()
    result[percentile_col] = 0.0
    result[tier_col] = "unknown"
    if not valid.any():
        return result

    rank_pct = scores[valid].rank(method="average", pct=True, ascending=True)
    result.loc[valid, percentile_col] = rank_pct.astype(float)

    def _tier(value: object) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "unknown"
        if numeric >= 0.8:
            return "high"
        if numeric <= 0.2:
            return "low"
        return "medium"

    result[tier_col] = result[percentile_col].apply(_tier)
    return result


def summarize_confidence_distribution(
    df: pd.DataFrame,
    *,
    score_col: str = "signal_score",
    percentile_col: str = "signal_confidence_pct",
    tier_col: str = "signal_confidence_tier",
) -> dict:
    if df.empty:
        return {
            "label": "unknown",
            "summary": "当前没有通过筛选的候选，暂无法评估置信度结构。",
            "counts": {"high": 0, "medium": 0, "low": 0, "unknown": 0},
            "ratios": {"high": 0.0, "medium": 0.0, "low": 0.0, "unknown": 0.0},
            "score_concentration": {
                "top1_share": None,
                "top2_share": None,
                "top3_share": None,
                "leader_gap": None,
            },
        }

    working = df.copy()
    if tier_col not in working.columns or percentile_col not in working.columns:
        working = annotate_confidence_tiers(
            working,
            score_col=score_col,
            percentile_col=percentile_col,
            tier_col=tier_col,
        )

    ordered_tiers = ["high", "medium", "low", "unknown"]
    counts = {tier: 0 for tier in ordered_tiers}
    raw_counts = working[tier_col].astype(str).value_counts(dropna=False).to_dict()
    for tier in ordered_tiers:
        counts[tier] = int(raw_counts.get(tier, 0))

    total = max(int(len(working)), 1)
    ratios = {tier: float(counts[tier] / total) for tier in ordered_tiers}

    scores = pd.to_numeric(working.get(score_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    positive_scores = scores.clip(lower=0.0).sort_values(ascending=False).reset_index(drop=True)
    score_sum = float(positive_scores.sum())

    if score_sum > 1e-9:
        top1_share = float(positive_scores.head(1).sum() / score_sum)
        top2_share = float(positive_scores.head(2).sum() / score_sum)
        top3_share = float(positive_scores.head(3).sum() / score_sum)
    else:
        top1_share = None
        top2_share = None
        top3_share = None

    if len(positive_scores) >= 2 and abs(float(positive_scores.iloc[0])) > 1e-9:
        leader_gap = float((positive_scores.iloc[0] - positive_scores.iloc[1]) / abs(float(positive_scores.iloc[0])))
    else:
        leader_gap = None

    if len(working) == 1:
        label = "single_candidate"
        summary = "当前只有 1 只候选，通过阈值后无需向后扩散，按头部单票处理即可。"
    elif counts["high"] == 0 and counts["medium"] <= 1:
        label = "weak_confidence"
        summary = "高置信度候选缺失，且可用中等置信度也偏少，今天更像是弱信号日。"
    elif (top1_share is not None and top1_share >= 0.65) or (top2_share is not None and top2_share >= 0.85):
        label = "leader_concentrated"
        summary = "信号强度明显集中在前 1 到 2 名，不建议把执行范围继续向后扩散。"
    elif counts["high"] >= 2 and top3_share is not None and top3_share <= 0.80:
        label = "balanced"
        summary = "高置信度与中等置信度分布相对均衡，头部并未出现极端挤压。"
    else:
        label = "mixed"
        summary = "候选存在头部优势，但集中度仍在可控范围内，维持当前 top-k 纪律更稳妥。"

    return {
        "label": label,
        "summary": summary,
        "counts": counts,
        "ratios": ratios,
        "score_concentration": {
            "top1_share": top1_share,
            "top2_share": top2_share,
            "top3_share": top3_share,
            "leader_gap": leader_gap,
        },
    }


def summarize_market_environment(
    df: pd.DataFrame,
    *,
    filtered_count: int | None = None,
) -> dict:
    if df.empty:
        return {
            "label": "unknown",
            "style": "neutral",
            "summary": "无可用预测，暂无法判断市场环境。",
            "metrics": {},
        }

    required = ["pred_short_up", "pred_take_profit", "pred_stop_loss"]
    if any(column not in df.columns for column in required):
        return {
            "label": "unknown",
            "style": "neutral",
            "summary": "预测结果缺少必要字段，暂无法判断市场环境。",
            "metrics": {},
        }

    short_mean = float(pd.to_numeric(df["pred_short_up"], errors="coerce").fillna(0.0).mean())
    take_profit_mean = float(pd.to_numeric(df["pred_take_profit"], errors="coerce").fillna(0.0).mean())
    stop_loss_mean = float(pd.to_numeric(df["pred_stop_loss"], errors="coerce").fillna(0.0).mean())
    bullish_ratio = float((pd.to_numeric(df["pred_short_up"], errors="coerce").fillna(0.0) >= 0.5).mean())
    setup_ratio = float(filtered_count / len(df)) if filtered_count is not None and len(df) > 0 else None

    if stop_loss_mean >= 0.52 or (setup_ratio is not None and setup_ratio <= 0.05 and short_mean < 0.50):
        label = "high_risk"
        style = "defensive"
        summary = "预测分布显示风险偏高，建议防御，优先控制仓位与止损。"
    elif short_mean >= 0.52 and take_profit_mean >= 0.53 and stop_loss_mean <= 0.48 and bullish_ratio >= 0.60:
        label = "trend_up"
        style = "aggressive"
        summary = "预测分布偏向上行趋势，可相对积极，但仍按默认纪律执行。"
    elif short_mean <= 0.48 and take_profit_mean <= 0.50 and stop_loss_mean >= 0.50:
        label = "trend_down"
        style = "defensive"
        summary = "预测分布偏弱，暂按下降市理解，宜偏防御。"
    else:
        label = "range"
        style = "neutral"
        summary = "预测分布未显示单边趋势，更接近震荡环境，宜中性执行。"

    return {
        "label": label,
        "style": style,
        "summary": summary,
        "metrics": {
            "pred_short_up_mean": short_mean,
            "pred_take_profit_mean": take_profit_mean,
            "pred_stop_loss_mean": stop_loss_mean,
            "bullish_ratio": bullish_ratio,
            "setup_ratio": setup_ratio,
        },
    }


def enrich_prediction_frame(df: pd.DataFrame, *, exit_label_mode: str = "window_any", prefix: str = "") -> pd.DataFrame:
    result = df.copy()
    take_profit_col = f"{prefix}pred_take_profit"
    stop_loss_col = f"{prefix}pred_stop_loss"
    prediction_prob_col = f"{prefix}prediction_prob"
    if take_profit_col not in result.columns or stop_loss_col not in result.columns:
        return result

    edge_col = f"{prefix}first_hit_edge"
    confidence_col = f"{prefix}first_hit_confidence"
    ratio_col = f"{prefix}first_hit_ratio"
    denom = (result[take_profit_col] + result[stop_loss_col]).clip(lower=1e-6)
    result[edge_col] = result[take_profit_col] - result[stop_loss_col]
    result[confidence_col] = result[take_profit_col] * (1.0 - result[stop_loss_col])
    result[ratio_col] = result[take_profit_col] / denom
    if exit_label_mode == "first_hit" and prediction_prob_col in result.columns:
        result[prediction_prob_col] = result[confidence_col]
    return result


def strategy_score_frame(df: pd.DataFrame, *, exit_label_mode: str = "window_any", prefix: str = "") -> pd.Series:
    take_profit_col = f"{prefix}pred_take_profit"
    stop_loss_col = f"{prefix}pred_stop_loss"
    short_up_col = f"{prefix}pred_short_up"
    max_gain_col = f"{prefix}pred_max_gain"
    max_drawdown_col = f"{prefix}pred_max_drawdown"
    gain_term = df[max_gain_col].clip(lower=0.0, upper=0.5)
    drawdown_penalty = df[max_drawdown_col].abs().clip(lower=0.01, upper=0.3)

    if exit_label_mode == "first_hit":
        edge_col = f"{prefix}first_hit_edge"
        confidence_col = f"{prefix}first_hit_confidence"
        edge = df[edge_col] if edge_col in df.columns else (df[take_profit_col] - df[stop_loss_col])
        confidence = df[confidence_col] if confidence_col in df.columns else (df[take_profit_col] * (1.0 - df[stop_loss_col]))
        edge = edge.clip(lower=-1.0, upper=1.0)
        return (
            0.45 * confidence
            + 0.30 * edge
            + 0.15 * df[short_up_col]
            + 0.10 * gain_term
            - 0.05 * drawdown_penalty
        )

    return (
        0.45 * df[short_up_col]
        + 0.35 * df[take_profit_col]
        + 0.20 * (1.0 - df[stop_loss_col])
        + 0.10 * gain_term
        - 0.05 * drawdown_penalty
    )