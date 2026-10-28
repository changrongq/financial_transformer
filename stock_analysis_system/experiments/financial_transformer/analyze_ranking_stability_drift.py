from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config


DEFAULT_PAIRS = [
    {
        "name": "baseline_micro",
        "old": "config_moneyflow_v1.yaml",
        "recent": "config_moneyflow_v1_recent_oos_v1.yaml",
    },
    {
        "name": "manual_micro",
        "old": "config_moneyflow_v1_concept_sentiment_manual_micro_v1.yaml",
        "recent": "config_moneyflow_v1_concept_sentiment_manual_micro_recent_oos_v1.yaml",
    },
]


def _strategy_score(df: pd.DataFrame) -> pd.Series:
    return 0.45 * df["pred_short_up"] + 0.35 * df["pred_take_profit"] - 0.20 * df["pred_stop_loss"]


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    if left.empty or right.empty:
        return None
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return None
    return _to_float(left.corr(right, method=method))


def _load_predictions(config_name: str) -> tuple[object, pd.DataFrame]:
    config = load_config(str((ROOT / config_name).resolve()))
    predictions_path = config.results_dir / "test_predictions.csv"
    if not predictions_path.exists():
        raise RuntimeError(f"missing predictions: {predictions_path}")
    df = pd.read_csv(predictions_path, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_columns = [
        "future_return",
        "pred_short_up",
        "pred_take_profit",
        "pred_stop_loss",
        "label_short_up",
        "label_take_profit",
        "label_stop_loss",
        "signal_score",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    required = ["date", "stock_code", "future_return", "pred_short_up", "pred_take_profit", "pred_stop_loss"]
    df = df.dropna(subset=[column for column in required if column in df.columns]).copy()
    if "signal_score" not in df.columns or df["signal_score"].isna().all():
        df["signal_score"] = _strategy_score(df)
    df["pass"] = (
        (df["pred_short_up"] > 0.5)
        & (df["pred_take_profit"] > 0.46)
        & (df["pred_stop_loss"] < 0.58)
    )
    return config, df


def _build_daily_pass_metrics(passed: pd.DataFrame) -> pd.DataFrame:
    if passed.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "pass_count",
                "pass_mean_score",
                "pass_mean_future_return",
                "top1_score",
                "top1_pred_take_profit",
                "top1_pred_stop_loss",
                "top1_future_return",
                "score_gap_top1_top2",
                "score_gap_top1_top5",
            ]
        )
    passed = passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).copy()
    daily = passed.groupby("date").agg(
        pass_count=("stock_code", "size"),
        pass_mean_score=("signal_score", "mean"),
        pass_mean_future_return=("future_return", "mean"),
    )
    top1 = passed.groupby("date").head(1).set_index("date")
    daily["top1_score"] = top1["signal_score"]
    daily["top1_pred_take_profit"] = top1["pred_take_profit"]
    daily["top1_pred_stop_loss"] = top1["pred_stop_loss"]
    daily["top1_future_return"] = top1["future_return"]
    spread_rows: list[dict] = []
    for date, group in passed.groupby("date"):
        scores = group["signal_score"].tolist()
        spread_rows.append(
            {
                "date": date,
                "score_gap_top1_top2": scores[0] - (scores[1] if len(scores) > 1 else scores[0]),
                "score_gap_top1_top5": scores[0] - scores[min(4, len(scores) - 1)],
            }
        )
    return daily.join(pd.DataFrame(spread_rows).set_index("date"), how="left").reset_index()


def _summarize_window(config_name: str) -> dict:
    config, predictions = _load_predictions(config_name)
    passed = predictions[predictions["pass"]].copy()
    daily = _build_daily_pass_metrics(passed)
    passed_ranked = passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).copy()
    passed_ranked["day_rank"] = passed_ranked.groupby("date").cumcount() + 1
    top1 = passed_ranked[passed_ranked["day_rank"] == 1].reset_index(drop=True)
    top1_stock_frequency = top1["stock_code"].value_counts() if not top1.empty else pd.Series(dtype="int64")
    top1_frequency_share = (
        (top1_stock_frequency / max(len(top1), 1)).round(4).to_dict() if not top1_stock_frequency.empty else {}
    )
    rank_profile = (
        passed_ranked[passed_ranked["day_rank"] <= 5]
        .groupby("day_rank", as_index=False)
        .agg(
            picks=("stock_code", "size"),
            avg_future_return=("future_return", "mean"),
            negative_ratio=("future_return", lambda series: float((series < 0).mean())),
        )
        .to_dict(orient="records")
    )
    return {
        "config_name": config_name,
        "results_dir": str(config.results_dir),
        "window": {
            "date_count": int(predictions["date"].nunique()),
            "row_count": int(len(predictions)),
            "pass_date_count": int(passed["date"].nunique()),
            "pass_row_count": int(len(passed)),
            "pass_row_ratio": _to_float(len(passed) / len(predictions)) if len(predictions) else None,
        },
        "ranking_quality": {
            "all_signal_future_pearson": _safe_corr(predictions["signal_score"], predictions["future_return"], "pearson"),
            "all_signal_future_spearman": _safe_corr(predictions["signal_score"], predictions["future_return"], "spearman"),
            "pass_signal_future_pearson": _safe_corr(passed["signal_score"], passed["future_return"], "pearson"),
            "pass_signal_future_spearman": _safe_corr(passed["signal_score"], passed["future_return"], "spearman"),
            "pass_mean_future_return": _to_float(passed["future_return"].mean()) if not passed.empty else None,
            "top1_mean_future_return": _to_float(top1["future_return"].mean()) if not top1.empty else None,
            "top1_negative_ratio": _to_float((top1["future_return"] < 0).mean()) if not top1.empty else None,
            "top1_take_profit_hit": _to_float(top1["label_take_profit"].mean()) if not top1.empty and "label_take_profit" in top1.columns else None,
        },
        "score_shape": {
            "pass_count_daily_mean": _to_float(daily["pass_count"].mean()) if not daily.empty else None,
            "pass_count_daily_median": _to_float(daily["pass_count"].median()) if not daily.empty else None,
            "top1_score_mean": _to_float(daily["top1_score"].mean()) if not daily.empty else None,
            "top1_score_p25": _to_float(daily["top1_score"].quantile(0.25)) if not daily.empty else None,
            "top1_score_p75": _to_float(daily["top1_score"].quantile(0.75)) if not daily.empty else None,
            "score_gap_top1_top2_mean": _to_float(daily["score_gap_top1_top2"].mean()) if not daily.empty else None,
            "score_gap_top1_top5_mean": _to_float(daily["score_gap_top1_top5"].mean()) if not daily.empty else None,
            "top1_pred_take_profit_mean": _to_float(daily["top1_pred_take_profit"].mean()) if not daily.empty else None,
            "top1_pred_stop_loss_mean": _to_float(daily["top1_pred_stop_loss"].mean()) if not daily.empty else None,
        },
        "concentration": {
            "top1_unique_stocks": int(top1["stock_code"].nunique()) if not top1.empty else 0,
            "top1_max_frequency_share": _to_float(top1_stock_frequency.iloc[0] / len(top1)) if not top1_stock_frequency.empty else None,
            "top1_top3_frequency_share": _to_float(top1_stock_frequency.head(3).sum() / len(top1)) if not top1_stock_frequency.empty else None,
            "top1_stock_frequency_share": dict(list(top1_frequency_share.items())[:10]),
        },
        "rank_profile": rank_profile,
        "worst_top1_rows": (
            top1[["date", "stock_code", "future_return", "signal_score", "pred_short_up", "pred_take_profit", "pred_stop_loss"]]
            .assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
            .nsmallest(5, "future_return")
            .to_dict(orient="records")
            if not top1.empty
            else []
        ),
    }


def _diff_sections(old: dict, recent: dict, section: str) -> dict:
    diff: dict = {}
    old_section = old.get(section, {})
    recent_section = recent.get(section, {})
    for key, recent_value in recent_section.items():
        old_value = old_section.get(key)
        if isinstance(recent_value, (int, float)) and isinstance(old_value, (int, float)):
            diff[key] = _to_float(recent_value - old_value)
    return diff


def _analyze_pair(name: str, old_config: str, recent_config: str) -> dict:
    old_summary = _summarize_window(old_config)
    recent_summary = _summarize_window(recent_config)
    return {
        "name": name,
        "old": old_summary,
        "recent": recent_summary,
        "delta_recent_minus_old": {
            "window": _diff_sections(old_summary, recent_summary, "window"),
            "ranking_quality": _diff_sections(old_summary, recent_summary, "ranking_quality"),
            "score_shape": _diff_sections(old_summary, recent_summary, "score_shape"),
            "concentration": _diff_sections(old_summary, recent_summary, "concentration"),
        },
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _append_window(lines: list[str], label: str, window: dict) -> None:
    lines.append(f"### {label}")
    lines.append("")
    lines.append(
        f"- Window: dates={window['window']['date_count']}, rows={window['window']['row_count']}, pass_dates={window['window']['pass_date_count']}, pass_rows={window['window']['pass_row_count']}, pass_ratio={_format_pct(window['window']['pass_row_ratio'])}"
    )
    lines.append(
        f"- Ranking quality: all_spearman={_format_float(window['ranking_quality']['all_signal_future_spearman'])}, pass_spearman={_format_float(window['ranking_quality']['pass_signal_future_spearman'])}, pass_avg_future_return={_format_pct(window['ranking_quality']['pass_mean_future_return'])}, top1_avg_future_return={_format_pct(window['ranking_quality']['top1_mean_future_return'])}, top1_negative_ratio={_format_pct(window['ranking_quality']['top1_negative_ratio'])}"
    )
    lines.append(
        f"- Score shape: daily_pass_mean={_format_float(window['score_shape']['pass_count_daily_mean'])}, top1_score_mean={_format_float(window['score_shape']['top1_score_mean'])}, gap12_mean={_format_float(window['score_shape']['score_gap_top1_top2_mean'])}, gap15_mean={_format_float(window['score_shape']['score_gap_top1_top5_mean'])}, top1_pred_tp_mean={_format_float(window['score_shape']['top1_pred_take_profit_mean'])}, top1_pred_sl_mean={_format_float(window['score_shape']['top1_pred_stop_loss_mean'])}"
    )
    lines.append(
        f"- Concentration: unique_top1_stocks={window['concentration']['top1_unique_stocks']}, top1_max_share={_format_pct(window['concentration']['top1_max_frequency_share'])}, top3_share={_format_pct(window['concentration']['top1_top3_frequency_share'])}"
    )
    lines.append(f"- Top1 frequency share: {json.dumps(window['concentration']['top1_stock_frequency_share'], ensure_ascii=False)}")
    if window["rank_profile"]:
        rank_chunks = []
        for row in window["rank_profile"]:
            rank_chunks.append(
                f"r{int(row['day_rank'])}: avg_ret={_format_pct(row['avg_future_return'])}, neg={_format_pct(row['negative_ratio'])}"
            )
        lines.append(f"- Top5 rank profile: {'; '.join(rank_chunks)}")
    if window["worst_top1_rows"]:
        lines.append("- Worst top1 rows:")
        for row in window["worst_top1_rows"][:3]:
            lines.append(
                f"  - {row['date']} {row['stock_code']} future_return={_format_pct(row['future_return'])} score={_format_float(row['signal_score'])} pred_tp={_format_float(row['pred_take_profit'])} pred_sl={_format_float(row['pred_stop_loss'])}"
            )
    lines.append("")


def _build_markdown(report: dict) -> str:
    lines = ["# Ranking Stability Drift Report", ""]
    for pair in report["pairs"]:
        lines.append(f"## {pair['name']}")
        lines.append("")
        _append_window(lines, f"Old ({pair['old']['config_name']})", pair["old"])
        _append_window(lines, f"Recent ({pair['recent']['config_name']})", pair["recent"])
        delta = pair["delta_recent_minus_old"]
        lines.append("### Delta Recent Minus Old")
        lines.append("")
        lines.append(
            f"- Ranking drift: all_spearman={_format_float(delta['ranking_quality'].get('all_signal_future_spearman'))}, pass_spearman={_format_float(delta['ranking_quality'].get('pass_signal_future_spearman'))}, pass_avg_future_return={_format_pct(delta['ranking_quality'].get('pass_mean_future_return'))}, top1_avg_future_return={_format_pct(delta['ranking_quality'].get('top1_mean_future_return'))}, top1_negative_ratio={_format_pct(delta['ranking_quality'].get('top1_negative_ratio'))}"
        )
        lines.append(
            f"- Shape drift: daily_pass_mean={_format_float(delta['score_shape'].get('pass_count_daily_mean'))}, top1_score_mean={_format_float(delta['score_shape'].get('top1_score_mean'))}, gap12_mean={_format_float(delta['score_shape'].get('score_gap_top1_top2_mean'))}, gap15_mean={_format_float(delta['score_shape'].get('score_gap_top1_top5_mean'))}, top1_pred_tp_mean={_format_float(delta['score_shape'].get('top1_pred_take_profit_mean'))}, top1_pred_sl_mean={_format_float(delta['score_shape'].get('top1_pred_stop_loss_mean'))}"
        )
        lines.append(
            f"- Concentration drift: unique_top1_stocks={_format_float(delta['concentration'].get('top1_unique_stocks'))}, top1_max_share={_format_pct(delta['concentration'].get('top1_max_frequency_share'))}, top3_share={_format_pct(delta['concentration'].get('top1_top3_frequency_share'))}"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ranking-stability drift between old and recent windows")
    parser.add_argument("--output-prefix", default="ranking_stability_drift_report")
    args = parser.parse_args()

    report = {
        "pairs": [
            _analyze_pair(pair["name"], pair["old"], pair["recent"])
            for pair in DEFAULT_PAIRS
        ]
    }

    output_json = ROOT / "results" / f"{args.output_prefix}.json"
    output_md = ROOT / "results" / f"{args.output_prefix}.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "pair_count": len(report["pairs"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()