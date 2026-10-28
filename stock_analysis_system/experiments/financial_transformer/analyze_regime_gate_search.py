from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.backtest import TransformerStrategyBacktester
from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.two_stage_execution import resolve_single_stage_execution_defaults


@dataclass(frozen=True)
class FairParams:
    short_up_threshold: float
    take_profit_threshold: float
    stop_loss_threshold: float
    top_k_per_day: int
    hold_days: int
    take_profit: float
    stop_loss: float


DEFAULT_CONFIGS = [
    "config_moneyflow_v1_recent_oos_v1.yaml",
    "config_moneyflow_v1_concept_sentiment_manual_micro_recent_oos_v1.yaml",
]


KNOWN_FAIR_PARAMS = {
    "config_moneyflow_v1_recent_oos_v1": FairParams(
        short_up_threshold=0.50,
        take_profit_threshold=0.46,
        stop_loss_threshold=0.58,
        top_k_per_day=1,
        hold_days=45,
        take_profit=0.14,
        stop_loss=-0.06,
    ),
}


METRICS_TO_SCAN = [
    "pass_count",
    "pass_mean_score",
    "pass_mean_pred_take_profit",
    "pass_mean_pred_stop_loss",
    "top1_score",
    "top1_pred_take_profit",
    "top1_pred_stop_loss",
    "score_gap_top1_top2",
    "score_gap_top1_top5",
]


class DateGatedBacktester(TransformerStrategyBacktester):
    def __init__(self, config, allowed_dates: set[pd.Timestamp]):
        super().__init__(config)
        self.allowed_dates = {pd.Timestamp(date) for date in allowed_dates}

    def _load_predictions(self, *args, **kwargs):
        df = super()._load_predictions(*args, **kwargs)
        if not self.allowed_dates:
            return df.iloc[0:0].copy()
        return df[df["date"].isin(self.allowed_dates)].copy()


def _resolve_fair_params(config_path: Path) -> FairParams:
    defaults = resolve_single_stage_execution_defaults(str(config_path))
    if defaults is not None:
        return FairParams(
            short_up_threshold=float(defaults.default_short_up_threshold),
            take_profit_threshold=float(defaults.default_take_profit_threshold),
            stop_loss_threshold=float(defaults.default_stop_loss_threshold),
            top_k_per_day=int(defaults.default_top_k_per_day),
            hold_days=int(defaults.default_execution_hold_days),
            take_profit=float(defaults.default_execution_take_profit),
            stop_loss=float(defaults.default_execution_stop_loss),
        )
    known = KNOWN_FAIR_PARAMS.get(config_path.stem)
    if known is not None:
        return known
    raise RuntimeError(f"no fair-params mapping for {config_path.name}")


def _load_prediction_frame(results_dir: Path) -> pd.DataFrame:
    predictions_path = results_dir / "test_predictions.csv"
    df = pd.read_csv(predictions_path, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ["pred_short_up", "pred_take_profit", "pred_stop_loss", "signal_score"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "pred_short_up", "pred_take_profit", "pred_stop_loss", "signal_score"]).copy()
    df["pass"] = (
        (df["pred_short_up"] > 0.5)
        & (df["pred_take_profit"] > 0.46)
        & (df["pred_stop_loss"] < 0.58)
    )
    return df


def _build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    passed = df[df["pass"]].copy()
    passed = passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
    daily = passed.groupby("date").agg(
        pass_count=("stock_code", "size"),
        pass_mean_score=("signal_score", "mean"),
        pass_mean_pred_take_profit=("pred_take_profit", "mean"),
        pass_mean_pred_stop_loss=("pred_stop_loss", "mean"),
    )
    top1 = passed.groupby("date").head(1).set_index("date")
    daily["top1_score"] = top1["signal_score"]
    daily["top1_pred_take_profit"] = top1["pred_take_profit"]
    daily["top1_pred_stop_loss"] = top1["pred_stop_loss"]
    spread_rows = []
    for date, group in passed.groupby("date"):
        scores = group["signal_score"].tolist()
        spread_rows.append(
            {
                "date": date,
                "score_gap_top1_top2": scores[0] - (scores[1] if len(scores) > 1 else scores[0]),
                "score_gap_top1_top5": scores[0] - scores[min(4, len(scores) - 1)],
            }
        )
    daily = daily.join(pd.DataFrame(spread_rows).set_index("date"), how="left").reset_index()
    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily


def _evaluate_gate(config_path: Path, fair: FairParams, allowed_dates: set[pd.Timestamp]) -> dict | None:
    config = load_config(str(config_path))
    backtester = DateGatedBacktester(config, allowed_dates)
    try:
        result = backtester.run(
            start_date=str((min(allowed_dates) if allowed_dates else pd.Timestamp(config.splits.val_end)).date()),
            end_date=config.splits.test_end,
            short_up_threshold=fair.short_up_threshold,
            take_profit_threshold=fair.take_profit_threshold,
            stop_loss_threshold=fair.stop_loss_threshold,
            top_k_per_day=fair.top_k_per_day,
            hold_days=fair.hold_days,
            take_profit=fair.take_profit,
            stop_loss=fair.stop_loss,
            initial_capital=1_000_000.0,
            commission=0.001,
            slippage=0.001,
            min_lot=100,
            min_position_weight=0.08,
            max_position_weight=0.20,
            max_total_position_fraction=1.0,
            cooldown_trigger_losses=0,
            cooldown_days=0,
            min_secondary_primary_ratio=0.0,
            rank_weight_power=0.0,
            min_score_ratio_to_leader=0.0,
        )
    except Exception:
        return None
    return result.summary


def _candidate_thresholds(series: pd.Series) -> list[float]:
    values = sorted(set(float(value) for value in series.dropna().tolist()))
    if len(values) > 8:
        values = sorted(set(float(series.quantile(q)) for q in [0.2, 0.35, 0.5, 0.65, 0.8]))
    return values


def _scan_config(config_path: Path, min_trade_count: int) -> dict:
    config = load_config(str(config_path))
    fair = _resolve_fair_params(config_path)
    daily = _build_daily_metrics(_load_prediction_frame(config.results_dir))
    base_summary = _evaluate_gate(config_path, fair, set(daily["date"]))
    candidates: list[dict] = []
    for metric in METRICS_TO_SCAN:
        for threshold in _candidate_thresholds(daily[metric]):
            for direction in ["ge", "le"]:
                if direction == "ge":
                    allowed_dates = set(daily.loc[daily[metric] >= threshold, "date"])
                else:
                    allowed_dates = set(daily.loc[daily[metric] <= threshold, "date"])
                if len(allowed_dates) < 3 or len(allowed_dates) >= len(daily):
                    continue
                summary = _evaluate_gate(config_path, fair, allowed_dates)
                if summary is None:
                    continue
                candidates.append(
                    {
                        "metric": metric,
                        "direction": direction,
                        "threshold": float(threshold),
                        "allowed_dates": [str(pd.Timestamp(date).date()) for date in sorted(allowed_dates)],
                        "total_return": float(summary["total_return"]),
                        "sharpe_ratio": float(summary["sharpe_ratio"]),
                        "max_drawdown": float(summary["max_drawdown"]),
                        "trade_count": int(summary["trade_count"]),
                        "signal_count": int(summary["signal_count"]),
                    }
                )
    with_trade = [candidate for candidate in candidates if candidate["trade_count"] >= min_trade_count]
    with_trade.sort(key=lambda item: (item["total_return"], item["sharpe_ratio"]), reverse=True)
    return {
        "config_name": config_path.name,
        "fair_params": fair.__dict__,
        "base": base_summary,
        "daily_metrics": daily[
            [
                "date_str",
                "pass_count",
                "pass_mean_score",
                "pass_mean_pred_take_profit",
                "pass_mean_pred_stop_loss",
                "top1_score",
                "top1_pred_take_profit",
                "top1_pred_stop_loss",
                "score_gap_top1_top2",
                "score_gap_top1_top5",
            ]
        ].to_dict(orient="records"),
        "top_candidates": with_trade[:10],
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _build_markdown(report: dict, min_trade_count: int) -> str:
    lines = ["# Regime Gate Report", ""]
    lines.append(f"Min trade count filter: {min_trade_count}")
    lines.append("")
    for item in report["configs"]:
        base = item["base"]
        lines.append(f"## {item['config_name']}")
        lines.append("")
        lines.append(
            f"- Base: total_return={_format_pct(base.get('total_return'))}, sharpe={_format_float(base.get('sharpe_ratio'))}, max_drawdown={_format_pct(base.get('max_drawdown'))}, trades={base.get('trade_count', 0)}"
        )
        if item["top_candidates"]:
            best = item["top_candidates"][0]
            lines.append(
                f"- Best gate: metric={best['metric']} {best['direction']} {best['threshold']:.6f}, total_return={_format_pct(best.get('total_return'))}, sharpe={_format_float(best.get('sharpe_ratio'))}, max_drawdown={_format_pct(best.get('max_drawdown'))}, trades={best.get('trade_count', 0)}"
            )
            lines.append(f"- Allowed dates: {json.dumps(best['allowed_dates'], ensure_ascii=False)}")
        else:
            lines.append("- Best gate: no candidate satisfied the minimum trade-count filter")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search simple date-level regime gates for recent-OOS configs")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--min-trade-count", type=int, default=2)
    parser.add_argument("--output-prefix", default="regime_gate_report")
    args = parser.parse_args()

    config_paths = [(ROOT / config_name).resolve() for config_name in args.configs]
    report = {"configs": [_scan_config(path, min_trade_count=args.min_trade_count) for path in config_paths]}

    output_json = ROOT / "results" / f"{args.output_prefix}.json"
    output_md = ROOT / "results" / f"{args.output_prefix}.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(report, min_trade_count=args.min_trade_count), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "config_count": len(report["configs"]),
                "min_trade_count": args.min_trade_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()