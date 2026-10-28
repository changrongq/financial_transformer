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

from financial_transformer_experiment.backtest import TransformerStrategyBacktester
from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.scoring import strategy_score_frame
from financial_transformer_experiment.two_stage_execution import resolve_single_stage_execution_defaults


KNOWN_FAIR_PARAMS = {
    "config_moneyflow_v1_recent_oos_v1": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
    "config_moneyflow_v1": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
    "config_moneyflow_selective_large_recent_oos_v1": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
    "config_moneyflow_selective_large_v1": {
        "short_up_threshold": 0.50,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    },
}


class RankSliceBacktester(TransformerStrategyBacktester):
    def __init__(self, config, min_rank: int, max_rank: int):
        super().__init__(config)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)

    def _load_predictions(self, *args, **kwargs):
        kwargs.pop("top_k_per_day", None)
        df = super()._load_predictions(*args, top_k_per_day=None, **kwargs)
        if df.empty:
            return df
        ranked = df.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False]).copy()
        ranked["day_rank"] = ranked.groupby("date").cumcount() + 1
        return ranked[(ranked["day_rank"] >= self.min_rank) & (ranked["day_rank"] <= self.max_rank)].copy()


def _strategy_summary(config_name: str, min_rank: int, max_rank: int) -> dict:
    config_path = (ROOT / config_name).resolve()
    config = load_config(str(config_path))
    defaults = resolve_single_stage_execution_defaults(str(config_path))
    if defaults is None:
        fallback = KNOWN_FAIR_PARAMS.get(config_path.stem)
        if fallback is None:
            raise RuntimeError(f"missing single-stage defaults for {config_name}")
        fair = fallback
    else:
        fair = {
            "short_up_threshold": float(defaults.default_short_up_threshold),
            "take_profit_threshold": float(defaults.default_take_profit_threshold),
            "stop_loss_threshold": float(defaults.default_stop_loss_threshold),
            "top_k_per_day": int(defaults.default_top_k_per_day),
            "hold_days": int(defaults.default_execution_hold_days),
            "take_profit": float(defaults.default_execution_take_profit),
            "stop_loss": float(defaults.default_execution_stop_loss),
        }

    backtester = RankSliceBacktester(config, min_rank=min_rank, max_rank=max_rank)
    result = backtester.run(
        short_up_threshold=float(fair["short_up_threshold"]),
        take_profit_threshold=float(fair["take_profit_threshold"]),
        stop_loss_threshold=float(fair["stop_loss_threshold"]),
        top_k_per_day=int(fair["top_k_per_day"]),
        hold_days=int(fair["hold_days"]),
        take_profit=float(fair["take_profit"]),
        stop_loss=float(fair["stop_loss"]),
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
    return result.summary


def _offline_rank_profile(config_name: str, max_rank: int) -> list[dict]:
    config = load_config(str((ROOT / config_name).resolve()))
    df = pd.read_csv(config.results_dir / "test_predictions.csv", dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ["future_return", "pred_short_up", "pred_take_profit", "pred_stop_loss"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "future_return", "pred_short_up", "pred_take_profit", "pred_stop_loss"]).copy()
    df["signal_score"] = strategy_score_frame(df, exit_label_mode=config.multitask.exit_label_mode)
    passed = df[
        (df["pred_short_up"] > 0.5)
        & (df["pred_take_profit"] > 0.46)
        & (df["pred_stop_loss"] < 0.58)
    ].copy()
    passed = passed.sort_values(["date", "signal_score", "pred_take_profit"], ascending=[True, False, False])
    passed["day_rank"] = passed.groupby("date").cumcount() + 1
    top = passed[passed["day_rank"] <= max_rank].copy()
    if top.empty:
        return []
    return (
        top.groupby("day_rank", as_index=False)
        .agg(
            picks=("stock_code", "size"),
            avg_future_return=("future_return", "mean"),
            negative_ratio=("future_return", lambda series: float((series < 0).mean())),
        )
        .to_dict(orient="records")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest rank slices for a fixed single-stage config")
    parser.add_argument("--config", default="config_moneyflow_v1_recent_oos_v1.yaml")
    parser.add_argument("--rank-slices", nargs="*", default=["1-1", "1-2", "1-3", "2-5", "3-5", "4-5", "5-5"])
    parser.add_argument("--output-prefix", default="rank_slice_backtest_report")
    args = parser.parse_args()

    report = {
        "config": args.config,
        "offline_rank_profile": _offline_rank_profile(args.config, max_rank=5),
        "slices": [],
    }
    for rank_slice in args.rank_slices:
        start_text, end_text = rank_slice.split("-", 1)
        min_rank = int(start_text)
        max_rank = int(end_text)
        try:
            summary = _strategy_summary(args.config, min_rank=min_rank, max_rank=max_rank)
            report["slices"].append(
                {
                    "rank_slice": rank_slice,
                    "summary": summary,
                }
            )
        except Exception as exc:
            report["slices"].append(
                {
                    "rank_slice": rank_slice,
                    "error": str(exc),
                }
            )

    output_json = ROOT / "results" / f"{args.output_prefix}.json"
    output_md = ROOT / "results" / f"{args.output_prefix}.md"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Rank Slice Backtest Report", "", f"Config: {args.config}", ""]
    if report["offline_rank_profile"]:
        lines.append("## Offline Top5 Rank Profile")
        lines.append("")
        for row in report["offline_rank_profile"]:
            lines.append(
                f"- r{int(row['day_rank'])}: picks={row['picks']}, avg_future_return={row['avg_future_return'] * 100:.2f}%, negative_ratio={row['negative_ratio'] * 100:.2f}%"
            )
        lines.append("")
    lines.append("## Backtest Slices")
    lines.append("")
    for item in report["slices"]:
        if "error" in item:
            lines.append(f"- {item['rank_slice']}: error={item['error']}")
            continue
        summary = item["summary"]
        lines.append(
            f"- {item['rank_slice']}: total_return={summary['total_return'] * 100:.2f}%, sharpe={summary['sharpe_ratio']:.4f}, max_drawdown={summary['max_drawdown'] * 100:.2f}%, trades={summary['trade_count']}, signals={summary['signal_count']}"
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "slice_count": len(report["slices"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()