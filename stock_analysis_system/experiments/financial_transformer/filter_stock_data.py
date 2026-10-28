from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import build_feature_frame, normalize_stock_code


@dataclass(frozen=True)
class FilterRuleResult:
    stock_code: str
    keep: bool
    reasons: list[str]
    pass_st_and_delist: bool
    pass_liquidity: bool
    pass_abnormal_volatility: bool
    pass_basic_fundamental: bool
    metrics: dict[str, float | int | str | None]


def _parse_stock_codes(raw_value: str | None) -> set[str]:
    if raw_value is None or not raw_value.strip():
        return set()
    return {normalize_stock_code(code) for code in raw_value.split(",") if code.strip()}


def _rule_st_and_delist(stock_df: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    name_series = stock_df.get("名称")
    if name_series is not None:
        latest_name = str(name_series.iloc[-1] or "").upper()
        if "ST" in latest_name or "*ST" in latest_name or "退市" in latest_name:
            reasons.append("st_or_delist")
    if "market" in stock_df.columns:
        latest_market = str(stock_df["market"].iloc[-1] or "").upper()
        if "退市" in latest_market:
            reasons.append("delist_market")
    return reasons


def _rule_liquidity(stock_df: pd.DataFrame, *, min_turnover_ma60: float, min_amount_ma60: float) -> list[str]:
    reasons: list[str] = []
    turnover = pd.to_numeric(stock_df.get("turnover_rate"), errors="coerce")
    amount = pd.to_numeric(stock_df.get("amount"), errors="coerce")
    turnover_ma60 = float(turnover.tail(60).mean()) if not turnover.tail(60).empty else 0.0
    amount_ma60 = float(amount.tail(60).mean()) if not amount.tail(60).empty else 0.0
    if turnover_ma60 < float(min_turnover_ma60):
        reasons.append(f"low_turnover_ma60={turnover_ma60:.4f}")
    if amount_ma60 < float(min_amount_ma60):
        reasons.append(f"low_amount_ma60={amount_ma60:.0f}")
    return reasons


def _rule_abnormal_volatility(
    stock_df: pd.DataFrame,
    *,
    streak_limit: int,
    streak_return_threshold: float,
    streak_sparse_amount: float,
    single_day_return_threshold: float,
    single_day_sparse_amount: float,
) -> list[str]:
    reasons: list[str] = []
    return_1d = pd.to_numeric(stock_df.get("return_1d"), errors="coerce").fillna(0.0)
    amount = pd.to_numeric(stock_df.get("amount"), errors="coerce").fillna(0.0)
    sparse_mask = amount < float(streak_sparse_amount)
    extreme_mask = return_1d.abs() > float(streak_return_threshold)
    streak = 0
    max_streak = 0
    for extreme, sparse in zip(extreme_mask.to_numpy(), sparse_mask.to_numpy()):
        if bool(extreme) and bool(sparse):
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    if max_streak >= int(streak_limit):
        reasons.append(f"suspected_controlled_stock_streak={max_streak}")
    if ((return_1d.abs() > float(single_day_return_threshold)) & (amount < float(single_day_sparse_amount))).any():
        reasons.append("single_day_extreme_sparse")
    return reasons


def _rule_basic_fundamental(stock_df: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    if "roe" not in stock_df.columns:
        return reasons
    roe = pd.to_numeric(stock_df["roe"], errors="coerce")
    if len(roe.dropna()) >= 3 and (roe.tail(12) < -10).rolling(3, min_periods=3).sum().max() >= 3:
        reasons.append("three_quarters_roe_below_minus_10")
    return reasons


def _score_stock(
    stock_df: pd.DataFrame,
    *,
    min_turnover_ma60: float,
    min_amount_ma60: float,
    streak_limit: int,
    streak_return_threshold: float,
    streak_sparse_amount: float,
    single_day_return_threshold: float,
    single_day_sparse_amount: float,
) -> FilterRuleResult:
    stock_code = normalize_stock_code(str(stock_df["stock_code"].iloc[0]))
    st_reasons = _rule_st_and_delist(stock_df)
    liquidity_reasons = _rule_liquidity(stock_df, min_turnover_ma60=min_turnover_ma60, min_amount_ma60=min_amount_ma60)
    volatility_reasons = (
        _rule_abnormal_volatility(
            stock_df,
            streak_limit=streak_limit,
            streak_return_threshold=streak_return_threshold,
            streak_sparse_amount=streak_sparse_amount,
            single_day_return_threshold=single_day_return_threshold,
            single_day_sparse_amount=single_day_sparse_amount,
        )
    )
    fundamental_reasons = _rule_basic_fundamental(stock_df)
    reasons: list[str] = []
    reasons.extend(st_reasons)
    reasons.extend(liquidity_reasons)
    reasons.extend(volatility_reasons)
    reasons.extend(fundamental_reasons)
    keep = len(reasons) == 0
    metrics = {
        "rows": int(len(stock_df)),
        "turnover_ma60": float(pd.to_numeric(stock_df["turnover_rate"], errors="coerce").tail(60).mean()) if "turnover_rate" in stock_df.columns else 0.0,
        "amount_ma60": float(pd.to_numeric(stock_df["amount"], errors="coerce").tail(60).mean()) if "amount" in stock_df.columns else 0.0,
        "last_date": str(pd.to_datetime(stock_df["date"]).max().date()) if "date" in stock_df.columns else None,
    }
    return FilterRuleResult(
        stock_code=stock_code,
        keep=keep,
        reasons=reasons,
        pass_st_and_delist=(len(st_reasons) == 0),
        pass_liquidity=(len(liquidity_reasons) == 0),
        pass_abnormal_volatility=(len(volatility_reasons) == 0),
        pass_basic_fundamental=(len(fundamental_reasons) == 0),
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter financial transformer training universe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allowed-stock-codes")
    parser.add_argument("--min-rows", type=int, default=120)
    parser.add_argument("--min-turnover-ma60", type=float, default=0.5)
    parser.add_argument("--min-amount-ma60", type=float, default=10_000_000)
    parser.add_argument("--streak-limit", type=int, default=5)
    parser.add_argument("--streak-return-threshold", type=float, default=0.09)
    parser.add_argument("--streak-sparse-amount", type=float, default=50_000_000)
    parser.add_argument("--single-day-return-threshold", type=float, default=0.15)
    parser.add_argument("--single-day-sparse-amount", type=float, default=50_000_000)
    args = parser.parse_args()

    config = load_config(args.config)
    allowed_stock_codes = _parse_stock_codes(args.allowed_stock_codes)
    feature_frame = build_feature_frame(config)
    if allowed_stock_codes:
        feature_frame = feature_frame[feature_frame["stock_code"].astype(str).map(normalize_stock_code).isin(allowed_stock_codes)].copy()

    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    kept_frames: list[pd.DataFrame] = []
    for stock_code, stock_df in feature_frame.groupby("stock_code"):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) < int(args.min_rows):
            rows.append({
                "stock_code": normalize_stock_code(str(stock_code)),
                "keep": False,
                "reasons": "min_rows",
                "pass_min_rows": False,
                "pass_st_and_delist": False,
                "pass_liquidity": False,
                "pass_abnormal_volatility": False,
                "pass_basic_fundamental": False,
                "rows": int(len(stock_df)),
                "turnover_ma60": float(pd.to_numeric(stock_df["turnover_rate"], errors="coerce").tail(60).mean()) if "turnover_rate" in stock_df.columns else 0.0,
                "amount_ma60": float(pd.to_numeric(stock_df["amount"], errors="coerce").tail(60).mean()) if "amount" in stock_df.columns else 0.0,
                "last_date": str(pd.to_datetime(stock_df["date"]).max().date()) if "date" in stock_df.columns else None,
            })
            continue
        result = _score_stock(
            stock_df,
            min_turnover_ma60=float(args.min_turnover_ma60),
            min_amount_ma60=float(args.min_amount_ma60),
            streak_limit=int(args.streak_limit),
            streak_return_threshold=float(args.streak_return_threshold),
            streak_sparse_amount=float(args.streak_sparse_amount),
            single_day_return_threshold=float(args.single_day_return_threshold),
            single_day_sparse_amount=float(args.single_day_sparse_amount),
        )
        rows.append({
            "stock_code": result.stock_code,
            "keep": result.keep,
            "reasons": "|".join(result.reasons),
            "pass_min_rows": True,
            "pass_st_and_delist": result.pass_st_and_delist,
            "pass_liquidity": result.pass_liquidity,
            "pass_abnormal_volatility": result.pass_abnormal_volatility,
            "pass_basic_fundamental": result.pass_basic_fundamental,
            **result.metrics,
        })
        if result.keep:
            kept_frames.append(stock_df)

    filtered = pd.concat(kept_frames, ignore_index=True) if kept_frames else feature_frame.iloc[0:0].copy()
    filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

    report = pd.DataFrame(rows).sort_values(["keep", "amount_ma60", "turnover_ma60"], ascending=[False, False, False]).reset_index(drop=True)
    summary = {
        "input_rows": int(len(feature_frame)),
        "output_rows": int(len(filtered)),
        "input_stocks": int(feature_frame["stock_code"].nunique()) if not feature_frame.empty else 0,
        "output_stocks": int(filtered["stock_code"].nunique()) if not filtered.empty else 0,
        "keep_ratio": float(len(filtered) / len(feature_frame)) if len(feature_frame) else 0.0,
    }

    stock_level = pd.DataFrame(rows)
    total_stocks = int(len(stock_level)) if not stock_level.empty else 0
    if total_stocks > 0:
        amount_ma60 = pd.to_numeric(stock_level.get("amount_ma60", pd.Series(dtype=float)), errors="coerce")
        amount_stats = {
            "min": float(amount_ma60.min()) if amount_ma60.notna().any() else None,
            "p25": float(amount_ma60.quantile(0.25)) if amount_ma60.notna().any() else None,
            "median": float(amount_ma60.median()) if amount_ma60.notna().any() else None,
            "p75": float(amount_ma60.quantile(0.75)) if amount_ma60.notna().any() else None,
            "max": float(amount_ma60.max()) if amount_ma60.notna().any() else None,
        }
        ratio_above_amount_threshold = float((amount_ma60 >= float(args.min_amount_ma60)).mean()) if amount_ma60.notna().any() else 0.0
        debug_breakdown = {
            "rule_pass_counts": {
                "pass_min_rows": int(stock_level["pass_min_rows"].sum()),
                "pass_st_and_delist": int(stock_level["pass_st_and_delist"].sum()),
                "pass_liquidity": int(stock_level["pass_liquidity"].sum()),
                "pass_abnormal_volatility": int(stock_level["pass_abnormal_volatility"].sum()),
                "pass_basic_fundamental": int(stock_level["pass_basic_fundamental"].sum()),
            },
            "cumulative_pass_counts": {
                "after_min_rows": int(stock_level[stock_level["pass_min_rows"]].shape[0]),
                "after_st_and_delist": int(stock_level[stock_level["pass_min_rows"] & stock_level["pass_st_and_delist"]].shape[0]),
                "after_liquidity": int(stock_level[stock_level["pass_min_rows"] & stock_level["pass_st_and_delist"] & stock_level["pass_liquidity"]].shape[0]),
                "after_abnormal_volatility": int(stock_level[stock_level["pass_min_rows"] & stock_level["pass_st_and_delist"] & stock_level["pass_liquidity"] & stock_level["pass_abnormal_volatility"]].shape[0]),
                "after_basic_fundamental": int(stock_level[stock_level["pass_min_rows"] & stock_level["pass_st_and_delist"] & stock_level["pass_liquidity"] & stock_level["pass_abnormal_volatility"] & stock_level["pass_basic_fundamental"]].shape[0]),
                "final_keep": int(stock_level[stock_level["keep"]].shape[0]),
            },
            "amount_unit_probe": {
                "threshold_min_amount_ma60": float(args.min_amount_ma60),
                "ratio_above_threshold": ratio_above_amount_threshold,
                "amount_ma60_stats": amount_stats,
            },
            "total_stock_groups": total_stocks,
        }
    else:
        debug_breakdown = {
            "rule_pass_counts": {},
            "cumulative_pass_counts": {},
            "amount_unit_probe": {},
            "total_stock_groups": 0,
        }
    report_payload = {
        "summary": summary,
        "debug_breakdown": debug_breakdown,
        "rows": report.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "report": str(report_path), **summary, "debug_breakdown": debug_breakdown}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()