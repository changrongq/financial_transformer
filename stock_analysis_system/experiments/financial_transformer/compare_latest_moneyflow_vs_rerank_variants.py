from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC, ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config  # noqa: E402
from financial_transformer_experiment.predict import predict_latest  # noqa: E402
from predict_latest_broader_history_rerank import VARIANT_RULES  # noqa: E402
from predict_latest_broader_history_rerank import run_latest_broader_history_rerank  # noqa: E402


BASE_LABEL = "moneyflow_v1_single_model"
BASE_DISPLAY_NAME = "Moneyflow V1 Single Model"


def _load_base_latest(eval_config_path: Path, fair: dict, *, refresh: bool = False) -> dict:
    eval_config = load_config(str(eval_config_path))
    if refresh:
        predict_latest(eval_config, top_n=None, use_calibration=False)
    latest_path = eval_config.results_dir / "latest_predictions.csv"
    if not latest_path.exists():
        raise FileNotFoundError(f"latest prediction file not found: {latest_path}")

    latest_df = pd.read_csv(latest_path, dtype={"stock_code": str})
    latest_df["stock_code"] = latest_df["stock_code"].astype(str).str.zfill(6)
    latest_df["date"] = pd.to_datetime(latest_df["date"], errors="coerce")
    latest_available_date = latest_df["date"].max()
    if pd.isna(latest_available_date):
        raise RuntimeError("base latest prediction file has no valid date")

    latest_df = latest_df[latest_df["date"] == latest_available_date].copy()
    filtered = latest_df[
        (latest_df["pred_short_up"] > float(fair["short_up_threshold"]))
        & (latest_df["pred_take_profit"] > float(fair["take_profit_threshold"]))
        & (latest_df["pred_stop_loss"] < float(fair["stop_loss_threshold"]))
    ].copy()
    filtered = filtered.sort_values(["signal_score", "pred_take_profit", "pred_short_up"], ascending=[False, False, False]).reset_index(drop=True)
    filtered["base_rank"] = filtered.index + 1
    selected = filtered.head(int(fair["top_k_per_day"]))

    return {
        "label": BASE_LABEL,
        "display_name": BASE_DISPLAY_NAME,
        "latest_available_date": str(pd.Timestamp(latest_available_date).date()),
        "candidate_count": int(len(filtered)),
        "selected_count": int(len(selected)),
        "top_stock": str(selected.iloc[0]["stock_code"]) if not selected.empty else None,
        "top_signal_score": float(selected.iloc[0]["signal_score"]) if not selected.empty else None,
        "fair": {
            "short_up_threshold": float(fair["short_up_threshold"]),
            "take_profit_threshold": float(fair["take_profit_threshold"]),
            "stop_loss_threshold": float(fair["stop_loss_threshold"]),
            "top_k_per_day": int(fair["top_k_per_day"]),
        },
        "top_picks": selected.head(3)[[
            "stock_code",
            "signal_score",
            "pred_short_up",
            "pred_take_profit",
            "pred_stop_loss",
            "close",
            "market_cap_yi",
            "base_rank",
        ]].to_dict(orient="records"),
    }


def _load_rerank_variant(result_root: Path, variant: str, *, refresh: bool = False) -> dict:
    if refresh:
        run_latest_broader_history_rerank(
            variant=variant,
            refresh_latest=True,
        )
    summary_path = result_root / variant / "latest" / "latest_summary.json"
    selected_path = result_root / variant / "latest" / "latest_selected.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"rerank summary not found: {summary_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"rerank selected file not found: {selected_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = pd.read_csv(selected_path, dtype={"stock_code": str})
    if not selected.empty:
        selected["stock_code"] = selected["stock_code"].astype(str).str.zfill(6)

    return {
        "label": variant,
        "display_name": variant,
        "latest_available_date": str(summary["latest_available_date"]),
        "candidate_count": int(summary["latest_candidate_count"]),
        "selected_count": int(summary["selected_count"]),
        "top_stock": str(summary["top_stock"]) if summary.get("top_stock") is not None else None,
        "top_rerank_score": float(summary["top_rerank_score"]) if summary.get("top_rerank_score") is not None else None,
        "fair": dict(summary["fair"]),
        "top_picks": selected.head(3)[[
            column for column in [
                "stock_code",
                "rerank_score",
                "signal_score",
                "pred_short_up",
                "pred_take_profit",
                "pred_stop_loss",
                "close",
                "market_cap_yi",
                "rerank_rank",
            ]
            if column in selected.columns
        ]].to_dict(orient="records"),
    }


def _build_alignment(base: dict, variants: list[dict]) -> dict:
    top_stocks = {entry["label"]: entry.get("top_stock") for entry in [base, *variants]}
    unique_stocks = sorted({stock for stock in top_stocks.values() if stock})
    return {
        "all_same_top_stock": len(unique_stocks) <= 1,
        "unique_top_stocks": unique_stocks,
        "top_stock_by_profile": top_stocks,
        "base_matches": {
            entry["label"]: bool(base.get("top_stock") and base.get("top_stock") == entry.get("top_stock"))
            for entry in variants
        },
    }


def build_qianxing_comparison_message(payload: dict) -> dict:
    base = payload["base"]
    variants = payload.get("variants", [])
    alignment = payload["alignment"]
    rerank_top_stock = variants[0].get("top_stock") if variants else None
    latest_date = base.get("latest_available_date")
    rerank_fair = variants[0].get("fair", {}) if variants else {}
    rerank_hold_days = rerank_fair.get("hold_days")

    if alignment.get("all_same_top_stock"):
        lines = [
            f"最新对比已更新，{latest_date} 当天单模型与 rerank 顶部候选一致。",
            f"当前统一 top stock 是 {alignment.get('unique_top_stocks', ['无'])[0] if alignment.get('unique_top_stocks') else '无'}。",
            "这说明 rerank 层当前没有改变最终执行候选，但仍保留历史验证意义。",
            "两条路径此刻没有分歧，不代表以后会一直一致；真正要看的是后续是否继续按同一条执行路径复盘。",
        ]
    else:
        lines = [
            f"最新对比已更新，{latest_date} 当天单模型与 rerank 顶部候选出现分叉。",
            f"单模型 moneyflow_v1 当前 top stock 是 {base.get('top_stock') or '无'}；rerank canonical variants 当前 top stock 是 {rerank_top_stock or '无'}。",
            "这类分歧更像是同一批短线候选在两套排序/执行路径下的优先级不同，不等于预测失败。",
            "如果决定跟 rerank 路径，就持续看 rerank latest profile 或 comparison 结果；如果决定跟单模型，就按单模型口径连续复盘，不要把“等分歧消失”当成买入前提。",
        ]

    if rerank_hold_days is not None:
        lines.append(
            f"对比里的 rerank 路径本质上也是短线执行口径：下一交易日择机入场，未触发止盈止损时最多持有 {int(rerank_hold_days)} 个交易日。"
        )

    return {
        "has_divergence": not bool(alignment.get("all_same_top_stock")),
        "latest_date": latest_date,
        "base_top_stock": base.get("top_stock"),
        "rerank_top_stock": rerank_top_stock,
        "lines": lines,
        "text": "\n".join(lines),
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Latest Moneyflow vs Rerank Variant Comparison",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Base profile: {payload['base']['label']}",
        f"- Comparison date: {payload['base']['latest_available_date']}",
        f"- All same top stock: {payload['alignment']['all_same_top_stock']}",
        f"- Unique top stocks: {', '.join(payload['alignment']['unique_top_stocks']) if payload['alignment']['unique_top_stocks'] else 'none'}",
        "",
        "| Profile | Latest Date | Candidates | Selected | Top Stock | Score | Notes |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]

    base = payload["base"]
    base_note = "single-model fair filter"
    lines.append(
        f"| {base['label']} | {base['latest_available_date']} | {base['candidate_count']} | {base['selected_count']} | {base.get('top_stock') or 'none'} | {base.get('top_signal_score') if base.get('top_signal_score') is not None else 'n/a'} | {base_note} |"
    )
    for variant in payload["variants"]:
        note = f"hold={variant['fair']['hold_days']}, TP={variant['fair']['take_profit']}, SL={variant['fair']['stop_loss']}"
        lines.append(
            f"| {variant['label']} | {variant['latest_available_date']} | {variant['candidate_count']} | {variant['selected_count']} | {variant.get('top_stock') or 'none'} | {variant.get('top_rerank_score') if variant.get('top_rerank_score') is not None else 'n/a'} | {note} |"
        )

    lines.extend([
        "",
        "## Top Picks",
        "",
        f"- {base['label']}: {json.dumps(base['top_picks'], ensure_ascii=False)}",
    ])
    for variant in payload["variants"]:
        lines.append(f"- {variant['label']}: {json.dumps(variant['top_picks'], ensure_ascii=False)}")

    return "\n".join(lines) + "\n"


def run_latest_profile_comparison(
    *,
    eval_config_path: str | Path | None = None,
    variants: list[str] | None = None,
    output_dir: str | Path | None = None,
    refresh: bool = False,
    write_outputs: bool = True,
) -> dict:
    resolved_variants = list(variants or ["default_exec", "transfer_exec"])
    resolved_output_dir = Path(output_dir).resolve() if output_dir else (ROOT / "results" / "broader_history_rerank_variants_v1").resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    base = _load_base_latest((ROOT / (eval_config_path or "config_moneyflow_v1.yaml")).resolve(), VARIANT_RULES["default_exec"], refresh=refresh)
    variant_payloads = [_load_rerank_variant(resolved_output_dir, variant, refresh=refresh) for variant in resolved_variants]
    payload = {
        "generated_at": str(pd.Timestamp.now()),
        "refresh": bool(refresh),
        "base": base,
        "variants": variant_payloads,
        "alignment": _build_alignment(base, variant_payloads),
    }
    payload["comparison_message"] = build_qianxing_comparison_message(payload)
    payload["qianxing_message"] = payload["comparison_message"]["text"]

    if write_outputs:
        json_path = resolved_output_dir / "latest_profile_comparison.json"
        markdown_path = resolved_output_dir / "latest_profile_comparison.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
        payload["output_files"] = {
            "json": str(json_path),
            "markdown": str(markdown_path),
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current moneyflow_v1 latest picks against broader-history rerank variants")
    parser.add_argument("--eval-config", default="config_moneyflow_v1.yaml")
    parser.add_argument("--variants", nargs="+", default=["default_exec", "transfer_exec"])
    parser.add_argument("--output-dir", default="results/broader_history_rerank_variants_v1")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    payload = run_latest_profile_comparison(
        eval_config_path=args.eval_config,
        variants=args.variants,
        output_dir=ROOT / args.output_dir,
        refresh=args.refresh,
        write_outputs=True,
    )
    print(json.dumps({**payload.get("output_files", {}), **payload["alignment"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()