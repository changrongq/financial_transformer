from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inverse_rl_market_simulation.config import load_config
from inverse_rl_market_simulation.data import load_stock_window
from inverse_rl_market_simulation.diagnostics import build_role_activity_vectors
from inverse_rl_market_simulation.optimize import run_calibration


def _parse_stock_codes(raw_value: str | None) -> list[str]:
    if raw_value is None or not raw_value.strip():
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _load_stock_codes_file(file_path: str | None) -> list[str]:
    if not file_path:
        return []
    path = Path(file_path).resolve()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _load_all_stock_codes(trade_db: Path) -> list[str]:
    query = "SELECT DISTINCT stock_code FROM stock_data ORDER BY stock_code"
    with sqlite3.connect(str(trade_db)) as conn:
        frame = pd.read_sql_query(query, conn)
    return sorted({str(value).strip() for value in frame["stock_code"].dropna().tolist() if str(value).strip()})


def _load_candidate_pool(
    candidate_file: str,
    stock_col: str,
    score_col: str,
    date_col: str,
    top_n: int | None,
) -> tuple[list[str], pd.DataFrame]:
    path = Path(candidate_file).resolve()
    frame = pd.read_csv(path, dtype={stock_col: str})
    if stock_col not in frame.columns:
        raise ValueError(f"candidate file missing required column: {stock_col}")

    frame = frame.copy()
    frame[stock_col] = frame[stock_col].astype(str).str.strip().str.zfill(6)
    frame = frame[frame[stock_col] != ""].reset_index(drop=True)

    if date_col in frame.columns:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        latest_date = frame[date_col].max()
        if pd.notna(latest_date):
            frame = frame[frame[date_col] == latest_date].copy()

    if score_col in frame.columns:
        frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
        frame = frame.sort_values([score_col], ascending=[False], na_position="last")

    frame = frame.drop_duplicates(subset=[stock_col], keep="first").reset_index(drop=True)
    if top_n is not None:
        frame = frame.head(max(int(top_n), 0)).copy()

    stock_codes = frame[stock_col].astype(str).tolist()
    keep_cols = [col for col in [stock_col, date_col, score_col, "layer", "prediction_prob", "pred_short_up", "pred_take_profit", "pred_stop_loss"] if col in frame.columns]
    return stock_codes, frame[keep_cols].copy()


def _resolve_stock_codes(trade_db: Path, inline_codes: list[str], file_codes: list[str], limit: int | None) -> list[str]:
    if inline_codes or file_codes:
        stock_codes = sorted({*inline_codes, *file_codes})
    else:
        stock_codes = _load_all_stock_codes(trade_db)
    if limit is not None:
        stock_codes = stock_codes[: max(int(limit), 0)]
    return stock_codes


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def _compute_snapshot_row(stock_code: str, real_frame: pd.DataFrame, best_simulation: pd.DataFrame, best_result: dict[str, object], n_trials: int) -> dict[str, object]:
    role_vectors = build_role_activity_vectors(best_simulation, stock_code)
    latest = role_vectors.iloc[-1]
    latest_sim = best_simulation.iloc[-1]

    latest_close = _safe_float(latest_sim["real_close"])
    ma20 = _safe_float(real_frame["close"].tail(20).mean(), default=latest_close)
    close_vs_ma20 = 0.0 if ma20 == 0.0 else latest_close / ma20 - 1.0
    above_cost_proxy = 1.0 if latest_close >= ma20 else 0.0

    role_5d = role_vectors[["herd_act", "slow_act", "trend_act"]].tail(5).mean()
    herd_act = _safe_float(latest["herd_act"])
    slow_act = _safe_float(latest["slow_act"])
    trend_act = _safe_float(latest["trend_act"])
    herd_act_5d = _safe_float(role_5d["herd_act"])
    slow_act_5d = _safe_float(role_5d["slow_act"])
    trend_act_5d = _safe_float(role_5d["trend_act"])

    herd_to_slow_weight = _safe_float(latest.get("herd_to_slow_weight", 0.0))
    slow_to_trend_weight = _safe_float(latest.get("slow_to_trend_weight", 0.0))
    total_activity = _safe_float(latest.get("total_activity", 0.0))

    dominant_role = str(latest.get("dominant_role", "none") or "none")
    herd_risk_score = 0.7 * herd_act + 0.3 * herd_act_5d - 0.25 * herd_to_slow_weight
    slow_build_score = 0.6 * slow_act + 0.4 * slow_act_5d + 0.35 * above_cost_proxy + 0.35 * slow_to_trend_weight
    trend_watch_score = 0.5 * trend_act + 0.3 * trend_act_5d + 0.45 * slow_to_trend_weight - 0.15 * herd_act
    review_priority_score = slow_build_score + 0.8 * trend_watch_score - 0.6 * herd_risk_score

    tags: list[str] = []
    if herd_risk_score >= 0.45:
        tags.append("crowded_herd")
    if slow_build_score >= 0.65:
        tags.append("slow_accumulation")
    if trend_watch_score >= 0.35:
        tags.append("trend_watch")
    if close_vs_ma20 > 0.03:
        tags.append("above_cost_proxy")

    return {
        "stock_code": stock_code,
        "scan_date": str(pd.Timestamp(latest["date"]).date()),
        "latest_close": latest_close,
        "close_vs_ma20": close_vs_ma20,
        "latest_return_1d": _safe_float(latest_sim.get("real_return", 0.0)),
        "latest_turnover_rate": _safe_float(latest_sim.get("real_turnover_rate", 0.0)),
        "best_total_loss": _safe_float(best_result.get("total_loss", 0.0)),
        "n_trials": int(n_trials),
        "herd_act": herd_act,
        "slow_act": slow_act,
        "trend_act": trend_act,
        "herd_act_5d": herd_act_5d,
        "slow_act_5d": slow_act_5d,
        "trend_act_5d": trend_act_5d,
        "dominant_role": dominant_role,
        "total_activity": total_activity,
        "herd_to_slow_weight": herd_to_slow_weight,
        "slow_to_trend_weight": slow_to_trend_weight,
        "trap_state": _safe_float(latest_sim.get("trap_state", 0.0)),
        "trend_switch_state": _safe_float(latest_sim.get("trend_switch_state", 0.0)),
        "slow_inventory": _safe_float(latest_sim.get("slow_inventory", 0.0)),
        "trend_inventory": _safe_float(latest_sim.get("trend_inventory", 0.0)),
        "herd_risk_score": herd_risk_score,
        "slow_build_score": slow_build_score,
        "trend_watch_score": trend_watch_score,
        "review_priority_score": review_priority_score,
        "tags": "|".join(tags),
    }


def _build_markdown(scan_rows: pd.DataFrame, failures: list[dict[str, str]], output_dir: Path) -> str:
    top_review = scan_rows.sort_values(["review_priority_score", "slow_build_score"], ascending=[False, False]).head(20)
    top_herd = scan_rows.sort_values(["herd_risk_score", "herd_act"], ascending=[False, False]).head(20)
    dominant_counts = scan_rows["dominant_role"].value_counts().to_dict()

    lines = [
        "# Market Role Snapshot Scan",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- scanned_count: `{len(scan_rows)}`",
        f"- failed_count: `{len(failures)}`",
        f"- output_dir: `{output_dir}`",
        "",
        "## Dominant Role Counts",
        "",
    ]
    for role_name, count in dominant_counts.items():
        lines.append(f"- {role_name}: `{int(count)}`")

    lines.extend(["", "## Top Review Priority", ""])
    for _, row in top_review.iterrows():
        lines.append(
            f"- {row['stock_code']}: dominant=`{row['dominant_role']}`, review_priority=`{float(row['review_priority_score']):.4f}`, slow_build=`{float(row['slow_build_score']):.4f}`, trend_watch=`{float(row['trend_watch_score']):.4f}`, herd_risk=`{float(row['herd_risk_score']):.4f}`, tags=`{row['tags']}`"
        )

    lines.extend(["", "## Top Herd Risk", ""])
    for _, row in top_herd.iterrows():
        lines.append(
            f"- {row['stock_code']}: dominant=`{row['dominant_role']}`, herd_risk=`{float(row['herd_risk_score']):.4f}`, herd_act=`{float(row['herd_act']):.4f}`, herd_act_5d=`{float(row['herd_act_5d']):.4f}`, herd_to_slow=`{float(row['herd_to_slow_weight']):.4f}`, tags=`{row['tags']}`"
        )

    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures[:20]:
            lines.append(f"- {item['stock_code']}: `{item['error']}`")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch scan market role snapshots from inverse-RL soft-switch calibration")
    parser.add_argument("--config-template", default=str(EXPERIMENT_ROOT / "configs" / "config_single_stock_soft_switch_v1_300124.yaml"))
    parser.add_argument("--stock-codes", help="Optional comma-separated stock codes")
    parser.add_argument("--stock-codes-file", help="Optional text file with one stock code per line")
    parser.add_argument("--candidate-file", help="Optional candidate CSV file (daily model recommendation list)")
    parser.add_argument("--candidate-stock-col", default="stock_code")
    parser.add_argument("--candidate-score-col", default="signal_score")
    parser.add_argument("--candidate-date-col", default="date")
    parser.add_argument("--candidate-top-n", type=int, help="Optional top N candidates from candidate file")
    parser.add_argument("--limit", type=int, help="Optional limit on number of scanned stocks")
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--end-date", help="Optional override for experiment.end_date")
    parser.add_argument("--output-dir", help="Optional explicit output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    template_config = load_config(args.config_template)
    candidate_frame: pd.DataFrame | None = None
    if args.candidate_file:
        stock_codes, candidate_frame = _load_candidate_pool(
            candidate_file=args.candidate_file,
            stock_col=args.candidate_stock_col,
            score_col=args.candidate_score_col,
            date_col=args.candidate_date_col,
            top_n=args.candidate_top_n,
        )
        if args.limit is not None:
            stock_codes = stock_codes[: max(int(args.limit), 0)]
    else:
        stock_codes = _resolve_stock_codes(
            template_config.databases.trade_db,
            _parse_stock_codes(args.stock_codes),
            _load_stock_codes_file(args.stock_codes_file),
            args.limit,
        )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (EXPERIMENT_ROOT / "results" / f"market_role_snapshot_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload = {
            "config_template": str(Path(args.config_template).resolve()),
            "candidate_file": str(Path(args.candidate_file).resolve()) if args.candidate_file else None,
            "stock_count": int(len(stock_codes)),
            "max_trials": int(args.max_trials),
            "sample_stock_codes": stock_codes[:20],
            "output_dir": str(output_dir),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    scan_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, stock_code in enumerate(stock_codes, start=1):
        try:
            config = replace(
                template_config,
                experiment=replace(
                    template_config.experiment,
                    stock_code=str(stock_code),
                    end_date=args.end_date if args.end_date is not None else template_config.experiment.end_date,
                ),
                optimization=replace(template_config.optimization, n_trials=int(args.max_trials)),
                results=replace(template_config.results, results_dir=output_dir / "per_stock" / str(stock_code)),
            )
            real_frame = load_stock_window(config)
            calibration = run_calibration(config, real_frame)
            scan_rows.append(
                _compute_snapshot_row(
                    str(stock_code),
                    real_frame,
                    calibration["best_simulation"],
                    calibration["best_result"],
                    int(config.optimization.n_trials),
                )
            )
        except Exception as exc:
            failures.append({"stock_code": str(stock_code), "error": str(exc)})

        if index % 25 == 0:
            print(json.dumps({"progress": f"{index}/{len(stock_codes)}", "success": len(scan_rows), "failures": len(failures)}, ensure_ascii=False))

    if not scan_rows:
        raise RuntimeError("No stocks were scanned successfully.")

    scan_df = pd.DataFrame(scan_rows).sort_values(["review_priority_score", "slow_build_score"], ascending=[False, False]).reset_index(drop=True)
    summary_csv = output_dir / "market_role_snapshot.csv"
    failures_csv = output_dir / "market_role_snapshot_failures.csv"
    with_signals_csv = output_dir / "market_role_snapshot_with_signals.csv"
    markdown_path = output_dir / "market_role_snapshot.md"
    metadata_path = output_dir / "market_role_snapshot.json"

    merged_df = scan_df
    if candidate_frame is not None and not candidate_frame.empty:
        candidate_view = candidate_frame.copy()
        candidate_view = candidate_view.rename(columns={
            args.candidate_stock_col: "stock_code",
            args.candidate_date_col: "signal_date",
        })
        if args.candidate_score_col in candidate_view.columns:
            candidate_view[args.candidate_score_col] = pd.to_numeric(candidate_view[args.candidate_score_col], errors="coerce")
        merged_df = candidate_view.merge(scan_df, on="stock_code", how="left")
        sort_cols = [col for col in [args.candidate_score_col, "review_priority_score", "slow_build_score"] if col in merged_df.columns]
        if sort_cols:
            merged_df = merged_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    scan_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    merged_df.to_csv(with_signals_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(failures_csv, index=False, encoding="utf-8-sig")
    markdown_path.write_text(_build_markdown(scan_df, failures, output_dir), encoding="utf-8")
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_template": str(Path(args.config_template).resolve()),
        "candidate_file": str(Path(args.candidate_file).resolve()) if args.candidate_file else None,
        "stock_count_requested": int(len(stock_codes)),
        "stock_count_succeeded": int(len(scan_df)),
        "stock_count_failed": int(len(failures)),
        "max_trials": int(args.max_trials),
        "end_date": args.end_date if args.end_date is not None else template_config.experiment.end_date,
        "summary_csv": str(summary_csv),
        "with_signals_csv": str(with_signals_csv),
        "failures_csv": str(failures_csv),
        "markdown": str(markdown_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()