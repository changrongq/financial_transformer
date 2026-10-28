from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
FT_SRC = EXPERIMENT_ROOT / "src"
IRL_SRC = PROJECT_ROOT / "experiments" / "inverse_rl_market_simulation" / "src"
IRL_RESULTS = PROJECT_ROOT / "experiments" / "inverse_rl_market_simulation" / "results"

for path in [PROJECT_SRC, FT_SRC, IRL_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inverse_rl_market_simulation.diagnostics import build_role_activity_vectors


DEFAULT_RESULT_DIRS = [
    "single_stock_soft_switch_v1_000001_recent",
    "single_stock_soft_switch_v1_300124_recent",
    "single_stock_soft_switch_v1_300344_recent",
    "single_stock_soft_switch_v1_601857_recent",
]


def _parse_result_dirs(raw_value: str | None) -> list[str]:
    if raw_value is None or not raw_value.strip():
        return list(DEFAULT_RESULT_DIRS)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _load_probe_frame(result_dir_name: str, horizon_days: int, threshold: float) -> pd.DataFrame:
    simulation_path = IRL_RESULTS / result_dir_name / "simulated_path.csv"
    if not simulation_path.exists():
        raise FileNotFoundError(f"missing simulation file: {simulation_path}")

    simulation = pd.read_csv(simulation_path)
    stock_code = result_dir_name.split("_")[-2]
    vectors = build_role_activity_vectors(simulation, stock_code)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(simulation["date"]),
            "stock_code": stock_code,
            "real_return": pd.to_numeric(simulation["real_return"], errors="coerce"),
            "real_turnover_z": pd.to_numeric(simulation["real_turnover_z"], errors="coerce"),
            "real_volatility_10": pd.to_numeric(simulation["real_volatility_10"], errors="coerce"),
            "close": pd.to_numeric(simulation["real_close"], errors="coerce"),
            "herd_act": vectors["herd_act"],
            "slow_act": vectors["slow_act"],
            "trend_act": vectors["trend_act"],
            "herd_to_slow_weight": pd.to_numeric(simulation.get("herd_to_slow_weight", 0.0), errors="coerce"),
            "slow_to_trend_weight": pd.to_numeric(simulation.get("slow_to_trend_weight", 0.0), errors="coerce"),
        }
    )
    frame["future_return_5d"] = frame["close"].shift(-horizon_days) / frame["close"] - 1.0
    frame["target_up_5d"] = (frame["future_return_5d"] >= threshold).astype(float)
    return frame.iloc[:-horizon_days].copy()


def _build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def _run_leave_one_stock_out(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model_name, features in feature_sets.items():
        aucs: list[float] = []
        aps: list[float] = []
        per_stock: dict[str, object] = {}
        for holdout_stock in sorted(df["stock_code"].unique()):
            train = df[df["stock_code"] != holdout_stock].copy()
            test = df[df["stock_code"] == holdout_stock].copy()
            if train["target_up_5d"].nunique() < 2 or test["target_up_5d"].nunique() < 2:
                continue
            model = _build_model()
            model.fit(train[features], train["target_up_5d"])
            probs = model.predict_proba(test[features])[:, 1]
            auc = float(roc_auc_score(test["target_up_5d"], probs))
            ap = float(average_precision_score(test["target_up_5d"], probs))
            aucs.append(auc)
            aps.append(ap)
            per_stock[holdout_stock] = {"auc": auc, "average_precision": ap}
        summary[model_name] = {
            "features": features,
            "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
            "mean_average_precision": float(np.mean(aps)) if aps else 0.0,
            "per_stock": per_stock,
        }
    return summary


def _compute_spearman(df: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for column_name in ["herd_act", "slow_act", "trend_act", "herd_to_slow_weight", "slow_to_trend_weight"]:
        output[column_name] = float(df[[column_name, "future_return_5d"]].corr(method="spearman").iloc[0, 1])
    return output


def _build_markdown(payload: dict[str, object]) -> str:
    loso = payload["leave_one_stock_out"]
    lines = [
        "# Role Vector Proxy Probe",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- result_dirs: `{', '.join(payload['result_dirs'])}`",
        f"- rows: `{payload['rows']}`",
        f"- positive_rate: `{float(payload['positive_rate']):.4f}`",
        f"- horizon_days: `{payload['horizon_days']}`",
        f"- target_threshold: `{float(payload['target_threshold']):.4f}`",
        "",
        "## Spearman vs future_return_5d",
        "",
    ]
    for key, value in payload["spearman_future_return"].items():
        lines.append(f"- {key}: `{float(value):.4f}`")

    lines.extend(["", "## Leave-One-Stock-Out Proxy Results", ""])
    for model_name in ["baseline", "role_only", "combined", "combined_plus_switch"]:
        summary = loso[model_name]
        lines.extend(
            [
                f"### {model_name}",
                "",
                f"- features: `{', '.join(summary['features'])}`",
                f"- mean_auc: `{float(summary['mean_auc']):.4f}`",
                f"- mean_average_precision: `{float(summary['mean_average_precision']):.4f}`",
                "",
            ]
        )
        for stock_code, metrics in summary["per_stock"].items():
            lines.append(
                f"- {stock_code}: auc=`{float(metrics['auc']):.4f}`, average_precision=`{float(metrics['average_precision']):.4f}`"
            )
        lines.append("")

    baseline_auc = float(loso["baseline"]["mean_auc"])
    role_auc = float(loso["role_only"]["mean_auc"])
    combined_auc = float(loso["combined"]["mean_auc"])
    combined_plus_switch_auc = float(loso["combined_plus_switch"]["mean_auc"])
    lines.extend(
        [
            "## Reading",
            "",
            f"- role_only mean_auc - baseline mean_auc = `{role_auc - baseline_auc:+.4f}`",
            f"- combined mean_auc - baseline mean_auc = `{combined_auc - baseline_auc:+.4f}`",
            f"- combined_plus_switch mean_auc - baseline mean_auc = `{combined_plus_switch_auc - baseline_auc:+.4f}`",
            "- Use this as a directional gate only. It is a lightweight proxy check, not a full transformer ablation.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether inverse-RL role vectors add signal for the financial transformer target")
    parser.add_argument("--result-dirs", help="Comma-separated inverse RL result directory names under experiments/inverse_rl_market_simulation/results")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--target-threshold", type=float, default=0.03)
    parser.add_argument("--output-stem", help="Optional output stem path without extension")
    args = parser.parse_args()

    if args.horizon_days <= 0:
        raise ValueError("--horizon-days must be positive")

    result_dirs = _parse_result_dirs(args.result_dirs)
    frames = [_load_probe_frame(result_dir_name, int(args.horizon_days), float(args.target_threshold)) for result_dir_name in result_dirs]
    df = pd.concat(frames, ignore_index=True)

    feature_sets = {
        "baseline": ["real_return", "real_turnover_z", "real_volatility_10"],
        "role_only": ["herd_act", "slow_act", "trend_act"],
        "combined": ["real_return", "real_turnover_z", "real_volatility_10", "herd_act", "slow_act", "trend_act"],
        "combined_plus_switch": [
            "real_return",
            "real_turnover_z",
            "real_volatility_10",
            "herd_act",
            "slow_act",
            "trend_act",
            "herd_to_slow_weight",
            "slow_to_trend_weight",
        ],
    }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "result_dirs": result_dirs,
        "rows": int(len(df)),
        "positive_rate": float(df["target_up_5d"].mean()),
        "horizon_days": int(args.horizon_days),
        "target_threshold": float(args.target_threshold),
        "spearman_future_return": _compute_spearman(df),
        "leave_one_stock_out": _run_leave_one_stock_out(df, feature_sets),
    }

    if args.output_stem:
        output_stem = Path(args.output_stem).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_stem = (EXPERIMENT_ROOT / "results" / f"role_vector_proxy_probe_{timestamp}").resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_stem.with_suffix(".json")
    markdown_path = output_stem.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_build_markdown(payload), encoding="utf-8")

    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()