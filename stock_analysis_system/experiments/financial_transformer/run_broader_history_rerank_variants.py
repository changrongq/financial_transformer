from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC, ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from apply_broader_history_rerank import main as apply_rerank_main  # noqa: E402
from analyze_broader_history_rerank_walkforward import main as walkforward_main  # noqa: E402
from execution_search_static_rerank import main as static_execution_main  # noqa: E402
from financial_transformer_experiment.config import load_config  # noqa: E402


def _invoke(main_func, argv0: str, args: list[str]) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [argv0, *args]
        main_func()
    finally:
        sys.argv = original_argv


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def _variant_source_dir(config_name: str, output_tag: str) -> Path:
    config = load_config(str((ROOT / config_name).resolve()))
    return ROOT / "results" / f"{config.results_dir.name}_{output_tag}"


def _run_apply(config_name: str, train_config: str, output_tag: str) -> tuple[Path, dict]:
    _invoke(
        apply_rerank_main,
        "apply_broader_history_rerank.py",
        [
            "--eval-config",
            config_name,
            "--train-config",
            train_config,
            "--train-split",
            "both",
            "--candidate-pool",
            "5",
            "--label-mode",
            "best_in_pool",
            "--output-tag",
            output_tag,
        ],
    )
    source_dir = _variant_source_dir(config_name, output_tag)
    payload = json.loads((source_dir / "lightweight_rerank_probe.json").read_text(encoding="utf-8"))
    return source_dir, payload


def _run_walkforward(output_prefix: str, fair: dict) -> dict:
    args = [
        "--history-config",
        "config_moneyflow_v1.yaml",
        "--eval-config",
        "config_moneyflow_v1_recent_oos_v1.yaml",
        "--history-split",
        "both",
        "--candidate-pool",
        "5",
        "--label-mode",
        "best_in_pool",
        "--window-size",
        "3",
        "--step-size",
        "3",
        "--output-prefix",
        output_prefix,
        "--short-up-threshold",
        str(fair["short_up_threshold"]),
        "--take-profit-threshold",
        str(fair["take_profit_threshold"]),
        "--stop-loss-threshold",
        str(fair["stop_loss_threshold"]),
        "--top-k-per-day",
        str(fair["top_k_per_day"]),
        "--hold-days",
        str(fair["hold_days"]),
        "--take-profit",
        str(fair["take_profit"]),
        "--stop-loss",
        str(fair["stop_loss"]),
    ]
    _invoke(walkforward_main, "analyze_broader_history_rerank_walkforward.py", args)
    return json.loads((ROOT / "results" / f"{output_prefix}.json").read_text(encoding="utf-8"))


def _run_transfer_execution(config_name: str, predictions_file: Path, output_tag: str, fair: dict) -> dict:
    _invoke(
        static_execution_main,
        "execution_search_static_rerank.py",
        [
            "--config",
            config_name,
            "--predictions-file",
            str(predictions_file),
            "--score-col",
            "rerank_score",
            "--short-up-threshold",
            str(fair["short_up_threshold"]),
            "--take-profit-threshold",
            str(fair["take_profit_threshold"]),
            "--stop-loss-threshold",
            str(fair["stop_loss_threshold"]),
            "--top-k-grid",
            str(fair["top_k_per_day"]),
            "--take-profit-exit-grid",
            str(fair["take_profit"]),
            f"--stop-loss-exit-grid={fair['stop_loss']}",
            "--hold-days-grid",
            str(fair["hold_days"]),
            "--min-trades",
            "2",
            "--min-signals",
            "4",
            "--output-tag",
            output_tag,
        ],
    )
    return json.loads((predictions_file.parent / f"execution_search_best_{output_tag}.json").read_text(encoding="utf-8"))


def _write_summary(canonical_root: Path, summary: dict) -> None:
    summary_json = canonical_root / "variant_summary.json"
    summary_md = canonical_root / "variant_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    default_recent = summary["default_exec"]["recent_eval"]["rerank_summary"]
    default_old = summary["default_exec"]["old_eval"]["rerank_summary"]
    transfer_recent = summary["transfer_exec"]["recent_eval"]["best"]
    transfer_old = summary["transfer_exec"]["old_eval"]["best"]
    default_walk = summary["default_exec"]["walkforward_3x3"]
    transfer_walk = summary["transfer_exec"]["walkforward_3x3"]

    lines = ["# Broader History Rerank Variants", ""]
    lines.append("## Default Exec")
    lines.append("")
    lines.append(
        f"- Recent: total_return={default_recent['total_return'] * 100:.2f}%, sharpe={default_recent['sharpe_ratio']:.4f}, trades={default_recent['trade_count']}"
    )
    lines.append(
        f"- Old: total_return={default_old['total_return'] * 100:.2f}%, sharpe={default_old['sharpe_ratio']:.4f}, trades={default_old['trade_count']}"
    )
    lines.append(
        f"- Walkforward 3x3: improved_return_windows={default_walk['improved_return_windows']}/{default_walk['window_count']}, avg_delta_return={default_walk['avg_delta_total_return'] * 100:.2f}%, avg_delta_sharpe={default_walk['avg_delta_sharpe_ratio']:.4f}"
    )
    lines.append("")
    lines.append("## Transfer Exec")
    lines.append("")
    lines.append(
        f"- Rule: top_k={summary['transfer_exec']['fair']['top_k_per_day']}, hold_days={summary['transfer_exec']['fair']['hold_days']}, take_profit={summary['transfer_exec']['fair']['take_profit']}, stop_loss={summary['transfer_exec']['fair']['stop_loss']}"
    )
    lines.append(
        f"- Recent: total_return={transfer_recent['total_return'] * 100:.2f}%, sharpe={transfer_recent['sharpe_ratio']:.4f}, trades={transfer_recent['trade_count']}"
    )
    lines.append(
        f"- Old: total_return={transfer_old['total_return'] * 100:.2f}%, sharpe={transfer_old['sharpe_ratio']:.4f}, trades={transfer_old['trade_count']}"
    )
    lines.append(
        f"- Walkforward 3x3: improved_return_windows={transfer_walk['improved_return_windows']}/{transfer_walk['window_count']}, avg_delta_return={transfer_walk['avg_delta_total_return'] * 100:.2f}%, avg_delta_sharpe={transfer_walk['avg_delta_sharpe_ratio']:.4f}"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Default exec remains the stronger recent-walkforward variant.")
    lines.append("- Transfer exec is the stronger full-window old/recent compromise, not a strict replacement.")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize canonical broader-history rerank execution variants")
    parser.add_argument("--output-name", default="broader_history_rerank_variants_v1")
    args = parser.parse_args()

    default_fair = {
        "short_up_threshold": 0.5,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 45,
        "take_profit": 0.14,
        "stop_loss": -0.06,
    }
    transfer_fair = {
        "short_up_threshold": 0.5,
        "take_profit_threshold": 0.46,
        "stop_loss_threshold": 0.58,
        "top_k_per_day": 1,
        "hold_days": 30,
        "take_profit": 0.14,
        "stop_loss": -0.04,
    }

    recent_source_dir, recent_payload = _run_apply(
        "config_moneyflow_v1_recent_oos_v1.yaml",
        "config_moneyflow_v1.yaml",
        f"{args.output_name}_recent",
    )
    old_source_dir, old_payload = _run_apply(
        "config_moneyflow_v1.yaml",
        "config_moneyflow_v1.yaml",
        f"{args.output_name}_old",
    )

    canonical_root = ROOT / "results" / args.output_name
    default_recent_dir = canonical_root / "default_exec" / "recent_eval"
    default_old_dir = canonical_root / "default_exec" / "old_eval"
    transfer_recent_dir = canonical_root / "transfer_exec" / "recent_eval"
    transfer_old_dir = canonical_root / "transfer_exec" / "old_eval"
    default_walk_dir = canonical_root / "default_exec" / "walkforward_3x3"
    transfer_walk_dir = canonical_root / "transfer_exec" / "walkforward_3x3"

    _copy_tree_contents(recent_source_dir, default_recent_dir)
    _copy_tree_contents(old_source_dir, default_old_dir)

    transfer_recent = _run_transfer_execution(
        "config_moneyflow_v1_recent_oos_v1.yaml",
        recent_source_dir / "test_candidate_pool.csv",
        f"{args.output_name}_transfer_recent",
        transfer_fair,
    )
    transfer_old = _run_transfer_execution(
        "config_moneyflow_v1.yaml",
        old_source_dir / "test_candidate_pool.csv",
        f"{args.output_name}_transfer_old",
        transfer_fair,
    )

    _copy_tree_contents(recent_source_dir, transfer_recent_dir)
    _copy_tree_contents(old_source_dir, transfer_old_dir)

    default_walk = _run_walkforward(f"{args.output_name}_default_walkforward_3x3", default_fair)
    transfer_walk = _run_walkforward(f"{args.output_name}_transfer_walkforward_3x3", transfer_fair)

    for prefix, target_dir in [
        (f"{args.output_name}_default_walkforward_3x3", default_walk_dir),
        (f"{args.output_name}_transfer_walkforward_3x3", transfer_walk_dir),
    ]:
        target_dir.mkdir(parents=True, exist_ok=True)
        for suffix in [".json", ".md"]:
            shutil.copy2(ROOT / "results" / f"{prefix}{suffix}", target_dir / f"{prefix}{suffix}")

    summary = {
        "output_name": args.output_name,
        "default_exec": {
            "fair": default_fair,
            "recent_eval": recent_payload,
            "old_eval": old_payload,
            "walkforward_3x3": default_walk,
        },
        "transfer_exec": {
            "fair": transfer_fair,
            "recent_eval": transfer_recent,
            "old_eval": transfer_old,
            "walkforward_3x3": transfer_walk,
        },
        "dirs": {
            "default_recent": str(default_recent_dir),
            "default_old": str(default_old_dir),
            "default_walkforward": str(default_walk_dir),
            "transfer_recent": str(transfer_recent_dir),
            "transfer_old": str(transfer_old_dir),
            "transfer_walkforward": str(transfer_walk_dir),
        },
    }
    canonical_root.mkdir(parents=True, exist_ok=True)
    _write_summary(canonical_root, summary)

    print(
        json.dumps(
            {
                "canonical_root": str(canonical_root),
                "summary_json": str(canonical_root / 'variant_summary.json'),
                "summary_md": str(canonical_root / 'variant_summary.md'),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()