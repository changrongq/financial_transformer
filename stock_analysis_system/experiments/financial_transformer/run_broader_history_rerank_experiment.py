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

from analyze_broader_history_rerank_walkforward import main as walkforward_main  # noqa: E402
from analyze_lightweight_rerank_probe import main as rerank_probe_main  # noqa: E402
from financial_transformer_experiment.config import load_config  # noqa: E402


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


def _run_probe(args: list[str]) -> dict:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_lightweight_rerank_probe.py", *args]
        rerank_probe_main()
    finally:
        sys.argv = original_argv

    output_tag = args[args.index("--output-tag") + 1]
    eval_config = args[args.index("--eval-config") + 1] if "--eval-config" in args else args[args.index("--config") + 1]
    results_dir_name = load_config(str((ROOT / eval_config).resolve())).results_dir.name
    output_dir = ROOT / "results" / f"{results_dir_name}_{output_tag}"
    payload_path = output_dir / "lightweight_rerank_probe.json"
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _run_walkforward(args: list[str]) -> dict:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_broader_history_rerank_walkforward.py", *args]
        walkforward_main()
    finally:
        sys.argv = original_argv

    output_prefix = args[args.index("--output-prefix") + 1]
    payload_path = ROOT / "results" / f"{output_prefix}.json"
    return json.loads(payload_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical broader-history rerank experiment and materialize fixed result directories")
    parser.add_argument("--output-name", default="broader_history_rerank_v1")
    parser.add_argument("--train-config", default="config_moneyflow_v1.yaml")
    parser.add_argument("--train-split", default="both", choices=["val", "test", "both"])
    parser.add_argument("--candidate-pool", type=int, default=5)
    parser.add_argument("--label-mode", default="best_in_pool", choices=["best_in_pool", "positive_return"])
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args()

    recent_output_tag = f"{args.output_name}_recent"
    old_output_tag = f"{args.output_name}_old"

    recent_payload = _run_probe(
        [
            "--config",
            "config_moneyflow_v1_recent_oos_v1.yaml",
            "--train-config",
            args.train_config,
            "--train-split",
            args.train_split,
            "--eval-config",
            "config_moneyflow_v1_recent_oos_v1.yaml",
            "--candidate-pool",
            str(args.candidate_pool),
            "--label-mode",
            args.label_mode,
            "--output-tag",
            recent_output_tag,
        ]
    )
    old_payload = _run_probe(
        [
            "--config",
            "config_moneyflow_v1.yaml",
            "--train-config",
            args.train_config,
            "--train-split",
            args.train_split,
            "--eval-config",
            "config_moneyflow_v1.yaml",
            "--candidate-pool",
            str(args.candidate_pool),
            "--label-mode",
            args.label_mode,
            "--output-tag",
            old_output_tag,
        ]
    )

    recent_source_dir = ROOT / "results" / f"moneyflow_v1_recent_oos_v1_{recent_output_tag}"
    old_source_dir = ROOT / "results" / f"moneyflow_v1_{old_output_tag}"
    canonical_root = ROOT / "results" / args.output_name
    recent_target_dir = canonical_root / "recent_eval"
    old_target_dir = canonical_root / "old_eval"
    walkforward_target_dir = canonical_root / "walkforward_3x3"
    _copy_tree_contents(recent_source_dir, recent_target_dir)
    _copy_tree_contents(old_source_dir, old_target_dir)

    walkforward_prefix = f"{args.output_name}_walkforward_3x3"
    walkforward_payload = _run_walkforward(
        [
            "--history-config",
            args.train_config,
            "--eval-config",
            "config_moneyflow_v1_recent_oos_v1.yaml",
            "--history-split",
            args.train_split,
            "--candidate-pool",
            str(args.candidate_pool),
            "--label-mode",
            args.label_mode,
            "--window-size",
            "3",
            "--step-size",
            "3",
            "--output-prefix",
            walkforward_prefix,
        ]
    )
    walkforward_json = ROOT / "results" / f"{walkforward_prefix}.json"
    walkforward_md = ROOT / "results" / f"{walkforward_prefix}.md"
    walkforward_target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(walkforward_json, walkforward_target_dir / walkforward_json.name)
    shutil.copy2(walkforward_md, walkforward_target_dir / walkforward_md.name)
    if not args.keep_intermediate:
        shutil.rmtree(recent_source_dir, ignore_errors=True)
        shutil.rmtree(old_source_dir, ignore_errors=True)
        walkforward_json.unlink(missing_ok=True)
        walkforward_md.unlink(missing_ok=True)

    report = {
        "output_name": args.output_name,
        "train_config": args.train_config,
        "train_split": args.train_split,
        "candidate_pool": args.candidate_pool,
        "label_mode": args.label_mode,
        "recent_eval": recent_payload,
        "old_eval": old_payload,
        "walkforward_3x3": walkforward_payload,
    }
    summary_path = canonical_root / "experiment_summary.json"
    summary_md_path = canonical_root / "experiment_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    recent_base = recent_payload["base_summary"]
    recent_rerank = recent_payload["rerank_summary"]
    old_base = old_payload["base_summary"]
    old_rerank = old_payload["rerank_summary"]
    lines = ["# Broader History Rerank Experiment", ""]
    lines.append(f"- Output name: {args.output_name}")
    lines.append(f"- Train config: {args.train_config}")
    lines.append(f"- Train split: {args.train_split}")
    lines.append(f"- Candidate pool: {args.candidate_pool}")
    lines.append(f"- Label mode: {args.label_mode}")
    lines.append("")
    lines.append("## Recent Eval")
    lines.append("")
    lines.append(
        f"- Base: total_return={recent_base['total_return'] * 100:.2f}%, sharpe={recent_base['sharpe_ratio']:.4f}, trades={recent_base['trade_count']}, signals={recent_base['signal_count']}"
    )
    lines.append(
        f"- Rerank: total_return={recent_rerank['total_return'] * 100:.2f}%, sharpe={recent_rerank['sharpe_ratio']:.4f}, trades={recent_rerank['trade_count']}, signals={recent_rerank['signal_count']}"
    )
    lines.append("")
    lines.append("## Old Eval")
    lines.append("")
    lines.append(
        f"- Base: total_return={old_base['total_return'] * 100:.2f}%, sharpe={old_base['sharpe_ratio']:.4f}, trades={old_base['trade_count']}, signals={old_base['signal_count']}"
    )
    lines.append(
        f"- Rerank: total_return={old_rerank['total_return'] * 100:.2f}%, sharpe={old_rerank['sharpe_ratio']:.4f}, trades={old_rerank['trade_count']}, signals={old_rerank['signal_count']}"
    )
    lines.append("")
    lines.append(f"- Recent artifacts: {recent_target_dir}")
    lines.append(f"- Old artifacts: {old_target_dir}")
    lines.append(f"- Walkforward artifacts: {walkforward_target_dir}")
    lines.append("")
    lines.append("## Walkforward 3x3")
    lines.append("")
    lines.append(
        f"- Improved return windows: {walkforward_payload['improved_return_windows']} / {walkforward_payload['window_count']}"
    )
    lines.append(
        f"- Improved sharpe windows: {walkforward_payload['improved_sharpe_windows']} / {walkforward_payload['window_count']}"
    )
    lines.append(f"- Avg delta total return: {walkforward_payload['avg_delta_total_return'] * 100:.2f}%")
    lines.append(f"- Avg delta sharpe: {walkforward_payload['avg_delta_sharpe_ratio']:.4f}")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "canonical_root": str(canonical_root),
                "summary_json": str(summary_path),
                "summary_md": str(summary_md_path),
                "recent_eval_dir": str(recent_target_dir),
                "old_eval_dir": str(old_target_dir),
                "walkforward_dir": str(walkforward_target_dir),
                "keep_intermediate": bool(args.keep_intermediate),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()