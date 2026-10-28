from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"

for path in [PROJECT_SRC, SRC, ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_lightweight_rerank_probe import main as rerank_probe_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the validated broader-history rerank recipe to an arbitrary evaluation config"
    )
    parser.add_argument("--eval-config", default="config_moneyflow_v1_recent_oos_v1.yaml")
    parser.add_argument("--train-config", default="config_moneyflow_v1.yaml")
    parser.add_argument("--train-configs", default=None)
    parser.add_argument("--train-split", default="both", choices=["val", "test", "both"])
    parser.add_argument("--candidate-pool", type=int, default=5)
    parser.add_argument("--label-mode", default="best_in_pool", choices=["best_in_pool", "positive_return"])
    parser.add_argument("--output-tag", default="broader_history_rerank_apply")
    parser.add_argument("--short-up-threshold", type=float, default=None)
    parser.add_argument("--take-profit-threshold", type=float, default=None)
    parser.add_argument("--stop-loss-threshold", type=float, default=None)
    parser.add_argument("--top-k-per-day", type=int, default=None)
    parser.add_argument("--hold-days", type=int, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--state-profile", type=str, default=None)
    args = parser.parse_args()

    probe_args = [
        "--config",
        args.eval_config,
        "--eval-config",
        args.eval_config,
        "--train-split",
        args.train_split,
        "--candidate-pool",
        str(args.candidate_pool),
        "--label-mode",
        args.label_mode,
        "--output-tag",
        args.output_tag,
    ]

    if args.train_configs:
        probe_args.extend(["--train-configs", args.train_configs])
    else:
        probe_args.extend(["--train-config", args.train_config])

    optional_args = {
        "--short-up-threshold": args.short_up_threshold,
        "--take-profit-threshold": args.take_profit_threshold,
        "--stop-loss-threshold": args.stop_loss_threshold,
        "--top-k-per-day": args.top_k_per_day,
        "--hold-days": args.hold_days,
        "--take-profit": args.take_profit,
        "--stop-loss": args.stop_loss,
        "--state-profile": args.state_profile,
    }
    for flag, value in optional_args.items():
        if value is not None:
            probe_args.extend([flag, str(value)])

    original_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_lightweight_rerank_probe.py", *probe_args]
        rerank_probe_main()
    finally:
        sys.argv = original_argv

    print(
        json.dumps(
            {
                "eval_config": args.eval_config,
                "train_config": args.train_config if not args.train_configs else None,
                "train_configs": args.train_configs.split(",") if args.train_configs else None,
                "train_split": args.train_split,
                "candidate_pool": args.candidate_pool,
                "label_mode": args.label_mode,
                "output_tag": args.output_tag,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()