from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.two_stage_execution import predict_latest_for_execution
from predict_latest_broader_history_rerank import VARIANT_RULES
from predict_latest_broader_history_rerank import run_latest_broader_history_rerank


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict latest financial transformer signals")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config_two_stage_rerank_best_v1.yaml"))
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--broader-history-rerank-variant", choices=sorted(VARIANT_RULES))
    args = parser.parse_args()
    if args.broader_history_rerank_variant:
        print(
            run_latest_broader_history_rerank(
                variant=args.broader_history_rerank_variant,
                use_calibration=args.use_calibration,
                top_n=args.top_n,
            )
        )
    else:
        print(predict_latest_for_execution(args.config, top_n=args.top_n, use_calibration=args.use_calibration))