from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.ensemble_optimizer import optimize_ensemble_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for better ensemble weights.")
    parser.add_argument("--step", type=float, default=0.05, help="Weight grid step size.")
    parser.add_argument("--percentile", type=float, default=0.05, help="Selection percentile.")
    parser.add_argument("--top-k", type=int, default=25, help="How many coarse candidates to validate realistically.")
    args = parser.parse_args()

    result_df, recommendation = optimize_ensemble_weights(
        ROOT / "config.yaml",
        percentile=args.percentile,
        step=args.step,
        top_k=args.top_k,
    )
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(result_df.head(10).to_string(index=False))