from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inverse_rl_market_simulation.config import load_config
from inverse_rl_market_simulation.data import load_stock_window
from inverse_rl_market_simulation.diagnostics import run_stability_check, write_outputs
from inverse_rl_market_simulation.optimize import run_calibration


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run single-stock market-generator MVP calibration")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "config_single_stock_mvp.yaml"))
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_trials is not None:
        config = replace(
            config,
            optimization=replace(config.optimization, n_trials=int(args.max_trials)),
        )

    real_frame = load_stock_window(config)
    if args.dry_run:
        summary = {
            "stock_code": config.experiment.stock_code,
            "rows": int(len(real_frame)),
            "start_date": str(real_frame["date"].iloc[0].date()),
            "end_date": str(real_frame["date"].iloc[-1].date()),
            "close_min": float(real_frame["close"].min()),
            "close_max": float(real_frame["close"].max()),
            "turnover_mean": float(real_frame["turnover_rate"].mean()),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    calibration = run_calibration(config, real_frame)
    stability = run_stability_check(
        real_frame,
        calibration["best_params"],
        config.experiment.stability_window_days,
        search_space=config.optimization.search_space,
    )
    outputs = write_outputs(
        config=config,
        real_frame=real_frame,
        best_simulation=calibration["best_simulation"],
        best_params=calibration["best_params"],
        best_result=calibration["best_result"],
        baseline_results=calibration["baseline_results"],
        trials=calibration["trials"],
        stability=stability,
        source_config=Path(args.config).resolve(),
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()