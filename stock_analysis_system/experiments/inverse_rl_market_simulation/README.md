# Inverse RL Market Simulation MVP

This experiment is the first executable version of the inverse-RL-style market simulation line.

The first version is intentionally narrow:

- one stock
- daily data only
- three agent classes
- nine trainable parameters
- black-box random search

It is not strict IRL. It is a single-stock market-generator calibration problem.

## Files

- `config_single_stock_mvp.yaml`: default single-stock config
- `run_single_stock_mvp.py`: runnable entrypoint
- `src/inverse_rl_market_simulation/`: experiment logic

## Usage

Dry run config and data extraction:

```bash
python experiments/inverse_rl_market_simulation/run_single_stock_mvp.py --dry-run
```

Run a short search:

```bash
python experiments/inverse_rl_market_simulation/run_single_stock_mvp.py --max-trials 20
```

## Outputs

The run writes the following to the configured results directory:

- `config.yaml`
- `best_params.json`
- `fit_summary.json`
- `simulated_path.csv`
- `role_activity_vectors.csv`
- `calibration_trials.csv`
- `stability_check.json`
- `real_vs_sim_plot.png`
- `interpretation.md`

## Daily Role Vectors

Each run now also writes `role_activity_vectors.csv`.

The three core columns are:

- `herd_act`
- `slow_act`
- `trend_act`

Definition:

- For each day, take `abs(herd_demand)`, `abs(slow_demand)`, `abs(trend_demand)` from `simulated_path.csv`
- Normalize them by the same-day total absolute demand
- If total demand is zero, all three values are set to `0.0`

So for each row, the vector either sums to `1.0` or, on fully inactive days, sums to `0.0`.

The exported csv also includes:

- `date`
- `stock_code`
- `total_activity`
- `dominant_role`
- `herd_to_slow_weight`
- `slow_to_trend_weight`

## Export From Existing Results

If a result directory already has `simulated_path.csv`, you can export vectors without rerunning calibration:

```bash
python experiments/inverse_rl_market_simulation/export_role_activity_vectors.py \
	--simulation-path experiments/inverse_rl_market_simulation/results/single_stock_soft_switch_v1_300124_recent/simulated_path.csv
```

This writes `role_activity_vectors.csv` beside the input file.

If the sibling `config.yaml` exists, the script reads `stock_code` from it automatically. You can also override it manually:

```bash
python experiments/inverse_rl_market_simulation/export_role_activity_vectors.py \
	--simulation-path path/to/simulated_path.csv \
	--output-path path/to/role_activity_vectors.csv \
	--stock-code 300124
```

## Reuse On Other Stocks

The exporter is stock-agnostic. As long as the input `simulated_path.csv` contains the standard demand columns, the same script works for any stock result directory.

Typical workflow for another stock:

1. Run or reuse that stock's single-stock simulation result
2. Locate its `simulated_path.csv`
3. Run `export_role_activity_vectors.py`
4. Consume `role_activity_vectors.csv` in downstream modeling