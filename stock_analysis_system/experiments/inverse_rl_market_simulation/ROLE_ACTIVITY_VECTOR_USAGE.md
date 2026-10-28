# Role Activity Vector Usage

This document explains how to export daily role vectors from the inverse RL market simulation results.

## What This Produces

The output file is `role_activity_vectors.csv`.

Core columns:

- `herd_act`
- `slow_act`
- `trend_act`

Definition:

- Read daily `herd_demand`, `slow_demand`, `trend_demand` from `simulated_path.csv`
- Convert each to absolute value
- Normalize by the same-day total absolute demand
- If the total absolute demand is zero, all three values are `0.0`

Interpretation:

- A row that sums to `1.0` means the market state that day is distributed across the three roles
- A row that sums to `0.0` means the simulator produced no role activity that day

Additional columns:

- `date`
- `stock_code`
- `total_activity`
- `dominant_role`
- `herd_to_slow_weight`
- `slow_to_trend_weight`

## Case 1: New Simulation Runs

If you run the normal single-stock experiment, the vector file is now produced automatically in the result directory.

Example:

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/inverse_rl_market_simulation/run_single_stock_mvp.py --config experiments/inverse_rl_market_simulation/configs/config_single_stock_soft_switch_v1_300124.yaml
```

After the run finishes, check the configured results directory for:

- `simulated_path.csv`
- `role_activity_vectors.csv`
- `role_heatmap_beta.png`

## Case 2: Existing Simulation Results

If the result directory already exists and you only want vectors, do not rerun calibration. Export directly from `simulated_path.csv`.

Example:

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/inverse_rl_market_simulation/export_role_activity_vectors.py --simulation-path experiments/inverse_rl_market_simulation/results/single_stock_soft_switch_v1_300124_recent/simulated_path.csv
```

Default behavior:

- output file is written beside the source simulation file
- output file name is `role_activity_vectors.csv`
- if a sibling `config.yaml` exists, `stock_code` is filled automatically

Optional explicit output path:

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/inverse_rl_market_simulation/export_role_activity_vectors.py --simulation-path experiments/inverse_rl_market_simulation/results/single_stock_soft_switch_v1_300124_recent/simulated_path.csv --output-path experiments/inverse_rl_market_simulation/results/single_stock_soft_switch_v1_300124_recent/role_activity_vectors_copy.csv
```

Optional explicit stock code:

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/inverse_rl_market_simulation/export_role_activity_vectors.py --simulation-path experiments/inverse_rl_market_simulation/results/some_result/simulated_path.csv --stock-code 300124
```

## Reuse On Other Stocks

The exporter is not tied to 300124.

Use the same process for any stock as long as its `simulated_path.csv` contains these columns:

- `slow_demand`
- `trend_demand`
- `herd_demand`

If the simulation also includes switch columns, they are carried into the output automatically:

- `herd_to_slow_weight`
- `slow_to_trend_weight`

## Recommended Team Workflow

For research use:

1. Run a single-stock simulation or reuse an accepted existing result
2. Export or locate `role_activity_vectors.csv`
3. Feed `herd_act`, `slow_act`, `trend_act` into downstream feature pipelines
4. Keep `total_activity` and `dominant_role` as auxiliary diagnostics rather than primary model labels

For operations use:

1. Keep one result directory per stock
2. Keep `config.yaml`, `simulated_path.csv`, `role_activity_vectors.csv`, and `role_heatmap_beta.png` together
3. If a new trading day is added and you want a refreshed vector file, rerun the simulation or rerun the exporter on a refreshed `simulated_path.csv`

## Notes

- The exporter does not automatically fetch new market data
- The exporter only transforms an existing `simulated_path.csv`
- Automatic daily refresh would require a separate scheduled job or pipeline trigger