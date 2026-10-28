from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inverse_rl_market_simulation.diagnostics import build_role_activity_vectors


def _resolve_stock_code(simulation_path: Path, explicit_stock_code: str | None) -> str:
    if explicit_stock_code:
        return explicit_stock_code

    config_path = simulation_path.parent / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return str(raw.get("experiment", {}).get("stock_code") or "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export daily herd/slow/trend role vectors from simulated_path.csv")
    parser.add_argument("--simulation-path", required=True, help="Path to simulated_path.csv")
    parser.add_argument("--output-path", help="Optional output csv path. Defaults to role_activity_vectors.csv beside the simulation file")
    parser.add_argument("--stock-code", help="Optional stock code override. If omitted, reads from sibling config.yaml when available")
    args = parser.parse_args()

    simulation_path = Path(args.simulation_path).resolve()
    if not simulation_path.exists():
        raise FileNotFoundError(f"simulation file not found: {simulation_path}")

    output_path = Path(args.output_path).resolve() if args.output_path else simulation_path.with_name("role_activity_vectors.csv")
    stock_code = _resolve_stock_code(simulation_path, args.stock_code)

    simulation = pd.read_csv(simulation_path)
    role_vectors = build_role_activity_vectors(simulation, stock_code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    role_vectors.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(output_path)


if __name__ == "__main__":
    main()