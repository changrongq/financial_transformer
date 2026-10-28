from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.trade_data import get_stock_concept_refresh_progress


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    progress = get_stock_concept_refresh_progress(config)

    total_concepts = int(progress["total_concepts"])
    covered_concepts = int(progress["covered_concepts"])
    processed_concepts = int(progress["processed_concepts"])
    covered_ratio = float(progress["covered_concept_ratio"]) * 100.0
    processed_ratio = float(progress["processed_ratio"]) * 100.0
    member_rows = int(progress["member_rows"])
    stock_rows = int(progress["stock_rows"])
    avg_concepts_per_stock = float(progress["avg_concepts_per_stock"])
    last_concept_id = str(progress["last_concept_id"] or "")
    next_concept_id = progress["next_concept_id"]
    cycle_completed = bool(progress["cycle_completed"])
    last_cycle_completed_at = progress["last_cycle_completed_at"]

    print(f"concept_refresh_processed: {processed_concepts}/{total_concepts} ({processed_ratio:.2f}%)")
    print(f"concept_refresh_covered: {covered_concepts}/{total_concepts} ({covered_ratio:.2f}%)")
    print(f"concept_member_rows: {member_rows}")
    print(f"concept_stock_rows: {stock_rows}")
    print(f"avg_concepts_per_stock: {avg_concepts_per_stock:.3f}")
    print(f"last_concept_id: {last_concept_id or 'none'}")
    print(f"next_concept_id: {next_concept_id or 'none'}")
    print(f"cycle_completed: {str(cycle_completed).lower()}")
    print(f"last_cycle_completed_at: {last_cycle_completed_at or 'none'}")
