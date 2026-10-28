from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.selection import write_daily_selection


if __name__ == "__main__":
    result = write_daily_selection(
        input_file=ROOT / "results" / "latest_picks.csv",
        output_file=ROOT / "results" / "daily_selection_ensemble_top5pct.csv",
        percentile=0.05,
    )
    print(result.to_string(index=False))