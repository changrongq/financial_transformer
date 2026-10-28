from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"

if __name__ == "__main__":
    stamp_dir = ROOT / "experiments" / "inverse_rl_market_simulation" / "results" / "market_role_snapshot_daily"
    cmd = [
        str(PYTHON_EXE),
        str(ROOT / "experiments" / "inverse_rl_market_simulation" / "scan_market_role_snapshot.py"),
        "--candidate-file",
        str(ROOT / "experiments" / "financial_transformer" / "results" / "latest_predictions.csv"),
        "--candidate-stock-col",
        "stock_code",
        "--candidate-score-col",
        "signal_score",
        "--candidate-date-col",
        "date",
        "--candidate-top-n",
        "20",
        "--max-trials",
        "20",
        "--output-dir",
        str(stamp_dir),
    ]
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)
