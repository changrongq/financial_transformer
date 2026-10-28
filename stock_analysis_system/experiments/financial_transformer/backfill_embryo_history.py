"""回填 6 月~7 月的台风胚胎追踪历史数据。
对每个交易日运行 typhoon_tracker.py --mode embryo --top-n 50，
将输出的 CSV 存入 tracker_dir 供持续性预测模型训练使用。
"""

from __future__ import annotations

import subprocess
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
TRACKER_SCRIPT = Path(__file__).resolve().parent / "typhoon_tracker.py"
TRADE_DB = ROOT.parent / "stock_history.db"


def trading_dates(start: str, end: str) -> list[str]:
    conn = sqlite3.connect(str(TRADE_DB))
    try:
        rows = conn.execute(
            "SELECT DISTINCT [日期] FROM stock_data WHERE [日期] >= ? AND [日期] <= ? ORDER BY [日期]",
            (start, end),
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows if row[0]]


def main():
    dates = trading_dates("2026-06-01", "2026-07-30")

    succeeded = 0
    failed = []

    for idx, date in enumerate(dates):
        print(f"[{idx+1}/{len(dates)}] {date} ...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [str(PYTHON), str(TRACKER_SCRIPT), "--mode", "embryo", "--top-n", "50", "--end-date", date],
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            if result.returncode != 0:
                failed.append((date, result.stderr.strip()[-200:] if result.stderr else "unknown"))
                print("FAIL")
            else:
                succeeded += 1
                print("OK")
        except subprocess.TimeoutExpired:
            failed.append((date, "timeout"))
            print("TIMEOUT")
        except Exception as exc:
            failed.append((date, str(exc)))
            print("ERROR", exc)

    print(f"\n完成: 成功 {succeeded}, 失败 {len(failed)}")
    if failed:
        for date, reason in failed:
            print(f"  {date}: {reason}")


if __name__ == "__main__":
    main()
