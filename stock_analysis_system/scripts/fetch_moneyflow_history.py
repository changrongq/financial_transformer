from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.trade_data import update_moneyflow_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="回填 Tushare 个股日度资金流向")
    parser.add_argument("--start-date", default=None, help="格式 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="格式 YYYY-MM-DD")
    args = parser.parse_args()

    config = load_config(ROOT / "config.yaml")
    df = update_moneyflow_history(config, start_date=args.start_date, end_date=args.end_date)
    print(
        {
            "rows": len(df),
            "start_date": str(df["日期"].min()) if not df.empty else None,
            "end_date": str(df["日期"].max()) if not df.empty else None,
        }
    )