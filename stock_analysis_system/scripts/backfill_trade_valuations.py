from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.trade_data import backfill_trade_valuation_fields


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按交易日回填 stock_data 中的估值与换手率字段")
    parser.add_argument("--start-date", required=True, help="格式 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="格式 YYYY-MM-DD")
    parser.add_argument("--sleep-seconds", type=float, default=0.12, help="每个交易日抓取后的休眠秒数")
    args = parser.parse_args()

    config = load_config(ROOT / "config.yaml")
    df = backfill_trade_valuation_fields(
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_seconds=args.sleep_seconds,
    )
    print(
        {
            "rows": len(df),
            "start_date": str(df["日期"].min()) if not df.empty else None,
            "end_date": str(df["日期"].max()) if not df.empty else None,
            "stocks": int(df["stock_code"].nunique()) if not df.empty else 0,
            "trade_dates": int(df["日期"].nunique()) if not df.empty else 0,
        }
    )