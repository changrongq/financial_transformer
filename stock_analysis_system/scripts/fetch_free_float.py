from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.free_float import (
    fetch_holder_structure,
    compute_real_free_float,
    compute_real_turnover_rate,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="抓取股东结构并计算真实自由流通盘/换手率")
    parser.add_argument("--period", default="20260331", help="十大股东披露期，默认20260331")
    parser.add_argument("--skip-holders", action="store_true", help="跳过股东结构抓取，复用已有表")
    parser.add_argument("--skip-turnover", action="store_true", help="跳过真实换手率落表")
    args = parser.parse_args()

    config = load_config(ROOT / "config.yaml")

    if not args.skip_holders:
        holders = fetch_holder_structure(config, period=args.period)
        print(f"股东结构抓取完成：{len(holders)} 条（十大股东）")

    free_float = compute_real_free_float(config)
    print(f"真实自由流通盘计算完成：{len(free_float)} 只股票")
    if not free_float.empty:
        print(free_float[["stock_code", "total_share", "real_free_float_share", "real_free_float_ratio"]].head(10).to_string())

    if not args.skip_turnover:
        turnover = compute_real_turnover_rate(config)
        print(f"真实换手率落表完成：{len(turnover)} 条")
        if not turnover.empty:
            print(turnover.head(5).to_string())
