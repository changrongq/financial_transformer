from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
EXPERIMENT_SRC = EXPERIMENT_ROOT / "src"

for path in [PROJECT_SRC, EXPERIMENT_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.data import _latest_market_caps
from financial_transformer_experiment.data import normalize_stock_code
from hierarchical_system.features import layer_from_market_cap_yi


def _load_basic_info(trade_db: str, stock_codes: list[str]) -> pd.DataFrame:
    if not stock_codes:
        return pd.DataFrame(columns=["stock_code", "名称", "行业"])
    normalized_codes = [normalize_stock_code(code) for code in stock_codes]
    placeholders = ", ".join(["?"] * len(normalized_codes))
    conn = sqlite3.connect(trade_db)
    try:
        frame = pd.read_sql_query(
            f"SELECT stock_code, 名称, 行业 FROM stock_basic_info WHERE stock_code IN ({placeholders})",
            conn,
            params=normalized_codes,
        )
    finally:
        conn.close()
    if frame.empty:
        return pd.DataFrame(columns=["stock_code", "名称", "行业"])
    frame["stock_code"] = frame["stock_code"].astype(str).map(normalize_stock_code)
    return frame.drop_duplicates("stock_code", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a manual concept label template for the current experiment universe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        raise RuntimeError(f"输出文件已存在: {output_path}。如需覆盖请加 --force。")

    market_caps = _latest_market_caps(config).copy()
    market_caps["stock_code"] = market_caps["stock_code"].astype(str).map(normalize_stock_code)
    market_caps["layer"] = market_caps["market_cap_yi"].apply(lambda value: layer_from_market_cap_yi(float(value)))
    basic_info = _load_basic_info(config.databases.trade_db, market_caps["stock_code"].tolist())
    template = market_caps.merge(basic_info, on="stock_code", how="left")
    template = template.rename(columns={"名称": "stock_name", "行业": "industry_name"})
    template["stock_name"] = template["stock_name"].fillna("")
    template["industry_name"] = template["industry_name"].fillna("")
    template["manual_concept_names"] = ""
    template["manual_concept_ids"] = ""
    template["notes"] = ""
    template = template[[
        "stock_code",
        "stock_name",
        "industry_name",
        "layer",
        "market_cap_yi",
        "manual_concept_names",
        "manual_concept_ids",
        "notes",
    ]].sort_values(["layer", "market_cap_yi", "stock_code"], ascending=[True, False, True]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"exported {len(template)} rows to {output_path}")


if __name__ == "__main__":
    main()