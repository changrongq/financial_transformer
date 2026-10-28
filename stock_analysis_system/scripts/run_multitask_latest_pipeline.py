from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.market_cap import fetch_and_store_market_caps
from hierarchical_system.multitask_inference import ProductionMultiTaskInferenceEngine
from hierarchical_system.trade_data import (
    latest_moneyflow_date,
    latest_top_inst_date,
    latest_trade_date,
    update_latest_moneyflow_data,
    update_latest_top_inst_data,
    update_latest_trade_data,
)


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    trade_df = update_latest_trade_data(config)
    moneyflow_df = update_latest_moneyflow_data(config)
    top_inst_df = update_latest_top_inst_data(config)
    fetch_and_store_market_caps(config, refresh_existing=True)
    latest, strategy_a = ProductionMultiTaskInferenceEngine(ROOT / "config.yaml").predict_latest(top_n=50)
    trade_max_date = latest_trade_date(config.databases.trade_db)
    moneyflow_max_date = latest_moneyflow_date(config.databases.trade_db)
    top_inst_max_date = latest_top_inst_date(config.databases.trade_db)
    prediction_max_date = str(latest["date"].max()) if not latest.empty else None
    if trade_max_date and prediction_max_date != trade_max_date:
        raise RuntimeError(
            f"多任务最新预测日期落后于交易库: trade_db={trade_max_date}, latest_multitask_picks={prediction_max_date}."
        )
    if trade_max_date and moneyflow_max_date and moneyflow_max_date != trade_max_date:
        raise RuntimeError(
            f"多任务资金流日期落后于交易库: trade_db={trade_max_date}, moneyflow_db={moneyflow_max_date}."
        )
    if trade_max_date and top_inst_max_date and top_inst_max_date != trade_max_date:
        raise RuntimeError(
            f"多任务龙虎榜机构日期落后于交易库: trade_db={trade_max_date}, top_inst_db={top_inst_max_date}."
        )
    print(
        {
            "trade_rows": len(trade_df),
            "trade_date": trade_df["日期"].max() if not trade_df.empty else None,
            "moneyflow_rows": len(moneyflow_df),
            "moneyflow_date": moneyflow_df["日期"].max() if not moneyflow_df.empty else None,
            "top_inst_rows": len(top_inst_df),
            "top_inst_date": top_inst_df["日期"].max() if not top_inst_df.empty else None,
            "prediction_count": len(latest),
            "strategy_a_count": len(strategy_a),
            "prediction_date": prediction_max_date,
            "top_stock": latest.iloc[0]["stock_code"] if not latest.empty else None,
            "top_strategy_a_stock": strategy_a.iloc[0]["stock_code"] if not strategy_a.empty else None,
            "output_file": str(config.results_dir / "latest_multitask_picks.csv"),
            "strategy_a_file": str(config.results_dir / "latest_multitask_strategy_a.csv"),
        }
    )