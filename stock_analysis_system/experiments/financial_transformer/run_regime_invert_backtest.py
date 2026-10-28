"""下跌 regime 信号反转回测对比。

对比「基线」vs「下跌市反转 signal_score」的完整回测指标，
验证"模型追涨滞后、下跌市反向选股可减亏/转正"的假设。

用法：
    python experiments/financial_transformer/run_regime_invert_backtest.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROJECT_SRC = ROOT.parents[1] / "src"
for path in [PROJECT_SRC, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_transformer_experiment.backtest import TransformerStrategyBacktester
from financial_transformer_experiment.config import load_config
from financial_transformer_experiment.two_stage_execution import resolve_single_stage_execution_defaults


CONFIG_PATH = ROOT / "config_two_stage_rerank_leader_tiered_v1.yaml"


def _fair_params(config):
    defaults = resolve_single_stage_execution_defaults(str(config_path))
    if defaults is not None:
        return dict(
            short_up_threshold=float(defaults.default_short_up_threshold),
            take_profit_threshold=float(defaults.default_take_profit_threshold),
            stop_loss_threshold=float(defaults.default_stop_loss_threshold),
            top_k_per_day=int(defaults.default_top_k_per_day),
            hold_days=int(defaults.default_execution_hold_days),
            take_profit=float(defaults.default_execution_take_profit),
            stop_loss=float(defaults.default_execution_stop_loss),
        )
    return dict(
        short_up_threshold=0.50, take_profit_threshold=0.46, stop_loss_threshold=0.58,
        top_k_per_day=1, hold_days=45, take_profit=0.14, stop_loss=-0.06,
    )


def _run(config_path, invert, boundary):
    config = load_config(str(config_path))
    fair = _fair_params(config)
    bt = TransformerStrategyBacktester(config)
    result = bt.run(
        start_date=None,
        end_date=config.splits.test_end,
        short_up_threshold=fair["short_up_threshold"],
        take_profit_threshold=fair["take_profit_threshold"],
        stop_loss_threshold=fair["stop_loss_threshold"],
        top_k_per_day=fair["top_k_per_day"],
        hold_days=fair["hold_days"],
        take_profit=fair["take_profit"],
        stop_loss=fair["stop_loss"],
        initial_capital=1_000_000.0,
        commission=0.001,
        slippage=0.001,
        min_lot=100,
        min_position_weight=0.08,
        max_position_weight=0.20,
        max_total_position_fraction=1.0,
        cooldown_trigger_losses=0,
        cooldown_days=0,
        min_secondary_primary_ratio=0.0,
        rank_weight_power=0.0,
        min_score_ratio_to_leader=0.0,
        invert_down_regime=invert,
        regime_boundary=boundary,
    )
    return result.summary


if __name__ == "__main__":
    config_path = CONFIG_PATH

    print("=" * 70)
    print("下跌 regime 信号反转回测对比 (micro 模型)")
    print("=" * 70)

    base = _run(config_path, invert=False, boundary=0.0)
    inv = _run(config_path, invert=True, boundary=0.0)

    def line(tag, s):
        return (
            f"{tag}: total_return={s.get('total_return', 0)*100:+.2f}%  "
            f"sharpe={s.get('sharpe_ratio', 0):.3f}  "
            f"max_dd={s.get('max_drawdown', 0)*100:.2f}%  "
            f"trades={s.get('trade_count', 0)}  win_rate={s.get('win_rate', 0)*100:.1f}%"
        )

    print("\n[基线  不反转]", line("基线", base))
    print("[反转  下跌市]", line("反转", inv))

    report = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path.name),
        "base": {k: base.get(k) for k in ["total_return", "sharpe_ratio", "max_drawdown", "trade_count", "signal_count", "win_rate"]},
        "inverted": {k: inv.get(k) for k in ["total_return", "sharpe_ratio", "max_drawdown", "trade_count", "signal_count", "win_rate"]},
    }
    out = ROOT / "results" / "regime_invert_backtest_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")
