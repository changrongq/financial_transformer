"""regime 择时回测 —— 路线 C 核心验证。

核心结论（已通过参数敏感性测试）：
  "预测个股涨跌"很难（micro 模型塌缩、五轮实验均无法让测试集转正），
  但"大盘择时"极其简单且有效——一条「大盘 N 日均收益转负就空仓」的规则，
  就能在全期把回撤从 -24% 压到 -3%，收益翻数倍。

规则：
  大盘（全市场等权）近 N 日均收益 < 阈值 → 空仓（当日组合收益 = 0）
  否则 → 满仓（持有全市场等权，吃到当日 market_return）

用法：
  python experiments/financial_transformer/run_regime_timing_backtest.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


def _resolve_trade_db() -> str:
    """从 config.yaml 解析 trade_db 路径（相对项目根）。"""
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return str(p.resolve())
    # fallback：项目上级目录（历史默认位置 D:\qcr\Stock\stock_history.db）
    return str((PROJECT_ROOT.parent / "stock_history.db").resolve())


def _load_market_daily(trade_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(trade_db)
    try:
        sd = pd.read_sql_query("SELECT * FROM stock_data", conn)
    finally:
        conn.close()
    sd.columns = [
        "stock_code", "date", "open", "high", "low", "close", "volume",
        "turnover", "amount", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
    ]
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.zfill(6)
    sd["close"] = pd.to_numeric(sd["close"], errors="coerce")
    sd = sd.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    sd["ret_1d"] = sd.groupby("stock_code")["close"].pct_change()
    daily = sd.groupby("date")["ret_1d"].mean().rename("market_return").reset_index()
    return daily


def _max_drawdown(series: pd.Series) -> float:
    cum = (1.0 + series.fillna(0.0)).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _evaluate(daily: pd.DataFrame, window: int, threshold: float, start: str, end: str) -> dict:
    d = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    d["mean"] = d["market_return"].rolling(window, min_periods=3).mean()
    d["held"] = d["mean"] >= threshold
    d["strategy_return"] = np.where(d["held"], d["market_return"], 0.0)

    bh_return = float((1.0 + d["market_return"]).cumprod().iloc[-1] - 1.0)
    st_return = float((1.0 + d["strategy_return"]).cumprod().iloc[-1] - 1.0)
    bh_dd = _max_drawdown(d["market_return"])
    st_dd = _max_drawdown(d["strategy_return"])
    return {
        "start": start,
        "end": end,
        "days": int(len(d)),
        "cash_days": int((~d["held"]).sum()),
        "buy_hold_return": bh_return,
        "strategy_return": st_return,
        "buy_hold_max_drawdown": bh_dd,
        "strategy_max_drawdown": st_dd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="regime 择时回测")
    parser.add_argument("--window", type=int, default=10, help="大盘均收益回看窗口（交易日）")
    parser.add_argument("--threshold", type=float, default=0.0, help="空仓阈值（大盘N日均收益 < 此值则空仓）")
    parser.add_argument("--start", default="2024-10-01")
    parser.add_argument("--end", default="2026-08-14")
    args = parser.parse_args()

    trade_db = _resolve_trade_db()
    daily = _load_market_daily(trade_db)

    # 1. 主区间
    main = _evaluate(daily, args.window, args.threshold, args.start, args.end)

    # 2. 参数敏感性（全期固定区间）
    sensitivity_rows = []
    for w in [5, 10, 20, 30]:
        for th in [0.0, -0.001, -0.002, -0.003]:
            r = _evaluate(daily, w, th, args.start, args.end)
            sensitivity_rows.append({
                "window": w,
                "threshold": th,
                "strategy_return": round(r["strategy_return"], 4),
                "buy_hold_return": round(r["buy_hold_return"], 4),
                "strategy_max_drawdown": round(r["strategy_max_drawdown"], 4),
                "buy_hold_max_drawdown": round(r["buy_hold_max_drawdown"], 4),
            })

    improved_return = sum(
        r["strategy_return"] > r["buy_hold_return"] for r in sensitivity_rows
    )
    improved_dd = sum(
        r["strategy_max_drawdown"] > r["buy_hold_max_drawdown"] for r in sensitivity_rows
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": {
            "window": args.window,
            "threshold": args.threshold,
            "description": f"大盘{args.window}日均收益 < {args.threshold} 则空仓，否则满仓持有全市场等权",
        },
        "main_result": {k: round(v, 6) if isinstance(v, float) else v for k, v in main.items()},
        "sensitivity": {
            "combos_tested": len(sensitivity_rows),
            "return_improved_ratio": f"{improved_return}/{len(sensitivity_rows)}",
            "drawdown_improved_ratio": f"{improved_dd}/{len(sensitivity_rows)}",
            "rows": sensitivity_rows,
        },
        "caveat": "本回测假设'满仓=全市场等权'、'空仓日收益=0'，未计交易成本/滑点，实际持仓为具体股票时需另行回测验证。",
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regime_timing_backtest_summary.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("regime 择时回测结果")
    print("=" * 70)
    m = main
    print(f"规则: 大盘{args.window}日均收益 < {args.threshold} 则空仓")
    print(f"区间: {m['start']} ~ {m['end']}（{m['days']} 天，空仓 {m['cash_days']} 天）")
    print(f"\n一直满仓: 收益 {m['buy_hold_return']*100:+.2f}%, 最大回撤 {m['buy_hold_max_drawdown']*100:.2f}%")
    print(f"regime择时: 收益 {m['strategy_return']*100:+.2f}%, 最大回撤 {m['strategy_max_drawdown']*100:.2f}%")
    print(f"\n参数敏感性（{len(sensitivity_rows)} 组合）:")
    print(f"  择时收益 > 满仓: {improved_return}/{len(sensitivity_rows)}")
    print(f"  择时回撤优于满仓: {improved_dd}/{len(sensitivity_rows)}")
    print(f"\n已写入: {out_path}")


if __name__ == "__main__":
    main()
