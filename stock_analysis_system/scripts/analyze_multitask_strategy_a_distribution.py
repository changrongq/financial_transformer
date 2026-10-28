from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _to_records(df: pd.DataFrame, sort_columns: list[str]) -> list[dict]:
    if df.empty:
        return []
    return df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze distribution of multitask Strategy A backtest trades")
    parser.add_argument("--trades", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    summary_path = Path(args.summary)
    output_tag = f"_{args.output_tag}" if args.output_tag else ""

    trades = pd.read_csv(trades_path, dtype={"stock_code": str, "layer": str})
    trades["stock_code"] = trades["stock_code"].astype(str).str.zfill(6)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    for column in ["signal_date", "buy_date", "sell_date"]:
        if column in trades.columns:
            trades[column] = pd.to_datetime(trades[column], errors="coerce")

    trades["buy_month"] = trades["buy_date"].dt.to_period("M").astype(str)
    trades["signal_month"] = trades["signal_date"].dt.to_period("M").astype(str) if "signal_date" in trades.columns else None

    layer_summary = (
        trades.groupby("layer", dropna=False)
        .agg(
            trade_count=("stock_code", "count"),
            total_profit=("profit", "sum"),
            avg_profit=("profit", "mean"),
            win_rate=("profit", lambda values: float((values > 0).mean())),
            avg_hold_days=("hold_days", "mean"),
        )
        .reset_index()
    ) if "layer" in trades.columns else pd.DataFrame()

    buy_month_summary = (
        trades.groupby("buy_month", dropna=False)
        .agg(
            trade_count=("stock_code", "count"),
            total_profit=("profit", "sum"),
            avg_profit=("profit", "mean"),
            win_rate=("profit", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )

    exit_summary = (
        trades.groupby("exit_reason", dropna=False)
        .agg(
            trade_count=("stock_code", "count"),
            total_profit=("profit", "sum"),
            avg_profit=("profit", "mean"),
        )
        .reset_index()
    )

    stock_summary = (
        trades.groupby(["stock_code", "layer"], dropna=False)
        .agg(
            trade_count=("stock_code", "count"),
            total_profit=("profit", "sum"),
            avg_profit=("profit", "mean"),
            win_rate=("profit", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    ) if "layer" in trades.columns else pd.DataFrame()

    payload = {
        "summary": summary,
        "layer_summary": _to_records(layer_summary, ["total_profit", "trade_count"]),
        "buy_month_summary": _to_records(buy_month_summary, ["total_profit", "trade_count"]),
        "exit_summary": _to_records(exit_summary, ["total_profit", "trade_count"]),
        "top_stocks": _to_records(stock_summary.head(10), ["total_profit", "trade_count"]) if not stock_summary.empty else [],
        "worst_stocks": stock_summary.sort_values(["total_profit", "trade_count"], ascending=[True, False]).head(10).to_dict(orient="records") if not stock_summary.empty else [],
    }

    results_dir = ROOT / "results"
    json_path = results_dir / f"multitask_strategy_a_distribution{output_tag}.json"
    markdown_path = results_dir / f"multitask_strategy_a_distribution{output_tag}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# 多任务方案 A 交易分布分析",
        "",
        f"- 总收益: {summary['total_return']:.4%}",
        f"- 最大回撤: {summary['max_drawdown']:.4%}",
        f"- 交易数: {summary['trade_count']}",
        f"- 胜率: {summary['win_rate']:.2%}",
        "",
        "## 按市值层",
    ]
    if layer_summary.empty:
        lines.append("- 无 layer 字段，无法分层统计")
    else:
        for row in layer_summary.sort_values(["total_profit", "trade_count"], ascending=[False, False]).to_dict(orient="records"):
            lines.append(
                f"- {row['layer']}: 交易 {int(row['trade_count'])} 笔, 总利润 {row['total_profit']:.2f}, 胜率 {row['win_rate']:.2%}, 平均持有 {row['avg_hold_days']:.2f} 天"
            )

    lines.append("")
    lines.append("## 按买入月份")
    for row in buy_month_summary.sort_values(["buy_month"], ascending=[True]).to_dict(orient="records"):
        lines.append(
            f"- {row['buy_month']}: 交易 {int(row['trade_count'])} 笔, 总利润 {row['total_profit']:.2f}, 胜率 {row['win_rate']:.2%}"
        )

    lines.append("")
    lines.append("## 按退出原因")
    for row in exit_summary.sort_values(["total_profit", "trade_count"], ascending=[False, False]).to_dict(orient="records"):
        lines.append(
            f"- {row['exit_reason']}: 交易 {int(row['trade_count'])} 笔, 总利润 {row['total_profit']:.2f}, 平均单笔 {row['avg_profit']:.2f}"
        )

    lines.append("")
    lines.append("## 贡献最高股票")
    if stock_summary.empty:
        lines.append("- 无 layer 字段，无法输出股票贡献")
    else:
        for row in stock_summary.sort_values(["total_profit", "trade_count"], ascending=[False, False]).head(10).to_dict(orient="records"):
            lines.append(
                f"- {row['stock_code']} ({row['layer']}): 交易 {int(row['trade_count'])} 笔, 总利润 {row['total_profit']:.2f}, 胜率 {row['win_rate']:.2%}"
            )

        lines.append("")
        lines.append("## 拖累最大股票")
        for row in stock_summary.sort_values(["total_profit", "trade_count"], ascending=[True, False]).head(10).to_dict(orient="records"):
            lines.append(
                f"- {row['stock_code']} ({row['layer']}): 交易 {int(row['trade_count'])} 笔, 总利润 {row['total_profit']:.2f}, 胜率 {row['win_rate']:.2%}"
            )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()