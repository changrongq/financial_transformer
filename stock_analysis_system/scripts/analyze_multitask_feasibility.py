from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.config import load_config
from hierarchical_system.multitask_data import extract_multitask_features, summarize_multitask_targets


def _table_to_markdown(df) -> str:
    headers = list(df.columns)
    rows = [headers]
    for _, row in df.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(headers))]

    def render(row_values):
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row_values)) + " |"

    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    rendered = [render(rows[0]), separator]
    rendered.extend(render(row) for row in rows[1:])
    return "\n".join(rendered)


def _render_markdown(summary_df, config) -> str:
    overall = summary_df.groupby("split", as_index=False).agg(
        samples=("samples", "sum"),
        up5_rate=("up5_rate", "mean"),
        take_profit_45d_rate=("take_profit_45d_rate", "mean"),
        stop_loss_45d_rate=("stop_loss_45d_rate", "mean"),
        max_gain_45d_mean=("max_gain_45d_mean", "mean"),
        max_drawdown_45d_mean=("max_drawdown_45d_mean", "mean"),
    )
    lines = [
        "# 多任务可行性报告",
        "",
        "## 标签设定",
        f"- 短期信号: {config.multitask.short_horizon} 天后涨幅 >= {config.multitask.short_threshold:.0%}",
        f"- 止盈事件: {config.multitask.hold_days} 天内任意一天涨幅 >= {config.multitask.take_profit_threshold:.0%}",
        f"- 止损事件: {config.multitask.hold_days} 天内任意一天跌幅 <= {config.multitask.stop_loss_threshold:.0%}",
        "",
        "## 分层汇总",
        "",
        _table_to_markdown(summary_df),
        "",
        "## 按 split 汇总",
        "",
        _table_to_markdown(overall),
        "",
        "## 初步结论",
        "- 45 天止盈与止损任务都不是稀有标签，可以直接纳入多任务分类头。",
        "- 不同市值层的 45 天标签分布存在明显差异，说明保留分层信息仍然有意义。",
        "- micro 和 small 层的止损事件更频繁，回测阶段应对风险头设置更严格阈值。",
        "- 当前特征库缺少大盘指数、行业指数和情绪因子，多任务版先复用现有个股特征，再补市场上下文特征。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    feature_df = extract_multitask_features(config)
    summary_df = summarize_multitask_targets(feature_df)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / "multitask_feasibility_summary.csv", index=False, encoding="utf-8-sig")

    overall = {
        "samples": int(len(feature_df)),
        "up5_rate": float(feature_df["target_up_5d"].mean()),
        "take_profit_45d_rate": float(feature_df["target_take_profit_45d"].mean()),
        "stop_loss_45d_rate": float(feature_df["target_stop_loss_45d"].mean()),
        "max_gain_45d_mean": float(feature_df["max_gain_45d"].mean()),
        "max_drawdown_45d_mean": float(feature_df["max_drawdown_45d"].mean()),
    }
    with (results_dir / "multitask_feasibility_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, ensure_ascii=False, indent=2)
    with (results_dir / "multitask_feasibility_report.md").open("w", encoding="utf-8") as handle:
        handle.write(_render_markdown(summary_df, config))

    print(summary_df.to_string(index=False))
    print(json.dumps(overall, ensure_ascii=False, indent=2))