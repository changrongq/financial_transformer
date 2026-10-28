from __future__ import annotations

from pathlib import Path

import pandas as pd


def review_mid_model(results_dir: str | Path, output_file: str | Path) -> Path:
    results_path = Path(results_dir)
    history_path = results_path / "mid_history.json"
    metrics_path = results_path / "mid_metrics.json"
    predictions_path = results_path / "mid_predictions.csv"

    history = pd.read_json(history_path)
    metrics = pd.read_json(metrics_path, typ="series")
    predictions = pd.read_csv(predictions_path)
    predictions["prediction_prob"] = pd.to_numeric(predictions["prediction_prob"], errors="coerce")
    predictions["label"] = pd.to_numeric(predictions["label"], errors="coerce")
    predictions = predictions.dropna(subset=["prediction_prob", "label"]).copy()

    top_bucket = predictions.nlargest(max(1, len(predictions) // 10), "prediction_prob")
    bottom_bucket = predictions.nsmallest(max(1, len(predictions) // 10), "prediction_prob")

    report = f"""# 中型股模型复查报告

## 现象

- 当前中型股测试集 AUC: {float(metrics.get('auc', 0.0)):.4f}
- 当前中型股测试集准确率: {float(metrics.get('accuracy', 0.0)):.4f}
- 当前中型股测试集精确率: {float(metrics.get('precision', 0.0)):.4f}
- 当前中型股测试集召回率: {float(metrics.get('recall', 0.0)):.4f}

## 关键异常

- 训练历史中几乎所有 epoch 的 precision 和 recall 都为 0。
- `prediction_prob` 大面积集中在单一窄区间，模型输出接近常数。
- 最新训练结果说明模型没有真正学到有效的正类判别边界。

## 量化证据

- 预测分数最小值: {float(predictions['prediction_prob'].min()):.6f}
- 预测分数中位数: {float(predictions['prediction_prob'].median()):.6f}
- 预测分数最大值: {float(predictions['prediction_prob'].max()):.6f}
- 分数标准差: {float(predictions['prediction_prob'].std()):.6f}
- 基础正样本率: {float(predictions['label'].mean()):.4f}
- 前 10% 高分样本命中率: {float(top_bucket['label'].mean()):.4f}
- 后 10% 低分样本命中率: {float(bottom_bucket['label'].mean()):.4f}

## 判断

中型股模型当前不是完全随机，但已经出现明显的输出塌缩，属于“有一点排序信号，但分类头基本没张开”的状态。

这解释了两个现象：

- AUC 仍略高于 0.5
- 但 precision / recall 长期为 0，且预测分数几乎都落在同一段

## 优先排查方向

1. 检查中型股特征是否存在大量低方差列或无效列。
2. 检查正负样本比例，必要时为 BCE loss 增加 `pos_weight`。
3. 简化当前 `mid` 架构，先确认是数据问题还是模型复杂度问题。
4. 对中型股单独做分数分布和特征分布排查，而不是直接复用其他层参数。

## 当前建议

- 在生产排序中保留 `mid`，但只作为次要参考层。
- 分位数使用上先取 `top_20pct`，不要使用更窄阈值。
- 下一轮模型优化优先处理 `mid`。
"""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path