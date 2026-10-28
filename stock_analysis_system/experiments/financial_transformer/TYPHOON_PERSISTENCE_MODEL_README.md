# 台风胚胎持续性预测模型 使用说明

> 给钳星的使用手册 — 2026-07-31

## 概览

这个模型在现有台风胚胎规则引擎之上加了一层机器学习（HistGradientBoosting），输出三个概率分数：

| 输出 | 含义 | 训练数据 |
|---|---|---|
| `persistence_prob` | 次日涨幅 > 3% 的概率 | 历史胚胎CSV |
| `risk_prob` | 次日跌幅 > 3% 的概率（风险预警） | 历史胚胎CSV |
| `survival_prob` | 3天内O值从胚胎升级到台风的概率 | 数据尚不足（暂不可靠） |

## 文件位置

| 内容 | 路径 |
|---|---|
| 训练脚本 | `experiments/financial_transformer/typhoon_persistence_model.py` |
| 回填脚本 | `experiments/financial_transformer/backfill_embryo_history.py` |
| 模型文件 | `experiments/financial_transformer/results/typhoon_persistence_model/*.joblib` |
| 训练数据 | `experiments/financial_transformer/results/typhoon_tracker/typhoon_embryo_tracker_*.csv` |

## 使用方式

### 重新训练

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/financial_transformer/typhoon_persistence_model.py --train
```

### 在最新胚胎上预测

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/financial_transformer/typhoon_persistence_model.py --train --predict
```

输出文件：`results/typhoon_persistence_model/persistence_predictions_YYYY-MM-DD.csv`

### 回填历史胚胎数据（新交易日增加后）

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/financial_transformer/backfill_embryo_history.py
```

## 当前模型表现 (2026-07-31)

训练数据：43个交易日，653行

| 模型 | AUC | 阳性率 |
|---|---|---|
| persistence_next_day | 0.597 | 16.9% |
| risk_next_day | 0.596 | 2.1% |
| survival_3d | 0.500 | 0.3% |

## 关键提醒

- **survival_3d 暂不可靠** — 标签太稀疏（只有2个正例），需要3-4周数据积累后才会有效。
- persistence 和 risk 已经可用，但 AUC 约 0.6 属于中等水平——会随着数据增加逐步提升。
- 建议每天收盘后跑 `backfill_embryo_history.py` 补充新数据，每周跑一次 `--train` 更新模型。

## Paper Trade API 快捷操作

新增了简洁接口，返回 JSON，适合脚本调用：

```bash
# 查询持仓（自动刷新腾讯行情价格）
curl -X POST http://localhost:8000/positions

# 快捷交易：自动判断买卖
curl -X POST http://localhost:8000/trade \
  -H "Content-Type: application/json" \
  -d '{"stock_code":"000566","quantity":500}'
```

- 无持仓 → 自动买入
- 有持仓 → 自动卖出
