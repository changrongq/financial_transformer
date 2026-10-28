# 台风持续性预测模型 v1.0 给钳星的手册

> 2026-08-08 | 基于睫的需求文档实现

## 核心变化 (v0.3 → v1.0)

| 维度 | v0.3 | v1.0 |
|---|---|---|
| 特征 | 14 个（动量+herding+角色） | 23 个（+ 资金结构 8 个 + 成交量突变 1 个） |
| risk AUC | 0.596 | **0.823** |
| 信号评级 | 无 | ⭐ 三等分级 |
| 风控提示 | 无 | 自动标注资金过热/机构反手/散户割肉 |
| 报告定位 | 概率输出 | 观察池信号（非买入指令） |

## 新增资金结构因子

每天自动从 `stock_moneyflow` 表 join 以下特征（无需手动处理）：

| 因子 | 列名 | 来自睫的需求 |
|---|---|---|
| 特大单净流入 | `elg_net_amount` | 8.1 资金结构信号 |
| 特大单流入流出比 | `elg_in_out_ratio` | 新北洋案例：<0.6=机构反手 |
| 大单净流入 | `lg_net_amount` | 与特大单分歧判断 |
| 大单流入流出比 | `lg_in_out_ratio` | 游资 vs 机构识别 |
| 中单净流入 | `md_net_amount` | 辅助判断老筹码出货 |
| 小单净流入 | `sm_net_amount` | 散户方向 |
| 小单连续流出天数 | `sm_consecutive_out` | ≥2天=散户持续割肉 |
| 大小单分歧 | `lg_elg_ratio` | 机构 vs 游资分歧度 |

## 风控自动提示

报告里会自动标注以下风险：

| 触发条件 | 标注 |
|---|---|
| herding > 15 | 资金过热 |
| 特大单流入流出比 < 0.6 | 机构反手 |
| 小单连续 ≥ 2 日净流出 | 散户割肉 |

## 报告信号等级

| 条件 | 等级 | 含义 |
|---|---|---|
| persistence > 85% + risk < 5% | `=== 强信号` | 优先进入观察池 |
| persistence 70-85% | `== 中等信号` | 进入观察池，等待更多确认 |
| persistence < 70% | `= 弱信号` | 仅跟踪，不进入观察池 |

## 运行命令

```powershell
cd D:\qcr\Stock\stock_analysis_system

# 训练 + 预测 一次性完成
.venv/Scripts/python.exe experiments/financial_transformer/typhoon_persistence_model.py --train --predict

# 生成完整总报告（含台风+胚胎+持续性预测）
.venv/Scripts/python.exe experiments/financial_transformer/signal_report.py --signal-profile all_split --skip-refresh-data --skip-predict-latest
```

## 已知限制

1. **归因暂时简化** — moneyflow 数据在预测阶段尚未并入 CSV（训练时有，预测时待修）。因此 Top3 因子归因列暂时不显示。下周完善。
2. **survival_3d 仍不可靠** — 标签稀疏（3 日内 O 值升级事件太少），需要更多交易日积累。
3. **基本面因子** — 业绩预增、行业分类等尚未接入，睫的需求中有提及但尚未实现。需要 tushare forecast 接口或本地数据源。
4. **波动率因子** — JumpModel 跳跃强度尚未加入，优先级较低。

## 建议的每日工作流

```
盘后 1: 回填历史 → .venv/Scripts/python.exe experiments/financial_transformer/backfill_embryo_history.py
盘后 2: 训练+预测 → .venv/Scripts/python.exe experiments/financial_transformer/typhoon_persistence_model.py --train --predict
盘后 3: 生成报告 → .venv/Scripts/python.exe experiments/financial_transformer/signal_report.py --signal-profile all_split --skip-refresh-data --skip-predict-latest
```
