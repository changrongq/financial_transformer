# 股票台风追踪系统 MVP

本文档对应 `typhoon_tracker.py`，实现设计文档《台风追踪系统（Typhoon Tracker）》的 Phase 1/Phase 2 最小可行版本。

## 运行命令

在项目根目录运行：

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/financial_transformer/typhoon_tracker.py --top-n 30
```

指定交易日：

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/financial_transformer/typhoon_tracker.py --end-date 2026-07-03 --top-n 30
```

## 输出文件

默认输出目录：`experiments/financial_transformer/results/typhoon_tracker/`

- `typhoon_tracker_<date>.md`：每日台风追踪报告
- `typhoon_tracker_<date>.csv`：可筛选明细
- `typhoon_tracker_<date>_summary.json`：结构化摘要

## 当前口径

### 编号热带低压

同时满足：

1. I1：近 5 日日均成交量 > 20 日日均成交量 × 1.5
2. I2：收盘价在 MA5 和 MA10 之上
3. I3：近 5 日日均振幅 > 5%，且高于 20 日日均振幅
4. I4：近 5 日主力净流入 > 0，主力按大单+特大单金额近似
5. I5：剔除泛概念后，同概念至少 2 只股票同时满足 I1-I4

### 组织强度 O

按系统 A 计算：

`O = 0.25*A + 0.20*C + 0.20*R + 0.20*S + 0.15*E`

- A：成交量加速度归一化
- C：收盘价/最高价，表示收敛度
- R：概念共振数量归一化
- S：连续满足编号条件的天数归一化
- E：涨停效率日频代理

### 热力学评分 S

按系统 B 的离散近似：

`S_t = (1 - alpha) * S_{t-1} + alpha * Q_t`

其中 `Q_t = L * H * sigma * demand`，作为对照实验指标。

## 当前首版验证

首版已在最新交易日 `2026-07-03` 跑通：

- 扫描股票数：5464
- 编号热带低压：105
- 报告跟踪数：30
- Top1：603717 天域生物，O=0.840，评级=台风成形
- 协同试验候选：000566 海南海药，O=0.825，角色=herd，稳定天数=5

## 注意

该脚本只输出跟踪和研究报告，不产生自动交易指令。真正协同试验仍需人工确认：角色、Filter 治理结论、市场环境和重大新闻。