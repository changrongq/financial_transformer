# 任务：执行主线 vs Filter 试验线对比回测

## 你的角色

你是一个**执行型助手**。你只需要按照下面的指令逐条执行，然后把结果告诉我。不需要你做任何推理或判断。

---

## 任务 1：检查环境

请依次执行以下 4 条命令，把每条命令的输出完整地回复给我：

**命令 1：**
```
python --version
```

**命令 2：**
```
pwd
```

**命令 3：**
```
cd d:\qcr\Stock\stock_analysis_system && dir experiments\financial_transformer\assistant_tasks
```

**命令 4：**
```
cd d:\qcr\Stock\stock_analysis_system && .venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('experiments/financial_transformer/results/formal_filtered_v5_e80_bsstop_seed33/latest_predictions.csv', dtype={'stock_code': str}); df['date'] = pd.to_datetime(df['date']); print('date range:', df['date'].min(), '~', df['date'].max()); print('rows:', len(df)); print(df[['stock_code','signal_score','pred_short_up','pred_take_profit','pred_stop_loss']].head(5).to_string(index=False))"
```

---

请现在执行任务 1 的 4 条命令，把输出回复给我。
