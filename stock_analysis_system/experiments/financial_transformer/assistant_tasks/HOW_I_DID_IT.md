# 我是怎么做"Filter 线 3-6 月回测对比"的 — 完整操作记录

> 这份文档记录了我（Coder）完成这个任务的**每一步实际操作**。
> 如果你是一个执行型 AI 助手，这份文档就是你的操作手册。

---

## 我用的工具

在工作中，我可以使用以下工具来完成操作：

| 工具 | 作用 | 例子 |
|---|---|---|
| `run_in_terminal` | 在终端执行命令 | 运行 Python 脚本、检查文件是否存在 |
| `create_file` | 创建新文件并写入内容 | 写一个新的 Python 脚本 |
| `read_file` | 读取已有文件的内容 | 查看配置文件、检查数据 |
| `replace_string_in_file` | 修改已有文件中的某段代码 | 修复脚本中的 bug |
| `file_search` | 搜索文件名 | 查找某个目录下有哪些文件 |
| `grep_search` | 在文件内容中搜索文本 | 查找某段代码在哪个文件里 |
| `list_dir` | 列出目录内容 | 查看某个文件夹里有什么 |

**核心原则：每一步都产生可验证的产出——要么是一个文件，要么是一段 terminal 输出。**

---

## 完整操作流程

### 第 1 步：确认环境可用

**我做了什么：**
- 用 `list_dir` 查看了助手工作目录
- 用 `run_in_terminal` 执行 Python 检查版本
- 确认 checkpoint 文件存在

**实际命令：**
```powershell
# 检查 Filter 的 checkpoint 是否存在
# 我在终端执行了：
dir experiments\financial_transformer\results\formal_filtered_v5_e80_bsstop_seed33\artifacts\
```

**产出：** 确认 `financial_transformer.pt` 存在 → 可以继续。

---

### 第 2 步：检查现有数据范围

**我做了什么：**
- 用 `run_in_terminal` 执行 Python 读取 CSV，查看日期范围

```powershell
.venv\Scripts\python.exe -c "
import pandas as pd
df = pd.read_csv('experiments/financial_transformer/results/formal_filtered_v5_e80_bsstop_seed33/test_predictions.csv',
                  dtype={'stock_code': str})
df['date'] = pd.to_datetime(df['date'])
print('date range:', df['date'].min(), '~', df['date'].max())
print('rows:', len(df))
"
```

**产出：** 发现 test_predictions 只到 3 月 4 日 → 需要补齐 3 月 5 日到 6 月 4 日的数据。

---

### 第 3 步：补齐缺失的预测数据

**我做了什么：**
1. 先用 `read_file` 读取 `predict.py` 的源码，理解 `predict_latest()` 的逻辑
2. 用 `read_file` 读取 `data.py` 中 `build_feature_frame()` 的逻辑
3. 用 **`create_file`** 创建一个新脚本 `backfill_filter_predictions.py`

**脚本的核心逻辑：**
```python
# 1. 加载 checkpoint 模型
checkpoint = torch.load("financial_transformer.pt")

# 2. 构建全量特征（build_feature_frame 返回所有历史数据）
feature_df = build_feature_frame(config)

# 3. 对 3 月 5 日到 6 月 4 日的每个交易日：
for target_date in target_dates:
    # 只取截至 target_date 的特征
    cut_df = feature_df[feature_df["date"] <= target_date]

    # 对每只股票：取最近 seq_len 天的窗口，跑预测
    for stock_code, group in cut_df.groupby("stock_code"):
        latest_window = group.tail(seq_len)[feature_cols].to_numpy()
        outputs = model(latest_window)
        # 保存预测结果...
```

4. 用 **`run_in_terminal`** 执行这个脚本：
```powershell
.venv\Scripts\python.exe experiments\financial_transformer\assistant_tasks\backfill_filter_predictions.py
```

**产出：** `backfill_predictions.csv`（4158 行，77 个交易日，从 1 月 5 日到 6 月 4 日完整覆盖）

---

### 第 4 步：确认补齐的数据正确

**我做了什么：**
用 `run_in_terminal` 快速检查新文件的日期范围：

```powershell
.venv\Scripts\python.exe -c "
import pandas as pd
df = pd.read_csv('.../backfill_predictions.csv')
df['date'] = pd.to_datetime(df['date'])
print(df['date'].min(), '~', df['date'].max())
print('unique dates:', df['date'].nunique())
"
```

**产出：** 确认 77 个交易日覆盖完整。

---

### 第 5 步：编写回测脚本

**我做了什么：**

1. 用 `read_file` 看了 `backtest.py` 中的回测逻辑
2. 用 **`create_file`** 创建 `run_filter_backtest_direct.py`

**脚本核心逻辑（手工回测）：**

```
1. 读取 backfill_predictions.csv
2. 对每个窗口（full / old / recent）：
   a. 日期过滤
   b. 阈值过滤（short_up>0.254, take_profit>0.572, stop_loss<0.50）
   c. top_k=2 每日选股
   d. 遍历每个交易日：
      - 检查已有持仓是否需要退出（止盈/止损/到期）
      - 检查是否有新信号入场（下一交易日开盘买入）
      - 每只股票仓位 8%-20%，按总资产的 15%×signal_score 分配
      - 最多同时持有 2 只（因为 top_k=2）
   e. 计算最终指标（收益、Sharpe、回撤、胜率、PF）
```

3. 用 **`run_in_terminal`** 执行：
```powershell
.venv\Scripts\python.exe experiments\financial_transformer\assistant_tasks\run_filter_backtest_direct.py
```

---

### 第 6 步：修正 bug

第一次运行时，我的回测逻辑有 bug——没有正确处理多持仓情况。我用 `replace_string_in_file` 修复了核心回测循环。

---

### 第 7 步：输出结果

**最终产出：**

| 文件 | 作用 |
|---|---|
| `backfill_predictions.py` | 预测补齐脚本 |
| `backfill_predictions.csv` | 完整预测序列（77 天） |
| `run_filter_backtest_direct.py` | 回测脚本 |
| terminal 输出 | 对比表格 |

---

## 总结：执行型助手的工作模式

```
理解任务 → 检查现有数据 → 补齐缺失数据 → 验证数据正确 → 编写分析脚本 → 执行 → 修正 → 输出结果
   │            │                │              │              │           │      │        │
   │         read_file      create_file    run_in_terminal  create_file  run_in_  replace_ 打印
   │         run_in_terminal                 (小验证)                    terminal  string
   │                                                                              _in_file
   │
   └── 每一个子任务都用工具完成，不凭空想象
```

**关键原则：**
1. **永远先读再写** — 先 `read_file` 理解现有代码，再 `create_file` 写新代码
2. **每一步都验证** — 写完文件后立即 `run_in_terminal` 执行，看有没有报错
3. **报错了就修** — 用 `replace_string_in_file` 改 bug，再跑
4. **不编造数字** — 所有指标都来自脚本的真实输出
5. **产出留在文件里** — 脚本、数据、结果都有文件留存，可复现
