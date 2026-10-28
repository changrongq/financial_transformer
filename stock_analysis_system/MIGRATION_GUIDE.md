# 分层市值系统迁移与异机训练说明

本文档用于将当前分层市值分析系统迁移到另一台电脑继续训练，适用于“本机无 GPU、目标机器有 GPU”的场景。

## 1. 迁移目标

当前本机已经完成：
- 市值数据抓取
- 股票独立分割
- 分层特征提取

因此迁移到新电脑时，**不建议重新抓取和重新提特征**，最省时间的做法是直接拷贝已经准备好的数据库和项目目录，然后从训练阶段继续。

---

## 2. 必须拷贝的文件和目录

请把以下内容完整复制到新电脑：

### 2.1 项目目录
- [stock_analysis_system](stock_analysis_system)

### 2.2 原始数据库
- [stock_history.db](../stock_history.db)
- [market_cap_full.db](../market_cap_full.db)

### 2.3 已生成的数据文件
- [data/hierarchical_features.db](data/hierarchical_features.db)
- [data/stock_splits.csv](data/stock_splits.csv)

### 2.4 可选：如果希望保留已有结果，也一起带走
- [models](models)
- [results](results)

---

## 3. 新电脑推荐目录结构

建议在新电脑上保持和当前一致的相对结构：

- 某个工作目录/
  - stock_history.db
  - market_cap_full.db
  - stock_analysis_system/

原因：
当前配置文件 [config.yaml](config.yaml) 使用的是相对路径：
- `trade_db: "../stock_history.db"`
- `market_cap_db: "../market_cap_full.db"`
- `features_db: "data/hierarchical_features.db"`

只要目录结构不变，迁移后通常**不需要改路径**。

---

## 4. 新电脑环境准备

### 4.1 安装 Python
建议安装：
- Python 3.9 及以上

### 4.2 安装依赖
进入 [stock_analysis_system](stock_analysis_system) 目录后安装依赖：
- `pip install -r requirements.txt`

如果目标机器有 NVIDIA GPU，建议安装对应 CUDA 版本的 PyTorch。

### 4.3 验证 PyTorch GPU
可在新电脑上验证：
- `python -c "import torch; print(torch.cuda.is_available())"`

如果输出为 `True`，说明可用 GPU。

---

## 5. 配置文件调整

配置文件位置：
- [config.yaml](config.yaml)

### 5.1 把训练设备改成 GPU
把：
- `device: "cpu"`

改成：
- `device: "cuda"`

对应位置在 [config.yaml](config.yaml) 的 `training.device`。

### 5.2 Tushare Token 建议
如果后续还要重新抓取市值，建议不要在配置里长期硬编码 token。

更推荐改为：
- `token: "${TUSHARE_TOKEN}"`

然后在新电脑中设置环境变量 `TUSHARE_TOKEN`。

如果你只是继续训练，而不再抓取数据，这一步不是必须的。

---

## 6. 到新电脑后不需要重新做的步骤

如果以下文件已经复制成功：
- [../stock_history.db](../stock_history.db)
- [../market_cap_full.db](../market_cap_full.db)
- [data/hierarchical_features.db](data/hierarchical_features.db)
- [data/stock_splits.csv](data/stock_splits.csv)

那么通常**不需要再执行**：
- `scripts/fetch_market_cap.py`
- `scripts/split_data.py`
- `scripts/extract_features.py`

---

## 7. 新电脑上建议执行顺序

### 第一步：训练四层模型
按顺序运行：
- [scripts/train_micro_model.py](scripts/train_micro_model.py)
- [scripts/train_small_model.py](scripts/train_small_model.py)
- [scripts/train_mid_model.py](scripts/train_mid_model.py)
- [scripts/train_large_model.py](scripts/train_large_model.py)

训练完成后，模型文件会输出到：
- [models](models)

### 第二步：生成分层集成结果
运行：
- [scripts/ensemble_model.py](scripts/ensemble_model.py)

结果会输出到：
- [results](results)

### 第三步：运行生产推理
运行：
- [scripts/run_inference.py](scripts/run_inference.py)

结果文件：
- [results/latest_picks.csv](results/latest_picks.csv)

### 第四步：运行历史回测
运行：
- [scripts/run_backtest.py](scripts/run_backtest.py)

结果文件：
- [results/backtest_trades.csv](results/backtest_trades.csv)
- [results/backtest_equity.csv](results/backtest_equity.csv)
- [results/backtest_summary.json](results/backtest_summary.json)

---

## 8. 如果要启用接口服务

可运行：
- [scripts/serve_api.py](scripts/serve_api.py)

接口包括：
- `GET /health`
- `POST /predict/latest`
- `POST /backtest/run`

Docker 入口文件：
- [Dockerfile](Dockerfile)

---

## 9. 迁移前检查清单

在旧电脑上，迁移前请确认：

- [ ] [stock_analysis_system](stock_analysis_system) 已完整复制
- [ ] [../stock_history.db](../stock_history.db) 已复制
- [ ] [../market_cap_full.db](../market_cap_full.db) 已复制
- [ ] [data/hierarchical_features.db](data/hierarchical_features.db) 已复制
- [ ] [data/stock_splits.csv](data/stock_splits.csv) 已复制
- [ ] 新电脑已安装 Python
- [ ] 新电脑已安装依赖
- [ ] 新电脑已将 [config.yaml](config.yaml) 中 `device` 改为 `cuda`（若有 GPU）

---

## 10. 常见问题

### 10.1 新电脑提示找不到数据库
优先检查目录结构是否保持为：
- 上一级目录放 `stock_history.db`
- 上一级目录放 `market_cap_full.db`
- 当前目录放 `stock_analysis_system`

如果结构不同，就需要修改 [config.yaml](config.yaml) 中的数据库路径。

### 10.2 新电脑可以启动，但训练没走 GPU
检查：
1. `torch.cuda.is_available()` 是否为 `True`
2. [config.yaml](config.yaml) 中 `training.device` 是否为 `cuda`
3. 安装的 PyTorch 是否是 CUDA 版本

### 10.3 只训练了一层模型，能不能先推理
可以。
系统会自动加载已经存在的模型，缺失的层只会告警，不会中断。

### 10.4 是否需要重新抓市值和重新提特征
一般不需要。
只要现有数据库和特征库完整复制过去，就可以直接训练。

---

## 11. 最短迁移路线

如果只追求最快开始训练，可以按下面做：

1. 复制：
   - [stock_analysis_system](stock_analysis_system)
   - [../stock_history.db](../stock_history.db)
   - [../market_cap_full.db](../market_cap_full.db)
2. 新电脑安装依赖
3. 修改 [config.yaml](config.yaml) 中 `training.device: "cuda"`
4. 直接运行四个训练脚本
5. 再运行集成、推理、回测

---

## 12. 当前建议

你现在最适合的方式是：
- 把整个项目和数据库拷贝到有 GPU 的电脑
- 只在新电脑上做训练和后续推理
- 不在新电脑上重复抓取与提特征

这样最省时间，也最稳定。
