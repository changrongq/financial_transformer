# 分层市值分析系统

这是一个基于真实市值的股票分层训练系统骨架，目标是避免股票重叠带来的数据泄露，并为不同市值层使用不同模型。

## 目录
- [config.yaml](config.yaml): 系统配置
- [scripts/fetch_market_cap.py](scripts/fetch_market_cap.py): 抓取真实市值
- [scripts/fetch_moneyflow_history.py](scripts/fetch_moneyflow_history.py): 回填 TuShare 个股日度资金流向
- [scripts/fetch_top_inst_history.py](scripts/fetch_top_inst_history.py): 回填 TuShare 龙虎榜机构席位数据
- [scripts/fetch_margin_history.py](scripts/fetch_margin_history.py): 回填 TuShare 个股融资融券明细
- [scripts/refresh_latest_data.py](scripts/refresh_latest_data.py): 刷新最新行情、资金流、龙虎榜、融资融券，并同步股票静态行业信息
- [scripts/refresh_concepts_with_progress.py](scripts/refresh_concepts_with_progress.py): 断点续刷股票-概念关系，并打印刷新前后进度摘要
- [scripts/split_data.py](scripts/split_data.py): 构建互斥训练/验证/测试股票集合
- [scripts/extract_features.py](scripts/extract_features.py): 提取分层特征并写入 SQLite
- [scripts/extract_multitask_features.py](scripts/extract_multitask_features.py): 提取 45 天多任务特征与标签
- [scripts/analyze_multitask_feasibility.py](scripts/analyze_multitask_feasibility.py): 生成多任务可行性报告
- [scripts/train_multitask_model.py](scripts/train_multitask_model.py): 训练统一版或分层独立版多任务五头模型
- [scripts/run_multitask_inference.py](scripts/run_multitask_inference.py): 使用分层多任务模型生成最新预测
- [scripts/run_multitask_latest_pipeline.py](scripts/run_multitask_latest_pipeline.py): 刷新最新行情/市值后执行多任务最新预测
- [scripts/run_multitask_strategy_a_backtest.py](scripts/run_multitask_strategy_a_backtest.py): 运行多任务方案 A 历史回测
- [scripts/run_multitask_strategy_a_layered_backtest.py](scripts/run_multitask_strategy_a_layered_backtest.py): 运行按市值层分别配置阈值的多任务方案 A 回测
- [scripts/run_multitask_strategy_a_threshold_search.py](scripts/run_multitask_strategy_a_threshold_search.py): 搜索多任务方案 A 的入场阈值组合，支持最小交易数/信号数约束
- [scripts/train_micro_model.py](scripts/train_micro_model.py): 微型股模型训练
- [scripts/train_small_model.py](scripts/train_small_model.py): 小型股模型训练
- [scripts/train_mid_model.py](scripts/train_mid_model.py): 中型股模型训练
- [scripts/train_large_model.py](scripts/train_large_model.py): 大型股模型训练
- [scripts/ensemble_model.py](scripts/ensemble_model.py): 结果加权集成
- [scripts/run_inference.py](scripts/run_inference.py): 生产推理与最新选股
- [scripts/run_backtest.py](scripts/run_backtest.py): 历史回测
- [scripts/serve_api.py](scripts/serve_api.py): FastAPI 服务入口

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 设置 Tushare Token：`set TUSHARE_TOKEN=你的token`
3. 抓市值：`python scripts/fetch_market_cap.py`
4. 回填资金流：`python scripts/fetch_moneyflow_history.py --start-date 2025-01-01`
5. 回填机构席位：`python scripts/fetch_top_inst_history.py --start-date 2025-01-01`
6. 回填融资融券：`python scripts/fetch_margin_history.py --start-date 2025-01-01`
7. 同步静态行业信息并刷新最新数据：`python scripts/refresh_latest_data.py`
8. 断点续刷概念关系并查看覆盖进度：`python scripts/refresh_concepts_with_progress.py`
9. 分割股票：`python scripts/split_data.py`
10. 提特征：`python scripts/extract_features.py`
11. 多任务标签分析：`python scripts/analyze_multitask_feasibility.py`
12. 多任务特征落库：`python scripts/extract_multitask_features.py`
13. 多任务训练：`python scripts/train_multitask_model.py --all-layers`
14. 多任务最新预测：`python scripts/run_multitask_inference.py`
15. 多任务刷新并预测：`python scripts/run_multitask_latest_pipeline.py`
16. 多任务方案 A 回测：`python scripts/run_multitask_strategy_a_backtest.py`
17. 多任务方案 A 阈值搜索：`python scripts/run_multitask_strategy_a_threshold_search.py`
18. 分层训练：依次运行四个训练脚本
19. 集成结果：`python scripts/ensemble_model.py`
20. 最新选股：`python scripts/run_inference.py`
21. 一键刷新并预测：`python scripts/run_latest_pipeline.py`
22. 分位数回测：`python scripts/run_percentile_backtest.py`
23. 每日 top 5% 选股：`python scripts/select_top_ensemble_daily.py`
24. 真实约束回测：`python scripts/run_realistic_ensemble_backtest.py`
25. 中型股模型复查：`python scripts/review_mid_model.py`
26. 集成权重优化：`python scripts/optimize_ensemble_weights.py`
27. 历史回测：`python scripts/run_backtest.py`
28. 启动接口：`python scripts/serve_api.py`

## 已实现能力
- SQLite 数据读写
- 股票级互斥数据分割
- 四层专用特征列
- 四种 PyTorch 分层模型
- 统一版与分层独立版多任务五任务训练入口
- 训练日志、早停、指标与预测文件输出
- 集成预测结果导出
- 最新日线行情刷新
- 生产级最新选股推理
- 历史回测结果、资金曲线与统计摘要
- FastAPI 服务化入口与 Dockerfile

## Transformer 实验线现状
- 独立实验线位于 [experiments/financial_transformer/README.md](experiments/financial_transformer/README.md)，完整实验结论汇总位于 [experiments/financial_transformer/results/experiment_report_2026-04-15.md](experiments/financial_transformer/results/experiment_report_2026-04-15.md)。
- 当前最优单模型底座仍是 `moneyflow_v1`。它在公平窗口 `2026-01-01 ~ 2026-03-04` 上的 frozen 基线为 `13` 笔交易、约 `+15.75%`、Sharpe `5.52`。
- 当前最优执行平台已经从单模型升级为两阶段重排 `two_stage_rerank_best_v1`：`moneyflow_v1` 负责候选池，`moneyflow_selective_v1` 负责 `0.55 / 0.45` 融合修正，候选池大小 `8`。
- 在此之外，最近又补了一条非默认的最新信号 sidecar：`broader_history_rerank_*`。它不替换当前默认两阶段执行线，而是把 `moneyflow_v1` 的 latest 候选池交给“broader-history rerank”做二次排序，当前有两个可选 profile：`broader_history_rerank_default_exec` 与 `broader_history_rerank_transfer_exec`。
- 这两个 profile 已接入 FastAPI 的 `POST /predict/latest` 与 `GET /market/realtime/latest-picks`。同时还新增了 `GET /market/realtime/latest-profile-comparison`，可直接返回 `moneyflow_v1` 单模型 fair-filter top1 与两条 canonical rerank variant 的最新并排对比。
- `signal_report.py` 侧目前还额外保留了一条显式 Large 实验线 profile：`large_experimental`。它不属于默认 `all_split` 聚合输出，主要用于隔离当前 Large 试验结果与已停用的旧默认 `large` 档位。
- 当前 live latest 对比也已经固定成 artifact：`experiments/financial_transformer/results/broader_history_rerank_variants_v1/latest_profile_comparison.json` 与 `.md`。在 `2026-05-08` 这一天，单模型 base 选的是 `300197`，而两条 rerank canonical variant 都选 `003020`，说明 rerank 层已经在最新口径下实际改变了 top pick。
- 当前默认两阶段执行口径为：`short_up > 0.254`、`take_profit > 0.572`、`stop_loss < 0.50`、`top_k_per_day = 2`，基础退出纪律为 `+14% / -6% / 35天`。这条线在整窗上约 `+20.08%`、Sharpe `10.08`，并保持 `11/11` 个滚动窗口全正。
- 当前最强候选执行升级是 `leader-only exit`：仅对当日第 `1` 名候选使用 `+16% / -4% / 20天`，其余候选保持 `+14% / -6% / 35天`。这条线在两阶段平台上约 `+21.62%`、Sharpe `10.77`、最大回撤 `-1.29%`，同样保持 `11/11` 个滚动窗口全正。
- 当前还保留一个保守执行分支：`top_k_per_day = 3` 且 `min_score_ratio_to_leader = 0.97`。它整窗收益略低，但滚动平均 Sharpe 更高、平均回撤更小。
- 缠论/中枢类技术结构特征已经完成首轮验证。`moneyflow_chan_v1` 说明这类弱结构特征并非完全无信息，最优可行结果约 `+6.87%`、Sharpe `4.21`；但它仍明显落后于 `moneyflow_v1`。进一步收缩到核心结构信号的 `moneyflow_chan_core_v1` 只剩约 `+3.62%`、Sharpe `0.68`，说明这类特征当前更适合作为辅助排序信息，而不适合作为默认主线特征。
- 最近已明确证伪的方向包括：第一版 `trade-quality filter`、`rank_weight_power`、`margin_detail` 增强、`top_inst` 增强，以及当前实现下的 ranking-loss probe。它们要么只复现基线，要么在交易层明显退化。
- 最近修复了一条关键工程问题：`moneyflow_v1` 历史结果与重算结果不一致，根因是动态股票池漂移，不是模型数值漂移。现在凡是依赖历史预测重算的实验，都必须优先复用训练时保存的 `selected_stock_universe.csv`。
- 现在已经补完一轮更严格的 `full6 nested CV`：对 `moneyflow_v1` 与 `moneyflow_fundamental_grouped_gated_medium_v1` 做了 `6` 个外层时间滚动窗口、`3` 个内层股票 fold 的公平比较。结果仍支持 `moneyflow_v1` 继续作为默认单模型主线；它的内层验证均值更高、fold 波动更小，外层未来窗口均值也略优。
- 这轮 nested CV 之后，Transformer 主线的下一阶段重点已经转向“抗时变”而不是“再找一条替代主线”。当前优先方向是围绕 `moneyflow_v1` 做 `time decay weighting`、`recent-window fine-tune` 与轻量 `regime context` 增强，详细方案见 [experiments/financial_transformer/results/anti_regime_mainline_plan_2026-04-17.md](experiments/financial_transformer/results/anti_regime_mainline_plan_2026-04-17.md)。
- 当前行业相关特征仍只是低成本 probe：交易库里的行业标签来自 Tushare 静态 `stock_basic_info.行业` 单标签字段，不能表达一只股票同时属于多个题材或概念。
- 概念层数据链路已经落地：当前可通过 [scripts/refresh_concepts_with_progress.py](scripts/refresh_concepts_with_progress.py) 做断点续刷，`2026-04-20` 最新一轮已跑完整个概念列表，进度达到 `879/879`，其中有成员的概念覆盖为 `873/879`，累计概念成员关系 `32840` 行、覆盖股票 `4001` 只，平均每只股票约 `8.208` 个概念。
- 基于这层 many-to-many 概念关系，实验线已新增 [experiments/financial_transformer/config_moneyflow_v1_concept_sentiment_v1.yaml](experiments/financial_transformer/config_moneyflow_v1_concept_sentiment_v1.yaml) 作为概念情绪 probe；当前数据构建已验证可用，但 `round1` 训练仍在等待正在运行中的主线训练结束后再启动。

## 注意
- 默认复用上层目录中的 [stock_history.db](../stock_history.db) 和 [market_cap_full.db](../market_cap_full.db)。
- 配置里不再硬编码 Token，优先使用环境变量。
- 如果原始交易库没有 `成交额`，系统会自动用 `收盘 * 成交量` 估算。

## 推理输出
- 最新选股结果会写入 [results/latest_picks.csv](results/latest_picks.csv)
- 默认包含：`stock_code`、`layer`、`prediction_prob`、`ensemble_score`、`amount`

## 最新数据刷新
- 刷新交易库最近交易日行情：[scripts/refresh_latest_data.py](scripts/refresh_latest_data.py)
- 断点续刷股票概念关系并打印覆盖进度：[scripts/refresh_concepts_with_progress.py](scripts/refresh_concepts_with_progress.py)
- 刷新后直接生成最新推荐：[scripts/run_latest_pipeline.py](scripts/run_latest_pipeline.py)
- 当前使用 Tushare 日线、daily_basic 和 moneyflow，属于最新日频数据，不是盘中逐笔行情
- 当前也会同步股票静态行业分类，并把 `pe_ttm`、`pb`、`ps_ttm`、`dv_ttm` 等估值字段写入交易库，供特征工程直接使用
- 概念续刷采用断点续跑；命中 `concept_detail` 每分钟限流时会停在当前概念，下一次运行会从该位置继续，而不是清空重刷

## 实时行情接口
- 当前 FastAPI 额外提供盘中实时行情查询，适合做监控、盯盘和信号联调
- `GET /market/realtime/quote?ts_codes=000001.SZ,600000.SH`
- `GET /market/realtime/quote?ts_codes=000001,600000&src=sina`
- `GET /market/realtime/list?src=auto&page_count=1`
- `GET /market/realtime/latest-picks?top_n=20`
- `GET /market/realtime/latest-picks?top_n=20&refresh=true`
- `GET /market/realtime/latest-picks?signal_profile=broader_history_rerank_default_exec&top_n=5`
- `GET /market/realtime/latest-picks?signal_profile=large_experimental&top_n=5`
- `GET /market/realtime/latest-profile-comparison?variants=default_exec,transfer_exec&write_outputs=true`
- `POST /market/realtime/latest-profile-comparison/refresh?variants=default_exec,transfer_exec&write_outputs=true`
- `GET /market/realtime/latest-profile-comparison/jobs/{job_id}`
- 个股接口支持传入 `ts_code` 或 6 位股票代码，服务端会自动补齐交易所后缀
- 个股返回包含 `price`、`pct_change`、`change`、`open`、`high`、`low`、`volume`、`amount` 以及五档买卖盘字段
- 列表接口默认使用 `auto`，会先尝试 `dc`，失败后自动回退到 `sina`
- `latest-picks` 接口会把模型最新推荐和盘中实时行情合并，直接返回 `prediction_prob`、`ensemble_score`、`signal_close`、`price` 以及相对信号收盘价的盘中涨跌幅
- `latest-picks` 响应还会返回 `signal_date`、`signal_age_days` 和 `signal_is_stale`，方便区分“旧推荐的盘中表现”与“今天刚生成的实时推荐”
- 如果 `signal_profile` 传入 `broader_history_rerank_default_exec` 或 `broader_history_rerank_transfer_exec`，接口会直接返回 broader-history rerank 最新入选结果，而不影响默认 `micro` 路径。
- 如果 `signal_profile` 传入 `large_experimental`，接口会返回当前显式 Large 实验线结果；它与默认 `micro` 路径隔离，不会重新启用已停用的旧 `large` 默认档位。
- `latest-profile-comparison` 会同时返回单模型 base 与 rerank canonical variants 的最新 side-by-side payload，并可选落地 comparison artifact 到 `experiments/financial_transformer/results/broader_history_rerank_variants_v1/`。
- 如果需要先重算 latest 再比对，优先使用异步入口 `POST /market/realtime/latest-profile-comparison/refresh`，再通过 `GET /market/realtime/latest-profile-comparison/jobs/{job_id}` 查询状态；当前不建议在交互请求里直接等待同步 `refresh=true` 完成。

## 回测输出
- 成交记录：[results/backtest_trades.csv](results/backtest_trades.csv)
- 资金曲线：[results/backtest_equity.csv](results/backtest_equity.csv)
- 汇总指标：[results/backtest_summary.json](results/backtest_summary.json)

## 分位数回测输出
- 汇总表：[results/percentile_backtest_summary.csv](results/percentile_backtest_summary.csv)
- 净值曲线：[results/percentile_backtest_equity.csv](results/percentile_backtest_equity.csv)
- 入选明细：[results/percentile_backtest_selections.csv](results/percentile_backtest_selections.csv)

## 策略执行输出
- 每日 `ensemble top_5pct` 选股：[results/daily_selection_ensemble_top5pct.csv](results/daily_selection_ensemble_top5pct.csv)
- 真实约束回测汇总：[results/realistic_ensemble_top5pct_summary.json](results/realistic_ensemble_top5pct_summary.json)
- 真实约束回测成交：[results/realistic_ensemble_top5pct_trades.csv](results/realistic_ensemble_top5pct_trades.csv)
- 中型股模型复查报告：[results/mid_model_review.md](results/mid_model_review.md)
- 集成权重粗筛结果：[results/ensemble_weight_search_coarse.csv](results/ensemble_weight_search_coarse.csv)
- 集成权重真实回测结果：[results/ensemble_weight_search_realistic.csv](results/ensemble_weight_search_realistic.csv)
- 集成权重推荐：[results/ensemble_weight_recommendation.json](results/ensemble_weight_recommendation.json)
- 多任务可行性汇总：[results/multitask_feasibility_summary.csv](results/multitask_feasibility_summary.csv)
- 多任务可行性报告：[results/multitask_feasibility_report.md](results/multitask_feasibility_report.md)
- 分层多任务指标示例：[results/multitask_micro_metrics.json](results/multitask_micro_metrics.json)
- 多任务最新预测：[results/latest_multitask_picks.csv](results/latest_multitask_picks.csv)
- 多任务方案 A 候选：[results/latest_multitask_strategy_a.csv](results/latest_multitask_strategy_a.csv)
- 多任务方案 A 交易记录：[results/multitask_strategy_a_trades.csv](results/multitask_strategy_a_trades.csv)
- 多任务方案 A 资金曲线：[results/multitask_strategy_a_equity.csv](results/multitask_strategy_a_equity.csv)
- 多任务方案 A 每日持仓：[results/multitask_strategy_a_daily_holdings.csv](results/multitask_strategy_a_daily_holdings.csv)
- 多任务方案 A 汇总：[results/multitask_strategy_a_summary.json](results/multitask_strategy_a_summary.json)
- 多任务方案 A 阈值搜索全量结果：[results/multitask_strategy_a_threshold_search.csv](results/multitask_strategy_a_threshold_search.csv)
- 多任务方案 A 阈值搜索最佳结果：[results/multitask_strategy_a_threshold_best.json](results/multitask_strategy_a_threshold_best.json)
- 钳星简化投资建议：[results/qianxing_micro_strategy_recommendation.md](results/qianxing_micro_strategy_recommendation.md)
- 钳星模拟盘接口说明：[results/qianxing_paper_trading_guide.md](results/qianxing_paper_trading_guide.md)
- 钳星模拟盘 Python 客户端：[scripts/paper_trading_client.py](scripts/paper_trading_client.py)

## API 接口
- `GET /health`
- `POST /predict/latest`
- `POST /backtest/run`
- `GET /market/realtime/quote`
- `GET /market/realtime/list`
- `GET /market/realtime/latest-picks`
- `GET /market/realtime/latest-profile-comparison`
- `POST /market/realtime/latest-profile-comparison/refresh`
- `GET /market/realtime/latest-profile-comparison/jobs/{job_id}`
- `POST /paper-trading/account/init`
- `GET /paper-trading/account`
- `GET /paper-trading/positions`
- `GET /paper-trading/trades`
- `POST /paper-trading/orders/buy`
- `POST /paper-trading/orders/sell`
- `POST /paper-trading/orders/batch-buy`
- `POST /paper-trading/orders/batch-sell`
- `GET /paper-trading/quote`

## 模拟盘接口
- 模拟盘数据会持久化到 [results/paper_trading.db](results) 下，重启 API 不会丢失
- 先初始化账户现金和当前持仓：`POST /paper-trading/account/init`
- 买入时可直接传 `quantity`，也可以传 `cash_amount` 或 `position_fraction` 让系统自动按当前现金换算股数
- 卖出时可传 `quantity`，也可以传 `position_fraction` 按仓位比例减仓
- 买卖都会先读取实时行情价格，且现在支持可选 `limit_price` 做价格保护：买入时实时价高于限价会拒单，卖出时实时价低于限价会拒单
- 持仓估值、总资产、已实现盈亏、未实现盈亏都可以从 `GET /paper-trading/account` 获取
- 查看单只或多只股票实时行情并附带当前持仓：`GET /paper-trading/quote?stock_code=603109` 或 `GET /paper-trading/quote?stock_codes=603109,603933,000001`

### 初始化账户示例
```json
POST /paper-trading/account/init
{
	"initial_cash": 500000,
	"reset": true,
	"positions": [
		{"stock_code": "603109", "quantity": 1000, "avg_cost": 20.8},
		{"stock_code": "603933", "quantity": 500, "avg_cost": 15.3}
	]
}
```

### 自动按仓位买入示例
```json
POST /paper-trading/orders/buy
{
	"stock_code": "603109",
	"position_fraction": 0.25,
	"limit_price": 23.50,
	"src": "sina",
	"note": "qianxing paper buy"
}
```

### 按比例卖出示例
```json
POST /paper-trading/orders/sell
{
	"stock_code": "603109",
	"position_fraction": 0.5,
	"limit_price": 23.00,
	"src": "sina",
	"note": "take partial profit"
}
```

## Docker
构建镜像后可直接启动 FastAPI 服务，Dockerfile 在 [Dockerfile](Dockerfile)。
