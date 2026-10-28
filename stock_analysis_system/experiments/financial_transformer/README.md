# 金融 Transformer 实验线

这个目录用于承载独立于主分层系统的 Transformer 实验，但现在已经对齐到 `micro-only` 多任务策略口径，方便和当前最佳方案做公平比较。

当前默认单模型实验线仍是 `moneyflow_v1`，也就是“资金流增强版 Transformer”；但执行类入口已经默认切到带 `leader` 分层退出和 `leader score ratio` 过滤的版本 `two_stage_rerank_leader_tiered_v1`。

从 `2026-05-12` 开始，`moneyflow_v1` 单模型训练主线开始显式转向“幅度排序主导”：训练仍保留多任务分类头，但默认早停指标改为 `signal_future_spearman`，并启用同日收益差驱动的 `short_up` pairwise ranking loss；同时上调 `max_gain / max_drawdown` 回归头权重，降低纯阈值分类头的相对主导权。

最近又新增了一条非默认的 latest sidecar 主线：`broader_history_rerank_variants_v1`。它不替换当前默认 two-stage 执行，而是把 `moneyflow_v1` 的 latest 候选池交给更广历史监督训练出的 lightweight rerank 层做二次排序，并保留两条 canonical execution variant：

- `default_exec`：当前 recent-robust 版本，latest API profile 名称是 `broader_history_rerank_default_exec`
- `transfer_exec`：当前 transfer-prioritized 版本，latest API profile 名称是 `broader_history_rerank_transfer_exec`

当前这两条 variant 已经具备完整的 latest operational path：CLI、日报、API profile，以及 base vs rerank 的 latest side-by-side comparison endpoint 都已打通。

截至 `2026-04-24`，最近几轮增量实验已经形成新的阶段性结论：

- `moneyflow_v1` 的重算可复现性问题已经修复。根因不是模型数值漂移，而是动态股票池在复算时发生了 `stock universe drift`；现在冻结 universe 时必须同时关闭 `max_stocks` 和 `layer_filter`，否则会把快照里的股票再次筛掉。
- 第一版 `trade-quality filter` 已分别在单模型 `moneyflow_v1` 和最优两阶段执行线 `two_stage_rerank_best_v1` 上验证，结论一致：最佳阈值都只会复现 no-filter 基线，更严格阈值会明显变差，因此这版过滤器当前不具备真实增量。
- 执行层继续细化后，leader-only exit 主线又做了两轮局部精修：先把 leader 退出纪律优化到 `+16.5% / -3.5% / 20天`，再在同一口径下加入 `min_score_ratio_to_leader = 0.95` 的跟随筛选。其余候选仍保持基础退出 `+14% / -6% / 35天`。这条线在 `two_stage_rerank_best_v1` 上达到约 `+21.92%`、Sharpe `11.50`、MaxDD `-1.19%`，并继续保持 `11/11` 个滚动窗口全正，因此已经替代旧的 `+16% / -4% / 20天` 成为当前最强且滚动可行的执行升级点。随后补做的更长 OOS 复核也继续站住：`20x5` 平均窗口收益约 `+9.44%`、`4/4` 个正窗，`25x5` 平均窗口收益约 `+13.03%`、`3/3` 个正窗，因此这条默认执行线当前不只是“短窗有效”，而是对更长验证窗也保持了明显韧性。
- 缠论/中枢类技术结构特征已经完成首轮验证。`moneyflow_chan_v1` 说明这类弱结构特征并非完全无信息，最优可行窗口约为 `5` 笔交易、`+6.87%`、Sharpe `4.21`；但它仍明显落后于 `moneyflow_v1`。进一步收缩到更核心结构信号的 `moneyflow_chan_core_v1` 则只剩约 `+3.62%`、Sharpe `0.68`，说明这类特征当前更适合作为辅助排序信息，而不适合作为默认主线特征。
- `full6 nested CV` 已经完成，对 `moneyflow_v1` 和 `moneyflow_fundamental_grouped_gated_medium_v1` 做了 `6` 个外层时间窗口、`3` 个内层股票 fold 的公平评估。结论是：`moneyflow_v1` 仍然是当前默认主线。它的 `inner_mean_best_validation_metric` 约 `0.5449`，明显高于 grouped/gated fundamental 的约 `0.5387`；`inner_std` 约 `0.0204`，也低于 grouped/gated fundamental 的约 `0.0315`。外层 `business_score` 均值两者非常接近，但主线仍略高，且更稳。
- 最新补做的一跳结构复核也说明，这类 grouped/gated fundamental 信息当前不适合直接拿来做默认执行线的辅助 rerank 头。把 `moneyflow_fundamental_grouped_gated_medium_v1` 作为 `moneyflow_v1` 的 secondary，并沿用当前 leader-tiered 默认执行口径后，整窗结果只有约 `+5.49%`、Sharpe `1.85`、最大回撤 `-4.31%`；`20x5` 虽然仍是 `4/4` 个正窗，但平均窗口收益只有约 `+4.31%`，显著弱于当前默认 leader-tiered 线。进一步把 secondary 权重降到 `0.25 / 0.35` 也没有救回来，分别只有约 `-6.52% / +1.42%`，而 `0.15` 甚至会把信号筛空。因此这条“grouped/gated fundamental 做辅助 rerank”分支当前也应收口。
 - 随后又补做了一轮“辅助 gate 而不是直接混头”的最小结构试验。结果同样没有形成可升级增量：把 grouped/gated fundamental 分数附着到当前 leader-tiered 默认预测上做全局 gate 或 leader-only gate，最好的阈值 `>= 0.48` 只能把整窗收益从约 `+21.92%` 轻微抬到约 `+22.21%`，但 Sharpe 从约 `11.50` 降到约 `10.98`，最大回撤也从约 `-1.19%` 变差到约 `-1.30%`，因此不值得替换默认。再换成 `chan_core_v1` 结构分数做同样的 gate probe，结果几乎完全不动；更贴近“事件确认器”假设的 `topinst` 模型分数 leader-only gate 也同样完全不动默认结果。最后再往前走一步，直接把原始 `top_inst` 事件字段做成 raw leader gate 也没有救回来：不管是 `top_inst_active == 1`、`top_inst_active_5d == 1`，还是 `top_inst_net_buy_ratio >= 0.005`、`top_inst_net_buy_5d >= 0.01`，只要真正启用过滤，整窗结果都会从约 `+21.92% / Sharpe 11.50 / MaxDD -1.19%` 退化到约 `+13.55% / Sharpe 6.10 / MaxDD -3.29%`，`20x5` 平均窗口收益也从约 `+10.29%` 掉到约 `+6.37%`。在这之后又补做了一版 execution-aware 的 `trade_quality_filter_v2`，标签直接来自验证期默认 leader-tiered 执行线的真实成交盈亏，而不再使用静态 `take_profit && !stop_loss`。这一步虽然修正了标签定义，但仍没有形成可升级增量：验证期真实成交只有 `8` 笔，正样本占比约 `25%`，训练后测试期的 `pred_trade_quality` 几乎全部挤在 `0.5` 附近，阈值 `0.35 / 0.40 / 0.45` 只会精确复现默认基线，阈值 `0.50` 则会直接退化到约 `8` 个信号、`2` 笔交易、整窗约 `+2.21%`。随后又补做了“多窗聚合标签”判别：把 leader-tiered 默认执行线的 `20x5` 真实成交去重后聚合成训练标签，测试期仍只有约 `14` 笔可用样本，学出的 `pred_trade_quality` 标准差仍只有约 `0.00016`，阈值 `0.35 ~ 0.50` 全部原样复现默认基线，`0.55` 以上则直接筛空；改成只用验证段 `20x5` 聚合标签训练、再去测试段评估也没有改善，测试期分数标准差仍只有约 `0.00068`，`0.35 ~ 0.45` 继续完全复现默认，`0.50` 则退化到只剩 `2` 个信号且为负收益。到这里可以把当前 trade-quality filter 线正式收口：问题已经不只是标签定义，而是当前默认线可提供的真实成交监督样本过少，无法支撑一个有分离度的默认坏交易过滤器。
- `window_f` 的 outer test 指标为 `0`，原因更像是最新日期边界上的标签尚未完全闭合，因此当前正式比较以 `window_a ~ window_e` 为主。按这 `5` 个有效窗口看，`moneyflow_v1` 的 outer business_score 均值约 `0.514`，`moneyflow_fundamental_grouped_gated_medium_v1` 约 `0.512`，差距不大，但方向上仍支持主线优先。
- 这轮 nested CV 也把后续优化方向收敛清楚了：下一轮不再优先寻找新的默认主线，而是围绕 `moneyflow_v1` 做更直接的抗时变改造，例如 `time decay weighting`、`recent-window fine-tune` 和轻量 `regime context` 特征。对应方案见 `results/anti_regime_mainline_plan_2026-04-17.md`。
- 三条抗时变主线 probe 里，`time_decay`、`recent_finetune`、`regime_context` 都已经完成 `full6 nested CV`。其中 `regime_context_v1` 在 `round1` 首次略胜主线，但放大到 `full6` 后没有维持优势：其 `inner_mean_best_validation_metric` 约 `0.5367`、`outer_mean_test_business_score` 约 `0.4209`，低于 `moneyflow_v1` 的约 `0.5449 / 0.4285`。
- 在此基础上，后续又验证了一条更窄的组合式近期适配线：`adaptive_recent_60_decay_240_v1`，也就是 `240` 天 half-life 的轻度 time decay 叠加最近 `60` 天小学习率 finetune。它在 `full6` 上给出约 `0.5398 / 0.0168 / 0.4312`：`outer_mean_test_business_score` 略高于主线的 `0.4285`，`inner_std` 也更低，但 `inner_mean_best_validation_metric` 仍低于主线的 `0.5449`。因此它是当前第一条在完整 `full6` 口径上略微超过主线外层均值的近期适配候选，但还不是“全面打穿主线”的结论，默认主线暂时仍保持 `moneyflow_v1` 不变。
- 在修复了微盘资金流缺口和训练前数据检查后，又补做了一组 post-fix `round1` 公平对照：`moneyflow_v1` 重跑后约为 `0.5385 / 0.0195 / 0.4298`，而 `config_moneyflow_v1_ar60d240_v1.yaml` 重跑后约为 `0.5347 / 0.0176 / 0.4324`。这说明数据链路修复并没有推翻此前判断：adaptive 组合线仍在 outer mean 和稳定性上占优，但 inner mean 仍低于修复后的 `moneyflow_v1`。
- 不过在随后专门补做的“固定阈值、尽量看未来而不是继续拟合过去”的交易验证里，结论反过来了：同样使用 `short_up=0.50 / take_profit=0.46 / stop_loss=0.58 / top_k_per_day=1` 的固定筛选口径，以及 `45d / +14% / -6%` 的统一执行口径时，`moneyflow_v1` 在最近区间单次回测约为 `+15.75%`、Sharpe `5.52`、最大回撤 `-3.18%`，而 `config_moneyflow_v1_ar60d240_v1.yaml` 约为 `-4.35%`、Sharpe `-1.31`、最大回撤 `-6.43%`。对应 rolling `10x5` / `15x5` 的平均窗口收益，主线约为 `+1.59% / +3.50%`，而 `ar60d240` 仅约 `+0.46% / +0.50%`。因此这条线现在应被定义为“训练端最强的近期适配候选，但未通过固定阈值交易验证”，默认主线继续保持 `moneyflow_v1`，不做替换。
- 随后补做的 `moneyflow_v1_industry_sentiment_v1` 也给出了相同方向的结论。它在 `round1 nested CV` 上约为 `0.5443 / 0.0216 / 0.4373`，说明行业状态特征并非完全无信息；但在相同固定筛选与统一执行口径下，最近区间单次回测只有约 `+3.18%`、Sharpe `0.94`、最大回撤 `-8.30%`，明显落后于 `moneyflow_v1` 的 `+15.75% / 5.52 / -3.18%`。对应 rolling `10x5 / 15x5` 的平均窗口收益约为 `+0.91% / +1.92%`，也低于主线的 `+1.59% / +3.50%`。因此这条线目前同样只保留为“有一定离线增益、但未通过固定阈值交易验证”的研究分支，不升级默认主线。
- 最新补做的四条“最小改动/近期适配”主线补充复核也都没有通过同口径交易验证。`moneyflow_v1_time_decay_240_v1` 虽然离线 `business_score` 约为 `0.5283`，高于主线的 `0.5164`，但固定阈值单次回测只有约 `-1.00%`、Sharpe `0.36`、最大回撤 `-20.25%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-3.01% / -2.88%`；`moneyflow_v1_recent_finetune_60_v1` 的离线 `business_score` 也约为 `0.5295`，但固定阈值单次回测约 `-4.13%`、Sharpe `0.15`、最大回撤 `-21.18%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-0.06% / -1.21%`；`moneyflow_v1_regime_context_v1` 的离线 `business_score` 约为 `0.5208`，也略高于主线，但固定阈值单次回测约 `-3.28%`、Sharpe `-0.82`、最大回撤 `-11.88%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-1.38% / -2.47%`；`moneyflow_v1_ar60d240_gentle_v1` 的离线 `business_score` 约为 `0.5183`，同样略高于主线，但固定阈值单次回测约 `-4.37%`、Sharpe `-1.13`、最大回撤 `-9.67%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-3.84% / -4.52%`。这进一步说明：当前这类“偏近期样本权重”“近期末端再适配”或“轻量环境上下文”式的主线改造，在线下分数上可以有亮点，但并没有把优势兑现成更好的未来交易结果，因此默认单模型主线仍应继续固定为 `moneyflow_v1`。
- 这之后又补做了一次更直接的“标签对齐交易规则”试验：`config_moneyflow_v1_firsthit_tp14_v1.yaml` 把 `take_profit / stop_loss` 标签改成与回测一致的 `first-hit` 语义，并把训练阈值直接对齐到 `45` 天内 `+14% / -6%`。结果依然没有超过主线：离线 `business_score` 约 `0.5099`，低于 `moneyflow_v1` 的 `0.5164`；固定阈值单次回测约 `-4.84%`、Sharpe `0.04`、最大回撤 `-26.79%`，rolling `10x5 / 15x5` 的平均窗口收益约 `-1.32% / -1.61%`，同样明显弱于主线。因此“训练标签更贴近退出规则”这个方向本身值得保留，但以当前实现直接替换主线标签，并没有带来更好的未来交易结果。
- 随后又补做了一版更接近“最终盈亏归因”的上游标签试验：`config_moneyflow_v1_realizedprofit_tp14_v1.yaml` 把 `take_profit / stop_loss` 标签改成未来 `45` 天内按 `+14% / -6%` 真实退出规则落地后的盈亏符号，并在 `multitask_data.py` 里新增了 `exit_label_mode = realized_profit`。先用 smoke 版 `12` epoch 快速验证后，离线 `business_score` 约为 `0.5003`，固定阈值单次回测约 `+4.47%`、Sharpe `1.55`、最大回撤 `-5.14%`、`10` 笔交易；rolling `10x5 / 15x5` 的平均窗口收益约为 `+0.61% / +1.03%`，都弱于 `moneyflow_v1` 的 `+15.75% / 5.52 / -3.18%` 与 `+1.59% / +3.50%`。这说明“直接用 realized-profit 重定义标签”比 `first-hit` 标签替换更稳一些，但目前仍没有形成可升级主线的交易增量，因此暂时只保留为标签工程备选分支，不继续优先放大。
- 按 `2026-04-25` 的收口方向，又补做了一版只围绕高密度派生特征的小包 smoke 试验 `config_moneyflow_v1_densepack_smoke_v1.yaml`：在 `moneyflow_v1` 基础上只增加 `amount_return_5d`、`mf_change_rate_3d`、`mf_change_rate_5d`、`return_vol_adj_5d`、`mf_volatility_adj`、`price_amount_divergence_5d` 六个密集特征。结果再次说明“离线增益不等于交易增益”：smoke 离线 `business_score` 虽然约为 `0.5189`，略高于主线，但统一固定阈值单次回测直接退化到约 `-11.60%`、Sharpe `-3.40`、最大回撤 `-13.46%`、`13` 笔交易。因此当前这组 densepack 特征不应继续放大，也不应直接并入默认单模型主线。
- 最新按同一固定筛选与统一执行口径补做的 `moneyflow_topinst_v1` 复核也没有通过。虽然它把龙虎榜机构席位事件成功接入到资金流底座里，离线 `business_score` 仍有约 `0.5109`，但最近区间单次回测只有约 `+0.67%`、Sharpe `0.51`、最大回撤 `-21.89%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-2.59% / -3.60%`，并且 `15x5` 只剩 `1/5` 个正收益窗口。这说明 `top_inst` 在当前任务里更像稀疏辅助事件，而不是能直接抬升默认单模型交易表现的主增强维度，因此这条线现阶段也应正式收口。
- 同样补做的 `moneyflow_margin_v1` 固定阈值复核也没有通过。虽然融资融券特征覆盖更高、理论上比事件型 `top_inst` 更容易稳定利用，但最近区间单次回测仍只有约 `-2.49%`、Sharpe `-0.86`、最大回撤 `-6.48%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-0.72% / -0.50%`，其中 `15x5` 也只剩 `1/5` 个正收益窗口。当前结论是：`margin_detail` 在现有任务定义下同样没有把离线信号兑现成更好的未来交易结果，这条增强线也应一起收口。
- `concept_sentiment_v2` 作为单模型方向已经完成闭合版 `full6` 对照。结论是：它证明了 concept 聚合确实有信息，但在公平闭合口径下仍未打穿 `moneyflow_v1`，因此不升级为新的默认单模型底座。
- 随后又连续做了两轮 concept 两阶段验证：先直接复用 `moneyflow_selective_v1` 做 secondary，再重训了一个 concept 专用 secondary。两轮结果都失败，而且第二轮更差。固定回测分别约为 `-1.53%` 和 `-4.29%`，显著落后于当前默认两阶段执行线 `two_stage_rerank_best_v1` 的约 `+20.08%`。因此这条 concept 两阶段分支现在正式收口，不再继续作为近期优先方向。

## 目标

- 复用现有交易库和市值库
- 在 `micro-only` 股票宇宙上训练多任务 Transformer
- 使用与现有 `micro-only` 多任务策略一致的标签和交易约束做公平回测

## 实验报告

- 当前阶段的实验总结和后续探索方向，见：`results/experiment_report_2026-04-15.md`

## 人工概念标签

- 训练/验证/实验阶段现在支持使用本地人工概念标签文件替代或覆盖数据库里的 `stock_concept_members`。
- 配置项放在 `experiment` 段：
	- `manual_concept_labels_path`: 本地 CSV 路径。
	- `concept_label_source`: `db_only`、`manual_first` 或 `manual_only`。
- 人工标签文件至少需要两列：`stock_code` 和 `manual_concept_names`。多概念使用 `|` 分隔；可选列 `manual_concept_ids` 也支持同样格式。
- 可以先用下面的脚本导出当前股票池的人工标注模板，再逐步手工补齐：

```powershell
d:/qcr/Stock/stock_analysis_system/.venv/Scripts/python.exe experiments/financial_transformer/prepare_manual_concept_labels.py --config experiments/financial_transformer/config_moneyflow_v1_large_v1.yaml --output experiments/financial_transformer/manual_labels/manual_concept_labels_large_v1.csv
```
- 04-25 对外部交易反馈的正式审阅与收口结论，见：`results/model_optimization_review_2026-04-25.md`

## 当前实现

- 默认训练配置文件 [config.yaml](d:\qcr\Stock\stock_analysis_system\experiments\financial_transformer\config.yaml)，当前等价于 `moneyflow_v1`
- 当前训练默认口径已经统一切到“幅度排序主导”：可训练配置默认使用 `signal_future_spearman` 做早停指标，并启用 `short_up` 的同日 pairwise ranking loss；`rank_v1 / rank_v2 / rank_mild` 这类专门 ranking 分支保留各自原始 ranking 强度，但同样切到新的早停口径
- 默认执行配置文件 `config_two_stage_rerank_leader_tiered_v1.yaml`，当前等价于“`moneyflow_v1` 候选池 + `moneyflow_selective_v1` 融合重排”，并对当日 leader 使用更强的分层退出纪律，同时要求跟随信号至少达到 leader 分数的 `95%`
- 新增保守执行分支 `config_two_stage_rerank_conservative_v1.yaml`，保持同样的两阶段融合底座，但把执行口径切到 `top_k_per_day = 3` 且 `min_score_ratio_to_leader = 0.97`
- 新增 leader 分层退出执行分支 `config_two_stage_rerank_leader_tiered_v1.yaml`，保持同样的两阶段融合底座，但把当日第 `1` 名候选切到独立退出纪律 `+16.5% / -3.5% / 20天`
- `micro-only` 动态股票池构建
- 与主系统一致的多任务标签：`target_up_5d / target_take_profit_45d / target_stop_loss_45d / max_gain_45d / max_drawdown_45d`
- 金融 Transformer + 时间注意力池化 + 五头任务输出
- 多任务训练、排序相关性早停、指标输出、预测输出
- 训练产物现在默认额外输出 `signal_future_pearson`、`signal_future_spearman`、`pass_signal_future_spearman`、`pass_mean_future_return`、`top1_mean_future_return`
- 与现有微盘策略同口径的交易回测

## 使用方式

1. 按需要修改 [config.yaml](config.yaml) 中的 `layer_filter`、`max_stocks` 和时间切分。
2. 在项目根目录安装依赖：`pip install -r requirements.txt`
3. 运行：`python experiments/financial_transformer/train.py`
4. 对比分层模型：`python experiments/financial_transformer/compare.py`
5. 运行多任务策略回测：`python experiments/financial_transformer/backtest.py`
6. 运行阈值搜索：`python experiments/financial_transformer/threshold_search.py`
7. 运行固定阈值的滚动窗口复核：`python experiments/financial_transformer/rolling_backtest.py --config experiments/financial_transformer/config_moneyflow_v1.yaml --short-up-threshold 0.50 --take-profit-threshold 0.46 --stop-loss-threshold 0.58 --top-k-per-day 1`
8. 生成最新多任务预测：`python experiments/financial_transformer/predict.py`
9. 生成最新信号日报：`python experiments/financial_transformer/signal_report.py`
10. 如果要显式生成 Large 实验线日报：`python experiments/financial_transformer/signal_report.py --signal-profile large_experimental --skip-refresh-data --skip-predict-latest`
11. 如果要直接跑 broader-history rerank latest 入口：`python experiments/financial_transformer/predict.py --broader-history-rerank-variant default_exec`
12. 如果要生成 broader-history rerank latest 日报：`python experiments/financial_transformer/signal_report.py --broader-history-rerank-variant default_exec --skip-refresh-data --skip-predict-latest`
13. 如果要生成最新 side-by-side 对比：`python experiments/financial_transformer/compare_latest_moneyflow_vs_rerank_variants.py`
14. 如果要直接生成 comparison 日报与钳星提示：`python experiments/financial_transformer/signal_report.py --latest-profile-comparison --skip-refresh-data --skip-predict-latest`
15. 对已训练模型补做概率校准：`python experiments/financial_transformer/calibrate.py`
16. 如果要按校准后概率运行预测/回测/搜索，需要显式传 `--use-calibration`
17. 如果要批量搜索窗口长度，可运行：`python experiments/financial_transformer/window_search.py --config experiments/financial_transformer/config_moneyflow_v1.yaml --candidates 45,60,75,90,105,120 --sample-size-only`
18. 如果要复查执行层参数，可运行：`python experiments/financial_transformer/execution_search.py --config experiments/financial_transformer/config_two_stage_rerank_best_v1.yaml --short-up-threshold 0.254 --take-profit-threshold 0.572 --stop-loss-threshold 0.50`
19. 如果要复查 leader 分层退出，可运行：`python experiments/financial_transformer/execution_tiered_exit_search.py`

如果要保留当前默认入口并尝试其他新版本，可以给这些入口传 `--config`，例如：`python experiments/financial_transformer/train.py --config experiments/financial_transformer/config_llm_v1.yaml`

## 输出目录

- 默认训练模型文件：`experiments/financial_transformer/artifacts/moneyflow_v1/financial_transformer.pt`
- 默认训练历史：`experiments/financial_transformer/results/moneyflow_v1/history.json`
- 默认训练评估指标：`experiments/financial_transformer/results/moneyflow_v1/metrics.json`，其中现在默认包含排序相关性与 top1 未来收益指标
- 默认训练概率校准参数：`experiments/financial_transformer/results/moneyflow_v1/probability_calibration.json`
- 默认训练测试集多任务预测：`experiments/financial_transformer/results/moneyflow_v1/test_predictions.csv`
- 分层模型对比指标：`experiments/financial_transformer/results/hierarchical_compare_metrics.json`
- 分层模型对比预测：`experiments/financial_transformer/results/hierarchical_compare_predictions.csv`
- 默认执行阈值搜索结果：`experiments/financial_transformer/results/two_stage_rerank_best_v1/transformer_multitask_threshold_search*.csv`
- 默认执行回测汇总：`experiments/financial_transformer/results/two_stage_rerank_best_v1/transformer_micro_multitask_backtest_summary.json`
- 默认执行回测成交：`experiments/financial_transformer/results/two_stage_rerank_best_v1/transformer_micro_multitask_backtest_trades.csv`
- 默认最新预测结果：`experiments/financial_transformer/results/two_stage_rerank_best_v1/latest_predictions.csv`
- 默认最新信号日报：`experiments/financial_transformer/results/two_stage_rerank_best_v1/latest_signal_report_*.md`
- broader-history rerank latest 输出目录：`experiments/financial_transformer/results/broader_history_rerank_variants_v1/default_exec/latest/` 与 `.../transfer_exec/latest/`
- broader-history rerank latest 对比输出：`experiments/financial_transformer/results/broader_history_rerank_variants_v1/latest_profile_comparison.json` 与 `.md`
- Signal_report comparison 日报输出：`experiments/financial_transformer/results/broader_history_rerank_variants_v1/latest_profile_comparison_signal_report.md`
- Signal_report 钳星 comparison 提示：`experiments/financial_transformer/results/broader_history_rerank_variants_v1/latest_profile_comparison_qianxing_message.md`
- 保守执行分支输出目录：`experiments/financial_transformer/results/two_stage_rerank_conservative_v1/`
- leader 分层退出执行分支输出目录：`experiments/financial_transformer/results/two_stage_rerank_best_v1/` 下的 `execution_search_*leader_tiered_exit_v1*`
- 单模型过滤器复现实验输出目录：`experiments/financial_transformer/results/moneyflow_v1_trade_filter_v1_frozen/`
- 两阶段过滤器实验输出目录：`experiments/financial_transformer/results/two_stage_rerank_best_v1_trade_filter_v1/`

## 最新 Rerank 入口

- CLI latest 预测入口：`python experiments/financial_transformer/predict.py --broader-history-rerank-variant default_exec`
- CLI latest 日报入口：`python experiments/financial_transformer/signal_report.py --broader-history-rerank-variant default_exec`
- CLI Large 实验线日报入口：`python experiments/financial_transformer/signal_report.py --signal-profile large_experimental --skip-refresh-data --skip-predict-latest`
- API latest profile：`broader_history_rerank_default_exec` 与 `broader_history_rerank_transfer_exec`
- `signal_report.py` 当前默认 `all_split` 只输出 `micro`；原 `large` 默认档位已临时停用，避免继续复用被失败重训覆盖的 checkpoint。
- 如果仍需查看 Large 最新信号，优先使用 `large_experimental`，它当前指向独立的 `config_moneyflow_selective_large_v1.yaml` 与结果目录。
- API latest 对比入口：`GET /market/realtime/latest-profile-comparison?variants=default_exec,transfer_exec`
- API latest 对比异步刷新入口：`POST /market/realtime/latest-profile-comparison/refresh?variants=default_exec,transfer_exec`
- API latest 对比任务状态入口：`GET /market/realtime/latest-profile-comparison/jobs/{job_id}`
- 当前 latest 对比结论：在 `2026-05-08` 这个最新可用日期上，单模型 `moneyflow_v1` fair-filter top1 是 `300197`，而两条 canonical rerank variant 的 top1 都是 `003020`。这说明 broader-history rerank 已经在 live latest 口径下改变了最终 top pick，但 `default_exec` 与 `transfer_exec` 当前仍彼此一致。
- 如果要先重算 base latest 与 rerank latest 再做 comparison，当前推荐走异步 API job，而不是直接等待同步 `refresh=true` 请求返回；后者在当前环境里属于重量级重算路径。

## 版本并行

- 默认 `config.yaml` 现在对应当前最强的单模型底座 `moneyflow_v1`。
- 默认执行入口则对应 `config_two_stage_rerank_best_v1.yaml`。
- 新实验建议新建独立配置文件，并把 `results_dir` / `artifacts_dir` 指向单独目录，避免覆盖当前结果。
- 当前已新增一个更强的 LLM-inspired 变体配置：`config_llm_v1.yaml`
- 当前也支持更轻量的相对位置偏置实验，可以用 `position_encoding: alibi` 新建独立配置做小步验证
- 当前也新增了资金流增强实验配置：`config_moneyflow_v1.yaml`
- 当前也新增了资金流细化增强实验配置：`config_moneyflow_v2.yaml`
- 当前也新增了资金流动态长窗口实验配置：`config_moneyflow_dynamic_120_v1.yaml`
- 当前也新增了主线抗时变的 time-decay 配置：`config_moneyflow_v1_time_decay_120_v1.yaml`、`config_moneyflow_v1_time_decay_180_v1.yaml`、`config_moneyflow_v1_time_decay_240_v1.yaml`
- 当前也新增了主线抗时变的 recent-finetune 配置：`config_moneyflow_v1_recent_finetune_40_v1.yaml`、`config_moneyflow_v1_recent_finetune_60_v1.yaml`
- 当前也新增了主线近期适配的组合配置：`config_moneyflow_v1_adaptive_recent_60_decay_240_v1.yaml`
- 为了避免 Windows 长路径问题，同一组合版也提供短名别名配置：`config_moneyflow_v1_ar60d240_v1.yaml`
- 当前也新增了主线抗时变的 regime-context 配置：`config_moneyflow_v1_regime_context_v1.yaml`
- 当前也新增了资金流 + 龙虎榜机构席位增强实验配置：`config_moneyflow_topinst_v1.yaml`
- 当前也新增了资金流 + 融资融券增强实验配置：`config_moneyflow_margin_v1.yaml`

## Time Decay 主线

- 现在训练配置支持 `training.time_decay_half_life_days`。只要给出正数，训练集样本就会按“距训练窗最新日期的天数”做指数半衰减加权，越新的样本权重越高。
- 当前首轮主线配置已经按计划补齐：`120`、`180`、`240` 三个 half-life 版本，输出目录分别落到独立的 `results/moneyflow_v1_time_decay_*` 与 `artifacts/moneyflow_v1_time_decay_*`。
- 这条改造不改模型结构，也不改 nested CV runner；后续直接把这些配置作为 candidate 跑同一套外层时间滚动 + 内层股票轮换即可。
- 但按最新补做的固定阈值交易验证看，`config_moneyflow_v1_time_decay_240_v1.yaml` 并没有把离线增益兑现成更好的未来收益：虽然离线 `business_score` 约为 `0.5283`，但最近区间单次回测约 `-1.00%`、Sharpe `0.36`、最大回撤 `-20.25%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-3.01% / -2.88%`。因此 time-decay 当前只能保留为“训练端有信号、交易端未通过验证”的研究分支，不升级默认主线。

## Recent Finetune 主线

- 现在训练配置也支持 `recent_finetune` 两阶段训练：先按原训练窗完成基础训练，再只用训练集最近窗口做一段小学习率微调。
- 可用参数包括 `training.recent_finetune_window_days`、`training.recent_finetune_epochs`、`training.recent_finetune_learning_rate_factor` 和可选的 `training.recent_finetune_patience`。
- 第一轮先补了 `40` 天和 `60` 天两个版本，默认学习率倍率为 `0.1`，用于验证“长期底座 + 近期再适配”是否比纯 time decay 更稳。
- 最新补做的 `config_moneyflow_v1_recent_finetune_60_v1.yaml` 固定阈值交易验证同样失败。它的离线 `business_score` 约为 `0.5295`，高于主线的 `0.5164`；但最近区间单次回测只有约 `-4.13%`、Sharpe `0.15`、最大回撤 `-21.18%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-0.06% / -1.21%`。这说明“长期底座 + 近期微调”在当前设定下并没有比纯 time decay 更稳，反而在真实交易验证里同样显著落后于 `moneyflow_v1`。

## Adaptive Recent 主线

- 当前新增了一个更窄的组合版近期适配配置：`config_moneyflow_v1_adaptive_recent_60_decay_240_v1.yaml`。
- 它不再引入新的主线探针，而是把目前三条路线里相对最温和、也最贴近实盘再适配的两个动作叠加起来：`240` 天 half-life 的 time decay + 最近 `60` 天的小学习率 finetune。
- 这条线的目的不是替代 `moneyflow_v1` 的信号底座，而是验证“轻度偏向近期样本 + 末端短窗再适配”能否比单独使用其中任一动作更稳。
- 为了避免 Windows 长路径导致 nested CV 中途失败，这条线在实际 runner 中同时使用了短名别名配置 `config_moneyflow_v1_ar60d240_v1.yaml`。
- 在此基础上又补了一条更温和的变体 `config_moneyflow_v1_ar60d240_gentle_v1.yaml`，保持 `60d recent + 240d decay` 不变，只把 recent finetune 力度收敛到 `epochs=3`、`lr_factor=0.05`、`patience=2`，用于验证是否能在保住 outer 表现的同时把 inner mean 拉回主线附近。
- `round1` 上它给出约 `0.5420 / 0.0157 / 0.5203`，同时优于 `moneyflow_v1`、`time_decay_240` 和 `recent_finetune_60` 的同口径结果，因此被推进到 `full6`。
- `full6` 上它的最终结果为：`inner_mean_best_validation_metric` 约 `0.5398`、`inner_std_best_validation_metric` 约 `0.0168`、`outer_mean_test_business_score` 约 `0.4312`。这说明它在主比较指标上略优于 `moneyflow_v1`，并且稳定性更好，但优势幅度还不够大，且 inner mean 仍未超过主线，因此当前更适合作为“最值得继续打磨的近期适配候选”，而不是立即替换默认主线。
- `gentle` 变体的 `round1` 结果为约 `0.5370 / 0.0197 / 0.4333`。这说明更弱的 recent finetune 并没有把 inner mean 拉回去，反而低于原版 `ar60d240_v1` 的约 `0.5420 / 0.0157 / 0.5203`；虽然 outer mean 仍高于旧主线 `moneyflow_v1 round1` 的约 `0.5141`，但相对原版 adaptive 组合线并没有形成更强证据，因此目前仍以 `config_moneyflow_v1_ar60d240_v1.yaml` 作为唯一重点候选继续保留。
- 在引入资金流缺口清洗与训练前数据检查后，这条线又补做了一次 post-fix `round1`：约 `0.5347 / 0.0176 / 0.4324`。对照同口径重跑后的 `moneyflow_v1` 约 `0.5385 / 0.0195 / 0.4298`，可以看到 adaptive 仍然保持了更高的 outer mean 和更低的 inner std；如果只看 `window_a ~ window_e` 这 `5` 个有效外层窗口，其 outer business_score 均值约 `0.5189`，也高于修复后 `moneyflow_v1` 的约 `0.5157`。
- 但把判断标准进一步收紧到固定阈值的真实交易验证后，这个优势没有兑现成更好的未来窗口收益。按统一固定口径补跑后，`moneyflow_v1` 的最近区间单次回测约 `+15.75% / Sharpe 5.52 / MDD -3.18%`，rolling `10x5` 与 `15x5` 的平均窗口收益约 `+1.59% / +3.50%`；而 `config_moneyflow_v1_ar60d240_v1.yaml` 约为 `-4.35% / -1.31 / -6.43%`，rolling `10x5` 与 `15x5` 仅约 `+0.46% / +0.50%`。因此这条线现在不再作为“可能升级主线”的候选推进，而是降级为“训练端有亮点、交易端未通过验证”的保留研究分支；默认主线继续维持 `moneyflow_v1`。
- 更温和的 `config_moneyflow_v1_ar60d240_gentle_v1.yaml` 随后也补做了同口径固定阈值交易验证，但结果比原版 adaptive 组合线还弱。它的离线 `business_score` 约为 `0.5183`，仍略高于主线的 `0.5164`；但最近区间单次回测约 `-4.37%`、Sharpe `-1.13`、最大回撤 `-9.67%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-3.84% / -4.52%`。这说明把 recent finetune 再收敛到更轻力度，也没有把 adaptive recent 这条路线拉回可交易区间，因此 `gentle` 变体同样只保留为研究分支，不再继续作为主线升级候选。

## Regime Context 主线

- 当前版本把环境状态特征作为轻量附加输入，而不是改模型主体。实现方式是在所选股票池内按日期做横截面聚合，再把环境状态回填到每只股票当日样本。
- 第一版只加 6 个代理特征：`regime_breadth_10d`、`regime_breadth_20d`、`regime_return_mean_10d`、`regime_return_vol_10d`、`regime_turnover_ratio_10_20`、`regime_net_mf_mean_10d`。
- 这些特征都来自当前实验已经在用的价格、换手和资金流字段，不依赖额外外部数据，便于直接沿用 nested CV 口径做公平比较。
- `round1_regime_context_v1` 曾在 `2` 个外层窗口上同时略优于主线，因此被推进到 `full6`。
- 但 `full6_regime_context_v1` 的最终结果没有打穿主线：`inner_mean_best_validation_metric` 约 `0.5367`，低于 `moneyflow_v1` 的 `0.5449`；`outer_mean_test_business_score` 约 `0.4209`，也低于主线的 `0.4285`。因此这条路线目前只能算有效 probe，还不能升级成默认主线。
- 随后补做的固定阈值交易验证同样失败。`config_moneyflow_v1_regime_context_v1.yaml` 的离线 `business_score` 约为 `0.5208`，仍高于主线的 `0.5164`；但最近区间单次回测约 `-3.28%`、Sharpe `-0.82`、最大回撤 `-11.88%`，rolling `10x5 / 15x5` 的平均窗口收益约为 `-1.38% / -2.47%`。这说明轻量 regime context 在当前实现里更像“离线有帮助的辅助上下文”，但还不足以支撑默认单模型主线升级。

## First-Hit 标签对齐主线

- 当前额外加了一条更贴近真实退出规则的试验配置：`config_moneyflow_v1_firsthit_tp14_v1.yaml`。
- 这条线不改模型结构，也不改输入特征，只改标签语义：把 `target_take_profit_45d / target_stop_loss_45d` 从原来的“45 天窗口内是否曾触发过”改成“按回测同样的优先级看，45 天内先触发的是止盈还是止损”，并把训练阈值直接对齐到当前真实执行的 `+14% / -6% / 45天`。
- 其中同日同时触发止盈和止损时，标签与回测保持一致，统一按止损优先处理。
- 最新这次固定阈值交易验证显示，这条线并没有把结果拉回主线之上。它的离线 `business_score` 约 `0.5099`，低于主线的 `0.5164`；最近区间单次回测约 `-4.84%`、Sharpe `0.04`、最大回撤 `-26.79%`，rolling `10x5 / 15x5` 的平均窗口收益约 `-1.32% / -1.61%`。虽然 rolling `10x5` 的正收益窗口数达到 `5/6`，但主要由低交易数窗口支撑，最终仍被尾部大幅亏损窗口拖成负均值。
- 随后又专门把这条线的 `signal_score` 和筛选逻辑也改成围绕 `first-hit` 头工作，并做了一轮独立阈值搜索。最优可行组合大致为 `short_up > 0.46`、`take_profit > 0.40`、`stop_loss < 0.40`、`first_hit_edge >= 0`、`top_k = 2`，单次回测可回到约 `+2.41%`、Sharpe `0.62`、最大回撤 `-20.96%`、`9` 笔交易、`74` 个信号。这说明“标签 + 打分 + 筛选一起围绕 first-hit 重构”这个方向比单独换标签要更合理，至少已经把结果从负收益拉回正收益；但它仍明显落后于 `moneyflow_v1` 的 `+15.75% / 5.52 / -3.18%`，而且回撤质量很差，因此当前仍不能升级默认主线。
- 因此现阶段可以确认：简单把训练标签改成 first-hit 并直接对齐 `+14% / -6% / 45天`，还不足以让 `moneyflow_v1` 的未来交易结果变好。这条路线目前更适合作为一个值得继续拆解的标签工程 probe，而不是立即升级默认主线。

## 容量配比实验

- 为了验证“新增信息维度是否需要同步扩模型脑容量”，当前新增了一组最小容量配比矩阵。
- 对照底座分两组：`moneyflow_v1` 代表当前最佳资金流底座；`moneyflow_fundamental_v1` 代表在同一底座上加入基本面与行业标签后的扩展特征版本。
- `small` 对照直接复用现有配置：`config_moneyflow_v1.yaml` 与 `config_moneyflow_fundamental_v1.yaml`。
- `medium` 容量配置：`d_model=96`、`nhead=6`、`num_layers=3`、`dim_feedforward=384`，对应 `config_moneyflow_v1_capacity_medium_v1.yaml` 与 `config_moneyflow_fundamental_capacity_medium_v1.yaml`。
- `large` 容量配置：`d_model=128`、`nhead=8`、`num_layers=4`、`dim_feedforward=512`，对应 `config_moneyflow_v1_capacity_large_v1.yaml` 与 `config_moneyflow_fundamental_capacity_large_v1.yaml`。
- 这组实验的核心不是只看谁的绝对指标最高，而是看“扩展特征版本相对基础版本”的差值是否会随着容量变大而改善；如果 small 下无增量，但 medium / large 开始转正，就说明当前结论里确实混入了容量瓶颈。
- 推荐运行顺序：先跑 `moneyflow_v1_capacity_medium_v1`，确认单纯扩容不会立即恶化；再跑 `moneyflow_fundamental_capacity_medium_v1`，判断基本面扩维是否在更高容量下开始释放；最后再补 large 版本做边界确认。
- 训练命令示例：`python experiments/financial_transformer/train.py --config experiments/financial_transformer/config_moneyflow_v1_capacity_medium_v1.yaml`
- 最新固定阈值交易复核已经给出当前阶段结论：`moneyflow_v1_capacity_medium_v1` 的离线 `business_score` 只有约 `0.5093`，单纯扩容没有带来增量；而离线更强的 `moneyflow_fundamental_capacity_medium_v1` 虽然达到约 `0.5257`，但统一交易口径下最近区间单次回测约 `-8.00%`、Sharpe `-2.59`、最大回撤 `-11.27%`，rolling `10x5 / 15x5` 的平均窗口收益约 `-1.34% / -2.50%`。因此“新增信息维度需要同步扩容”这条结构假设在当前设定下暂未成立，不建议继续作为近期优先方向。

## 序列长度实验

- 默认仍然是固定 `60` 天窗口，也就是 `seq_len: 60`、`min_seq_len: 60`。
- 如果只想做“更长的固定窗口”，把 `seq_len` 和 `min_seq_len` 同时改成同一个更大值，例如 `90` 或 `120`。
- 如果想做“动态长度”，把 `seq_len` 设成最大窗口，把 `min_seq_len` 设成最短可接受历史长度。例如 `config_moneyflow_dynamic_120_v1.yaml` 表示最大看 `120` 天，但允许样本实际长度在 `60-120` 天之间。
- 当前实现已经会在训练时对补位部分加 mask，attention 和时间池化都不会把 padding 当成真实交易日。
- 现在也提供了 `window_search.py`，可以把窗口写成候选列表统一搜索。固定窗口直接写 `45,60,75,90`；动态窗口可以写成 `120:80,120:60` 这种 `seq_len:min_seq_len` 形式。

## 当前默认执行口径

- 当前默认单模型 Transformer 仍是资金流增强版 `moneyflow_v1`，模型主体不变，只是在输入特征里加入资金流维度。
- 当前最优执行覆盖层已经升级为两阶段重排：先用 `moneyflow_v1` 做候选池，再用 `moneyflow_selective_v1` 做 `0.55 / 0.45` 融合修正。
- 当前最优执行口径为：`short_up > 0.254`、`take_profit > 0.572`、`stop_loss < 0.50`、`top_k_per_day = 2`，候选池大小 `8`。
- 当前还冻结了一个保守执行分支：同样使用候选池 `8` 和 `0.55 / 0.45` 融合，但执行层改成 `top_k_per_day = 3` 且 `min_score_ratio_to_leader = 0.97`。这条线整窗收益略低于新的默认 leader 分层口径，但在滚动复核里仍给出了较高的平均 Sharpe 和较小的平均回撤。
- 退出纪律已经支持通过 `backtest.py` 与 `rolling_backtest.py` 显式传参做独立搜索；最近滚动感知搜索确认，默认执行纪律可从 `+12% / -6% / 45天` 小幅升级到 `+14% / -6% / 45天`。更激进的候选 `+16% / -2.5% / 20天` 虽然单窗更强，但滚动稳定性不足，因此仍只保留为候选。
- 进一步做的 leader 分层退出验证确认：如果只对当日第 `1` 名候选使用更激进的退出纪律 `+16.5% / -3.5% / 20天`，其余候选仍保持 `+14% / -6% / 35天`，并额外要求保留下来的跟随信号满足 `min_score_ratio_to_leader = 0.95`，则整窗收益可进一步提升到约 `+21.92%`、Sharpe `11.50`、最大回撤 `-1.19%`，同时继续保持 `11/11` 个滚动窗口全正。随后补做的 `20x5` / `25x5` 更长 OOS 滚动窗口也分别保持 `4/4`、`3/3` 全正，平均窗口收益约 `+9.44% / +13.03%`。旧的 `+16% / -4% / 20天` 和无 ratio 过滤的 `+16.5% / -3.5% / 20天` 都依然可行，但都已被这一更细化的执行纪律替代。
- 这代表“默认底座模型”和“当前最优执行平台”已经分开：底座仍是 `moneyflow_v1`，但执行层最强的是两阶段重排版本。
- 概率校准已经支持，但当前不作为默认口径，因为在公平窗口里它低于未校准的默认策略；只建议作为单独实验能力使用。

## 近期补充结论

### 复现性修复

- `moneyflow_v1` 历史结果和重算结果此前不一致，根因已经定位为股票池漂移，而不是模型数值漂移。
- 共同键 `stock_code/date` 上的预测数值原本几乎完全一致，但动态 `max_stocks=200` 会按最新市值快照重建股票池，导致整只股票被替换。
- 现在训练端已经支持保存 `selected_stock_universe.csv`，后续重算或过滤器实验应优先复用 frozen universe。

### 过滤器方向的最新判断

- 单模型 `moneyflow_v1` 上，第一版 `trade-quality filter` 的最佳阈值 `quality >= 0.45` 只会复现 no-filter 基线：`13` 笔交易、约 `+15.75%`、Sharpe `5.52`。
- 两阶段最优执行线 `two_stage_rerank_best_v1` 上，同样的第一版过滤器也只会复现 no-filter 基线：`9` 笔交易、约 `+20.08%`、Sharpe `10.08`。
- 因此这版过滤器当前不再是优先方向；后续若重启过滤器探索，必须先换标签定义或模型设定，而不是继续微调这版 logistic 过滤器。

### 缠论/中枢特征的最新判断

- 已完成两个版本的 Chan-style probe：`moneyflow_chan_v1` 是在资金流底座上叠加摆动高低点、趋势结构、中枢宽度/相对位置、背驰弱代理等特征；`moneyflow_chan_core_v1` 则进一步压缩到更核心的缠论结构信号。
- `moneyflow_chan_v1` 的结果说明这类技术结构特征不是完全没信息，它能给离线排序补进一点“结构感”，但在公平窗口里的最优可行回测也只有约 `+6.87%`、Sharpe `4.21`，仍显著弱于 `moneyflow_v1`。
- `moneyflow_chan_core_v1` 更能说明边界：当模型更大程度依赖缠论/中枢类信号时，交易层表现会明显变差，最优结果只有约 `+3.62%`、Sharpe `0.68`，且回撤显著放大。
- 当前最合理的定位是：这类特征可以保留为辅助特征池，但不应升级为默认核心方向，更不应替代资金流主线。

### 执行层的最新判断

- 当前最有效的增量仍来自执行层，而不是继续堆同型特征或微调第一版过滤器。
- `top_k=3 + min_score_ratio_to_leader=0.97` 继续作为保守执行分支保留。
- `leader-only exit` 则成为当前新的强候选升级：第 `1` 名使用 `+16.5% / -3.5% / 20天`，其余候选使用 `+14% / -6% / 35天`，并要求跟随信号至少达到 leader 分数的 `95%`。
- concept 分支的两阶段验证已经补完，而且连续两轮失败：`concept_sentiment_v2 + moneyflow_selective_v1` 固定回测约 `-1.53%`、`10x5` 仅 `4/6` 正窗；`concept_sentiment_v2 + concept_sentiment_selective_v1` 固定回测约 `-4.29%`、`10x5` 仅 `3/6` 正窗、`15x5` 仅 `2/5` 正窗。当前结论是：concept 方向可保留为“有信息的单模型 probe”，但不应再沿用现有两阶段执行配方继续深挖。

## 设计约束

- 标签定义已经对齐主系统多任务策略，但模型结构仍然是实验性质的 Transformer 版本，不代表主系统正式方案
- 只使用当前代码库可直接构造的特征，不引入额外外部数据
- 数据切分按预测日进行，适合 PoC 验证，不替代正式生产方案