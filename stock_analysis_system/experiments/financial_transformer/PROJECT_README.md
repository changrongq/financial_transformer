# Financial Transformer · A股量化研究

一组关于**A股预测与择时**的系统性实验，包含完整的**假设验证 → 实验 → 结论**链条。

> ⚠️ **2026-10-31 重要修订**
>
> 知微复核后发现原"个股预测"研究链存在 4 个未排除的实现缺陷
> （标签跨期、不完整标签、**输入尺度淹没特征**、回测 T+1 违规）。
> 我逐条复现确认，其中最致命的是输入尺度：
> 把 `ret5d` 从 -0.1 改到 +0.1，原模型输出差 **0.0000000000**。
>
> **因此下文"预测个股方向几乎不可行"这一结论目前不成立**，
> 降级为「在这组实现下的观察，原因待查」。
>
> 详见 [ZHIWEI_TRANSFORMER_REVIEW_BUGFIX.md](../../ZHIWEI_TRANSFORMER_REVIEW_BUGFIX.md)。
> 修复件：[`scripts/transformer_fixes.py`](../../scripts/transformer_fixes.py)，
> 验证 17/17 PASS。
>
> **择时部分（ETF）不受影响**，因为那条线用的是 `etf_daily`，
> 有独立的引擎与回归测试（`verify_engine_fixes.py` 4/4 PASS）。

> 原核心结论：~~预测个股涨跌几乎不可行（技术形态因子全部失效），
> 但"择时躲暴跌"是唯一经得起逐年检验的稳定超额来源。~~
>
> 修正后：**择时部分经知微三轮审查后已修正为「3/6 收益胜、6/6 回撤胜、
> 主标的跑输」，个股预测部分待重新验证。**

---

## 一、这个项目在研究什么

从"能不能用深度学习预测个股涨跌"出发，逐步验证了几乎所有类型的信号，最终收敛到"择时 + 指数ETF"这个可落地的方向。

> ⚠️ 上面这句是**原始研究意图**。2026-10-31 复核后，
> "个股预测"这条线因实现缺陷**尚未得出有效结论**。

### 研究主线

```
换手率修正 → 模型输出塌缩诊断 → 五轮结构实验 → 因子可预测性分层
→ 日历因子探索 → 择时策略 → 情绪 vs 规则对照实验
→ [2026-10-31] 知微复核：实现缺陷 → 修复测量工具
```

---

## 二、核心结论（均有数据支撑）

> ⚠️ 本节前两条涉及个股预测，**已在 2026-10-31 降级**，见顶部修订说明。

### 1. 预测个股方向：~~几乎不可行~~ **待重新验证**

| 因子/模型 | 与未来收益的相关性 |
|---|---|
| 技术动量因子 | IC ≈ 0.07（近噪声）|
| Transformer（micro，32特征）| 测试集 AUC 0.435（反向随机）|
| 台风模型 O 值 | Pearson +0.0014（近零）|
| 台风持续模型 persistence_prob | Spearman **-0.16（负相关）** |

**为什么？** ~~股票收益是「非线性、对称、条件化」的——同样的技术形态，涨跌都可能。用二分类标签去拟合对称系统，必然塌缩到 0.5。~~

> ⚠️ **2026-10-31 修订**：上面这句因果解释**已撤回**。
>
> 知微复核后发现：默认 32 特征里金额/数量与收益率量级差 **9 个数量级**，
> 而 flat 分支先 `Linear` 混合再 `LayerNorm`。复现结果：
> 把 `ret5d` 从 -0.1 改到 +0.1，原模型输出差 **0.0000000000**；
> 逐列标准化后差 **0.039**。
>
> **「塌缩到 0.5」首先是一个输入尺度问题，其次才轮得到"目标定义问题"。**
> 上表的 AUC 0.435 等数字来自有缺陷的实现，**不能用来支持"不可预测"**。
>
> 详见 `ZHIWEI_TRANSFORMER_REVIEW_BUGFIX.md`。

### 2. 换手率修正（有价值的数据工程）

- 官方"流通股"≠"真正会交易的股"：国资/实控人长锁筹码事实不流通
- 真实换手率普遍是名义换手率的 **1.6–2.5 倍**
- 实现：`src/hierarchical_system/free_float.py`

### 3. 预测波动幅度：排序可测、量化不可测

| 预测目标 | IC | MAE 改善 |
|---|---|---|
| 方向 | 0.07 | 不可预测 |
| **波动幅度** | **0.28** | 仅 4% |

能判断"谁波动大"（排序），但难预测"具体波动多少"（幅度）。

### 4. 日历/季节因子：几乎全部失效

| 因子 | 逐年一致性 |
|---|---|
| 旅游×节假日提前期 | ❌ 多数年份为负 |
| 电力×夏季 | ❌ 是"高股息风格"非季节 |
| 家电×夏季 | ❌ 无效 |
| 农业×两会 | ⚠️ 弱正但不稳定 |

**教训**：只看"跨年累计均值"会严重误导，必须**逐年检验**。

### 5. 择时躲暴跌：唯一稳定有效 ⭐

规则：**大盘 10 日均收益 < 0 → 空仓**

| 区间 | 满仓 | 择时 |
|---|---|---|
| 全期（431天）| +29.5%，回撤 -23.9% | **+152%，回撤 -7.4%** |
| 近一年 | +7.0% | **+59.2%** |

- 参数敏感性：**16/16 组合全部跑赢满仓**
- 暴跌躲避率：**30 个暴跌日躲开 24 个（80%）**

### 6. 情绪 vs 规则对照实验（回答一个认知问题）

同一套行情、同一套成本，三种智能体 2019-2026 表现：

| 智能体 | 总收益 | 最大回撤 | 亏损后仓位 | 盈利后仓位 |
|---|---|---|---|---|
| **RULE 规则型**（按规则）| **+822%** | **-9.2%** | — | — |
| 满仓持有 | +58% | -27.3% | — | — |
| EMOTION 情绪化 | +95% | -7.1% | **14%** | **64%** |
| CONST 恒定型 | +29% | -14.2% | — | — |

**RULE 逐年 8/8 全胜情绪化。** 情绪化的行为模式就是"高买低卖"：亏损后仓位降到 14%（割肉踏空）、盈利后加到 64%（追高被套）。

> 情绪在交易中是**包袱**——不是因为情绪本身坏，而是它干扰纪律。但"完全不思考"（恒定型）更糟。

---

## 三、关键文档

| 文档 | 内容 |
|---|---|
| `experiments/financial_transformer/MICRO_MODEL_DIAGNOSIS_REPORT.md` | 完整诊断报告（从换手率失真到 ETF 落地） |
| `experiments/financial_transformer/NONLINEAR_MODEL_UPGRADE_PLAN.md` | 非线性/对称性问题与模型升级方案 |
| `experiments/financial_transformer/CALENDAR_FACTOR_REPORT.md` | 日历因子探索全过程 |
| `experiments/financial_transformer/ETF_TIMING_USER_GUIDE.md` | ETF 择时落地使用说明 |
| `experiments/financial_transformer/TYPHOON_BACKTEST_NOTE.md` | 台风模型回测说明 |

---

## 四、核心代码

### 数据工程
- `src/hierarchical_system/free_float.py` — 真实自由流通盘 / 真实换手率
- `scripts/fetch_free_float.py` — 股东结构批量抓取
- `scripts/fetch_historical_daily.py` — 历史日线回补

### 择时策略（核心产出）
- `experiments/financial_transformer/daily_regime_timing_signal.py` — 每日 HOLD/CASH 信号
- `experiments/financial_transformer/review_regime_timing_signal.py` — 三维度复盘
- `experiments/financial_transformer/run_regime_timing_backtest.py` — 择时回测
- `experiments/financial_transformer/signal_report.py` — 集成到日报

### 实验
- `experiments/financial_transformer/compare_emotion_vs_rules.py` — 情绪 vs 规则对照
- `experiments/financial_transformer/rl_position_timing.py` — RL 仓位择时实验
- `experiments/financial_transformer/typhoon_tracker.py` — 台风模型
- `experiments/financial_transformer/typhoon_persistence_model.py` — 持续预测模型

---

## 五、环境与运行

### 依赖
```bash
pip install -r requirements.txt
```

### Token 配置（必需）
Tushare token 不硬编码在代码中。通过以下任一方式提供：

```powershell
# 方式1：环境变量
$env:TUSHARE_TOKEN = "your_token"

# 方式2：本地文件（不入版本控制）
# 将 token 写入 config/tushare_token.local
```

### 快速开始

```bash
# 1. 生成每日择时信号
python experiments/financial_transformer/daily_regime_timing_signal.py

# 2. 复盘信号质量
python experiments/financial_transformer/review_regime_timing_signal.py

# 3. 情绪 vs 规则对照实验
python experiments/financial_transformer/compare_emotion_vs_rules.py
```

---

## 六、数据说明

**本仓库不含数据文件**（数据库、模型权重、回测结果均未入库）。数据需自行通过 `scripts/` 下的脚本从 Tushare 拉取。

---

## 七、方法论沉淀

本项目最重要的产出不是某个模型，而是**验证方法论**：

1. **逐年一致性检验** — 任何因子只看"几年平均"都会被个别大年份误导，必须逐年分解
2. **区分"逻辑正确"与"数据支持"** — 很多假设逻辑上成立（如"节前炒旅游"），数据上不成立
3. **警惕"假象"** — 反向信号、累计均值、单因子 IC 高，都可能是噪声或过拟合
4. **诚实标注负结果** — 大部分实验是失败的，失败结论同样有价值

---

## 免责声明

本项目为个人量化研究，所有结论基于历史数据回测，**不构成任何投资建议**。历史表现不代表未来收益。
