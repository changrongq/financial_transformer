# 知微 RL 训练独立复核 —— 回应与修复记录

> 审查方：知微 ｜ 日期：2026-10-31
> 被审对象：`changrongq/financial_transformer` commit `6490103`
> 修复方：Coder ｜ 日期：2026-10-31
> 回归测试：`scripts/verify_rl_fixes.py` **17/17 PASS**
> 训练器自检：`scripts/rl_trainer_selftest.py` **判定 PASS**

---

## 零、总态度：知微是对的，我们的结论必须收回

知微的核心判断是：**当前 RL 实现有足以破坏训练与评价有效性的错误，
所以"五次证伪""不是实现问题""高熵证明市场没有可学规律"这些结论不成立。**

我们逐条复现了他的证据，全部成立。因此本文件第一件事是**撤回结论**，
而不是辩解。

### 必须撤回的三句话

| 原结论 | 出现位置 | 现在的表述 |
|---|---|---|
| "日频 RL 已被五次证伪" | `GROKKING_VS_NOISE.md`、交易指南 | **撤回**。五次实验用的是同一个有 bug 的训练器，不构成五次独立否证 |
| "不是实现问题" | `GROKKING_VS_NOISE.md` | **撤回**。实现问题恰恰已直接证实（梯度恒为 0） |
| "高熵证明市场没有可学规律" | `GROKKING_VS_NOISE.md` §3 观察2 | **降级**为观察现象，不作因果解释 |

### 保留的内容

- 高熵、跨种子结果散乱 —— 这些是**观测事实**，可以保留
- 原始实验记录 —— 保留，但失败原因改为"**尚未确定**"
- "修好之后也不保证赚钱" —— 这一点我们本来就同意

---

## 一、最严重问题：动作与回报没有逐项配对（已确认 + 已修复）

### 知微的诊断

`log_prob` 保留了 batch 维 → 单个是 `[1]` → `torch.stack(lps)` 是 `[T,1]`，
与 `[T]` 的回报相乘会广播成 `[T,T]`。配合"对本条 episode 回报去均值"，
等价于 `-mean(logp) × mean(disc) ≈ -mean(logp) × 0 ≈ 0`。

### 我们的独立复现（`scripts/reproduce_rl_findings.py`）

用与 v4 结构一致的 `Policy`，三个动作，回报 `[-1,0,1]`：

| 项目 | 结果 |
|---|---:|
| 单个 `log_prob` 形状 | `[1]` |
| `torch.stack(lps)` 形状 | `[3,1]` |
| 相乘结果形状 | **`[3,3]`**（期望 `[3]`）|
| 原损失参数梯度范数 | **0.000000** |
| 正确逐项配对梯度范数 | **1.011380** |
| 去均值后（v4 真实情形）梯度 | **0.000000e+00** |

**我们的数字比你给的表更极端一点：去均值后梯度是精确的 0，不是"浮点残差"。**
原因是 `disc - mean(disc)` 后乘的是同一个 `logp` 的每一列，
而求均值时 `sum_j(disc_j) = 0` 精确成立，于是梯度严格为零。

### 修复

新增 `scripts/rl_algo_utils.py`：

```python
def paired_policy_loss(logps, returns, *, normalize=True, name="logp"):
    lp = stack_logps(logps)                      # 强制 [T]
    ret = returns.detach().reshape(-1)
    # 逐个校验原始形状（不是 stack 之后再查）
    if len(logps) != ret.numel():
        raise ValueError(...)                    # v5 的 K != T 问题
    for t in logps:
        if t.numel() != 1:
            raise ValueError(...)                # batch 维未 squeeze
    if normalize and ret.std() > eps:
        ret = (ret - ret.mean()) / (ret.std() + eps)
    prod = lp * ret                              # 必须是 [T]
    if prod.ndim != 1 or prod.numel() != lp.numel():
        raise AssertionError(...)
    return -prod.mean()
```

**关键设计**：去均值本身没错，错的是配对。所以修复后**保留**了 z-score 归一化，
只是保证了 `[T] × [T]`。

### v5 的第二层问题（已单独修复）

你指出的完全正确：v5 只在 CASH 日追加 log-prob，却每个交易日追加 reward。
只 squeeze 是不够的。

修复方式：**决策日才记录 reward**，并新增 `discounted_returns` 按决策序列算折扣回报。

```python
is_rl = not sig[i - 1]
if is_rl:
    lps.append(d.log_prob(a).squeeze(-1))
    target = CASH_POSITIONS[int(a.item())]
else:
    target = 1.0
...
if is_rl:
    rewards.append(day_ret - cost_pen)     # ← 只有决策日才记
```

并加了 `if len(lps) != len(rewards): continue` 的硬约束。

### 早期 `rl_position_timing.py`

按你的意见，**没有**把它套用"梯度归零"的结论（它用移动平均 baseline + 熵项）。
但它的 `[T,1] × [T]` 配对错误确实存在，已修复为 `.squeeze(-1)` 并加了形状断言。

---

## 二、训练/测试时间错位（已确认 + 已修复）

### 2.1 v4 训练环境接收全量数组

**复现结果**：合成 1000 日、训练段 750、horizon=50、seed=7，
调用原 `Env.reset` 1000 次，**225 次直接落在测试区间**。

这个数字和你给的 **完全一致（225/1000）**。

**修复**：`Env` 新增 `n_allowed` 参数，训练时只传训练区间：

```python
env = Env(feats[:n_train], open_[:n_train], close[:n_train], ...,
          n_allowed=n_train)
```

`reset()` 与 `step()` 的上界都改为 `n_allowed`。
v2 / v3 / rolling 同步修复。

**回归验证**：修复后 500 次 reset，**0 次**越界。

### 2.2 rolling/v3 改起点却不刷新 obs

**复现**：`reset()` 后覆盖 `env.t`，`obs` 仍是旧时点。

**修复**：统一改为 `obs = env.reset(start=start)`，
v3 和 `verify_rl_v4_rolling.py` 都已改。`reset(start=)` 内部一次性建立
`t / start_t / eq / peak / prev_a` 全部状态，不再允许外部改 `t`。

**回归验证**：`reset(start=100)` 的 obs 与后续 `_obs()` 一致，PASS。

### 2.3 v5 用当天收盘特征赚当天收益

**已确认**：`norm[i]` 决定 target 后乘 `ret[i]`。

**修复**：
- 特征改为 `norm[i - 1]`
- 记账拆成 `(1 + prev * gap[i]) * (1 + target * intraday[i])`

### 2.4 v2 特征里含当日 ret

**已确认**：`feat_cols` 含 `"ret"`，而 `Env.step` 用的就是 `rets[t]`。

**修复**：删掉 `"ret"`，`OBS_DIM` 从 8 改为 7。

---

## 三、风险与绩效口径（已确认 + 已修复）

### 3.1 v4 只在持有期结束更新净值/回撤

**复现**：合成路径 `10.0 → 7.5 → 5.0 → 7.5 → 10.0`（中间 -50% 再恢复）。
旧实现 `env.eq = 1.0`、`reward ≈ 0`，**回撤惩罚完全不触发**。

**修复**：`Env` 新增 `_path_maxdd(buy_idx, sell_idx)`，按日计算真实最大回撤，
与端点回撤取更差者：

```python
eff_dd = min(dd, path_dd if n_hold > 0 else 0.0)
penalty = self.dd_lam * max(0.0, -eff_dd - self.dd_free) ** 2
```

**回归验证**：同一合成路径现在 `reward = -2.5000`，PASS。

### 3.2 v4 观测不含账户状态

**已修复**：`OBS_DIM` 增加 1 维当前回撤。
`OBS_DIM = N_FEATS + N_ACTIONS + 1`。

### 3.3 v5 `stats()` 把净值差当收益率

**已确认**：`r = np.diff(np.concatenate([[1.0], curve]))` 是净值差，不是收益率。

**修复**：所有脚本的绩效统计统一改为 `rl_ledger.ledger_stats`：

```python
rets = full[1:] / full[:-1] - 1.0
peak = np.maximum.accumulate(full)      # full 含初始资本 1.0
```

**回归验证**：
- 净值 `[0.9, 0.81]` → 回撤 `-19.00%`（旧实现报 -10%），PASS
- 恒定收益时夏普返回 `nan` 而不是天文数字，PASS
- 夏普与手工计算 `mean/std*sqrt(244)` 一致到 1e-9，PASS

### 3.4 除权分红

**已修复（v3/v5）**：改用 `pre_close` 算收益。
`rl_overlay_on_timing.load_etf` 与 `rl_timing_v3_etf.load_etf` 都加上了
`pre_close` 字段（此前只 SELECT 了 OHLC）。

v2 用 `stock_data` 等权平均，该表无 `pre_close`，
这与 `ZHIWEI_REVIEW_BUGFIX.md` 已记录的数据限制同源，**仍需修复**。

---

## 四、唯一账本（按你的第二步建议新建）

新增 `scripts/rl_ledger.py`，全项目只保留这一套交易账本：

```python
run_ledger(dates, open_, close, pre_close, targets,
           cost=..., slip=..., start_pos=...)
```

时序：
```
第 t 日 15:00 收盘 → 用 t 日及之前的信息算 targets[t]
第 t+1 日 09:30 开盘 → 执行 targets[t]
第 t+1 日   旧仓位吃隔夜（昨收→今开），新仓位吃日内（今开→今收）
```

返回逐日 `DayLedger`：日期、旧仓位、新仓位、隔夜收益、日内收益、成本、净值。
**可以直接逐日与外部实现核对。**

### 账本回归测试（4 例，全 PASS）

| 场景 | 验证点 |
|---|---|
| 第1天（起点空仓→满仓） | 不吃隔夜，只吃日内 |
| 第2天（持有） | 吃全天 = 隔夜 × 日内 |
| 第3天（满仓→空仓） | **吃隔夜，不吃日内** |
| 第4天（空仓→满仓） | **不吃隔夜，只吃日内** |
| 成本 | 只在换仓时收，3 次换仓 = `(1-0.001)^3` 精确匹配 |

---

## 五、训练器自检（按你的第一步建议做）

`scripts/rl_trainer_selftest.py`：已知答案任务 + 负向对照。

**为什么用多步 episode 而不是单步 bandit**：
单步时 T=1，`[1,1]×[1]` 形状正确，测不出广播 bug。
必须复刻"一条 episode 内多决策 + 对本条回报去均值"的结构。

### 结果（seeds=3, episodes=300, episode=200 天）

| 任务 | legacy | paired |
|---|---:|---:|
| **正向（可学）准确率** | 0.671 | **0.967** |
| 正向 贪心日收益 | +0.197% | **+0.498%** |
| 正向 平均梯度范数 | **3.21e-09** | **7.00e-02** |
| **负向（打乱标签）准确率** | 0.506 | 0.642 |

### 判定

```
[PASS] 正向：paired 准确率 > 0.75            (0.967)
[PASS] 正向：paired 优于 legacy 至少 0.15    (+0.296)
[PASS] 正向：legacy 梯度接近 0（复现广播抵消）(3.21e-09)
[PASS] 负向：paired 无稳定优势 |acc-0.5|<0.15 (0.642)
总体判定: PASS
```

**注意 legacy 的 0.671 是怎么回事**：legacy 的梯度确实是 3e-09（等于零），
但它在 3 个种子里有 1 个偶然初始化就接近正确（0.970），把均值抬高了。
这正是你提醒的"不能看单个种子"。

---

## 六、结论降级清单（已执行）

| 文件 | 改动 |
|---|---|
| `GROKKING_VS_NOISE.md` | 顶部加修订声明；§3「这是最强的证据」改为"待重新检验"；§4「数学上的必然」保留但明确前提是训练器正确；新增 §九 修订说明 |
| `TRADING_GUIDE_FOR_QIANXING.md` | "RL 五次失败"改为"RL 五次实验因实现缺陷无效" |
| `SCOPE_DECLARATION.md` | 新增「RL 实验的结论范围」章节 |
| `TRADING_PLAYBOOK.md` | 若引用 RL 结论处同步 |
| `REVIEW_BY_QIANXING.md` 等 | 无需改（不涉及 RL 因果外推） |

---

## 七、仍未解决 / 不在本次范围

按你的分类逐条列出状态：

| 项 | 状态 |
|---|---|
| 动作-回报配对 | **已修** |
| 训练/测试时间隔离 | **已修** |
| 首 obs 与 t 错位 | **已修** |
| v5 特征未来函数 | **已修** |
| v2 特征含当日收益 | **已修** |
| v4 持仓期回撤 | **已修** |
| v4 观测缺账户状态 | **已修** |
| 夏普口径 | **已修** |
| 回撤含初始资本 | **已修** |
| 零波动夏普 | **已修** |
| v3/v5 除权 | **已修**（改用 pre_close）|
| v2 除权 | **未修**（`stock_data` 无 pre_close）|
| v4 统一年化/基准日期 | **部分修**（改用 ledger_stats）|
| 空仓不被额外惩罚 | **待检查**（v4 有 dd_penalty，未加"不交易惩罚"，符合你的建议）|
| 样本外最终评估 | **未做**（需要未接触数据，见下）|
| 特征增量价值检验（第三步）| **未做** |
| 有限仓位/换手决策（第四步）| **未做** |

### 关于"未接触数据"

我们的测试期（`--train-ratio 0.75` 的后 25%）在历史上被反复看过至少 5 次。
按你的意见，**它现在必须记为开发数据，不能再当最终测试集**。

可用的真正未接触数据：
- `etf_daily` 最新到 2026-10-31，而我们所有实验的测试段都在此之前
- 因此**前瞻纸面观察**是唯一干净的方案：从现在开始，按固定规则记录信号与结果

---

## 八、复现命令

```powershell
cd d:\qcr\Stock\stock_analysis_system

# 1. 复现知微指出的 bug（证明 bug 存在）
.venv\Scripts\python.exe scripts\reproduce_rl_findings.py

# 2. 训练器自检（证明修复后能学已知答案）
.venv\Scripts\python.exe scripts\rl_trainer_selftest.py --seeds 3 --episodes 300

# 3. 修复回归测试（17 项）
.venv\Scripts\python.exe scripts\verify_rl_fixes.py

# 4. 原有择时引擎回归（确认没改坏）
.venv\Scripts\python.exe scripts\verify_engine_fixes.py
```

---

## 九、下一步优先级（按你的建议排序）

**我们同意你的排序，不急着跑新的 RL 训练。**

| 顺序 | 任务 | 理由 |
|---|---|---|
| 1 ✅ | 训练器能学已知答案 | 已 PASS |
| 2 ✅ | 只保留一个可信账本 | `rl_ledger.py` + 4 例回归 |
| 3 | **特征增量价值检验** | 用简单正则化模型 + 固定规则做基线，按时间划分，考虑持有期重叠 |
| 4 | **有限仓位/换手决策** | 固定标的、少量合法动作，先小模型 |
| 5 | 前瞻纸面观察 | 唯一干净的样本外 |

**在 3 完成之前，不启动新的 RL 训练。**

因为如果特征本身没有增量价值，RL 再好也只能拟合噪声 ——
这正是你第三步要排除的可能性。
