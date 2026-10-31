# -*- coding: utf-8 -*-
"""RL 修复后的回归测试（知微 2026-10-31 审查）。

验证四件事：
  1. 策略梯度配对正确（无广播、无零梯度）
  2. 环境不会越界到测试区间
  3. 账本逐日正确（隔夜/日内/成本/净值）
  4. 绩效口径正确（夏普用收益率、回撤含初始资本）

全部 PASS 才算修复完成。
"""

from __future__ import annotations

import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_algo_utils import (paired_policy_loss, discounted_returns,
                           assert_paired, stack_logps)
from rl_ledger import run_ledger, ledger_stats

RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


print("=" * 76)
print("RL 修复回归测试")
print("=" * 76)

# ============================================================
# 1. 策略梯度配对
# ============================================================
print("\n[1] 策略梯度配对")


class P(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Linear(4, 8)
        self.o = nn.Linear(8, 3)

    def forward(self, x):
        return torch.softmax(self.o(torch.tanh(self.h(x))), dim=-1)


pol = P()
lps = []
for _ in range(5):
    x = torch.randn(1, 4)
    d = Categorical(pol(x))
    lps.append(d.log_prob(d.sample()).squeeze(-1))     # 标量
ret = torch.tensor([-0.02, 0.01, 0.03, -0.01, 0.0])
loss = paired_policy_loss(lps, ret, normalize=True)
pol.zero_grad()
loss.backward()
gn = sum(q.grad.norm().item() ** 2 for q in pol.parameters() if q.grad is not None) ** 0.5
check("配对后梯度非零", gn > 1e-6, f"|g|={gn:.4e}")

# 形状必须 [T]
lp_t = stack_logps(lps)
check("stack_logps 输出 1 维", lp_t.ndim == 1, f"shape={tuple(lp_t.shape)}")

# 错误用法必须抛错：单个 log_prob 里塞了多个元素（典型的 batch 维未 squeeze）
try:
    bad = [torch.tensor([0.1, 0.2]), torch.tensor([0.3])]
    assert_paired(bad, torch.tensor([1.0, 2.0]))
    check("多元素 log_prob 能被捕获", False, "未抛异常")
except ValueError:
    check("多元素 log_prob 能被捕获", True, "抛出 ValueError")

# [1] 形状（单元素）是允许的，会被 stack 压平
try:
    assert_paired([torch.tensor([0.1]), torch.tensor([0.2])],
                  torch.tensor([1.0, 2.0]))
    check("[1] 形状被允许", True)
except ValueError as e:
    check("[1] 形状被允许", False, str(e))

# 长度不一致必须抛错（v5 的 K != T 问题）
try:
    assert_paired([torch.tensor(0.1)] * 3, torch.tensor([1.0, 2.0]))
    check("长度不一致能被捕获", False, "未抛异常")
except ValueError:
    check("长度不一致能被捕获", True, "抛出 ValueError")


# ============================================================
# 2. 环境时间隔离
# ============================================================
print("\n[2] 环境时间隔离")

import rl_timing_v4_variable_holding as V4

N, N_TRAIN = 1000, 750
feats = np.random.randn(N, V4.N_FEATS).astype(np.float32)
open_ = np.random.uniform(10, 12, N)
close = open_ * (1 + np.random.randn(N) * 0.01)
mu, sd = feats.mean(0), feats.std(0) + 1e-8

env = V4.Env(feats[:N_TRAIN], open_[:N_TRAIN], close[:N_TRAIN],
             0.0003, 0.0, 1.0, 50, mu, sd, n_allowed=N_TRAIN)
np.random.seed(7)
starts = []
for _ in range(500):
    env.reset()
    starts.append(env.t)
in_test = sum(1 for s in starts if s >= N_TRAIN)
check("训练环境不越界到测试区间", in_test == 0, f"{in_test}/500 次")

# reset(start=) 与 t 覆盖应一致
obs_a = env.reset(start=100)
env_t_100 = env.t
env.reset(start=100)
obs_b = env._obs()
check("reset(start=) 状态一致", np.allclose(obs_a, obs_b) and env_t_100 == 100)


# ============================================================
# 3. 持仓期回撤
# ============================================================
print("\n[3] 持仓期回撤")

open2 = np.full(200, 10.0)
close2 = np.full(200, 10.0)
close2[20:25] = [10.0, 7.5, 5.0, 7.5, 10.0]          # 中间跌 50%
open2[25] = 10.0
env2 = V4.Env(feats[:200], open2, close2, 0.0000, 10.0, 0.0, 100, mu, sd, n_allowed=200)
env2.reset(start=18)
env2.prev_a = 0
a_idx = V4.HOLD_OPTIONS.index(5)
_, reward, _ = env2.step(a_idx)
check("持仓中间回撤触发惩罚", reward < -0.01, f"reward={reward:.4f}")


# ============================================================
# 4. 账本逐日核对
# ============================================================
print("\n[4] 账本逐日核对")

n = 6
dates = list(range(n))
o = np.array([10.0, 10.2, 10.0, 10.5, 10.3, 10.8])
c = np.array([10.1, 10.1, 10.4, 10.2, 10.7, 10.9])
pc = np.array([10.0, 10.1, 10.1, 10.4, 10.2, 10.7])
targets = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

r = run_ledger(dates, o, c, pc, targets, cost=0.0, slip=0.0)
# 时序：nav[i] 对应第 i+1 天（第 0 天无调仓，净值起点为 1）
# 第1天：targets[0]=1, 昨仓位0 → 不吃隔夜(10.2/10.1)，吃日内(10.1/10.2)
exp1 = (1 + 0.0 * (10.2 / 10.1 - 1)) * (1 + 1.0 * (10.1 / 10.2 - 1))
check("第1天净值", abs(r.nav[0] - exp1) < 1e-12, f"{r.nav[0]:.10f} vs {exp1:.10f}")

# 第2天：targets[1]=1, 昨仓位1 → 吃隔夜(10.0/10.1)，吃日内(10.4/10.0)
exp2 = exp1 * (1 + 1.0 * (10.0 / 10.1 - 1)) * (1 + 1.0 * (10.4 / 10.0 - 1))
check("第2天净值（持有日吃全天）", abs(r.nav[1] - exp2) < 1e-12,
      f"{r.nav[1]:.10f} vs {exp2:.10f}")

# 第3天：targets[2]=0, 昨仓位1 → 【吃隔夜】(10.5/10.4)，不吃日内
exp3 = exp2 * (1 + 1.0 * (10.5 / 10.4 - 1)) * (1 + 0.0 * (10.2 / 10.5 - 1))
check("第3天净值（卖出日吃隔夜）", abs(r.nav[2] - exp3) < 1e-12,
      f"{r.nav[2]:.10f} vs {exp3:.10f}")

# 第4天：targets[3]=1, 昨仓位0 → 不吃隔夜，吃日内(10.7/10.3)
exp4 = exp3 * (1 + 0.0 * (10.3 / 10.2 - 1)) * (1 + 1.0 * (10.7 / 10.3 - 1))
check("第4天净值（买入日不吃隔夜）", abs(r.nav[3] - exp4) < 1e-12,
      f"{r.nav[3]:.10f} vs {exp4:.10f}")

# 成本只在实际换仓时收
r2 = run_ledger(dates, o, c, pc, targets, cost=0.001, slip=0.0)
check("成本只在换仓时收", r2.trades == 3, f"trades={r2.trades}")
expect_ratio = (1 - 0.001) ** 3
check("成本金额正确", abs(r2.nav[-1] / r.nav[-1] - expect_ratio) < 1e-9,
      f"{r2.nav[-1] / r.nav[-1]:.8f} vs {expect_ratio:.8f}")


# ============================================================
# 5. 绩效口径
# ============================================================
print("\n[5] 绩效口径")

# 净值 [0.9, 0.81]：真实回撤 -19%
s = ledger_stats(np.array([0.9, 0.81]))
check("回撤含初始资本 1.0", abs(s["dd"] - (-0.19)) < 1e-12, f"{s['dd']:.4f}")

# 夏普必须是收益率：净值恒定增长 10% 两天
s2 = ledger_stats(np.array([1.1, 1.21]))
check("恒定收益夏普为 nan（无波动）", np.isnan(s2["sharpe"]), f"{s2['sharpe']}")

s3 = ledger_stats(np.array([1.1, 1.05, 1.2]))
manual = np.array([0.10, 1.05 / 1.1 - 1, 1.2 / 1.05 - 1])
exp_sharpe = manual.mean() / manual.std(ddof=1) * np.sqrt(244)
check("夏普用收益率计算", abs(s3["sharpe"] - exp_sharpe) < 1e-9,
      f"{s3['sharpe']:.4f} vs {exp_sharpe:.4f}")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 76)
passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print(f"结果: {passed}/{total} PASS")
print("=" * 76)
if passed != total:
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  ({detail})")
    sys.exit(1)
print("全部通过")
