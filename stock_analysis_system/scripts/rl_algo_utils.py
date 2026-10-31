# -*- coding: utf-8 -*-
"""RL 训练器正确性工具（知微 2026-10-31 审查后新增）。

本模块只负责两件事：

  1. `paired_policy_loss`  —— 保证「每个动作 ↔ 它实际产生的回报」严格逐项配对。
  2. `assert_paired`       —— 训练入口的强制断言，防止 [T,1]×[T] 广播 bug 复现。

背景
----
旧实现：

    lps = []                       # 每次 d.log_prob(a)，a 来自带 batch 维的分布
    ...                            # 单个 log_prob 形状是 [1]
    loss = -(torch.stack(lps) * disc).mean()

`torch.stack(lps)` 是 [T,1]，`disc` 是 [T]，相乘按 PyTorch 广播规则变成 [T,T]。
在 disc 被去均值（v4/v5 都这么做）时，等价于

    - mean(log_prob) × mean(disc) ≈ - mean(log_prob) × 0 ≈ 0

策略梯度因此被抵消。实测该损失对参数的梯度范数为 0.000000，
而正确配对时为 1.011380（见 scripts/reproduce_rl_findings.py）。

用法
----
    from rl_algo_utils import paired_policy_loss, assert_paired

    logps = []                     # 每个元素都是标量 tensor
    ...
    logps.append(d.log_prob(a).squeeze(-1))     # [1] -> []
    loss = paired_policy_loss(logps, returns, normalize=True)
"""

from __future__ import annotations

from typing import Sequence

import torch


def assert_paired(logps: Sequence[torch.Tensor],
                  returns: torch.Tensor,
                  *, name: str = "logp") -> None:
    """强制检查 log_prob 与回报是 1 维、同长度、逐项对应。

    任何形状不符立刻抛错，而不是靠广播静默算出一个错误的值。
    """
    returns = returns.reshape(-1)
    if len(logps) != returns.numel():
        raise ValueError(
            f"[{name}] 动作数 {len(logps)} != 回报数 {returns.numel()}。"
            "v5 的问题就是只在 CASH 日追加动作、却每天追加回报。")
    for i, lp in enumerate(logps):
        # 严格：每个元素必须只有 1 个数（标量或 [1]），且不能是多元素张量
        if lp.numel() != 1:
            raise ValueError(
                f"[{name}] 第 {i} 个 log_prob 有 {lp.numel()} 个元素，"
                "期望 1 个。原因通常是 `d.log_prob(a)` 保留了 batch 维，"
                "请改用 `.squeeze(-1)` 或构造不带 batch 维的分布。")
    if returns.ndim != 1:
        raise ValueError(f"[{name}] 回报必须是 1 维，实际 {tuple(returns.shape)}")


def stack_logps(logps: Sequence[torch.Tensor]) -> torch.Tensor:
    """把一批标量 log_prob 堆成形状 [T] 的一维张量。"""
    return torch.stack([lp.reshape(()) for lp in logps])


def paired_policy_loss(logps: Sequence[torch.Tensor],
                       returns: torch.Tensor,
                       *, normalize: bool = True,
                       eps: float = 1e-8,
                       name: str = "logp") -> torch.Tensor:
    """REINFORCE 损失，严格的 [T] × [T] 逐项配对。

    参数
    ----
    logps      : 每个动作的 log_prob（标量，或形状 [1] 会被压平）
    returns    : 与 logps 同长度的回报（或优势），形状 [T]
    normalize  : 是否对优势做 z-score。注意：去均值本身没有错，
                错的是去均值 + 广播把梯度整体抵消。配对正确后可以放心去均值。
    """
    lp = stack_logps(logps)
    ret = returns.detach().reshape(-1).to(lp.dtype)

    # ★ 关键：在 reshape 之前逐个校验原始形状，不能先 stack 再检查
    if len(logps) != ret.numel():
        raise ValueError(
            f"[{name}] 动作数 {len(logps)} != 回报数 {ret.numel()}（v5 的 K!=T 问题）")
    for i, t in enumerate(logps):
        if t.numel() != 1:
            raise ValueError(
                f"[{name}] 第 {i} 个 log_prob 有 {t.numel()} 个元素，"
                "期望 1 个。通常是 `d.log_prob(a)` 保留了 batch 维，"
                "请改用 `.squeeze(-1)` 或构造不带 batch 维的分布。")

    if normalize and ret.numel() > 1 and ret.std() > eps:
        ret = (ret - ret.mean()) / (ret.std() + eps)

    prod = lp * ret          # 形状必须是 [T]
    if prod.ndim != 1 or prod.numel() != lp.numel():
        raise AssertionError(
            f"[{name}] 配对后形状异常: lp={tuple(lp.shape)} ret={tuple(ret.shape)} "
            f"prod={tuple(prod.shape)}")

    return -prod.mean()


def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> torch.Tensor:
    """从后往前算折扣回报，返回与 rewards 同序的 [T] 张量。"""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return torch.tensor(list(reversed(out)), dtype=torch.float32)
