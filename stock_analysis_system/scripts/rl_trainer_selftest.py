# -*- coding: utf-8 -*-
"""训练器自检：它能不能学会一个「已知答案」的任务？（知微第六节第一步）

知微的核心要求：
  「用同一训练调用路径做小型合成任务：已知可见信号可靠决定后续涨跌，
    训练器应在预先定义的种子与预算下改善……再做随机信号/打乱关联的负对照。」

设计说明（为什么用多步 episode 而不是单步 bandit）
--------------------------------------------------
旧 bug 的伤害来自 `torch.stack(lps)` 是 [T,1]、`disc` 是 [T] → 广播成 [T,T]，
再对本条 episode 的回报**去均值**。在单步 bandit 里 T=1，[1,1]×[1] 形状正确，
测不出问题。v4/v5 是「一条 episode 内多个决策、再对本条 episode 的回报去均值」，
所以自检必须复刻这个结构。

两组对照：
  legacy  —— 复刻旧代码：`obs.unsqueeze(0)` → `log_prob` 是 [1]，
             最后 `-(torch.stack(lps) * disc).mean()`，disc 已去均值。
  paired  —— 修正后的 `rl_algo_utils.paired_policy_loss`（[T]×[T] 逐项配对）。

两个任务：
  正向（可学）  合成收益由可见特征 x_t 的符号驱动，最优策略是「x_t>0 就持仓」。
  负向（不可学）收益被打乱，与特征无关；正确训练器不应稳定获得优势。

用法：
  python scripts/rl_trainer_selftest.py
  python scripts/rl_trainer_selftest.py --seeds 5 --episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from rl_algo_utils import discounted_returns, paired_policy_loss


# ============================================================
# 合成环境：多步 episode，结构对齐 v4
# ============================================================
class EpisodeEnv:
    """每日一个二元决策（空仓 / 满仓）。

    inform=True : ret_t = edge * sign(x_t) + 噪声，x_t 在 t 日收盘可观测
                  → 最优策略：x_t > 0 时持仓
    inform=False: ret_t 与 x_t 无关（标签打乱）→ 无最优策略
    """

    def __init__(self, n_days: int = 1200, inform: bool = True,
                 edge: float = 0.01, noise: float = 0.005, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.x = rng.randn(n_days).astype(np.float32)
        signal = np.sign(self.x)
        if not inform:
            signal = rng.permutation(signal)
        self.ret = (signal * edge + rng.randn(n_days).astype(np.float32) * noise)
        self.n = n_days

    def episode(self, rng: np.random.RandomState, length: int):
        hi = self.n - length - 2
        start = rng.randint(0, max(1, hi)) if hi > 0 else 0
        idx = np.arange(start, min(start + length, self.n - 1))
        return self.x[idx], self.ret[idx]


class Policy(nn.Module):
    def __init__(self, n_in: int = 1, n_out: int = 2, hidden: int = 16):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, n_out)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return torch.softmax(self.head(self.body(x)), dim=-1)


# ============================================================
# 一条 episode 的策略梯度：legacy vs paired
# ============================================================
def run_episode(env: EpisodeEnv, pol: Policy, rng: np.random.RandomState,
                length: int, hold: int, mode: str):
    x, r = env.episode(rng, length)

    # 先把整条 episode 的特征一次前向（快），拿到分布
    xs = torch.from_numpy(x).reshape(-1, 1)          # [L,1]
    with torch.no_grad():
        probs = pol(xs)                              # [L,2]

    logps, rewards = [], []
    t = 0
    while t + hold <= len(r):
        pv = probs[t:t + 1]                          # [1,2] 保留 batch 维
        d = Categorical(pv)
        a = d.sample()                               # [1]
        if mode == "legacy":
            logps.append(d.log_prob(a))              # [1]  <- 关键：保留 batch 维
            # legacy 需要梯度，重新前向一次
            pv2 = pol(torch.from_numpy(x[t:t + 1]).reshape(1, -1))
            logps[-1] = Categorical(pv2).log_prob(a)
        else:
            # paired：用可求导的路径，形状压成标量
            pv2 = pol(torch.from_numpy(x[t:t + 1]).reshape(1, -1))
            logps.append(Categorical(pv2).log_prob(a).squeeze(-1))   # []

        pos = float(a.item())
        seg = r[t:t + hold]
        rewards.append(float(np.prod(1 + pos * seg) - 1.0))
        t += hold

    if len(logps) < 2:
        return None

    disc = discounted_returns(rewards, gamma=0.99)

    if mode == "legacy":
        # 复刻旧代码：episode 内去均值 + [T,1]*[T] 广播
        d0 = (disc - disc.mean()) / (disc.std() + 1e-8)
        loss = -(torch.stack(logps) * d0).mean()
    else:
        loss = paired_policy_loss(logps, disc, normalize=True)

    return loss


def train_one(env: EpisodeEnv, mode: str, seed: int, episodes: int,
              length: int = 200, hold: int = 1, lr: float = 1e-2) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 10000)

    pol = Policy()
    opt = optim.Adam(pol.parameters(), lr=lr)
    grad_norms: List[float] = []

    for _ in range(episodes):
        loss = run_episode(env, pol, rng, length, hold, mode)
        if loss is None:
            continue
        opt.zero_grad()
        loss.backward()
        gn = sum(q.grad.norm().item() ** 2 for q in pol.parameters()
                 if q.grad is not None) ** 0.5
        grad_norms.append(gn)
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        xt = torch.from_numpy(env.x).reshape(-1, 1)
        act = pol(xt).argmax(dim=-1).numpy()
    acc = float(np.mean(act == (env.x > 0).astype(int)))
    greedy_ret = float(np.mean(np.where(act == 1, env.ret, 0.0)))

    return {
        "mode": mode, "seed": seed, "accuracy": acc,
        "greedy_ret": greedy_ret,
        "grad_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "n_episodes": len(grad_norms),
    }


def summarize(rows: List[Dict]) -> Dict:
    accs = np.array([r["accuracy"] for r in rows])
    rets = np.array([r["greedy_ret"] for r in rows])
    return {
        "n_seeds": len(rows),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "accuracy_min": float(accs.min()), "accuracy_max": float(accs.max()),
        "greedy_ret_mean": float(rets.mean()),
        "grad_mean": float(np.mean([r["grad_mean"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--hold", type=int, default=1)
    ap.add_argument("--out", default="results/rl_trainer_selftest.json")
    args = ap.parse_args()

    print("=" * 76)
    print("训练器自检  已知答案任务（知微第六节第一步）")
    print(f"seeds={args.seeds} episodes={args.episodes} "
          f"episode长度={args.length} 决策间隔={args.hold}")
    print("=" * 76)

    report: Dict = {}
    for label, inform in [("positive", True), ("negative", False)]:
        print(f"\n--- {label} ({'可学' if inform else '打乱标签/不可学'}) ---")
        for mode in ["legacy", "paired"]:
            env = EpisodeEnv(args.days, inform=inform, seed=42)
            rows = [train_one(env, mode, seed=s, episodes=args.episodes,
                              length=args.length, hold=args.hold)
                    for s in range(args.seeds)]
            s = summarize(rows)
            s["seeds_detail"] = rows
            report[f"{label}|{mode}"] = s
            print(f"  {mode:7s}  acc={s['accuracy_mean']:.3f} "
                  f"[{s['accuracy_min']:.3f},{s['accuracy_max']:.3f}]  "
                  f"贪心日收益={s['greedy_ret_mean']*100:+.4f}%  "
                  f"梯度={s['grad_mean']:.2e}")

    print("\n" + "=" * 76)
    print("判定")
    print("=" * 76)
    p_paired = report["positive|paired"]
    p_legacy = report["positive|legacy"]
    n_paired = report["negative|paired"]

    checks = [
        ("正向：paired 准确率 > 0.75（能学到已知答案）",
         p_paired["accuracy_mean"] > 0.75, f"{p_paired['accuracy_mean']:.3f}"),
        ("正向：paired 优于 legacy 至少 0.15",
         p_paired["accuracy_mean"] - p_legacy["accuracy_mean"] > 0.15,
         f"{p_paired['accuracy_mean'] - p_legacy['accuracy_mean']:+.3f}"),
        ("正向：legacy 梯度接近 0（复现广播抵消）",
         p_legacy["grad_mean"] < 0.01, f"{p_legacy['grad_mean']:.2e}"),
        ("负向：paired 无稳定优势 |acc-0.5| < 0.15",
         abs(n_paired["accuracy_mean"] - 0.5) < 0.15,
         f"{n_paired['accuracy_mean']:.3f}"),
    ]

    all_ok = True
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  (实际 {val})")
        all_ok &= ok

    report["verdict"] = "PASS" if all_ok else "FAIL"
    report["checks"] = [{"name": n, "ok": bool(o), "value": v} for n, o, v in checks]
    report["args"] = vars(args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.out}")
    print(f"总体判定: {report['verdict']}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
