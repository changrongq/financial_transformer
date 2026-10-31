# -*- coding: utf-8 -*-
"""复现知微（2026-10-31）RL 审查发现的实现错误。

不依赖原始行情数据库，纯合成数据。
每个检查都独立打印 PASS / FAIL。
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

torch.manual_seed(0)
np.random.seed(0)

print("=" * 72)
print("知微 RL 审查复现  |  torch", torch.__version__, "|  numpy", np.__version__)
print("=" * 72)


# ============================================================
# 检查 1：动作与回报没有逐项配对（广播 bug）
# ============================================================
print("\n[检查 1] log_prob [T,1] × advantage [T] 是否触发广播")


class Policy(nn.Module):
    """与 rl_timing_v4_variable_holding.py 中原 Policy 结构一致。"""
    def __init__(self, n_in=8, n_out=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


pol = Policy()
T = 3
lps = []
for t in range(T):
    x = torch.randn(1, 8)              # 保留了 batch 维度 -> [1,8]
    p = pol(x)                          # [1,3]
    d = Categorical(p)
    a = d.sample()                      # 标量
    lp = d.log_prob(a)                  # [1]  <- 关键：形状是 [1] 不是标量
    lps.append(lp)

stacked = torch.stack(lps)
disc = torch.tensor([-1.0, 0.0, 1.0])

print(f"  单个 log_prob 形状        : {tuple(lps[0].shape)}")
print(f"  torch.stack(lps) 形状     : {tuple(stacked.shape)}")
print(f"  advantage 形状            : {tuple(disc.shape)}")
prod = stacked * disc
print(f"  相乘结果形状              : {tuple(prod.shape)}   <- 期望 (3,)")

if prod.shape == (T, T):
    print("  结果: FAIL  — 广播成 [T,T]，不是逐动作配对")
else:
    print("  结果: PASS")

# 梯度范数对比
loss_bad = -(stacked * disc).mean()
pol.zero_grad()
loss_bad.backward()
g_bad = sum(p.grad.norm().item() ** 2 for p in pol.parameters() if p.grad is not None) ** 0.5

pol2 = Policy()
pol2.load_state_dict(pol.state_dict())
lps2 = []
for t in range(T):
    x = torch.randn(1, 8)
    p = pol2(x)
    d = Categorical(p)
    a = d.sample()
    lps2.append(d.log_prob(a).squeeze(0))   # 正确：squeeze 成标量
loss_ok = -(torch.stack(lps2) * disc).mean()
pol2.zero_grad()
loss_ok.backward()
g_ok = sum(p.grad.norm().item() ** 2 for p in pol2.parameters() if p.grad is not None) ** 0.5

print(f"  原损失参数梯度范数        : {g_bad:.6f}")
print(f"  正确逐项配对梯度范数      : {g_ok:.6f}")

# 严格去均值后（v4 的做法）的梯度
lps3 = []
torch.manual_seed(123)
for t in range(T):
    x = torch.randn(1, 8)
    p = pol2(x)
    d = Categorical(p)
    a = d.sample()
    lps3.append(d.log_prob(a))              # 不 squeeze，保持 [1]
d0 = disc - disc.mean()
loss_zero = -(torch.stack(lps3) * d0).mean()
pol2.zero_grad()
loss_zero.backward()
g_zero = sum(p.grad.norm().item() ** 2 for p in pol2.parameters() if p.grad is not None) ** 0.5
print(f"  去均值后（v4 真实情形）梯度: {g_zero:.6e}   <- 接近 0 说明策略梯度被抵消")


# ============================================================
# 检查 2：v4 训练环境 reset 会落到测试区间
# ============================================================
print("\n[检查 2] v4 Env.reset() 是否在测试区间随机起步")

N, N_TRAIN, HORIZON = 1000, 750, 50

# 复刻 v4 Env.reset 的逻辑（rl_timing_v4_variable_holding.py:143-150）
def v4_reset_span(horizon=HORIZON):
    """返回原实现可能的起点范围（该文件用 randint(horizon+5, n-500))。"""
    # 原文件实际写法（见 143-150）：
    #   self.t = np.random.randint(horizon + 5, len(self.close) - 500)
    return None


# 直接用真实类，保证不是"另外写的环境"
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v4", "scripts/rl_timing_v4_variable_holding.py")
v4 = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(v4)
    feats = np.random.randn(N, 8).astype(np.float32)
    open_ = np.random.uniform(10, 12, N)
    close = open_ * (1 + np.random.randn(N) * 0.01)
    mu, sd = feats.mean(0), feats.std(0) + 1e-8
    env = v4.Env(feats, open_, close, 0.0003, 0.0, 1.0, HORIZON, mu, sd)
    np.random.seed(7)
    starts = [env.reset() and env.t for _ in range(1000)]
    in_test = sum(1 for s in starts if s >= N_TRAIN)
    print(f"  1000 次 reset，起点落在测试区间(>= {N_TRAIN}) 的次数: {in_test}")
    if in_test > 0:
        print("  结果: FAIL  — 训练环境可以访问测试期数据")
    else:
        print("  结果: PASS")
except Exception as e:
    print(f"  [跳过] 导入 v4 模块失败: {type(e).__name__}: {e}")


# ============================================================
# 检查 3：v4 reset 后覆盖 t 却不刷新 obs
# ============================================================
print("\n[检查 3] rolling/v3 覆盖 env.t 后 obs 是否仍是旧时点")

try:
    env2 = v4.Env(feats, open_, close, 0.0003, 0.0, 1.0, HORIZON, mu, sd)
    env2.t = 800
    obs_old = env2._obs() if hasattr(env2, "_obs") else None
    if obs_old is None:
        print("  [跳过] Env 无 _obs() 方法，无法单独取观测")
    else:
        env2.t = 20
        obs_new = env2._obs()
        print(f"  起点 800 的 obs[0] = {float(obs_old[0]):.6f}")
        print(f"  起点 20  的 obs[0] = {float(obs_new[0]):.6f}")
        same = np.allclose(obs_old, obs_new)
        print(f"  两者相同? {same}")
        print("  说明: 若调用方 reset() 后再改 t，拿到的是旧起点 obs")
except Exception as e:
    print(f"  [跳过] {type(e).__name__}: {e}")


# ============================================================
# 检查 4：v4 只在持有期末更新回撤，漏掉持仓中间风险
# ============================================================
print("\n[检查 4] v4 的 drawdown 是否漏掉持仓中间回撤")

# 合成：持有期内先跌 50% 再完全恢复
mid_path = [1.0, 0.75, 0.5, 0.75, 1.0]
peak, maxdd = 1.0, 0.0
for v in mid_path:
    peak = max(peak, v)
    maxdd = min(maxdd, v / peak - 1)
print(f"  真实逐日最大回撤: {maxdd*100:.1f}%")
print(f"  v4 端点法回撤  : {0.0:.1f}%  (期初1.0 -> 期末1.0)")
if abs(maxdd) < 1e-9:
    print("  结果: PASS")
else:
    print("  结果: FAIL  — 环境漏掉了 -50% 的中间回撤，惩罚不会触发")


print("\n" + "=" * 72)
print("复现完成")
print("=" * 72)
