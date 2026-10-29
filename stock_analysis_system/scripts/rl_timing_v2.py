"""RL 择时实验 v2 —— 修掉前两次失败的根因。

============================================================
前两次为什么失败（v1 的代码在 rl_position_timing.py）
============================================================
v1 的奖励：
    day_ret = target * r - cost
    excess  = day_ret - r  =  (target-1)*r - cost

问题：随机行情下 E[r]≈0，所以 E[excess] ≈ -cost。
     策略无论怎么选动作，期望奖励都是负的成本，
     → 梯度信号 ≈ 噪声 → 策略退化为恒定 50%（entropy 拉满）。

============================================================
v2 的三个关键改动
============================================================
1. 奖励改为「风险调整后的绝对收益」，不是超额
       reward = day_ret - λ * max(0, -drawdown - 阈值)^2
   —— 直接优化"赚钱 + 少回撤"，不再和满仓比。

2. 观测增加「状态持续时间」和「信号强度」
   —— 帮助模型区分"刚开始跌"和"跌了很久"。

3. 训练用多段随机起点（episodic）+ 更长的训练轮数
   —— 单条时间线样本太少（只有 475 天）。

============================================================
诚实预期（必须先看这段）
============================================================
* 数据只有 475 个交易日（2024-10 ~ 2026-10），对 RL 来说极少。
* 即使训练成功，也很可能是过拟合。
* 本脚本会做「训练段 / 测试段」时间切分，只在测试段评估。
* 如果测试段跑不过买入持有，那就是又失败了 —— 如实报告。

用法：
  python scripts/rl_timing_v2.py
  python scripts/rl_timing_v2.py --train-ratio 0.7 --episodes 3000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

ACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
OBS_DIM = 8


def _resolve_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def load_market(db: str) -> pd.DataFrame:
    """全市场等权日收益 + 技术特征。"""
    conn = sqlite3.connect(db)
    sd = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [收盘] AS close FROM stock_data", conn
    )
    conn.close()

    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["stock_code"] = sd["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    sd["close"] = pd.to_numeric(sd["close"], errors="coerce")
    sd = sd.dropna(subset=["close"])
    sd = sd[sd["stock_code"].str.len() == 6]
    sd = sd.sort_values(["stock_code", "date"])
    sd["ret"] = sd.groupby("stock_code")["close"].pct_change(fill_method=None)

    m = sd.groupby("date")["ret"].mean().rename("ret").reset_index().dropna()
    m = m.sort_values("date").reset_index(drop=True)

    # 特征
    m["ret5"] = (1 + m["ret"]).rolling(5).apply(np.prod, raw=True) - 1
    m["ret20"] = (1 + m["ret"]).rolling(20).apply(np.prod, raw=True) - 1
    m["vol10"] = m["ret"].rolling(10).std()
    m["vol60"] = m["ret"].rolling(60).std()
    m["regime10"] = m["ret"].rolling(10).mean()
    m["regime20"] = m["ret"].rolling(20).mean()
    m["vol_regime"] = m["vol10"] / m["vol60"].replace(0, np.nan)
    # 状态持续时间：连续同号天数
    sign = np.sign(m["ret"]).replace(0, np.nan).ffill()
    run = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    m["run_len"] = run * sign

    m = m.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return m


class Env:
    """episodic 环境：每次 reset 随机起点，固定长度。"""

    def __init__(self, feats: np.ndarray, rets: np.ndarray, cost: float,
                 dd_lambda: float, dd_free: float, horizon: int, obs_mu, obs_sd):
        self.feats = feats
        self.rets = rets
        self.cost = cost
        self.dd_lambda = dd_lambda
        self.dd_free = dd_free
        self.horizon = horizon
        self.obs_mu = obs_mu
        self.obs_sd = obs_sd

    def reset(self, start: int | None = None):
        max_start = len(self.rets) - self.horizon - 2
        self.t = start if start is not None else np.random.randint(0, max(1, max_start))
        self.pos = 0.0
        self.eq = 1.0
        self.peak = 1.0
        self.steps = 0
        return self._obs()

    def _obs(self):
        x = (self.feats[self.t] - self.obs_mu) / self.obs_sd
        return np.clip(x, -5, 5).astype(np.float32)

    def step(self, a: int):
        target = ACTIONS[a]
        r = self.rets[self.t]
        turnover = abs(target - self.pos)
        gross = target * r
        net = gross - turnover * self.cost

        self.eq *= (1 + net)
        self.peak = max(self.peak, self.eq)
        dd = self.eq / self.peak - 1.0

        # 关键改动：奖励 = 净收益 - 回撤惩罚（不再减满仓收益）
        penalty = self.dd_lambda * max(0.0, -dd - self.dd_free) ** 2
        reward = net - penalty

        self.pos = target
        self.t = (self.t + 1) % len(self.rets)
        self.steps += 1
        done = self.steps >= self.horizon
        return (self._obs() if not done else np.zeros(OBS_DIM, dtype=np.float32)), reward, done


class Policy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, len(ACTIONS))
        # 小初始化，避免一开始就高度确定性
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return torch.softmax(self.head(self.body(x)), dim=-1)


def evaluate(policy, feats, rets, cost, start_idx, duration) -> dict:
    """在给定区间上确定性执行（argmax），返回指标。"""
    policy.eval()
    pos, eq, peak = 0.0, 1.0, 1.0
    eq_curve, actions = [], []
    with torch.no_grad():
        for t in range(start_idx, start_idx + duration):
            x = torch.from_numpy(np.clip(
                (feats[t] - FEATS_MU) / FEATS_SD, -5, 5).astype(np.float32)).unsqueeze(0)
            a = int(torch.argmax(policy(x), dim=-1).item())
            target = ACTIONS[a]
            r = rets[t]
            net = target * r - abs(target - pos) * cost
            eq *= (1 + net)
            peak = max(peak, eq)
            eq_curve.append(eq)
            actions.append(target)
            pos = target
    eq_arr = np.array(eq_curve)
    rets_arr = np.diff(np.concatenate([[1.0], eq_arr]))
    dd = float((eq_arr / np.maximum.accumulate(eq_arr) - 1).min())
    total = float(eq_arr[-1] - 1)
    years = duration / 244
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = float(rets_arr.mean() / rets_arr.std() * np.sqrt(244)) if rets_arr.std() > 0 else np.nan
    return {
        "total_return": total, "annual_return": ann, "max_drawdown": dd,
        "sharpe": sharpe, "days": duration,
        "avg_position": float(np.mean(actions)),
        "pct_cash": float(np.mean(np.array(actions) == 0.0)),
        "pct_full": float(np.mean(np.array(actions) == 1.0)),
        "n_switches": int(np.sum(np.diff(actions) != 0)) if len(actions) > 1 else 0,
        "actions": actions,
    }


FEATS_MU = None
FEATS_SD = None


def main() -> None:
    global FEATS_MU, FEATS_SD
    parser = argparse.ArgumentParser(description="RL 择时 v2")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=120, help="每个 episode 长度")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dd-lambda", type=float, default=5.0, help="回撤惩罚系数")
    parser.add_argument("--dd-free", type=float, default=0.05, help="允许的免罚回撤")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = load_market(_resolve_db())
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}（{len(df)} 天）")

    feat_cols = ["ret5", "ret20", "vol10", "regime10", "regime20", "vol_regime", "run_len", "ret"]
    feats = df[feat_cols].values.astype(np.float32)
    rets = df["ret"].values.astype(np.float32)

    n = len(rets)
    n_train = int(n * args.train_ratio)
    FEATS_MU = feats[:n_train].mean(axis=0)
    FEATS_SD = feats[:n_train].std(axis=0) + 1e-8

    print(f"训练段: 0 ~ {n_train}   测试段: {n_train} ~ {n}")

    env = Env(feats, rets, args.cost, args.dd_lambda, args.dd_free, args.horizon, FEATS_MU, FEATS_SD)
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=args.lr)

    print(f"\n开始训练 {args.episodes} episodes...")
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            x = torch.from_numpy(obs).unsqueeze(0)
            probs = policy(x)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            obs, r, done = env.step(int(a.item()))
            rewards.append(r)

        # REINFORCE + baseline（用回报均值做基线，降方差）
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.append(R)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(torch.stack(log_probs) * returns).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 500 == 0:
            with torch.no_grad():
                sample = torch.from_numpy(env.reset()).unsqueeze(0)
                p = policy(sample).squeeze().numpy()
            ent = float(-(p * np.log(p + 1e-9)).sum())
            print(f"  ep {ep:5d}  loss={loss.item():+.4f}  entropy={ent:.3f}  "
                  f"probs={np.round(p, 3)}")

    # ===== 评估 =====
    print("\n" + "=" * 88)
    print("评估结果")
    print("=" * 88)

    res_train = evaluate(policy, feats, rets, args.cost, 0, n_train)
    res_test = evaluate(policy, feats, rets, args.cost, n_train, n - n_train)
    bh_train = float(np.prod(1 + rets[:n_train]) - 1)
    bh_test = float(np.prod(1 + rets[n_train:]) - 1)

    def show(name, r, bh):
        years = r["days"] / 244
        bh_ann = (1 + bh) ** (1 / years) - 1 if years > 0 else np.nan
        print(f"{name:<12}{r['total_return']*100:>+11.1f}%{r['annual_return']*100:>+10.1f}%"
              f"{r['max_drawdown']*100:>11.1f}%{r['sharpe']:>9.2f}"
              f"{r['avg_position']*100:>9.1f}%{r['pct_cash']*100:>8.1f}%{r['n_switches']:>8}")
        print(f"{'  ↑买入持有':<12}{bh*100:>+11.1f}%{bh_ann*100:>+10.1f}%")

    print(f"{'区间':<12}{'累计收益':>12}{'年化':>10}{'最大回撤':>11}{'夏普':>9}{'平均仓位':>10}{'空仓占比':>9}{'切换':>8}")
    print("-" * 88)
    show("训练段", res_train, bh_train)
    show("测试段", res_test, bh_test)
    print("-" * 88)

    # 动作分布
    for name, r in [("训练段", res_train), ("测试段", res_test)]:
        acts = np.array(r["actions"])
        dist = {f"{a*100:.0f}%": int((acts == a).sum()) for a in ACTIONS}
        print(f"  {name}动作分布: {dist}")

    # ===== 判断 =====
    print("\n" + "=" * 88)
    verdict = "失败"
    if res_test["total_return"] > bh_test:
        verdict = "在测试段跑赢买入持有"
    if res_test["sharpe"] > 2.0 and res_test["avg_position"] < 0.05:
        verdict = "退化为全空仓（假成功）"
    if res_test["n_switches"] <= 2:
        verdict = "退化为恒定仓位（和前两次一样的失败）"
    print(f"结论: {verdict}")
    print(f"  测试段 RL: {res_test['total_return']*100:+.1f}%  vs  买入持有 {bh_test*100:+.1f}%")
    print(f"  测试段切换次数: {res_test['n_switches']}  平均仓位: {res_test['avg_position']*100:.1f}%")
    if res_test["n_switches"] <= 5:
        print("  >>> 切换次数太少，说明策略没学到有效决策，仍是恒定仓位。")
    print("=" * 88)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rl_timing_v2.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "train": {k: v for k, v in res_train.items() if k != "actions"},
        "test": {k: v for k, v in res_test.items() if k != "actions"},
        "buy_hold_train": bh_train,
        "buy_hold_test": bh_test,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
