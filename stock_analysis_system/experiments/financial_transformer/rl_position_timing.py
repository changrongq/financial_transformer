"""RL 仓位择时实验。

目标（与豆包讨论后的收窄版）：
  不做「选哪只股」（已证明方向不可预测），只做「仓位择时」——
  让 RL 智能体学习每日仓位（0/25/50/75/100%），标的为上证指数（可对应 ETF）。

设计：
  - 观测：近 LOOKBACK 日的行情统计（收益、波动、量能、regime 趋势）
  - 动作：5 档仓位
  - 奖励：当日组合收益 - 交易成本 - 回撤惩罚
  - 算法：REINFORCE（策略梯度，无依赖，torch 手写）
  - 验证：时间切分 + 逐年一致性检验

用法：
  python experiments/financial_transformer/rl_position_timing.py --train-end 2023-12-31
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]

LOOKBACK = 20
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _resolve_db() -> str:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return str(p.resolve())
    return str((PROJECT_ROOT.parent / "stock_history.db").resolve())


def load_index_series(db_path: str) -> pd.DataFrame:
    """加载上证指数日线，构造观测特征。"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT date, close, pct_chg FROM index_daily_sh ORDER BY date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ret"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
    df = df.dropna(subset=["close", "ret"]).reset_index(drop=True)

    # 观测特征
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["vol10"] = df["ret"].rolling(10).std()
    df["ret5"] = df["close"].pct_change(5)
    df["ret20"] = df["close"].pct_change(20)
    df["regime10"] = df["ret"].rolling(10).mean()
    df["dist_ma20"] = df["close"] / df["ma20"] - 1.0
    df["vol_ratio"] = df["vol10"] / df["vol10"].rolling(60).mean()
    df = df.dropna().reset_index(drop=True)
    return df


OBS_DIM = 6


class TimingEnv:
    """单资产仓位择时环境。

    关键设计（修正版）：
      - 观测做标准化（用训练段统计量），避免量纲差异导致网络学不动
      - 奖励 = 组合日收益 - 满仓日收益（超额），让信用分配更聚焦
      - 加「空仓机会成本」：连续空仓时轻微惩罚，逼迫模型必须决策
    """

    def __init__(
        self,
        df: pd.DataFrame,
        start: int,
        end: int,
        cost: float = 0.0005,
        dd_penalty: float = 0.1,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        idle_penalty: float = 1e-6,
    ):
        self.df = df.reset_index(drop=True)
        self.start = start
        self.end = end
        self.cost = cost
        self.dd_penalty = dd_penalty
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.idle_penalty = idle_penalty
        self.reset()

    def _raw_obs(self) -> np.ndarray:
        r = self.df.iloc[self.t]
        return np.array([
            r["ret5"], r["ret20"], r["vol10"], r["regime10"],
            r["dist_ma20"], r["vol_ratio"],
        ], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        raw = self._raw_obs()
        if self.obs_mean is not None and self.obs_std is not None:
            raw = (raw - self.obs_mean) / self.obs_std
        return np.clip(raw, -5.0, 5.0).astype(np.float32)

    def reset(self):
        self.t = self.start
        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0
        return self._obs()

    def step(self, action: int):
        target = POSITIONS[action]
        r = self.df.iloc[self.t]["ret"]

        turnover = abs(target - self.position)
        trade_cost = turnover * self.cost

        day_ret = target * r - trade_cost
        self.equity *= (1.0 + day_ret)
        self.peak = max(self.peak, self.equity)
        drawdown = self.equity / self.peak - 1.0

        # 奖励 = 超额（相对满仓） - 回撤惩罚 - 轻微空仓惩罚（逼迫决策）
        excess = day_ret - r
        reward = excess - self.dd_penalty * max(0.0, -drawdown - 0.05) ** 2
        if target == 0.0:
            reward -= self.idle_penalty

        self.position = target
        self.t += 1
        done = self.t >= self.end
        return (self._obs() if not done else np.zeros(OBS_DIM, dtype=np.float32)), reward, done


class Policy(nn.Module):
    def __init__(self, obs_dim=6, n_actions=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def evaluate(
    policy: Policy,
    df: pd.DataFrame,
    start: int,
    end: int,
    cost: float,
    dd_penalty: float,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> dict:
    env = TimingEnv(df, start, end, cost, dd_penalty, obs_mean, obs_std)
    obs = env.reset()
    done = False
    rets = []
    actions = []
    while not done:
        with torch.no_grad():
            probs = policy(torch.tensor(obs).unsqueeze(0))
        a = int(torch.argmax(probs, dim=-1).item())
        actions.append(a)
        obs, r, done = env.step(a)
        rets.append(env.df.iloc[env.t - 1]["ret"] * POSITIONS[a])

    equity = env.equity
    # 基准：满仓持有
    sub = env.df.iloc[start:end]
    bh_equity = float(np.prod(1.0 + sub["ret"].values))

    def max_dd(series):
        cum = np.cumprod(1.0 + np.array(series))
        peak = np.maximum.accumulate(cum)
        return float((cum / peak - 1).min())

    return {
        "equity": float(equity),
        "total_return": float(equity - 1.0),
        "buy_hold_return": float(bh_equity - 1.0),
        "max_drawdown": max_dd(rets),
        "avg_position": float(np.mean([POSITIONS[a] for a in actions])),
        "days": int(end - start),
    }


def _compute_obs_stats(df: pd.DataFrame, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """用训练段统计量做观测标准化。"""
    sub = df.iloc[start:end]
    raw = np.stack([
        sub["ret5"].values, sub["ret20"].values, sub["vol10"].values,
        sub["regime10"].values, sub["dist_ma20"].values, sub["vol_ratio"].values,
    ], axis=1).astype(np.float32)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0) + 1e-8
    return mean, std


def train(
    df: pd.DataFrame,
    start: int,
    end: int,
    epochs: int,
    cost: float,
    dd_penalty: float,
    seed: int = 42,
    n_trajectories: int = 8,
    entropy_coef: float = 0.02,
    gamma: float = 0.95,
):
    """REINFORCE + baseline + 熵正则 + 多轨迹（修正版）。

    修正点：
      1. 观测标准化（否则网络学不动）
      2. 多轨迹/epoch（降方差）
      3. baseline（滚动均值）做 advantage（降方差）
      4. 熵正则（防止策略坍缩成恒定动作 = "逼迫模型必须决策"）
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = Policy(obs_dim=OBS_DIM)
    opt = optim.Adam(policy.parameters(), lr=3e-3)
    obs_mean, obs_std = _compute_obs_stats(df, start, end)

    baseline = 0.0
    baseline_decay = 0.9

    for epoch in range(epochs):
        all_log_probs, all_advantages, all_entropies = [], [], []
        epoch_rewards = []

        for _ in range(n_trajectories):
            env = TimingEnv(df, start, end, cost, dd_penalty, obs_mean, obs_std)
            obs = env.reset()
            log_probs, rewards, entropies = [], [], []
            done = False
            while not done:
                probs = policy(torch.tensor(obs).unsqueeze(0))
                dist = torch.distributions.Categorical(probs)
                a = dist.sample()
                log_probs.append(dist.log_prob(a))
                entropies.append(dist.entropy())
                obs, r, done = env.step(int(a.item()))
                rewards.append(r)
            epoch_rewards.append(sum(rewards))

            # 折扣回报
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            all_log_probs.extend(log_probs)
            all_advantages.extend(returns)
            all_entropies.extend(entropies)

        adv = torch.tensor(all_advantages, dtype=torch.float32)
        # baseline 用 epoch 平均回报（简单但有效）
        ep_mean = float(adv.mean())
        baseline = baseline_decay * baseline + (1 - baseline_decay) * ep_mean
        adv = adv - baseline
        adv = adv / (adv.std() + 1e-8)

        logp = torch.stack(all_log_probs)
        ent = torch.stack(all_entropies).mean()
        policy_loss = -(logp * adv).mean()
        loss = policy_loss - entropy_coef * ent

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                probs_test = policy(torch.zeros(1, OBS_DIM))
            print(
                f"  epoch {epoch+1}: loss={loss.item():+.4f}, policy={policy_loss.item():+.4f}, "
                f"entropy={ent.item():.3f}, avg_reward={np.mean(epoch_rewards):+.5f}, "
                f"probs={np.round(probs_test.squeeze().numpy(), 2)}"
            )
    return policy, obs_mean, obs_std


def main() -> None:
    parser = argparse.ArgumentParser(description="RL 仓位择时实验")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--dd-penalty", type=float, default=0.1)
    args = parser.parse_args()

    db = _resolve_db()
    df = load_index_series(db)
    print(f"数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}, {len(df)} 个交易日")

    # 索引区间
    train_start = int(df.index[df["date"] >= pd.Timestamp(args.train_start)][0])
    train_end = int(df.index[df["date"] <= pd.Timestamp(args.train_end)][-1]) + 1
    test_start = int(df.index[df["date"] >= pd.Timestamp(args.test_start)][0])
    test_end = len(df)

    print(f"\n训练区间: {df.iloc[train_start]['date'].date()} ~ {df.iloc[train_end-1]['date'].date()} ({train_end-train_start} 天)")
    print(f"测试区间: {df.iloc[test_start]['date'].date()} ~ {df.iloc[test_end-1]['date'].date()} ({test_end-test_start} 天)")

    print("\n开始训练...")
    policy, obs_mean, obs_std = train(df, train_start, train_end, args.epochs, args.cost, args.dd_penalty)

    # 检查策略是否退化（对不同市场状态是否有不同动作）
    print("\n=== 策略对不同市场状态的响应（检查是否学到东西）===")
    scenarios = [
        ("强势上涨", [0.05, 0.10, 0.01, 0.008, 0.03, 1.2]),
        ("弱势下跌", [-0.05, -0.10, 0.03, -0.008, -0.03, 1.5]),
        ("横盘", [0.0, 0.01, 0.015, 0.0, 0.0, 0.9]),
        ("暴跌", [-0.08, -0.15, 0.05, -0.02, -0.06, 2.0]),
    ]
    action_spread = []
    for name, raw in scenarios:
        normed = (np.array(raw, dtype=np.float32) - obs_mean) / obs_std
        normed = np.clip(normed, -5, 5)
        with torch.no_grad():
            probs = policy(torch.tensor(normed).unsqueeze(0)).squeeze().numpy()
        best = int(np.argmax(probs))
        action_spread.append(best)
        print(f"  {name}: 仓位={POSITIONS[best]*100:.0f}% 概率={np.round(probs,2)}")
    print(f"  动作区分度: {len(set(action_spread))} 种不同动作（1=退化，>1=学到东西）")

    # 评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    train_res = evaluate(policy, df, train_start, train_end, args.cost, args.dd_penalty, obs_mean, obs_std)
    test_res = evaluate(policy, df, test_start, test_end, args.cost, args.dd_penalty, obs_mean, obs_std)

    for name, r in [("训练集", train_res), ("测试集", test_res)]:
        print(f"\n【{name}】{r['days']} 天")
        print(f"  RL策略: 收益 {r['total_return']*100:+.2f}%, 最大回撤 {r['max_drawdown']*100:.2f}%, 平均仓位 {r['avg_position']*100:.0f}%")
        print(f"  满仓基准: 收益 {r['buy_hold_return']*100:+.2f}%")
        print(f"  超额: {(r['total_return']-r['buy_hold_return'])*100:+.2f}%")

    # 逐年检验（关键！）
    print("\n" + "=" * 60)
    print("逐年一致性检验（核心）")
    print("=" * 60)
    yearly = {}
    for year in sorted(df["date"].dt.year.unique()):
        if year < 2019:
            continue
        ydf = df[df["date"].dt.year == year]
        if len(ydf) < 60:
            continue
        s = int(ydf.index[0]); e = int(ydf.index[-1]) + 1
        r = evaluate(policy, df, s, e, args.cost, args.dd_penalty, obs_mean, obs_std)
        yearly[int(year)] = r
        print(f"  {year}: RL {r['total_return']*100:+7.2f}% | 满仓 {r['buy_hold_return']*100:+7.2f}% | 超额 {(r['total_return']-r['buy_hold_return'])*100:+7.2f}% | 平均仓位 {r['avg_position']*100:.0f}%")

    wins = sum(1 for r in yearly.values() if r["total_return"] > r["buy_hold_return"])
    print(f"\n  逐年跑赢满仓: {wins}/{len(yearly)} 年")

    out = ROOT / "results" / "rl_position_timing_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": vars(args),
        "train": train_res,
        "test": test_res,
        "yearly": yearly,
        "yearly_win_ratio": f"{wins}/{len(yearly)}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
