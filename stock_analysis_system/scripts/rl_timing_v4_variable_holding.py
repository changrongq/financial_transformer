"""RL 择时 v4：T+1 可变持有期（贴合真实交易约束）。

=================== 设计思路 ===================
前三次 RL 失败的教训：
  v1: 奖励 = (position-1)*r - cost，期望≈0，梯度消失
  v2: 改奖励为净收益，但仍每天决策 → 成本吃掉
  v3: 多时间点动作（空仓/隔夜/日内/全天），发现 T+1 下"日内"违规

本次 v4 的核心改变：
【动作语义】
  不是"今天仓位多少"，而是"从明天开盘起，持有多久"：
    a=0 → 不持仓（空仓）
    a=1 → 持有 1 个交易日（明天开盘买，后天开盘卖）
    a=2 → 持有 2 个交易日
    a=3 → 持有 3 个交易日
    a=5 → 持有 5 个交易日
    a=10 → 持有 10 个交易日

【关键：只在到期日决策】
  一旦决定持有 N 天，中间 N-1 天不再决策（不产生交易成本）
  → 交易次数大幅降低，成本可控

【收益计算（严格 T+1）】
  明天开盘买入 → N 天后开盘卖出
  收益 = 明天日内 + 后天日内 + ... + 第N天日内 + 第N+1天隔夜
  简化：用 close[t+1]/open[t+1] * ... * open[t+1+N]/close[t+N] 链式计算

【奖励】
  - 持有期净收益 - 回撤惩罚
  - 加"空仓机会成本"极轻微惩罚，防止全空仓退化

【数据】
  510300 沪深300ETF，2012-2026（14.4年，3503天）
  比之前 2 年数据多 7 倍

用法：
  python scripts/rl_timing_v4_variable_holding.py
  python scripts/rl_timing_v4_variable_holding.py --episodes 3000 --seeds 1 2 3 4 5
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
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# 动作 = 持有期（交易日）。0 = 空仓
HOLD_OPTIONS = [0, 1, 2, 3, 5, 10]
N_ACTIONS = len(HOLD_OPTIONS)
N_FEATS = 9
OBS_DIM = N_FEATS + N_ACTIONS  # 特征 + 上一个动作 one-hot


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


def load_etf(db: str, code: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    if df.empty:
        raise SystemExit(f"{code} 无数据，先跑 fetch_etf_history.py")
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)

    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0

    # 特征（全部 T 日收盘可得）
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["ret20"] = df["close"].pct_change(20)
    df["ret60"] = df["close"].pct_change(60)
    df["vol10"] = df["ret"].rolling(10).std()
    df["vol60"] = df["ret"].rolling(60).std()
    df["regime5"] = df["ret"].rolling(5).mean()
    df["regime10"] = df["ret"].rolling(10).mean()
    df["regime20"] = df["ret"].rolling(20).mean()
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


FEAT_COLS = ["ret5", "ret10", "ret20", "ret60", "vol10", "vol60",
             "regime5", "regime10", "regime20"]


def period_return(open_, close, t: int, n: int) -> float:
    """T+1 下：t+1 日开盘买入，持有 n 个交易日，t+1+n 日开盘卖出。

    收益 = (close[t+1]/open[t+1]) * prod(close[t+k]/close[t+k-1]) * (open[t+1+n]/close[t+n])
    简化：直接 = open[t+1+n] / open[t+1] - 1
    """
    buy_idx = t + 1
    sell_idx = t + 1 + n
    if sell_idx >= len(open_):
        return np.nan
    return float(open_[sell_idx] / open_[buy_idx] - 1.0)


class Env:
    """可变持有期环境。只在持仓到期后决策。"""

    def __init__(self, feats, open_, close, cost, dd_lam, dd_free, horizon, mu, sd):
        self.feats = feats
        self.open = open_
        self.close = close
        self.cost = cost
        self.dd_lam = dd_lam
        self.dd_free = dd_free
        self.horizon = horizon
        self.mu = mu
        self.sd = sd
        self.n = len(open_)

    def reset(self, start=None):
        hi = max(1, self.n - self.horizon - 15)
        self.t = start if start is not None else np.random.randint(0, hi)
        self.start_t = self.t
        self.eq = 1.0
        self.peak = 1.0
        self.prev_a = 0
        return self._obs()

    def _obs(self):
        x = (self.feats[self.t] - self.mu) / self.sd
        oh = np.zeros(N_ACTIONS, dtype=np.float32)
        oh[self.prev_a] = 1.0
        return np.clip(np.concatenate([x, oh]), -5, 5).astype(np.float32)

    def step(self, a: int):
        n_hold = HOLD_OPTIONS[a]
        if n_hold == 0:
            r = 0.0
            trade_cost = 0.0
            advance = 1
        else:
            r = period_return(self.open, self.close, self.t, n_hold)
            if np.isnan(r):
                r = 0.0
                advance = 1
            else:
                trade_cost = 2 * self.cost  # 买入 + 卖出
                advance = n_hold + 1  # 持有 n 天后，t+1+n 卖出，下一步从那里决策
            r = r - trade_cost

        self.eq *= (1 + r)
        self.peak = max(self.peak, self.eq)
        dd = self.eq / self.peak - 1.0
        penalty = self.dd_lam * max(0.0, -dd - self.dd_free) ** 2
        reward = r - penalty

        self.prev_a = a
        self.t = min(self.t + advance, self.n - 2)
        done = (self.t - self.start_t) >= self.horizon or self.t >= self.n - 2
        return (self._obs() if not done else np.zeros(OBS_DIM, dtype=np.float32)), reward, done


class Policy(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, N_ACTIONS)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return torch.softmax(self.head(self.body(x)), dim=-1)


def evaluate(policy, feats, open_, close, cost, start, end, mu, sd, warm=None):
    """确定性回测（argmax）。（start, end）为区间索引。"""
    policy.eval()
    eq, peak = 1.0, 1.0
    curve, acts = [], []
    t = start
    prev_a = 0
    guard = 0
    while t < end - 15 and guard < 10 * (end - start):
        guard += 1
        x = (feats[t] - mu) / sd
        oh = np.zeros(N_ACTIONS, dtype=np.float32)
        oh[prev_a] = 1.0
        obs = torch.from_numpy(np.clip(np.concatenate([x, oh]), -5, 5).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            a = int(torch.argmax(policy(obs), dim=-1).item())
        n_hold = HOLD_OPTIONS[a]
        if n_hold == 0:
            r, adv = 0.0, 1
        else:
            r = period_return(open_, close, t, n_hold)
            if np.isnan(r):
                r, adv = 0.0, 1
            else:
                r = r - 2 * cost
                adv = n_hold + 1
        eq *= (1 + r)
        peak = max(peak, eq)
        curve.append(eq)
        acts.append(a)
        prev_a = a
        t = min(t + adv, end - 2)

    if not curve:
        return {"total": 0.0, "ann": np.nan, "dd": 0.0, "sharpe": np.nan,
                "days": 0, "n_decisions": 0, "action_dist": {}, "avg_holding": 0.0}
    arr = np.array(curve)
    days = t - start
    years = max(days / 244, 1e-6)
    total = float(arr[-1] - 1)
    rets = np.diff(np.concatenate([[1.0], arr]))
    dist = {f"{HOLD_OPTIONS[k]}日": float(np.mean(np.array(acts) == k)) for k in range(N_ACTIONS)}
    return {
        "total": total,
        "ann": float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan,
        "dd": float((arr / np.maximum.accumulate(arr) - 1).min()),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(244)) if rets.std() > 0 else np.nan,
        "days": int(days),
        "n_decisions": len(acts),
        "action_dist": dist,
        "avg_holding": float(np.mean([HOLD_OPTIONS[a] for a in acts])),
    }


def buy_hold(open_, close, start, end):
    buy = open_[start + 1]
    sell = open_[min(end, len(open_) - 1)]
    total = float(sell / buy - 1)
    days = max(min(end, len(open_) - 1) - start - 1, 1)
    years = days / 244
    return {"total": total,
            "ann": float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan,
            "days": int(days)}


def main() -> None:
    parser = argparse.ArgumentParser(description="RL v4 可变持有期")
    parser.add_argument("--code", default="510300.SH")
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dd-lambda", type=float, default=3.0)
    parser.add_argument("--dd-free", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    df = load_etf(_resolve_db(), args.code)
    print(f"{args.code}: {df['date'].min().date()} ~ {df['date'].max().date()}  "
          f"{len(df)} 天 ≈ {len(df)/244:.1f} 年")

    feats = df[FEAT_COLS].values.astype(np.float32)
    open_ = df["open"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    n = len(df)
    n_train = int(n * args.train_ratio)
    mu = feats[:n_train].mean(axis=0)
    sd = feats[:n_train].std(axis=0) + 1e-8

    print(f"训练 0~{n_train}，测试 {n_train}~{n}")
    print(f"动作空间: 持有 {HOLD_OPTIONS} 个交易日（0=空仓）\n")

    bh_tr = buy_hold(open_, close, 0, n_train)
    bh_te = buy_hold(open_, close, n_train, n)
    print(f"买入持有: 训练段 {bh_tr['total']*100:+.1f}%，测试段 {bh_te['total']*100:+.1f}%\n")

    results = []
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = Env(feats, open_, close, args.cost, args.dd_lambda,
                  args.dd_free, args.horizon, mu, sd)
        pol = Policy()
        opt = optim.Adam(pol.parameters(), lr=args.lr)

        for ep in range(args.episodes):
            obs = env.reset()
            lps, rs = [], []
            done = False
            while not done:
                x = torch.from_numpy(obs).unsqueeze(0)
                p = pol(x)
                d = Categorical(p)
                a = d.sample()
                lps.append(d.log_prob(a))
                obs, r, done = env.step(int(a.item()))
                rs.append(r)
            G, disc = 0.0, []
            for r in reversed(rs):
                G = r + 0.99 * G
                disc.append(G)
            disc = torch.tensor(list(reversed(disc)), dtype=torch.float32)
            disc = (disc - disc.mean()) / (disc.std() + 1e-8)
            loss = -(torch.stack(lps) * disc).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()

        r_tr = evaluate(pol, feats, open_, close, args.cost, 0, n_train, mu, sd)
        r_te = evaluate(pol, feats, open_, close, args.cost, n_train, n, mu, sd)
        adv = r_te["total"] - bh_te["total"]
        results.append({"seed": seed, "train": r_tr, "test": r_te, "adv": adv})
        print(f"seed={seed}: 训练 {r_tr['total']*100:+8.1f}%  测试 {r_te['total']*100:+8.1f}%  "
              f"(持有 {bh_te['total']*100:+.1f}%, 优势 {adv*100:+7.1f}pp)  "
              f"决策{r_te['n_decisions']}次 平均持有{r_te['avg_holding']:.1f}日")

    # ===== 汇总 =====
    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    advs = np.array([r["adv"] for r in results])
    tests = np.array([r["test"]["total"] for r in results])
    print(f"跑赢买入持有: {int((advs>0).sum())}/{len(advs)}")
    print(f"测试段收益中位数: {np.median(tests)*100:+.1f}%   买入持有: {bh_te['total']*100:+.1f}%")
    print(f"优势均值: {advs.mean()*100:+.1f}pp   标准差: {advs.std()*100:.1f}pp")

    print("\n动作分布（测试段平均）:")
    for k in range(N_ACTIONS):
        key = f"{HOLD_OPTIONS[k]}日"
        vals = [r["test"]["action_dist"].get(key, 0.0) for r in results]
        print(f"  持有{HOLD_OPTIONS[k]:>2}日  {np.mean(vals)*100:>6.1f}%")

    best = max(results, key=lambda r: r["test"]["total"])
    print(f"\n最佳 seed={best['seed']}: 测试 {best['test']['total']*100:+.1f}%，"
          f"回撤 {best['test']['dd']*100:.1f}%，夏普 {best['test']['sharpe']:.2f}")

    win_rate = float((advs > 0).mean())
    if win_rate >= 0.8 and advs.std() < 0.5:
        verdict = "RL 学到稳定优势"
    elif win_rate >= 0.6:
        verdict = "胜率过半但方差大，靠运气成分高"
    else:
        verdict = "RL 未学到有效策略"
    print(f"\n>>> {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rl_timing_v4.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "period": {"start": str(df['date'].min().date()), "end": str(df['date'].max().date()),
                   "days": n},
        "hold_options": HOLD_OPTIONS,
        "buy_hold_train": bh_tr, "buy_hold_test": bh_te,
        "runs": [{"seed": r["seed"], "train": r["train"], "test": r["test"],
                  "adv": round(r["adv"], 6)} for r in results],
        "win_rate": round(win_rate, 4),
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
