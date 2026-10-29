"""RL 择时 v3：多时间点执行 + 14年真实 ETF 数据。

船长的新设想：
  "买入和卖出不一定按开盘价/收盘价，可以是不同时间点"

可用的时间点（日线数据能支持的）：
  - 昨收 (T-1 close)     —— 已有
  - 今开 (T open)        —— 已有（隔夜跳空后的价格）
  - 今收 (T close)       —— 已有

关键发现（本脚本会复用）：
  沪深300ETF 14年年化：隔夜 -10.76%，日内 +15.92%
  → 策略的真正价值在于「避开隔夜、捕捉日内」

所以动作空间设计为：
  0 = 空仓（不持有，避开隔夜+日内）
  1 = 隔夜仓（昨收买，今开卖：只吃隔夜）      ← 反向实验
  2 = 日内仓（今开买，今收卖：只吃日内）      ← 重点
  3 = 全天仓（昨收/今开持有到今收：吃隔夜+日内）
  4 = 满仓持有（今天开盘买，一直持有）

奖励：净收益 - 回撤惩罚（不用超额，避免 v1 的零梯度陷阱）

数据：510300 的 3503 天（14.4年），对 RL 来说是 478 天的 7.3 倍。

诚实预期：
  即便 3503 天，对深度 RL 仍然偏少，但配合「小网络 + 少动作」有可能学到。

用法：
  python scripts/rl_timing_v3_etf.py
  python scripts/rl_timing_v3_etf.py --episodes 4000 --horizon 250
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

# 动作 -> (是否吃隔夜, 是否吃日内)
ACTION_SPEC = {
    0: (False, False),  # 空仓
    1: (True, False),   # 只吃隔夜（今开卖出）
    2: (False, True),   # 只吃日内（今开买入，今收卖出）
    3: (True, True),    # 全天（昨收持有到今收）
}
ACTIONS = sorted(ACTION_SPEC.keys())
ACTION_NAMES = {0: "空仓", 1: "隔夜仓", 2: "日内仓", 3: "全天仓"}
N_FEATS = 10
OBS_DIM = N_FEATS + len(ACTIONS)  # 10 特征 + 4 one-hot = 14


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
        raise SystemExit(f"{code} 无数据")
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)

    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)

    # 技术特征（全部用 T 日收盘可得信息，避免未来函数）
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["ret20"] = df["close"].pct_change(20)
    df["vol5"] = df["ret"].rolling(5).std()
    df["vol20"] = df["ret"].rolling(20).std()
    df["vol60"] = df["ret"].rolling(60).std()
    df["regime5"] = df["ret"].rolling(5).mean()
    df["regime10"] = df["ret"].rolling(10).mean()
    df["gap_ma5"] = df["gap"].rolling(5).mean()
    df["intraday_ma5"] = df["intraday"].rolling(5).mean()

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


FEAT_COLS = ["ret5", "ret10", "ret20", "vol5", "vol20", "vol60",
             "regime5", "regime10", "gap_ma5", "intraday_ma5"]


class Env:
    """多时间点执行环境。

    关键：T 日收盘时看到特征，决定 T+1 日的动作。
    动作在 T+1 日执行，吃到 T+1 日的隔夜/日内收益。
    """

    def __init__(self, feats, gap, intra, ret, cost, dd_lam, dd_free, horizon, mu, sd):
        self.feats, self.gap, self.intra, self.ret = feats, gap, intra, ret
        self.cost = cost
        self.dd_lam = dd_lam
        self.dd_free = dd_free
        self.horizon = horizon
        self.mu, self.sd = mu, sd

    def reset(self, start=None):
        max_start = len(self.ret) - self.horizon - 2
        self.t = start if start is not None else np.random.randint(0, max(1, max_start))
        self.eq = 1.0
        self.peak = 1.0
        self.steps = 0
        self.prev_action = 0
        return self._obs()

    def _obs(self):
        x = (self.feats[self.t] - self.mu) / self.sd
        onehot = np.zeros(len(ACTIONS), dtype=np.float32)
        onehot[self.prev_action] = 1.0
        return np.clip(np.concatenate([x, onehot]), -5, 5).astype(np.float32)

    def step(self, a: int):
        # a 决定 T+1 日的持仓方式；T+1 的收益 = gap + intraday
        t1 = min(self.t + 1, len(self.ret) - 1)
        overnight, day = ACTION_SPEC[a]
        r = 0.0
        if overnight:
            r += self.gap[t1]
        if day:
            r += self.intra[t1]
        trade = self.cost if a != self.prev_action else 0.0
        net = r - trade

        self.eq *= (1 + net)
        self.peak = max(self.peak, self.eq)
        dd = self.eq / self.peak - 1.0
        penalty = self.dd_lam * max(0.0, -dd - self.dd_free) ** 2
        reward = net - penalty

        self.prev_action = a
        self.t = t1
        self.steps += 1
        done = self.steps >= self.horizon or self.t >= len(self.ret) - 1
        return (self._obs() if not done else np.zeros(OBS_DIM, dtype=np.float32)), reward, done


class Policy(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, len(ACTIONS))
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return torch.softmax(self.head(self.body(x)), dim=-1)


def evaluate(policy, feats, gap, intra, ret, cost, start, duration, mu, sd):
    policy.eval()
    eq, peak, curve, acts = 1.0, 1.0, [], []
    prev = 0
    with torch.no_grad():
        for t in range(start, min(start + duration, len(ret) - 1)):
            x = (feats[t] - mu) / sd
            oh = np.zeros(len(ACTIONS), dtype=np.float32)
            oh[prev] = 1.0
            obs = torch.from_numpy(np.clip(np.concatenate([x, oh]), -5, 5).astype(np.float32)).unsqueeze(0)
            a = int(torch.argmax(policy(obs), dim=-1).item())
            overnight, day = ACTION_SPEC[a]
            r = 0.0
            if overnight:
                r += gap[t + 1]
            if day:
                r += intra[t + 1]
            r -= cost if a != prev else 0.0
            eq *= (1 + r)
            peak = max(peak, eq)
            curve.append(eq)
            acts.append(a)
            prev = a
    arr = np.array(curve)
    r = np.diff(np.concatenate([[1.0], arr]))
    total = float(arr[-1] - 1)
    years = len(arr) / 244
    return {
        "total": total,
        "ann": (1 + total) ** (1 / years) - 1 if years > 0 else np.nan,
        "dd": float((arr / np.maximum.accumulate(arr) - 1).min()),
        "sharpe": float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan,
        "days": len(arr),
        "action_dist": {ACTION_NAMES[a]: float(np.mean(np.array(acts) == a)) for a in ACTIONS},
        "n_switches": int(np.sum(np.diff(acts) != 0)) if len(acts) > 1 else 0,
        "acts": acts,
    }


def bench_hold(gap, intra, start, duration):
    """满仓持有的基准（吃隔夜+日内）。"""
    t = np.arange(start + 1, min(start + duration + 1, len(gap)))
    r = gap[t] + intra[t]
    eq = np.cumprod(1 + r)
    total = float(eq[-1] - 1)
    years = len(eq) / 244
    return {
        "total": total,
        "ann": (1 + total) ** (1 / years) - 1 if years > 0 else np.nan,
        "dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
        "sharpe": float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RL 多时间点择时 (14年ETF)")
    parser.add_argument("--code", default="510300.SH")
    parser.add_argument("--episodes", type=int, default=4000)
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
    gap = df["gap"].values.astype(np.float32)
    intra = df["intraday"].values.astype(np.float32)
    ret = df["ret"].values.astype(np.float32)

    n = len(df)
    n_train = int(n * args.train_ratio)
    mu = feats[:n_train].mean(axis=0)
    sd = feats[:n_train].std(axis=0) + 1e-8
    print(f"训练 {n_train} 天 / 测试 {n - n_train} 天")
    print(f"动作空间: {[f'{k}={v}' for k, v in ACTION_NAMES.items()]}\n")

    bh_train = bench_hold(gap, intra, 0, n_train)
    bh_test = bench_hold(gap, intra, n_train, n - n_train)
    print(f"买入持有: 训练段 {bh_train['total']*100:+.1f}%  测试段 {bh_test['total']*100:+.1f}%\n")

    results = []
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = Env(feats, gap, intra, ret, args.cost, args.dd_lambda,
                  args.dd_free, args.horizon, mu, sd)
        pol = Policy()
        opt = optim.Adam(pol.parameters(), lr=args.lr)

        for ep in range(args.episodes):
            obs = env.reset()
            env.t = np.random.randint(0, max(1, n_train - args.horizon - 2))
            lps, rs = [], []
            done = False
            while not done:
                x = torch.from_numpy(obs).unsqueeze(0)
                p = pol(x)
                dist = Categorical(p)
                a = dist.sample()
                lps.append(dist.log_prob(a))
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

        r_tr = evaluate(pol, feats, gap, intra, ret, args.cost, 0, n_train, mu, sd)
        r_te = evaluate(pol, feats, gap, intra, ret, args.cost, n_train, n - n_train, mu, sd)
        results.append({"seed": seed, "train": r_tr, "test": r_te,
                        "adv_test": r_te["total"] - bh_test["total"]})
        print(f"seed={seed}: 训练 {r_tr['total']*100:+7.1f}%  测试 {r_te['total']*100:+7.1f}%  "
              f"(买入持有 {bh_test['total']*100:+.1f}%, 优势 {results[-1]['adv_test']*100:+.1f}pp)  "
              f"切换 {r_te['n_switches']}")

    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    advs = np.array([r["adv_test"] for r in results])
    tests = np.array([r["test"]["total"] for r in results])
    print(f"跑赢买入持有: {int((advs>0).sum())}/{len(advs)}")
    print(f"测试段收益中位数: {np.median(tests)*100:+.1f}%  买入持有: {bh_test['total']*100:+.1f}%")
    print(f"优势均值: {advs.mean()*100:+.1f}pp   标准差: {advs.std()*100:.1f}pp")

    print("\n各动作使用比例（测试段）:")
    for a in ACTIONS:
        vals = [r["test"]["action_dist"][ACTION_NAMES[a]] for r in results]
        print(f"  {ACTION_NAMES[a]:<6} {np.mean(vals)*100:>6.1f}%")

    best = max(results, key=lambda r: r["test"]["total"])
    print(f"\n最佳 seed={best['seed']}: 测试 {best['test']['total']*100:+.1f}%, "
          f"回撤 {best['test']['dd']*100:.1f}%, 夏普 {best['test']['sharpe']:.2f}")

    verdict = ("RL 学到稳定优势" if (advs > 0).mean() >= 0.8 and advs.std() < 0.5
               else "结果不稳定（方差大，靠运气）")
    print(f"\n>>> {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rl_timing_v3_etf.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "period": {"start": str(df['date'].min().date()), "end": str(df['date'].max().date()),
                   "days": n},
        "buy_hold_train": bh_train, "buy_hold_test": bh_test,
        "runs": [{"seed": r["seed"],
                  "train": {k: v for k, v in r["train"].items() if k != "acts"},
                  "test": {k: v for k, v in r["test"].items() if k != "acts"},
                  "adv_test": r["adv_test"]} for r in results],
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
