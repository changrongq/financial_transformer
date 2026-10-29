"""RL v5：择时骨架 + RL 只学"CASH 时卖多少"。

=================== 与前四次的本质区别 ===================
v1-v4 都在「从零学决策」：学仓位、学持有期，动作空间大，样本几千个 → 必然过拟合
v5 把骨架焊死，RL 只学一个细节

【骨架（固定，不可改）】
  大盘（全市场等权）10日均收益 >= 0 → HOLD（满仓）
  < 0 → CASH（当前实现：全部卖出）

【RL 只学一件事】
  CASH 信号出现时，卖多少？
    动作 0 = 全卖（0% 仓位）      ← 当前规则的行为
    动作 1 = 卖一半（50% 仓位）
    动作 2 = 不卖（100% 仓位）

【为什么这个设计能学】
  · 参数量：只有 3 个动作输出，网络极小
  · 样本量：14年有 241 次 CASH 信号，足够
  · 骨架已提供 90% 的收益，RL 只需优化边际
  · 即使 RL 完全学不出来，退化为"全卖"也是原策略，不会更差

【特征（CASH 时刻的状态）】
  · 大盘10日均收益（信号强度）
  · 近5日收益、近20日收益
  · 波动率、波动率相对水平
  · 连续下跌天数
  · 距离上次 CASH 天数

【奖励】
  相对"全卖"基准的超额收益（让 RL 专注于改进，而非重复实现择时）

用法：
  python scripts/rl_overlay_on_timing.py
  python scripts/rl_overlay_on_timing.py --episodes 2000 --seeds 1 2 3 4 5
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

# 动作 = CASH 时的目标仓位
CASH_POSITIONS = [0.0, 0.5, 1.0]
N_ACTIONS = len(CASH_POSITIONS)
HOLD_POSITION = 1.0
N_FEATS = 7
OBS_DIM = N_FEATS


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


def load_etf(db: str, code: str, window: int = 10) -> pd.DataFrame:
    """加载 ETF 并计算择时信号（用 ETF 自身的 10 日均收益，避免口径混杂）。"""
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)

    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0

    # 择时信号（T日收盘算）
    df["regime"] = df["ret"].rolling(window, min_periods=3).mean()
    df["signal_hold"] = df["regime"] >= 0

    # CASH 时刻的特征（全部 T 日可得）
    df["ret5"] = df["close"].pct_change(5)
    df["ret20"] = df["close"].pct_change(20)
    df["vol10"] = df["ret"].rolling(10).std()
    df["vol60"] = df["ret"].rolling(60).std()
    df["vol_ratio"] = df["vol10"] / df["vol60"].replace(0, np.nan)
    # 连续下跌天数
    neg = (df["ret"] < 0).astype(int)
    run = neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1
    df["down_run"] = np.where(neg == 1, run, 0)
    # 距上次 CASH 的天数
    cash = (~df["signal_hold"]).astype(int)
    grp = cash.groupby((cash != cash.shift()).cumsum()).cumcount() + 1
    df["cash_age"] = np.where(cash == 1, grp, 0)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


FEAT_COLS = ["regime", "ret5", "ret20", "vol10", "vol_ratio", "down_run", "cash_age"]


def rollout(df: pd.DataFrame, local_idx, cost: float, mu, sd, policy=None,
            greedy: bool = True, p_override: float | None = None):
    """执行一次完整回测。

    local_idx: 只在满足条件的时刻做 RL 决策的行索引列表（CASH 转 CASH 的每日）
    返回净值曲线 + 动作记录。

    逻辑：
      每个交易日 T：
        - 若 T-1 信号为 HOLD → 仓位 = 1.0，吃 T 日收益（T 开盘已持有）
        - 若 T-1 信号为 CASH → RL 决定仓位 ∈ {0, 0.5, 1}，吃 T 日收益
      成本：仓位变动时收 2×cost（买卖各一次，保守）
    """
    n = len(df)
    ret = df["ret"].values.astype(np.float64)
    sig = df["signal_hold"].values
    feats = df[FEAT_COLS].values.astype(np.float32)
    norm = np.clip((feats - mu.astype(np.float32)) / sd.astype(np.float32), -5, 5).astype(np.float32)

    eq = 1.0
    curve = []
    pos = 1.0  # 起始假设满仓
    acts = []
    probs_log = []

    for i in range(1, n):
        if sig[i - 1]:
            target = HOLD_POSITION
        else:
            if p_override is not None:
                # 固定仓位覆盖（用于基线：总是半仓）
                target = p_override
            elif policy is not None:
                x = torch.from_numpy(norm[i]).unsqueeze(0)
                with torch.no_grad():
                    p = policy(x).squeeze().numpy()
                if greedy:
                    a = int(np.argmax(p))
                else:
                    a = int(np.random.choice(N_ACTIONS, p=p))
                target = CASH_POSITIONS[a]
                acts.append(a)
                probs_log.append(p)
            else:
                target = 0.0  # 基准：全卖

        if target != pos:
            eq *= (1 - 2 * cost)
        eq *= (1 + target * ret[i])
        pos = target
        curve.append(eq)

    return np.array(curve), acts, probs_log


def stats(curve: np.ndarray, n_days: int) -> dict:
    if len(curve) < 2:
        return {"total": 0.0, "ann": np.nan, "dd": 0.0, "sharpe": np.nan}
    r = np.diff(np.concatenate([[1.0], curve]))
    total = float(curve[-1] - 1)
    years = max(n_days / 244, 1e-6)
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"total": total,
            "ann": float((1 + total) ** (1 / years) - 1),
            "dd": dd, "sharpe": sharpe}


class Policy(nn.Module):
    def __init__(self, hidden=16):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="RL v5: 择时骨架 + RL 学 CASH 时卖多少")
    parser.add_argument("--code", default="510300.SH")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    df = load_etf(_resolve_db(), args.code, args.window)
    n = len(df)
    print(f"{args.code}: {df['date'].min().date()} ~ {df['date'].max().date()}  "
          f"{n} 天 ≈ {n/244:.1f} 年")
    n_cash = int((~df["signal_hold"]).sum())
    print(f"HOLD 天数 {int(df['signal_hold'].sum())}，CASH 天数 {n_cash}")
    print(f"动作空间: CASH 时仓位 {CASH_POSITIONS}\n")

    n_train = int(n * args.train_ratio)
    mu = df[FEAT_COLS].values[:n_train].mean(axis=0).astype(np.float32)
    sd = (df[FEAT_COLS].values[:n_train].std(axis=0) + 1e-8).astype(np.float32)

    # ===== 基线 =====
    print("=" * 96)
    print("基线（训练段 / 测试段）")
    print("=" * 96)
    curve_base_tr, _, _ = rollout(df.iloc[:n_train].reset_index(drop=True), None, args.cost, mu, sd, p_override=0.0)
    curve_base_te, _, _ = rollout(df.iloc[n_train:].reset_index(drop=True), None, args.cost, mu, sd, p_override=0.0)
    s_base_tr = stats(curve_base_tr, n_train)
    s_base_te = stats(curve_base_te, n - n_train)

    curve_half_tr, _, _ = rollout(df.iloc[:n_train].reset_index(drop=True), None, args.cost, mu, sd, p_override=0.5)
    curve_half_te, _, _ = rollout(df.iloc[n_train:].reset_index(drop=True), None, args.cost, mu, sd, p_override=0.5)
    s_half_tr = stats(curve_half_tr, n_train)
    s_half_te = stats(curve_half_te, n - n_train)

    bench = [
        ("买入持有", None),
        ("择时-全卖（当前规则）", (s_base_tr, s_base_te)),
        ("择时-总是半仓", (s_half_tr, s_half_te)),
    ]
    # 买入持有
    bh_tr = float(df["close"].iloc[n_train - 1] / df["close"].iloc[0] - 1)
    bh_te = float(df["close"].iloc[-1] / df["close"].iloc[n_train - 1] - 1)

    print(f"{'策略':<26}{'训练段':>14}{'测试段':>14}{'训练回撤':>12}{'测试回撤':>12}")
    print("-" * 96)
    print(f"{'买入持有':<26}{bh_tr*100:>+13.1f}%{bh_te*100:>+13.1f}%{'-':>12}{'-':>12}")
    print(f"{'择时-全卖（当前规则）':<26}{s_base_tr['total']*100:>+13.1f}%{s_base_te['total']*100:>+13.1f}%"
          f"{s_base_tr['dd']*100:>11.1f}%{s_base_te['dd']*100:>11.1f}%")
    print(f"{'择时-总是半仓':<26}{s_half_tr['total']*100:>+13.1f}%{s_half_te['total']*100:>+13.1f}%"
          f"{s_half_tr['dd']*100:>11.1f}%{s_half_te['dd']*100:>11.1f}%")
    print("-" * 96)

    # ===== 训练 RL =====
    print(f"\n训练 {args.episodes} episodes × {len(args.seeds)} seeds\n")
    results = []
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        pol = Policy()
        opt = optim.Adam(pol.parameters(), lr=args.lr)

        # 预计算训练段特征
        sub_tr = df.iloc[:n_train].reset_index(drop=True)
        norm = np.clip((sub_tr[FEAT_COLS].values.astype(np.float32) - mu) / sd, -5, 5).astype(np.float32)
        sig = sub_tr["signal_hold"].values
        ret = sub_tr["ret"].values.astype(np.float64)

        for ep in range(args.episodes):
            # 从随机时点开始一条 episode
            start = np.random.randint(5, max(6, n_train - 300))
            eq = 1.0
            pos = 1.0
            lps, rewards = [], []
            for i in range(start + 1, min(start + 300, n_train)):
                if sig[i - 1]:
                    target = 1.0
                else:
                    x = torch.from_numpy(norm[i]).unsqueeze(0)
                    p = pol(x)
                    d = Categorical(p)
                    a = d.sample()
                    lps.append(d.log_prob(a))
                    target = CASH_POSITIONS[int(a.item())]

                # 奖励 = 该日组合收益 - "全卖"基准的收益（专门优化边际改进）
                r_port = target * ret[i]
                r_base = 0.0 if not sig[i - 1] else ret[i]  # CASH时全卖收益=0
                cost_pen = 0.0
                if target != pos:
                    cost_pen = 2 * args.cost
                reward = r_port - r_base - cost_pen
                rewards.append(reward)

                eq *= (1 + r_port - cost_pen)
                pos = target

            if not lps:
                continue
            G, disc = 0.0, []
            for r in reversed(rewards):
                G = r + 0.99 * G
                disc.append(G)
            disc = torch.tensor(list(reversed(disc)), dtype=torch.float32)
            if disc.std() > 1e-8:
                disc = (disc - disc.mean()) / disc.std()
            loss = -(torch.stack(lps) * disc).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()

        # 评估（训练段 + 测试段）
        c_tr, acts_tr, _ = rollout(sub_tr, None, args.cost, mu, sd, policy=pol, greedy=True)
        sub_te = df.iloc[n_train:].reset_index(drop=True)
        c_te, acts_te, _ = rollout(sub_te, None, args.cost, mu, sd, policy=pol, greedy=True)
        s_tr = stats(c_tr, n_train)
        s_te = stats(c_te, n - n_train)

        # 动作分布
        dist = {}
        if acts_te:
            arr = np.array(acts_te)
            for k in range(N_ACTIONS):
                dist[f"仓位{CASH_POSITIONS[k]*100:.0f}%"] = round(float(np.mean(arr == k)), 4)

        adv_base = s_te["total"] - s_base_te["total"]
        results.append({"seed": seed, "train": s_tr, "test": s_te,
                        "action_dist": dist, "adv_vs_base": adv_base,
                        "n_cash_decisions": len(acts_te)})
        print(f"seed={seed}: 训练 {s_tr['total']*100:+8.1f}%  测试 {s_te['total']*100:+8.1f}%  "
              f"(全卖基准 {s_base_te['total']*100:+.1f}%, 优势 {adv_base*100:+7.1f}pp)  "
              f"CASH决策{len(acts_te)}次")

    # ===== 汇总 =====
    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    advs = np.array([r["adv_vs_base"] for r in results])
    tests = np.array([r["test"]["total"] for r in results])
    print(f"相对「全卖」基准: 跑赢 {int((advs>0).sum())}/{len(advs)}，"
          f"平均优势 {advs.mean()*100:+.1f}pp，标准差 {advs.std()*100:.1f}pp")
    print(f"测试段收益中位数: {np.median(tests)*100:+.1f}%  （全卖基准 {s_base_te['total']*100:+.1f}%）")

    print("\n动作分布（测试段平均）:")
    for k in range(N_ACTIONS):
        key = f"仓位{CASH_POSITIONS[k]*100:.0f}%"
        vals = [r["action_dist"].get(key, 0.0) for r in results]
        print(f"  CASH时{k:}→{CASH_POSITIONS[k]*100:>3.0f}%仓位  {np.mean(vals)*100:>6.1f}%")

    win_rate = float((advs > 0).mean())
    if win_rate >= 0.8 and advs.mean() > 0.01:
        verdict = "RL 稳定改进了择时（有价值！）"
    elif win_rate >= 0.6:
        verdict = "多数 seed 有改进，但幅度小/不稳定"
    else:
        verdict = "RL 未能改进择时，建议用固定规则"

    print(f"\n>>> {verdict}")
    print("=" * 96)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rl_overlay_on_timing.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "baselines": {
            "buy_hold": {"train": bh_tr, "test": bh_te},
            "timing_full_sell": {"train": s_base_tr, "test": s_base_te},
            "timing_half": {"train": s_half_tr, "test": s_half_te},
        },
        "runs": [{"seed": r["seed"],
                  "train": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r["train"].items()},
                  "test": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r["test"].items()},
                  "action_dist": r["action_dist"],
                  "adv_vs_base": round(r["adv_vs_base"], 6),
                  "n_cash_decisions": r["n_cash_decisions"]} for r in results],
        "win_rate": round(win_rate, 4),
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
