"""验证 RL v2 的结果是真的学到，还是运气好。

检查三件事：
  1. 换 5 个随机种子，测试段结果是否稳定？
  2. 换不同训练/测试切分点，是否还能赢？
  3. RL 的决策和什么特征相关？（看它是否学到了可解释的规律）

用法：
  python scripts/verify_rl_robustness.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_timing_v2 as R


def train_one(feats, rets, n_train, args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    mu = feats[:n_train].mean(axis=0)
    sd = feats[:n_train].std(axis=0) + 1e-8
    env = R.Env(feats, rets, args.cost, args.dd_lambda, args.dd_free, args.horizon, mu, sd)
    policy = R.Policy()
    opt = optim.Adam(policy.parameters(), lr=args.lr)

    for ep in range(args.episodes):
        obs = env.reset()
        # 只在训练段内采起点
        env.t = np.random.randint(0, max(1, n_train - args.horizon - 2))
        lps, rs = [], []
        done = False
        while not done:
            x = torch.from_numpy(obs).unsqueeze(0)
            probs = policy(x)
            d = Categorical(probs)
            a = d.sample()
            lps.append(d.log_prob(a))
            obs, r, done = env.step(int(a.item()))
            rs.append(r)
        G, rets_disc = 0.0, []
        for r in reversed(rs):
            G = r + 0.99 * G
            rets_disc.append(G)
        rets_disc = torch.tensor(list(reversed(rets_disc)), dtype=torch.float32)
        rets_disc = (rets_disc - rets_disc.mean()) / (rets_disc.std() + 1e-8)
        loss = -(torch.stack(lps) * rets_disc).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    return policy, mu, sd


def eval_policy(policy, feats, rets, cost, start, duration, mu, sd):
    policy.eval()
    pos, eq = 0.0, 1.0
    curve, acts = [], []
    with torch.no_grad():
        for t in range(start, start + duration):
            x = torch.from_numpy(np.clip((feats[t] - mu) / sd, -5, 5).astype(np.float32)).unsqueeze(0)
            a = int(torch.argmax(policy(x), dim=-1).item())
            target = R.ACTIONS[a]
            eq *= (1 + target * rets[t] - abs(target - pos) * cost)
            curve.append(eq)
            acts.append(target)
            pos = target
    arr = np.array(curve)
    r = np.diff(np.concatenate([[1.0], arr]))
    dd = float((arr / np.maximum.accumulate(arr) - 1).min())
    total = float(arr[-1] - 1)
    years = duration / 244
    return {
        "total": total,
        "ann": (1 + total) ** (1 / years) - 1 if years > 0 else np.nan,
        "dd": dd,
        "sharpe": float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan,
        "avg_pos": float(np.mean(acts)),
        "switches": int(np.sum(np.diff(acts) != 0)),
        "acts": acts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dd-lambda", type=float, default=5.0)
    parser.add_argument("--dd-free", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42, 123, 2024])
    args = parser.parse_args()

    df = R.load_market(R._resolve_db())
    feat_cols = ["ret5", "ret20", "vol10", "regime10", "regime20", "vol_regime", "run_len", "ret"]
    feats = df[feat_cols].values.astype(np.float32)
    rets = df["ret"].values.astype(np.float32)
    n = len(rets)
    n_train = int(n * 0.7)

    print(f"数据 {len(rets)} 天，训练 0~{n_train}，测试 {n_train}~{n}")
    bh_test = float(np.prod(1 + rets[n_train:]) - 1)
    print(f"测试段买入持有: {bh_test*100:+.1f}%\n")

    print("=" * 95)
    print("【检查1】换 5 个随机种子，看测试段是否稳定")
    print("=" * 95)
    print(f"{'seed':<8}{'测试段收益':>13}{'买入持有':>12}{'优势':>11}{'回撤':>10}{'夏普':>9}{'平均仓位':>10}{'切换':>7}")
    rows = []
    for s in args.seeds:
        pol, mu, sd = train_one(feats, rets, n_train, args, s)
        r = eval_policy(pol, feats, rets, args.cost, n_train, n - n_train, mu, sd)
        adv = r["total"] - bh_test
        flag = "WIN " if adv > 0 else "LOSE"
        print(f"{s:<8}{r['total']*100:>+12.1f}%{bh_test*100:>+11.1f}%"
              f"{flag}{adv*100:>+7.1f}pp{r['dd']*100:>9.1f}%{r['sharpe']:>9.2f}"
              f"{r['avg_pos']*100:>9.1f}%{r['switches']:>7}")
        rows.append({"seed": s, **{k: v for k, v in r.items() if k != "acts"}})

    n_win = sum(1 for r in rows if r["total"] > bh_test)
    print(f"\n  跑赢买入持有: {n_win}/{len(rows)}")
    print(f"  收益中位数: {np.median([r['total'] for r in rows])*100:+.1f}%  "
          f"vs 买入持有 {bh_test*100:+.1f}%")
    print(f"  平均仓位中位数: {np.median([r['avg_pos'] for r in rows])*100:.1f}%")

    print()
    print("=" * 95)
    print("【检查2】换不同切分点（训练比例）")
    print("=" * 95)
    print(f"{'训练比例':<12}{'训练天数':>10}{'测试天数':>10}{'测试RL':>12}{'测试买入持有':>14}{'优势':>12}")
    split_rows = []
    for ratio in [0.5, 0.6, 0.7, 0.8]:
        nt = int(n * ratio)
        pol, mu, sd = train_one(feats, rets, nt, args, 42)
        r = eval_policy(pol, feats, rets, args.cost, nt, n - nt, mu, sd)
        bh = float(np.prod(1 + rets[nt:]) - 1)
        adv = r["total"] - bh
        flag = "WIN " if adv > 0 else "LOSE"
        print(f"{ratio:<12.1f}{nt:>10}{n-nt:>10}{r['total']*100:>+11.1f}%"
              f"{bh*100:>+13.1f}%{flag}{adv*100:>+7.1f}pp")
        split_rows.append({"ratio": ratio, "rl": r["total"], "bh": bh, "adv": adv})

    print()
    print("=" * 95)
    print("【检查3】RL 在什么情况下减仓？（用最佳种子分析决策规律）")
    print("=" * 95)
    best = max(rows, key=lambda r: r["total"])
    pol, mu, sd = train_one(feats, rets, n_train, args, best["seed"])
    r = eval_policy(pol, feats, rets, args.cost, n_train, n - n_train, mu, sd)
    acts = np.array(r["acts"])
    seg = feats[n_train:n_train + len(acts)]
    names = ["ret5", "ret20", "vol10", "regime10", "regime20", "vol_regime", "run_len", "ret"]
    print(f"用 seed={best['seed']}（测试段 {r['total']*100:+.1f}%）")
    print(f"\n{'特征':<14}{'满仓时均值':>14}{'减仓时均值':>14}{'差异':>12}{'相关性':>10}")
    full = acts >= 0.75
    reduced = acts < 0.75
    if full.sum() > 5 and reduced.sum() > 5:
        for i, nm in enumerate(names):
            f_mean = seg[full, i].mean()
            r_mean = seg[reduced, i].mean()
            corr = float(np.corrcoef(seg[:, i], acts)[0, 1])
            print(f"{nm:<14}{f_mean:>14.4f}{r_mean:>14.4f}{f_mean-r_mean:>12.4f}{corr:>10.3f}")
    else:
        print("  样本不足，无法分析")

    out = Path(__file__).resolve().parents[1] / "results" / "rl_robustness.json"
    out.write_text(json.dumps({
        "seeds": rows, "n_win": n_win, "buy_hold_test": bh_test,
        "splits": split_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
