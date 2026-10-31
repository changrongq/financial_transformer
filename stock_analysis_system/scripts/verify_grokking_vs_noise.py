"""验证：RL 的好结果，是「顿悟」还是「噪声运气」？

=================== 判断标准 ===================
"顿悟"(grokking) 的特征：
  1. 训练集和验证集同时改善（泛化突然发生）
  2. 学习曲线有明显"突变点"（前慢后快，然后稳住）
  3. 多个种子都能达到相似水平（最优解可复现）
  4. 训练集结果集中（不分散）

"噪声运气" 的特征：
  1. 只有少数种子碰巧好，多数差
  2. 学习曲线平坦无特征（纯随机游走）
  3. 换种子结果天差地别
  4. 训练集本身就分散（连过拟合都做不到）

=================== 本实验设计 ===================
用 RL v5（择时骨架 + CASH 时卖多少，3 动作），因为它：
  · 动作空间最小（3个）
  · 骨架已经很好，不需要 RL 做太多
  · 如果连这个都学不出，说明问题不可学

跑 20 个种子，每个记录：
  · 训练集最终收益
  · 测试集最终收益
  · 训练过程中的收益曲线（每 N 轮采样一次）
  · 最终策略的确定性程度（entropy）

然后判断：
  · 如果训练集分布集中在正区间 → 地形有信号
  · 如果训练集分布散乱（跨度大）→ 无信号
  · 如果曲线有突变 → 可能顿悟

用法：
  python scripts/verify_grokking_vs_noise.py
  python scripts/verify_grokking_vs_noise.py --seeds 20 --episodes 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_overlay_on_timing as R
from rl_algo_utils import paired_policy_loss, discounted_returns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def train_with_curve(df, n_train, mu, sd, args, seed, snapshot_every=100):
    """训练并记录学习曲线。"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    pol = R.Policy()
    opt = optim.Adam(pol.parameters(), lr=args.lr)

    sub_tr = df.iloc[:n_train].reset_index(drop=True)
    norm = np.clip((sub_tr[R.FEAT_COLS].values.astype(np.float32) - mu) / sd, -5, 5).astype(np.float32)
    sig = sub_tr["signal_hold"].values
    ret = sub_tr["ret"].values.astype(np.float64)
    gap_a = sub_tr["gap"].values.astype(np.float64)
    intra_a = sub_tr["intraday"].values.astype(np.float64)

    curve_snapshots = []
    ent_snapshots = []

    for ep in range(1, args.episodes + 1):
        start = np.random.randint(5, max(6, n_train - 300))
        end = min(start + 300, n_train)
        pos = 1.0
        lps, rewards = [], []
        for i in range(start + 1, end):
            # ★ 知微审查 2026-10-31 修正：
            #   (1) 特征用 norm[i-1]，不用 norm[i]（避免未来函数）
            #   (2) 只在 RL 决策日记录 log_prob，保证 K == T
            #   (3) 记账拆成 隔夜(旧仓位) + 日内(新仓位)
            is_rl = not sig[i - 1]
            if is_rl:
                x = torch.from_numpy(norm[i - 1]).unsqueeze(0)
                p = pol(x)
                d = Categorical(p)
                a = d.sample()
                lps.append(d.log_prob(a).squeeze(-1))    # ★ 压成标量
                target = R.CASH_POSITIONS[int(a.item())]
            else:
                target = 1.0

            prev = pos
            cost_pen = (2 * args.cost) if target != prev else 0.0
            day_ret = (1 + prev * gap_a[i]) * (1 + target * intra_a[i]) - 1.0

            if is_rl:
                rewards.append(day_ret - cost_pen)
            pos = target

        if len(lps) >= 2 and len(lps) == len(rewards):
            disc = discounted_returns(rewards, gamma=0.99)
            loss = paired_policy_loss(lps, disc, normalize=True)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()

        # 快照
        if ep % snapshot_every == 0 or ep == args.episodes:
            c_tr, _, _ = R.rollout(sub_tr, None, args.cost, mu, sd, policy=pol, greedy=True)
            s_tr = R.stats(c_tr, n_train)
            # entropy：在 CASH 时刻的平均策略熵
            # ★ 与训练口径一致：用 i-1 的特征（第 i-1 日收盘可观测）
            cash_idx = np.where(~sig)[0] - 1
            cash_idx = cash_idx[(cash_idx > 0)][:200]
            if len(cash_idx):
                with torch.no_grad():
                    xs = torch.from_numpy(norm[cash_idx])
                    ps = pol(xs).numpy()
                    ent = float(np.mean(-(ps * np.log(ps + 1e-9)).sum(axis=1)))
            else:
                ent = np.nan
            curve_snapshots.append({"ep": ep, "train_ret": float(s_tr["total"]), "entropy": ent})

    # 最终评估
    c_tr, acts_tr, _ = R.rollout(sub_tr, None, args.cost, mu, sd, policy=pol, greedy=True)
    sub_te = df.iloc[n_train:].reset_index(drop=True)
    c_te, acts_te, _ = R.rollout(sub_te, None, args.cost, mu, sd, policy=pol, greedy=True)
    s_tr = R.stats(c_tr, n_train)
    s_te = R.stats(c_te, len(df) - n_train)

    dist = {}
    if acts_te:
        arr = np.array(acts_te)
        for k in range(R.N_ACTIONS):
            dist[str(k)] = float(np.mean(arr == k))

    return {
        "seed": seed,
        "train_total": float(s_tr["total"]),
        "train_dd": float(s_tr["dd"]),
        "test_total": float(s_te["total"]),
        "test_dd": float(s_te["dd"]),
        "action_dist": dist,
        "curve": curve_snapshots,
        "final_entropy": curve_snapshots[-1]["entropy"] if curve_snapshots else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="顿悟 vs 噪声 验证")
    parser.add_argument("--code", default="510300.SH")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    df = R.load_etf(R._resolve_db(), args.code, args.window)
    n = len(df)
    n_train = int(n * args.train_ratio)
    mu = df[R.FEAT_COLS].values[:n_train].mean(axis=0).astype(np.float32)
    sd = (df[R.FEAT_COLS].values[:n_train].std(axis=0) + 1e-8).astype(np.float32)

    print(f"{args.code}: {n} 天，训练 {n_train} / 测试 {n - n_train}")
    print(f"{args.seeds} 个种子 × {args.episodes} episodes\n")

    # 基线
    c_tr, _, _ = R.rollout(df.iloc[:n_train].reset_index(drop=True), None, args.cost, mu, sd, p_override=0.0)
    c_te, _, _ = R.rollout(df.iloc[n_train:].reset_index(drop=True), None, args.cost, mu, sd, p_override=0.0)
    base_tr = R.stats(c_tr, n_train)
    base_te = R.stats(c_te, n - n_train)
    print(f"基线（择时-全卖）: 训练 {base_tr['total']*100:+.1f}%  测试 {base_te['total']*100:+.1f}%\n")

    # 训练
    print("=" * 100)
    print(f"{'seed':<6}{'训练集':>12}{'测试集':>12}{'训练熵':>10}{'动作分布(0%/50%/100%)':>28}")
    print("-" * 100)

    results = []
    for s in range(1, args.seeds + 1):
        r = train_with_curve(df, n_train, mu, sd, args, s)
        results.append(r)
        d = r["action_dist"]
        dist_str = f"{d.get('0',0)*100:.0f}% / {d.get('1',0)*100:.0f}% / {d.get('2',0)*100:.0f}%"
        print(f"{s:<6}{r['train_total']*100:>+11.1f}%{r['test_total']*100:>+11.1f}%"
              f"{r['final_entropy']:>10.3f}{dist_str:>28}")

    # ===== 分析 =====
    print("\n" + "=" * 100)
    print("分析")
    print("=" * 100)

    tr = np.array([r["train_total"] for r in results])
    te = np.array([r["test_total"] for r in results])

    print(f"\n【训练集分布】（判断地形是否有信号）")
    print(f"  均值: {tr.mean()*100:+.1f}%   中位数: {np.median(tr)*100:+.1f}%")
    print(f"  标准差: {tr.std()*100:.1f}pp   范围: [{tr.min()*100:+.1f}%, {tr.max()*100:+.1f}%]")
    print(f"  为正的比例: {int((tr>0).sum())}/{len(tr)}")
    print(f"  超过基线({base_tr['total']*100:+.1f}%)的比例: {int((tr>base_tr['total']).sum())}/{len(tr)}")

    print(f"\n【测试集分布】")
    print(f"  均值: {te.mean()*100:+.1f}%   中位数: {np.median(te)*100:+.1f}%")
    print(f"  标准差: {te.std()*100:.1f}pp   范围: [{te.min()*100:+.1f}%, {te.max()*100:+.1f}%]")
    print(f"  超过基线({base_te['total']*100:+.1f}%)的比例: {int((te>base_te['total']).sum())}/{len(te)}")

    # 相关性：训练集好的，测试集是否也好（泛化能力的标志）
    corr = float(np.corrcoef(tr, te)[0, 1])
    print(f"\n【训练-测试相关性】{corr:+.3f}")
    if corr > 0.5:
        print(f"  → 强正相关：训练好的确实泛化好（说明有可学的信号）")
    elif corr > 0.1:
        print(f"  → 弱正相关：有一点泛化，但主要是噪声")
    else:
        print(f"  → 无相关/负相关：训练好不代表测试好，**纯噪声**")

    # 熵
    ents = np.array([r["final_entropy"] for r in results])
    print(f"\n【最终策略熵】（log(3)=1.099 表示完全随机）")
    print(f"  均值: {ents.mean():.3f}  范围: [{ents.min():.3f}, {ents.max():.3f}]")
    if ents.mean() > 1.0:
        print(f"  → 熵接近最大值，策略几乎均匀随机，**没学到东西**")
    elif ents.mean() > 0.8:
        print(f"  → 熵偏高，策略确定性弱")
    else:
        print(f"  → 熵较低，策略有一定确定性")

    # 学习曲线分析（顿悟检测）
    print("\n【学习曲线：是否有「突变」（顿悟特征）】")
    best_seed = max(results, key=lambda r: r["train_total"])
    curve = best_seed["curve"]
    print(f"  最好的种子 (seed={best_seed['seed']}, 训练 {best_seed['train_total']*100:+.1f}%)")
    print(f"  {'轮次':<8}{'训练收益':>12}{'熵':>10}")
    print("  " + "-" * 32)
    step = max(1, len(curve) // 10)
    for c in curve[::step]:
        print(f"  {c['ep']:<8}{c['train_ret']*100:>+11.1f}%{c['entropy']:>10.3f}")

    # 突变检测：找到最大单步提升
    rets_curve = [c["train_ret"] for c in curve]
    if len(rets_curve) > 3:
        diffs = np.diff(rets_curve)
        max_jump = float(diffs.max())
        median_jump = float(np.median(np.abs(diffs)))
        print(f"\n  最大单步提升: {max_jump*100:+.1f}pp")
        print(f"  单步提升中位数: {median_jump*100:.1f}pp")
        if max_jump > 5 * max(median_jump, 1e-6):
            print(f"  → 存在明显突变（{max_jump/median_jump:.1f}倍于中位数），**可能有顿悟迹象**")
        else:
            print(f"  → 无突变，曲线平滑（随机游走特征）")

    # ===== 最终判断 =====
    print("\n" + "=" * 100)
    print("最终判断：顿悟 or 噪声？")
    print("=" * 100)

    evidence = []
    # 判据1：训练集是否集中且为正
    train_concentrated = tr.std() < 0.30
    train_positive = (tr > 0).mean() > 0.7
    evidence.append(("训练集标准差 < 30pp", f"{tr.std()*100:.1f}pp", train_concentrated))
    evidence.append(("训练集70%以上为正", f"{(tr>0).mean()*100:.0f}%", train_positive))

    # 判据2：训练-测试相关性
    evidence.append(("训练-测试相关性 > 0.3", f"{corr:+.3f}", corr > 0.3))

    # 判据3：熵
    evidence.append(("熵 < 1.0（非随机）", f"{ents.mean():.3f}", ents.mean() < 1.0))

    # 判据4：超过基线的比例
    beat = (te > base_te["total"]).mean()
    evidence.append(("测试集超过基线 > 50%", f"{beat*100:.0f}%", beat > 0.5))

    print(f"\n  {'判据':<32}{'实测':>12}{'通过':>8}")
    print("  " + "-" * 54)
    for name, val, ok in evidence:
        print(f"  {name:<32}{val:>12}{'YES' if ok else 'NO ':>8}")

    n_pass = sum(1 for _, _, ok in evidence if ok)
    print(f"\n  通过 {n_pass}/{len(evidence)} 条判据")

    if n_pass >= 4:
        verdict = "可能是顿悟（多项判据支持存在可学信号）"
    elif n_pass >= 2:
        verdict = "混合：有部分信号，但不足以稳定泛化"
    else:
        verdict = "**噪声运气** —— 无证据支持可学习信号"

    print(f"\n  >>> {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "grokking_vs_noise.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "baseline": {"train": base_tr, "test": base_te},
        "runs": [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                  for k, v in r.items()} for r in results],
        "analysis": {
            "train_mean": round(float(tr.mean()), 6),
            "train_std": round(float(tr.std()), 6),
            "train_range": [round(float(tr.min()), 6), round(float(tr.max()), 6)],
            "test_mean": round(float(te.mean()), 6),
            "test_std": round(float(te.std()), 6),
            "train_test_corr": round(corr, 4),
            "entropy_mean": round(float(ents.mean()), 4),
            "beat_baseline_ratio": round(float(beat), 4),
            "criteria_passed": n_pass,
            "criteria_total": len(evidence),
        },
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
