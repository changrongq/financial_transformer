"""对 RL v2 做严格的「无 lucky seed」检验。

之前的问题：
  单次跑出 +22.3%，用的是 seed=42（5个种子里最好的）
  → 典型的 lucky seed 偏差

本脚本做：
  1. 每个切分点 × 每个种子，全部跑一遍（矩阵）
  2. 统计：所有组合里跑赢买入持有的比例
  3. 报告「期望收益」而不是「最好那个」

另外做「集成」检验：5个种子的动作取平均（软投票），
  看是否能消除随机性 —— 这是判断"是否学到规律"的标准做法。

用法：
  python scripts/verify_rl_strict.py
  python scripts/verify_rl_strict.py --episodes 1500 --seeds 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_rl_robustness import eval_policy, train_one
import rl_timing_v2 as R


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dd-lambda", type=float, default=5.0)
    parser.add_argument("--dd-free", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=6)
    args = parser.parse_args()

    df = R.load_market(R._resolve_db())
    feat_cols = ["ret5", "ret20", "vol10", "regime10", "regime20", "vol_regime", "run_len", "ret"]
    feats = df[feat_cols].values.astype(np.float32)
    rets = df["ret"].values.astype(np.float32)
    n = len(rets)

    seeds = list(range(1, args.seeds + 1))
    ratios = [0.5, 0.6, 0.7, 0.8]

    print(f"数据 {n} 天 | {len(ratios)} 个切分 × {len(seeds)} 个种子 = {len(ratios)*len(seeds)} 次训练")
    print(f"每次 {args.episodes} episodes\n")

    print("=" * 92)
    print("【矩阵检验】每格 = 测试段 RL收益 (优势 vs 买入持有)")
    print("=" * 92)
    header = f"{'训练比例':<10}{'买入持有':>11}" + "".join(f"{'s'+str(s):>14}" for s in seeds)
    print(header)
    print("-" * 92)

    all_adv = []
    matrix = []
    policies = {}

    for ratio in ratios:
        nt = int(n * ratio)
        bh = float(np.prod(1 + rets[nt:]) - 1)
        line = f"{ratio:<10.1f}{bh*100:>+10.1f}%"
        row = {"ratio": ratio, "bh": bh, "cells": []}
        for s in seeds:
            pol, mu, sd = train_one(feats, rets, nt, args, s)
            r = eval_policy(pol, feats, rets, args.cost, nt, n - nt, mu, sd)
            adv = r["total"] - bh
            all_adv.append(adv)
            line += f"{r['total']*100:>+9.1f}%({adv*100:>+4.0f})"
            row["cells"].append({"seed": s, "ret": r["total"], "adv": adv})
            policies[(ratio, s)] = (pol, mu, sd)
        print(line)
        matrix.append(row)

    print("-" * 92)
    adv_arr = np.array(all_adv)
    n_win = int((adv_arr > 0).sum())
    print(f"\n总计 {len(adv_arr)} 个组合:")
    print(f"  跑赢买入持有: {n_win}/{len(adv_arr)} ({n_win/len(adv_arr)*100:.0f}%)")
    print(f"  平均优势:     {adv_arr.mean()*100:+.1f}pp")
    print(f"  优势中位数:   {np.median(adv_arr)*100:+.1f}pp")
    print(f"  最好:         {adv_arr.max()*100:+.1f}pp")
    print(f"  最差:         {adv_arr.min()*100:+.1f}pp")
    print(f"  标准差:       {adv_arr.std()*100:.1f}pp  ← 越大说明越靠运气")

    # ===== 集成检验 =====
    print()
    print("=" * 92)
    print("【集成检验】把多个种子的动作平均（软投票），检验随机性是否是主要因素")
    print("=" * 92)
    print(f"{'训练比例':<12}{'单种子中位数':>14}{'集成后':>12}{'买入持有':>12}{'集成优势':>12}{'集成切换':>10}")
    ens_rows = []
    for ratio in ratios:
        nt = int(n * ratio)
        bh = float(np.prod(1 + rets[nt:]) - 1)
        dur = n - nt

        # 累计各模型的动作概率
        sum_probs = None
        mu_ref = sd_ref = None
        for s in seeds:
            pol, mu, sd = policies[(ratio, s)]
            mu_ref, sd_ref = mu, sd
            pol.eval()
            with torch.no_grad():
                xs = torch.from_numpy(
                    np.clip((feats[nt:nt+dur] - mu) / sd, -5, 5).astype(np.float32))
                p = pol(xs).numpy()
            sum_probs = p if sum_probs is None else sum_probs + p
        avg_probs = sum_probs / len(seeds)
        acts = np.array([R.ACTIONS[i] for i in avg_probs.argmax(axis=1)])

        pos, eq, curve = 0.0, 1.0, []
        for i, t in enumerate(range(nt, nt + dur)):
            target = acts[i]
            eq *= (1 + target * rets[t] - abs(target - pos) * args.cost)
            curve.append(eq)
            pos = target
        arr = np.array(curve)
        ens_ret = float(arr[-1] - 1)
        sw = int(np.sum(np.diff(acts) != 0)) if len(acts) > 1 else 0
        single_med = float(np.median([c["ret"] for c in matrix[ratios.index(ratio)]["cells"]]))

        print(f"{ratio:<12.1f}{single_med*100:>+13.1f}%{ens_ret*100:>+11.1f}%"
              f"{bh*100:>+11.1f}%{(ens_ret-bh)*100:>+11.1f}pp{sw:>10}")
        ens_rows.append({"ratio": ratio, "single_median": single_med, "ensemble": ens_ret,
                         "bh": bh, "adv": ens_ret - bh, "switches": sw})

    print()
    print("=" * 92)
    ens_adv = np.array([r["adv"] for r in ens_rows])
    print(f"集成后跑赢买入持有: {int((ens_adv>0).sum())}/{len(ens_adv)}")
    print(f"集成后平均优势: {ens_adv.mean()*100:+.1f}pp")

    # ===== 最终判断 =====
    print()
    print("=" * 92)
    print("最终判断")
    print("=" * 92)
    single_win_rate = n_win / len(adv_arr)
    if single_win_rate >= 0.7 and adv_arr.mean() > 0.02:
        verdict = "RL 可能学到了东西（胜率高且平均优势明显）"
    elif single_win_rate >= 0.5 and adv_arr.std() > 0.15:
        verdict = "结果不稳定：赢的比例过半，但方差极大，靠运气成分高"
    elif adv_arr.mean() > 0 and adv_arr.std() < 0.10:
        verdict = "有小幅稳定优势（值得进一步研究）"
    else:
        verdict = "RL 没有学到稳定规律（不值得实盘）"
    print(f"单种子胜率: {single_win_rate*100:.0f}%   平均优势: {adv_arr.mean()*100:+.1f}pp   "
          f"标准差: {adv_arr.std()*100:.1f}pp")
    print(f"\n>>> {verdict}")
    print("=" * 92)

    out = Path(__file__).resolve().parents[1] / "results" / "rl_strict_verification.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "matrix": [{"ratio": m["ratio"], "bh": m["bh"],
                    "cells": m["cells"]} for m in matrix],
        "summary": {
            "n_combos": len(adv_arr), "n_win": n_win,
            "win_rate": round(single_win_rate, 4),
            "mean_adv": round(float(adv_arr.mean()), 6),
            "median_adv": round(float(np.median(adv_arr)), 6),
            "std_adv": round(float(adv_arr.std()), 6),
            "best_adv": round(float(adv_arr.max()), 6),
            "worst_adv": round(float(adv_arr.min()), 6),
        },
        "ensemble": ens_rows,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
