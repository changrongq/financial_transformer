"""RL v4 的滚动窗口验证（判断是否真有效）。

问题：
  单次"训练前75% / 测试后25%"的切分，结果极度依赖选中的测试段。
  我们之前就踩过这个坑（2年数据得出错误结论）。

本脚本：
  用「滚动窗口」——多个训练段/测试段组合，看 RL 是否稳定胜过基线。

设计：
  · 3 个窗口：
    W1: 训练 2012-2017，测试 2018-2019
    W2: 训练 2012-2019，测试 2020-2021
    W3: 训练 2012-2021，测试 2022-2023
    W4: 训练 2012-2023，测试 2024-2026
  · 每个窗口 × 3 个种子
  · 基线：买入持有、固定持有2日、固定持有5日、简单择时（10日均线）

对比：RL 相对每个基线的优势

用法：
  python scripts/verify_rl_v4_rolling.py
  python scripts/verify_rl_v4_rolling.py --episodes 1500
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
import rl_timing_v4_variable_holding as R
from rl_algo_utils import paired_policy_loss, discounted_returns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

WINDOWS = [
    ("2012-2017 → 2018-2019", "2018-01-01", "2020-01-01"),
    ("2012-2019 → 2020-2021", "2020-01-01", "2022-01-01"),
    ("2012-2021 → 2022-2023", "2022-01-01", "2024-01-01"),
    ("2012-2023 → 2024-2026", "2024-01-01", "2027-01-01"),
]


def train_policy(feats, open_, close, n_train, args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    mu = feats[:n_train].mean(axis=0)
    sd = feats[:n_train].std(axis=0) + 1e-8
    env = R.Env(feats, open_, close, args.cost, args.dd_lambda, args.dd_free,
                args.horizon, mu, sd, n_allowed=n_train)
    pol = R.Policy()
    opt = optim.Adam(pol.parameters(), lr=args.lr)
    for ep in range(args.episodes):
        # ★ 知微审查 2026-10-31 修正：
        #   旧写法先 `obs = env.reset()` 再覆盖 `env.t`，导致传给模型的首个 obs
        #   仍来自旧时点（可能在测试期），而收益从新时点开始算 —— 状态与收益错位。
        #   正确做法：用 reset(start=...) 一次性建立一致状态。
        start = np.random.randint(0, max(1, n_train - args.horizon - 15))
        obs = env.reset(start=start)
        lps, rs = [], []
        done = False
        while not done:
            x = torch.from_numpy(obs).unsqueeze(0)
            p = pol(x)
            d = Categorical(p)
            a = d.sample()
            # ★ 压成标量，杜绝 [T,1]×[T] 广播
            lps.append(d.log_prob(a).squeeze(-1))
            obs, r, done = env.step(int(a.item()))
            rs.append(r)
        if len(lps) < 2:
            continue
        disc = discounted_returns(rs, gamma=0.99)
        loss = paired_policy_loss(lps, disc, normalize=True)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
    return pol, mu, sd


def fixed_hold(open_, close, cost, start, end, n_hold):
    """固定持有 n 天的策略。"""
    eq = 1.0
    t = start
    while t < end - 15:
        r = R.period_return(open_, close, t, n_hold)
        if np.isnan(r):
            t += 1
            continue
        eq *= (1 + r - 2 * cost)
        t += n_hold + 1
    return eq - 1


def ma_timing(open_, close, cost, start, end, window=10):
    """T+1 择时：T日收盘 10日均收益>0 → T+1开盘买入；<0 → T+1开盘卖出。"""
    rets = np.diff(close) / close[:-1]
    eq = 1.0
    buy_price = None
    for t in range(max(start, window), min(end, len(open_) - 2)):
        m = rets[t - window:t].mean()
        sig = m >= 0
        if sig and buy_price is None:
            buy_price = float(open_[t + 1])          # T+1 开盘买入
        elif (not sig) and buy_price is not None:
            sell = float(open_[t + 1])               # T+1 开盘卖出
            eq *= (1 + sell / buy_price - 1 - 2 * cost)
            buy_price = None
    if buy_price is not None:
        sell = float(open_[min(end, len(open_) - 1)])
        eq *= (1 + sell / buy_price - 1 - 2 * cost)
    return eq - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dd-lambda", type=float, default=3.0)
    parser.add_argument("--dd-free", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    df = R.load_etf(R._resolve_db(), "510300.SH")
    feats = df[R.FEAT_COLS].values.astype(np.float32)
    open_ = df["open"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    dates = df["date"].values

    print(f"510300: {df['date'].min().date()} ~ {df['date'].max().date()}  {len(df)} 天")
    print(f"{len(WINDOWS)} 个窗口 × {len(args.seeds)} 个种子 × {args.episodes} episodes\n")

    rows = []
    for wname, test_start, test_end in WINDOWS:
        tr_mask = dates < np.datetime64(test_start)
        te_mask = (dates >= np.datetime64(test_start)) & (dates < np.datetime64(test_end))
        n_train = int(tr_mask.sum())
        te_idx = np.where(te_mask)[0]
        if n_train < 500 or len(te_idx) < 100:
            print(f"{wname}: 数据不足，跳过")
            continue
        start, end = int(te_idx[0]), int(te_idx[-1])

        bh = R.buy_hold(open_, close, start, end)["total"]
        fh2 = fixed_hold(open_, close, args.cost, start, end, 2)
        fh5 = fixed_hold(open_, close, args.cost, start, end, 5)
        ma = ma_timing(open_, close, args.cost, start, end, 10)

        rl_rets = []
        for seed in args.seeds:
            pol, mu, sd = train_policy(feats, open_, close, n_train, args, seed)
            r = R.evaluate(pol, feats, open_, close, args.cost, start, end, mu, sd)
            rl_rets.append(r["total"])

        rl_med = float(np.median(rl_rets))
        rows.append({
            "window": wname, "test_days": int(end - start),
            "buy_hold": round(bh, 6), "fixed_hold2": round(fh2, 6),
            "fixed_hold5": round(fh5, 6), "ma_timing": round(ma, 6),
            "rl_median": round(rl_med, 6),
            "rl_all": [round(x, 6) for x in rl_rets],
            "rl_vs_bh": round(rl_med - bh, 6),
            "rl_vs_ma": round(rl_med - ma, 6),
        })
        print(f"{wname}")
        print(f"  测试天数 {int(end-start):>5}")
        print(f"  买入持有   {bh*100:>+8.1f}%    固定持有2日 {fh2*100:>+8.1f}%    固定持有5日 {fh5*100:>+8.1f}%")
        print(f"  均线择时   {ma*100:>+8.1f}%    RL中位数    {rl_med*100:>+8.1f}%  "
              f"（各seed: {', '.join(f'{x*100:+.1f}%' for x in rl_rets)}）")
        print(f"  RL vs 买入持有: {(rl_med-bh)*100:+.1f}pp    RL vs 均线择时: {(rl_med-ma)*100:+.1f}pp")
        print()

    if not rows:
        raise SystemExit("无有效窗口")

    # 汇总
    print("=" * 96)
    print("汇总")
    print("=" * 96)
    rdf = pd.DataFrame(rows)
    print(f"{'窗口':<28}{'买入持有':>12}{'均线择时':>12}{'RL中位':>12}{'RL-BH':>11}{'RL-MA':>11}")
    print("-" * 96)
    for _, r in rdf.iterrows():
        print(f"{r['window']:<28}{r['buy_hold']*100:>+11.1f}%{r['ma_timing']*100:>+11.1f}%"
              f"{r['rl_median']*100:>+11.1f}%{r['rl_vs_bh']*100:>+10.1f}pp{r['rl_vs_ma']*100:>+10.1f}pp")
    print("-" * 96)

    n_beat_bh = int((rdf["rl_vs_bh"] > 0).sum())
    n_beat_ma = int((rdf["rl_vs_ma"] > 0).sum())
    print(f"\nRL 跑赢买入持有: {n_beat_bh}/{len(rdf)} 个窗口")
    print(f"RL 跑赢均线择时: {n_beat_ma}/{len(rdf)} 个窗口")
    print(f"RL 平均优势(vs BH): {rdf['rl_vs_bh'].mean()*100:+.1f}pp")
    print(f"RL 平均优势(vs MA): {rdf['rl_vs_ma'].mean()*100:+.1f}pp")

    # 对比基线本身
    print(f"\n基线表现:")
    print(f"  买入持有平均: {rdf['buy_hold'].mean()*100:+.1f}%")
    print(f"  均线择时平均: {rdf['ma_timing'].mean()*100:+.1f}%")
    print(f"  固定持有2日平均: {rdf['fixed_hold2'].mean()*100:+.1f}%")
    print(f"  固定持有5日平均: {rdf['fixed_hold5'].mean()*100:+.1f}%")
    print(f"  RL 中位数平均: {rdf['rl_median'].mean()*100:+.1f}%")

    verdict = ("RL 稳定优于所有基线" if n_beat_bh == len(rdf) and n_beat_ma == len(rdf)
               else "RL 不稳定（部分窗口优于基线）" if n_beat_bh >= len(rdf) // 2
               else "RL 未能胜过简单基线")
    print(f"\n>>> {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rl_v4_rolling.json"
    out.write_text(json.dumps({
        "params": vars(args), "windows": rows, "verdict": verdict,
        "n_beat_bh": n_beat_bh, "n_beat_ma": n_beat_ma, "n_windows": len(rdf),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
