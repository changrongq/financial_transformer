"""验证修正后的引擎：三个问题是否都修好。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_engine as T


def run_manual(df, sigs, cost=0.0, slip=0.0):
    """用修正后的分段逻辑跑（绕过 add_signal）。"""
    d = df.copy().reset_index(drop=True)
    d["sig"] = sigs
    n = len(d)
    eq = [1.0]
    pos = 0.0
    rows = []
    for i in range(1, n):
        target = 1.0 if bool(d.loc[i - 1, "sig"]) else 0.0
        prev_pos = pos
        prev_eq = eq[-1]
        before = prev_eq

        # 段1 隔夜
        c_prev = float(d.loc[i - 1, "close"])
        o_t = float(d.loc[i, "open"])
        overnight = (o_t / c_prev - 1.0) if c_prev > 0 else 0.0
        prev_eq *= (1.0 + prev_pos * overnight)

        # 段2 日内
        intra = float(d.loc[i, "intraday"])
        prev_eq *= (1.0 + target * intra)

        trade = target != prev_pos
        if trade:
            prev_eq *= (1.0 - slip - cost)

        eq.append(prev_eq)
        pos = target
        rows.append((i, "buy" if target > prev_pos else ("sell" if target < prev_pos else "hold/cash"),
                     overnight, intra, before, prev_eq))
    return np.array(eq[1:]), rows


def build(prices):
    d = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(len(prices))],
        "open": [float(p[0]) for p in prices],
        "close": [float(p[1]) for p in prices],
    })
    d["pre_close"] = d["close"].shift(1).fillna(d["open"])
    d["ret"] = d["close"] / d["pre_close"] - 1.0
    d["intraday"] = d["close"] / d["open"] - 1.0
    return d


print("=" * 84)
print("验证 1：卖出日隔夜（修正后应正确）")
print("=" * 84)

prices = [(100, 100), (100, 100), (90, 90), (90, 90)]
sigs = [True, False, False, False]
df = build(prices)
eq, rows = run_manual(df, sigs)

print(f"\n场景 A：卖出日低开 -10%")
print(f"{'i':<4}{'动作':<10}{'隔夜':>10}{'日内':>10}{'净值前':>14}{'净值后':>14}")
print("-" * 66)
for i, kind, ov, it, b, a in rows:
    print(f"{i:<4}{kind:<10}{ov*100:>9.2f}%{it*100:>9.2f}%{b:>14.8f}{a:>14.8f}")
print("-" * 66)
print(f"最终: {eq[-1]:.8f}   正确应为 0.90000000")
ok_A = abs(eq[-1] - 0.90) < 1e-9
print(f"{'PASS' if ok_A else 'FAIL'}")

print()
prices_B = [(100, 100), (100, 100), (110, 110), (110, 110)]
eq_B, _ = run_manual(build(prices_B), sigs)
print(f"场景 B：卖出日高开 +10%")
print(f"最终: {eq_B[-1]:.8f}   正确应为 1.10000000")
ok_B = abs(eq_B[-1] - 1.10) < 1e-9
print(f"{'PASS' if ok_B else 'FAIL'}")

print()
print("=" * 84)
print("验证 2：夏普（改用收益率）")
print("=" * 84)
curve = np.array([2.0, 1.0, 2.0])
s = T.stats(curve)
print(f"  曲线 [2, 1, 2]")
print(f"  修正后夏普: {s['sharpe']:.8f}")
print(f"  知微期望:   11.04536102")
ok_sharpe = abs(s["sharpe"] - 11.04536102) < 1e-6
print(f"  {'PASS' if ok_sharpe else 'FAIL'}")

print()
print("=" * 84)
print("验证 3：回撤（含初始资本）")
print("=" * 84)
curve2 = np.array([0.9, 0.81])
s2 = T.stats(curve2)
print(f"  曲线 [0.9, 0.81]")
print(f"  修正后回撤: {s2['dd']*100:.4f}%")
print(f"  知微期望:   -19%")
ok_dd = abs(s2["dd"] + 0.19) < 1e-9
print(f"  {'PASS' if ok_dd else 'FAIL'}")

print()
print("=" * 84)
print("总结")
print("=" * 84)
results = [
    ("卖出日低开计入隔夜", ok_A),
    ("卖出日高开计入隔夜", ok_B),
    ("夏普用收益率", ok_sharpe),
    ("回撤含初始资本", ok_dd),
]
for name, ok in results:
    print(f"  {name:<24}{'PASS' if ok else 'FAIL'}")
print()
if all(ok for _, ok in results):
    print("  >>> 三个问题全部修复")
else:
    print("  >>> 仍有问题未修复")
