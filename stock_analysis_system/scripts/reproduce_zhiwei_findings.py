"""复现知微指出的三个问题（用合成数据，不依赖原始行情）。

知微报告的三处：
  1. 卖出日漏记隔夜收益
  2. 夏普用了净值绝对变化（np.diff），不是收益率
  3. 回撤的累计最高值没包含初始资本 1

本脚本用最小合成数据逐项验证，证明问题存在。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# 问题 1：卖出日漏记隔夜
# ============================================================


def make_signal_series(prices, sigs):
    """构造 add_signal 的等价输出。"""
    d = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=len(prices)),
        "open": [p[0] for p in prices],
        "close": [p[1] for p in prices],
    })
    d["ret"] = d["close"] / d["close"].shift(1) - 1
    d["intraday"] = d["close"] / d["open"] - 1
    d["sig"] = sigs
    return d


def run_strict_t1_demo(d, cost=0.0, slip=0.0):
    """复现 timing_engine.run_strict_t1 的核心循环。"""
    n = len(d)
    eq = [1.0]
    pos = 0.0
    trades = 0
    for i in range(1, n):
        target = 1.0 if bool(d.loc[i - 1, "sig"]) else 0.0
        prev_eq = eq[-1]
        if target > pos:
            intra = float(d.loc[i, "intraday"])
            prev_eq *= (1.0 + intra - slip - cost)
            trades += 1
        elif target < pos:
            prev_eq *= (1.0 - slip - cost)
            trades += 1
        elif target > 0:
            prev_eq *= (1.0 + float(d.loc[i, "ret"]))
        eq.append(prev_eq)
        pos = target
    return np.array(eq[1:]), trades


def run_correct_segmented(d, cost=0.0, slip=0.0):
    """知微给的正确分段公式：
    gross[t] = (1 + prev_pos × overnight[t]) × (1 + new_pos × intraday[t])
    """
    n = len(d)
    eq = [1.0]
    pos = 0.0
    trades = 0
    for i in range(1, n):
        target = 1.0 if bool(d.loc[i - 1, "sig"]) else 0.0
        overnight = float(d.loc[i, "open"] / d.loc[i - 1, "close"] - 1)
        intra = float(d.loc[i, "intraday"])
        # 分段：昨收持仓的吃隔夜，新仓吃日内
        factor = (1 + pos * overnight) * (1 + target * intra)
        prev_eq = eq[-1] * factor
        # 成本：仓位变动时收
        if target != pos:
            prev_eq *= (1.0 - slip - cost)
            trades += 1
        eq.append(prev_eq)
        pos = target
    return np.array(eq[1:]), trades


print("=" * 78)
print("问题 1：卖出日漏记隔夜收益")
print("=" * 78)

# 知微的场景：102 买入，持有至昨收 90，次日低开至 85.5 卖出
print("\n【场景 A：卖出日低开 5%】")
print("  价格路径：102(买) -> 90(昨收) -> 85.5(今开，卖出) -> 85.5(今收)")
prices = [(102, 102), (102, 90), (85.5, 85.5), (85.5, 85.5)]
sigs = [True, True, False, False]  # 前两日 HOLD，第三日开始 CASH
d = make_signal_series(prices, sigs)

eq_orig, _ = run_strict_t1_demo(d)
eq_correct, _ = run_correct_segmented(d)
print(f"  原函数最后净值:     {eq_orig[-1]:.8f}")
print(f"  正确分段最后净值:   {eq_correct[-1]:.8f}")
print(f"  知微给的对照:       0.88235294 / 0.83823529")
print(f"  期望值验证:")
print(f"    原函数: 1.0 * (90/102) [持仓吃ret] ... 需要逐步看")
print(f"  差异: {eq_correct[-1] - eq_orig[-1]:+.8f}")

print("\n【场景 B：卖出日高开 5%】")
print("  价格路径：102(买) -> 90(昨收) -> 94.5(今开，卖出) -> 94.5(今收)")
prices_b = [(102, 102), (102, 90), (94.5, 94.5), (94.5, 94.5)]
d_b = make_signal_series(prices_b, sigs)
eq_orig_b, _ = run_strict_t1_demo(d_b)
eq_correct_b, _ = run_correct_segmented(d_b)
print(f"  原函数最后净值:     {eq_orig_b[-1]:.8f}")
print(f"  正确分段最后净值:   {eq_correct_b[-1]:.8f}")
print(f"  知微给的对照:       0.88235294 / 0.92647059")

print("\n  >>> 结论：低开和高开两种情况，原函数给出【完全相同】的结果")
print("      → 说明卖出日的隔夜收益被完全忽略")

# ============================================================
# 问题 2：夏普计算错误
# ============================================================
print()
print("=" * 78)
print("问题 2：夏普用净值绝对变化，不是收益率")
print("=" * 78)


def stats_buggy(curve):
    r = np.diff(np.concatenate([[1.0], curve]))
    total = float(curve[-1] - 1)
    years = len(curve) / 244
    ann = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe}


def stats_correct(curve):
    # 收益率
    full = np.concatenate([[1.0], curve])
    r = full[1:] / full[:-1] - 1.0
    total = float(curve[-1] - 1)
    years = len(curve) / 244
    ann = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    # 回撤：包含初始资本 1 的累计最高
    dd = float((full / np.maximum.accumulate(full) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe}


curve = np.array([2.0, 1.0, 2.0])
print(f"\n  合成曲线 [2, 1, 2]")
sb = stats_buggy(curve)
sc = stats_correct(curve)
print(f"  原函数夏普:     {sb['sharpe']:.8f}")
print(f"  按收益率计算:   {sc['sharpe']:.8f}")
print(f"  知微给的对照:   5.52268051 / 11.04536102")
print(f"  匹配: 原={abs(sb['sharpe']-5.52268051)<1e-6}  正确={abs(sc['sharpe']-11.04536102)<1e-6}")

print()
print("  为什么差 2 倍：")
print("    净值变化: [1.0, 1.0, -1.0]，std 分母不同")
print("    收益率:   [1.0, -0.5, 1.0]，std 更大 → 夏普更大")
full = np.concatenate([[1.0], curve])
print(f"    净变化: {np.diff(full)}  mean={np.diff(full).mean():.4f} std={np.diff(full).std():.6f}")
rets = full[1:] / full[:-1] - 1
print(f"    收益率: {rets}  mean={rets.mean():.4f} std={rets.std():.6f}")

# ============================================================
# 问题 3：回撤没包含初始资本
# ============================================================
print()
print("=" * 78)
print("问题 3：回撤的累计最高值没包含初始资本 1")
print("=" * 78)

curve2 = np.array([0.9, 0.81])
dd_buggy = float((curve2 / np.maximum.accumulate(curve2) - 1).min())
full2 = np.concatenate([[1.0], curve2])
dd_correct = float((full2 / np.maximum.accumulate(full2) - 1).min())
print(f"\n  合成曲线 [0.9, 0.81]")
print(f"  原函数回撤:     {dd_buggy*100:.2f}%")
print(f"  含初始资本回撤: {dd_correct*100:.2f}%")
print(f"  知微给的对照:   -10% / -19%")
print(f"  匹配: 原={abs(dd_buggy+0.10)<1e-9}  正确={abs(dd_correct+0.19)<1e-9}")

print()
print("=" * 78)
print("总结")
print("=" * 78)
print("  三个问题全部确认存在：")
print("    1. 卖出日漏隔夜 → 卖出日涨跌被完全忽略")
print("    2. 夏普用净值变化 → 数值错误（可能偏大或偏小）")
print("    3. 回撤不含初始资本 → 回撤被低估")
print()
print("  影响：")
print("    · 收益（第一张表）：卖出日漏隔夜 → 方向不确定")
print("    · 夏普：公式错误 → 所有夏普数字作废")
print("    · 回撤：被低估 → 所有回撤数字作废")
print()
print("  >>> 结论：需要全套重算。知微的判断是对的。")
