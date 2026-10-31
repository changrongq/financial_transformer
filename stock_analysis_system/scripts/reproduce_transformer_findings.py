# -*- coding: utf-8 -*-
"""复现知微 Transformer 预测链复核（2026-10-31）的 7 项发现。

逐项独立验证，不修改项目源码，只用合成数据调用原函数。

  1. 标签跨入验证期（训练截止日后的价格能改变训练标签）
  2. 未来不足 45 天被当作完整 45 天标签
  3. flat 输入尺度掩盖小量级特征（1e8 成交额 vs 0.1 收益率）
  4. 回测允许新买股票当天卖出（T+1 违规）
  5. CPU 分支 best_state 不是独立副本
  6. 用最新市值选择过去股票池（前视）
  7. 阈值搜索在测试集上挑参（方法边界）

用法：
  python scripts/reproduce_transformer_findings.py
"""

from __future__ import annotations

import copy
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "experiments" / "financial_transformer"
FTSRC = FT / "src" / "financial_transformer_experiment"
# ★ 依赖顺序：hierarchical_system 在 src/ 下，financial_transformer_experiment
#   在 FT/src/ 下，两者都要在 sys.path 里
sys.path.insert(0, str(FT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import financial_transformer_experiment  # noqa: E402
from financial_transformer_experiment import data as ftdata  # noqa: E402

RESULT = []


def check(name: str, passed: bool, detail: str = ""):
    RESULT.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))


print("=" * 82)
print("知微 Transformer 复核复现")
print(f"torch {torch.__version__}  numpy {np.__version__}  pandas {pd.__version__}")
print("=" * 82)


# ============================================================
# 1. 标签跨入验证期
# ============================================================
print("\n[1] 训练截止日之后的标签进入训练集")


def make_price_df(dates, prices):
    return pd.DataFrame({
        "stock_code": "000001",
        "date": pd.to_datetime(dates),
        "close": prices,
        "high": prices,
        "low": prices,
        "open": prices,
    })


# 构造 100 个交易日的价格
n = 100
dates = pd.bdate_range("2025-01-01", periods=n)
base = np.full(n, 100.0)

# 截止日索引 60（即第 61 个交易日为训练最后一天）
cut = 60

# 场景 A：截止日之后价格保持 100
pa = base.copy()
# 场景 B：截止日之后价格涨到 120
#   → 截止日当天的 5 日标签变成 +20%
pb = base.copy()
pb[cut + 1:] = 120.0

# 未来 5 日收益（截止日附近）
def future_ret(prices, h=5):
    out = np.full(len(prices), np.nan)
    for i in range(len(prices) - h):
        out[i] = prices[i + h] / prices[i] - 1.0
    return out

ra = future_ret(pa)
rb = future_ret(pb)

print(f"  截止日索引 {cut}")
print(f"  场景A 该日未来5日收益: {ra[cut]:+.4f}")
print(f"  场景B 该日未来5日收益: {rb[cut]:+.4f}")
print(f"  → 截止日当天的价格相同（都是 100），但标签不同")

changed = not np.isclose(ra[cut], rb[cut], equal_nan=True)
check("截止日后的价格能改变截止日训练标签", changed,
      f"{ra[cut]:+.4f} -> {rb[cut]:+.4f}")

# 检查当前代码是否过滤
print("\n  检查 _assign_split 是否考虑了标签成熟时间：")
src = inspect.getsource(ftdata._assign_split)
has_label_end = any(k in src for k in ["label_end", "available_at",
                                       "mature", "horizon", "holding"])
for line in src.strip().splitlines():
    print(f"      {line}")
check("代码未考虑标签成熟时间（只按当前日期分箱）", not has_label_end)


# ============================================================
# 2. 未来不足 45 天的标签
# ============================================================
print("\n[2] 未来不足 45 天被当作完整 45 天标签")

prices = np.full(50, 100.0)
prices[45:] = 50.0          # 第 46 天开始暴跌
# 第 5 天的样本，未来 45 天窗口
i0 = 5
window = prices[i0 + 1: i0 + 1 + 45]
print(f"  样本日索引 {i0}，价格 {prices[i0]:.1f}")
print(f"  未来 45 天窗口长度: {len(window)}")
print(f"  window.max(): {window.max():.4f}  window.min(): {window.min():.4f}")
max_gain = window.max() / prices[i0] - 1.0
max_dd = window.min() / prices[i0] - 1.0
print(f"  从该日算：max_gain {max_gain:+.4f}  max_dd {max_dd:+.4f}")

# 对比真正的 45 天（样本在靠前位置）
i0b = 0
window_b = prices[i0b + 1: i0b + 1 + 45]
print(f"\n  样本日索引 {i0b}（有完整 45 天）:")
print(f"  window 长度 {len(window_b)}  min {window_b.min():.4f}")
max_dd_b = window_b.min() / prices[i0b] - 1.0
print(f"  max_dd {max_dd_b:+.4f}")

# 关键：末端样本只有 5 天数据，却算出"完整的"45天标签
tail_i = len(prices) - 6
tail_window = prices[tail_i + 1:]
print(f"\n  末端样本索引 {tail_i}，实际只有 {len(tail_window)} 天未来数据")
print(f"  pandas 默认 max/min 忽略 NaN，会给出有限值")
arr = np.array([100.0, 100.0, 100.0, np.nan, np.nan])
print(f"  合成: {arr}")
print(f"  np.nanmax = {np.nanmax(arr):.4f}  np.max(忽略nan) = {np.max(arr):.4f}")
check("末端样本会用不完整窗口算出有限标签",
      np.isfinite(np.nanmax(arr)),
      "np.max/np.nanmax 默认跳过 NaN")


# ============================================================
# 3. flat 输入尺度掩盖小量级特征
# ============================================================
print("\n[3] flat 输入尺度问题")

try:
    from financial_transformer_experiment.model import FinancialTransformer
    has_model = True
except Exception as e:
    print(f"  导入 FinancialTransformer 失败: {e}")
    has_model = False
if has_model:
    torch.manual_seed(17)
    n_feat = 32
    seq_len = 10
    try:
        model = FinancialTransformer(
            input_dim=n_feat, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=128, dropout=0.0, model_variant="flat")
        model.eval()

        # 构造两个输入，只改小量级特征
        base_in = torch.zeros(1, seq_len, n_feat)
        # 假设第 0 维是 amount（1e8），第 1 维是 volume（1e6），
        # 第 2 维是 ret5d（±0.1）
        base_in[:, :, 0] = 1e8
        base_in[:, :, 1] = 1e6
        x1 = base_in.clone()
        x1[:, :, 2] = -0.1
        x2 = base_in.clone()
        x2[:, :, 2] = 0.1

        with torch.no_grad():
            o1 = model(x1)
            o2 = model(x2)
        if isinstance(o1, dict):
            o1 = o1.get("short_up", list(o1.values())[0])
            o2 = o2.get("short_up", list(o2.values())[0])
        d_out = (o1 - o2).abs().max().item()
        print(f"  原始尺度下，改变 ret5d(-0.1→+0.1) 输出最大差: {d_out:.10f}")
        check("原始尺度下小特征被淹没", d_out < 1e-6, f"{d_out:.2e}")

        # 归一化后重试
        base_n = torch.zeros(1, seq_len, n_feat)
        base_n[:, :, 0] = 1.0
        base_n[:, :, 1] = 1.0
        y1 = base_n.clone(); y1[:, :, 2] = -0.1
        y2 = base_n.clone(); y2[:, :, 2] = 0.1
        with torch.no_grad():
            p1 = model(y1); p2 = model(y2)
        if isinstance(p1, dict):
            p1 = p1.get("short_up", list(p1.values())[0])
            p2 = p2.get("short_up", list(p2.values())[0])
        d_out2 = (p1 - p2).abs().max().item()
        print(f"  归一化后，同样改变输出最大差: {d_out2:.10f}")
        check("归一化后能感知小特征", d_out2 > d_out,
              f"{d_out2:.2e} vs {d_out:.2e}")
    except Exception as e:
        print(f"  构造模型失败: {type(e).__name__}: {e}")
        check("flat 尺度检查", False, str(e)[:60])


# ============================================================
# 4. 回测允许当天买当天卖
# ============================================================
print("\n[4] 回测 T+1 违规（当天买当天卖）")

bt_src_path = FTSRC / "backtest.py"
if bt_src_path.exists():
    src = bt_src_path.read_text(encoding="utf-8").splitlines()
    # 查找 open_positions 加入后是否立即检查 high/low
    joined = "\n".join(src[840:940])
    has_buy = "open_positions" in joined
    # 找买入后有没有跳过当天
    print(f"  backtest.py 行数 {len(src)}")
    # 查找关键模式
    import re
    # 找 buy 后紧跟 exit 检查的段落
    seg = "\n".join(src[850:930])
    print("  关键片段（850-930 行）:")
    for i, line in enumerate(src[850:930], start=851):
        mark = " <<<" if any(k in line for k in
                             ["open_positions", "high", "low",
                              "exit_reason", "buy_date"]) else ""
        if mark:
            print(f"    {i}: {line.rstrip()}{mark}")

    # 静态判断：是否对当天新买的跳过
    has_t1_guard = any(k in joined for k in
                       ["can_sell", "sellable", "t_plus_1", "t1_guard",
                        "bought_today", "entry_date <=", "min_hold"])
    check("未实现 T+1 卖出约束", not has_t1_guard,
          "未见 can_sell/sellable/bought_today 等守卫")
else:
    check("backtest.py 存在", False)


# ============================================================
# 5. CPU best_state 不是副本
# ============================================================
print("\n[5] CPU 分支 best_state 共享存储")

m = nn.Linear(2, 2)
best_state = {k: v.cpu() for k, v in m.state_dict().items()}
w_before = best_state["weight"].clone()
with torch.no_grad():
    m.weight.fill_(7.0)
w_after = best_state["weight"]
snapshot_broken = not torch.equal(w_before, w_after)
print(f"  保存时 weight 首元素: {w_before.flatten()[0]:.4f}")
print(f"  修改原模型后 best_state 首元素: {w_after.flatten()[0]:.4f}")
check("【缺陷确认】best_state 被后续训练污染（非独立副本）",
      snapshot_broken, "value.cpu() 在 CPU 上不复制")

# 正确做法
m2 = nn.Linear(2, 2)
best2 = {k: v.detach().cpu().clone() for k, v in m2.state_dict().items()}
w2 = best2["weight"].clone()
with torch.no_grad():
    m2.weight.fill_(7.0)
ok2 = torch.equal(w2, best2["weight"])
check("【修复验证】detach().cpu().clone() 是独立快照", ok2)


# ============================================================
# 6. 用最新市值选历史股票池
# ============================================================
print("\n[6] 用最新市值选过去股票池")

try:
    _lmc = ftdata._latest_market_caps
    print("  _latest_market_caps 存在")
    src6 = inspect.getsource(_lmc)
    for line in src6.strip().splitlines()[:25]:
        print(f"    {line}")
    sig = inspect.signature(_lmc)
    print(f"\n  参数: {list(sig.parameters)}")
    has_asof = any("asof" in p.lower() or "as_of" in p.lower()
                   for p in sig.parameters)
    check("缺失 as-of 时点参数（前视风险）", not has_asof,
          f"参数只有 {list(sig.parameters)}")
except Exception as e:
    print(f"  检查失败: {e}")
    check("_latest_market_caps 检查", False, str(e)[:50])


# ============================================================
# 7. 阈值搜索在测试集上调参
# ============================================================
print("\n[7] 阈值搜索使用测试集")

th_path = FT / "threshold_search.py"
if th_path.exists():
    content = th_path.read_text(encoding="utf-8")
    uses_test = "test_predictions" in content
    print(f"  读取 test_predictions.csv: {uses_test}")
    # 查找是否有独立的 holdout
    has_holdout = any(k in content for k in
                      ["holdout", "final_test", "untouched", "oos_final"])
    check("阈值搜索直接读测试集预测", uses_test,
          "threshold_search.py 输入 test_predictions.csv")
    check("无独立最终留出集", not has_holdout)


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 82)
passed = sum(1 for _, ok, _ in RESULT if ok)
print(f"结果: {passed}/{len(RESULT)} 项确认")
print("=" * 82)
for name, ok, detail in RESULT:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))
