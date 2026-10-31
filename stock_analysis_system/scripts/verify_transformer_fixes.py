# -*- coding: utf-8 -*-
"""验证 Transformer 修复件（对应知微 7 项发现）。

每个测试都成对出现：
  【缺陷确认】原实现有问题
  【修复验证】新实现解决该问题

用法：
  python scripts/verify_transformer_fixes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from transformer_fixes import (label_maturity_mask,
                               label_end_from_trade_calendar,
                               incomplete_label_mask,
                               ColumnScaler, snapshot_state_dict,
                               T1PositionBook)

RESULT = []


def check(name, ok, detail=""):
    RESULT.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))


print("=" * 84)
print("Transformer 修复验证")
print("=" * 84)


# ============================================================
# 1. 标签成熟度
# ============================================================
print("\n[1] 标签成熟度（对应知微第 1 项）")

# 构造交易日历
cal = pd.bdate_range("2025-01-01", periods=120)
df = pd.DataFrame({
    "stock_code": "000001",
    "date": cal,
    "close": 100.0,
})
df = label_end_from_trade_calendar(df, cal, horizon=5)

cutoff = cal[60]
print(f"  交易日 {len(cal)} 天，训练截止 {pd.Timestamp(cutoff).date()}")

mask = label_maturity_mask(df, cutoff)
n_mature = int(mask.sum())
# 截止日及之前共 61 个样本，其中最后 5 个的标签要用到截止日之后
expected = 61 - 5
print(f"  成熟样本 {n_mature}，期望 {expected}（截止日及之前 61 天 - 最后 5 天）")
check("【修复验证】未成熟标签被剔除", n_mature == expected,
      f"{n_mature} vs {expected}")

# 对比：旧逻辑（只按 date <= cutoff 分箱）
old_mask = (df["date"] <= cutoff).values
n_old = int(old_mask.sum())
print(f"  旧逻辑会保留 {n_old} 个样本（含 5 个未成熟）")
check("【缺陷确认】旧逻辑保留了未成熟标签", n_old == 61, f"{n_old}")
check("【修复验证】新逻辑更严格", n_mature < n_old,
      f"{n_mature} < {n_old}")

# 验证「截止日后的价格能改变标签」这个现象
def fwd_ret(prices, h=5):
    out = np.full(len(prices), np.nan)
    for i in range(len(prices) - h):
        out[i] = prices[i + h] / prices[i] - 1.0
    return out

pa = np.full(len(cal), 100.0)
pb = np.full(len(cal), 100.0)
pb[61:] = 120.0
i_cut = 60
ra, rb = fwd_ret(pa)[i_cut], fwd_ret(pb)[i_cut]
check("【缺陷确认】截止日后价格改变标签", ra != rb, f"{ra:+.2f} -> {rb:+.2f}")

# 用 mask 后，该样本被剔除
idx = list(df["date"]).index(cutoff)
check("【修复验证】截止日样本被剔除", not mask[idx], f"index {idx}")


# ============================================================
# 2. 不完整标签
# ============================================================
print("\n[2] 不完整标签（对应知微第 2 项）")

n_future = np.array([50, 45, 44, 10, 5, 0])
bad = incomplete_label_mask(None, n_future, horizon=45)
print(f"  未来天数: {n_future.tolist()}")
print(f"  不完整:   {bad.tolist()}")
expected_bad = [False, False, True, True, True, True]
check("【修复验证】未来不足 45 天全部标记", bad.tolist() == expected_bad)

# 旧逻辑会把这些当完整
prices = np.array([100.0, 100.0, 100.0, 50.0])
nan_padded = np.concatenate([prices, np.full(41, np.nan)])
old_max = np.nanmax(nan_padded)
old_min = np.nanmin(nan_padded)
print(f"  旧逻辑对 5 天真实数据算出 max={old_max:.2f} min={old_min:.2f}")
check("【缺陷确认】旧逻辑用不足数据算出有限标签",
      np.isfinite(old_max) and np.isfinite(old_min),
      f"max={old_max:.2f}, min={old_min:.2f}")


# ============================================================
# 3. 逐列尺度
# ============================================================
print("\n[3] 逐列尺度（对应知微第 3 项）")

try:
    sys.path.insert(0, str(ROOT / "experiments" / "financial_transformer" / "src"))
    sys.path.insert(0, str(ROOT / "src"))
    from financial_transformer_experiment.model import FinancialTransformer

    torch.manual_seed(17)
    n_feat, seq_len = 32, 10
    model = FinancialTransformer(
        input_dim=n_feat, d_model=64, nhead=4, num_layers=2,
        dim_feedforward=128, dropout=0.0, model_variant="flat")
    model.eval()

    base = torch.zeros(1, seq_len, n_feat)
    base[:, :, 0] = 1e8      # amount
    base[:, :, 1] = 1e6      # volume
    x1 = base.clone(); x1[:, :, 2] = -0.1
    x2 = base.clone(); x2[:, :, 2] = 0.1

    with torch.no_grad():
        o1, o2 = model(x1), model(x2)
    if isinstance(o1, dict):
        k = "short_up" if "short_up" in o1 else list(o1.keys())[0]
        o1, o2 = o1[k], o2[k]
    d_raw = (o1 - o2).abs().max().item()
    print(f"  原始尺度输出差: {d_raw:.10f}")
    check("【缺陷确认】原始尺度下小特征被完全淹没", d_raw < 1e-8, f"{d_raw:.2e}")

    # 修复：先标准化
    scaler = ColumnScaler()
    cols = [f"f{i}" for i in range(n_feat)]
    fit_df = pd.DataFrame(
        np.random.RandomState(0).randn(200, n_feat) * 0.1, columns=cols)
    fit_df["f0"] = 1e8
    fit_df["f1"] = 1e6
    scaler.fit(fit_df[cols])

    def to_scaled(v2):
        d = fit_df.iloc[:1][cols].copy()
        d["f2"] = v2
        return torch.from_numpy(
            scaler.transform(d).values.astype(np.float32)
        ).unsqueeze(1).repeat(1, seq_len, 1)

    s1, s2 = to_scaled(-0.1), to_scaled(0.1)
    with torch.no_grad():
        p1, p2 = model(s1), model(s2)
    if isinstance(p1, dict):
        k = "short_up" if "short_up" in p1 else list(p1.keys())[0]
        p1, p2 = p1[k], p2[k]
    d_norm = (p1 - p2).abs().max().item()
    print(f"  标准化后输出差: {d_norm:.10f}")
    check("【修复验证】标准化后小特征生效", d_norm > d_raw,
          f"{d_norm:.2e} > {d_raw:.2e}")

    # scaler 可序列化
    d = scaler.to_dict()
    sc2 = ColumnScaler.from_dict(d)
    same = np.allclose(sc2.transform(fit_df[cols]).values,
                       scaler.transform(fit_df[cols]).values)
    check("【修复验证】scaler 可随 checkpoint 保存/恢复", same)

except Exception as e:
    check("第 3 项检查", False, f"{type(e).__name__}: {e}")


# ============================================================
# 4. T+1 账本
# ============================================================
print("\n[4] T+1 卖出约束（对应知微第 4 项）")

book = T1PositionBook()
d1 = pd.Timestamp("2025-02-04")
book.buy("000001", d1, 100.0, 1000)
print(f"  {d1.date()} 买入 000001 @100 x1000")
check("【修复验证】当天不可卖", not book.can_sell("000001", d1),
      f"可卖 {book.sellable_qty('000001', d1):.0f}")

# 当天卖出必须抛错
try:
    book.sell("000001", d1, 94.0, 1000)
    check("【修复验证】当天卖出被拒绝", False, "未抛异常")
except ValueError as e:
    check("【修复验证】当天卖出被拒绝", True, str(e)[:50])

d2 = pd.Timestamp("2025-02-05")
check("【修复验证】次日可卖", book.can_sell("000001", d2),
      f"可卖 {book.sellable_qty('000001', d2):.0f}")

pnl = book.sell("000001", d2, 94.0, 1000)
check("【修复验证】次日卖出成功", abs(pnl - (-6000.0)) < 1e-6, f"pnl={pnl:.0f}")

# T+0 例外（ETF）显式开启
book2 = T1PositionBook()
book2.buy("510300", d1, 4.0, 10000, t0_ok=True)
check("【修复验证】T+0 品种可显式放开", book2.can_sell("510300", d1))


# ============================================================
# 5. 独立快照
# ============================================================
print("\n[5] 模型快照（对应知微第 5 项）")

m = nn.Linear(2, 2)
old_snap = {k: v.cpu() for k, v in m.state_dict().items()}
w0 = old_snap["weight"].clone()
with torch.no_grad():
    m.weight.fill_(7.0)
broken = not torch.equal(w0, old_snap["weight"])
check("【缺陷确认】旧写法快照被污染", broken,
      f"{w0.flatten()[0]:.3f} -> {old_snap['weight'].flatten()[0]:.3f}")

m2 = nn.Linear(2, 2)
new_snap = snapshot_state_dict(m2)
w1 = new_snap["weight"].clone()
with torch.no_grad():
    m2.weight.fill_(7.0)
ok = torch.equal(w1, new_snap["weight"])
check("【修复验证】新写法快照不变", ok,
      f"仍是 {new_snap['weight'].flatten()[0]:.3f}")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 84)
passed = sum(1 for _, ok, _ in RESULT if ok)
print(f"结果: {passed}/{len(RESULT)} PASS")
print("=" * 84)
for name, ok, detail in RESULT:
    if not ok:
        print(f"  FAILED: {name}  ({detail})")
if passed == len(RESULT):
    print("全部通过")
else:
    sys.exit(1)
