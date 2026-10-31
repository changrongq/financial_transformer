# -*- coding: utf-8 -*-
"""Transformer 预测链修复工具（知微 2026-10-31 复核后新增）。

本模块提供四个可独立测试的修复件，不侵入原训练流程：

  1. `label_maturity_mask`  —— 标签成熟度：训练/验证不得使用未成熟标签
  2. `ColumnScaler`          —— 逐列尺度变换，随 checkpoint 保存
  3. `snapshot_state_dict`   —— 独立快照，避免 CPU 上共享存储
  4. `T1PositionBook`        —— 账本层面的 T+1 可卖约束

设计原则
--------
知微的核心要求是「**每个任务保存 label_start / label_end / available_at**」。
所以本模块不猜测交易日历，而是要求调用方显式传入「每个样本的标签观察结束日」。

用法
----
    from transformer_fixes import (label_maturity_mask, ColumnScaler,
                                   snapshot_state_dict, T1PositionBook)

    # 1. 标签成熟度
    mask = label_maturity_mask(df, cutoff="2025-09-30",
                               label_end_col="label_end_date")
    df_train = df[mask]

    # 2. 逐列尺度
    scaler = ColumnScaler().fit(df_train[feat_cols])
    X = scaler.transform(df_all[feat_cols])
    torch.save({"model": sd, "scaler": scaler.to_dict()}, path)

    # 3. 快照
    best = snapshot_state_dict(model)

    # 4. T+1 账本
    book = T1PositionBook()
    book.buy("000001", date, price, qty)
    assert not book.can_sell("000001", date)      # 当天不能卖
    book.buy("000002", date2, price, qty)
    assert book.can_sell("000001", date2)         # 次日可以
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. 标签成熟度
# ============================================================
def label_maturity_mask(
    df: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    *,
    date_col: str = "date",
    label_end_col: str = "label_end_date",
    horizon_days: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """返回布尔数组：True 表示该样本的标签在 cutoff 之前已经成熟。

    参数
    ----
    df            : 至少包含 date_col 和 label_end_col
    cutoff        : 数据截止日（如训练截止日）
    label_end_col : 标签观察结束日。若不存在，用 horizon_days 估算

    为什么必须这样做
    ----------------
    训练截止日 2025-09-30 当天的「未来 5 日收益」需要 10 月的数据。
    如果直接用它训练，等于让模型看到截止日之后发生的事情。
    实测：截止日价格不变，只改之后的价格，标签从 0.00 变成 +0.20。

    返回
    ----
    np.ndarray[bool]，长度 = len(df)
    """
    cutoff = pd.Timestamp(cutoff)

    if label_end_col in df.columns:
        end = pd.to_datetime(df[label_end_col], errors="coerce")
    elif horizon_days is not None:
        # 没有显式结束日时，按最长 horizon 的**自然日**保守估计
        # （不精确，但方向正确；正式实验应提供 label_end_col）
        max_h = int(max(horizon_days))
        end = pd.to_datetime(df[date_col], errors="coerce") + pd.Timedelta(days=max_h * 2)
    else:
        raise ValueError(
            "必须提供 label_end_col，或传入 horizon_days 做保守估计。"
            "知微明确要求：不能仅调整 train_end 字符串。")

    start = pd.to_datetime(df[date_col], errors="coerce")
    ok = end.notna() & start.notna() & (end <= cutoff)
    return ok.values


def label_end_from_trade_calendar(
    df: pd.DataFrame,
    trade_dates: Sequence,
    horizon: int,
    *,
    date_col: str = "date",
    out_col: str = "label_end_date",
) -> pd.DataFrame:
    """用交易日历精确计算每个样本的标签观察结束日。

    这是 label_maturity_mask 的前置步骤。与「加 N 个自然日」不同，
    它严格按交易日计数，不会因周末/长假估错。
    """
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(list(trade_dates))))
    pos = cal.searchsorted(pd.to_datetime(df[date_col]), side="left")

    end = []
    for p in pos:
        j = p + int(horizon)
        end.append(cal[j] if j < len(cal) else pd.NaT)

    out = df.copy()
    out[out_col] = end
    return out


def incomplete_label_mask(
    df: pd.DataFrame,
    n_future: Sequence[int],
    horizon: int,
    *,
    out_col: str = "_n_future",
) -> np.ndarray:
    """标记「未来观测不足 horizon 天」的样本。

    知微第 2 项：`future_window.max(axis=1)` 默认跳过 NaN，
    所以只有 5 天数据也能算出「完整的 45 天最大涨幅」。

    n_future 是每个样本在数据中实际拥有的未来交易日数。
    返回 True 表示**标签不完整，必须剔除**。
    """
    arr = np.asarray(n_future, dtype=float)
    return arr < int(horizon)


# ============================================================
# 2. 逐列尺度
# ============================================================
class ColumnScaler:
    """逐列尺度变换（z-score 或 log 后 z-score）。

    为什么需要
    ----------
    默认 32 特征含 `amount`(1e8)、`volume`(1e6) 与 `ret5d`(±0.1)。
    flat 分支先 `Linear` 混合再 `LayerNorm`，但 **投影后的 LayerNorm
    不等于投影前的逐列尺度处理**：大数值先在 float32 里主导线性组合，
    小特征的贡献已经丢失。

    实测：原架构下把 ret5d 从 -0.1 改到 +0.1，输出差 **0.0000000000**；
    归一化后差 **1.98e-03**。

    保存
    ----
    必须随 checkpoint 一起保存（`to_dict` / `from_dict`），
    推理时加载同一份变换，**禁止在全量数据上重新拟合**。
    """

    def __init__(self, log_cols: Optional[Iterable[str]] = None):
        self.log_cols = list(log_cols or [])
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> "ColumnScaler":
        X = X.copy()
        for c in self.log_cols:
            if c in X.columns:
                X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
        self.columns_ = list(X.columns)
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("必须先 fit 或 from_dict")
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"缺少列: {missing[:5]}")
        X = X[self.columns_].copy()
        for c in self.log_cols:
            if c in X.columns:
                X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
        return (X - self.mean_) / self.std_

    def to_dict(self) -> dict:
        return {
            "log_cols": self.log_cols,
            "columns": self.columns_,
            "mean": None if self.mean_ is None else self.mean_.to_dict(),
            "std": None if self.std_ is None else self.std_.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnScaler":
        s = cls(d.get("log_cols"))
        s.columns_ = d.get("columns")
        if d.get("mean") is not None:
            s.mean_ = pd.Series(d["mean"])
            s.std_ = pd.Series(d["std"])
        return s


# ============================================================
# 3. 独立快照
# ============================================================
def snapshot_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """真正的独立快照。

    原代码 `{k: v.cpu() for k, v in model.state_dict().items()}`
    在参数已在 CPU 时**不复制**，`.cpu()` 直接返回同一存储。
    实测：保存后把模型 weight 改成 7，best_state 里也变成 7。

    正确写法必须 `detach().cpu().clone()`。
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# 4. T+1 账本
# ============================================================
@dataclass
class _Lot:
    """一批可卖/不可卖的持仓。"""
    stock_code: str
    buy_date: pd.Timestamp
    price: float
    qty: float
    sellable_from: pd.Timestamp


@dataclass
class T1PositionBook:
    """A 股 T+1 持仓账本。

    规则：T 日买入的股票，T+1 日才能卖。
    （ETF/可转债等例外需显式传 `t0_ok=True`，不要默认放开。）

    原回测在「加入 open_positions 之后」立刻检查当天 high/low 止盈止损，
    等价于允许当天买当天卖。合成复现得到的成交记录是：
        buy_date = sell_date = 2025-02-04, 止损出场
    这在普通 A 股不合法。
    """

    lots: List[_Lot] = field(default_factory=list)
    realized: float = 0.0

    def buy(self, code: str, date, price: float, qty: float,
            *, t0_ok: bool = False) -> None:
        d = pd.Timestamp(date)
        # 下一个交易日才能卖：这里用「日期严格大于买入日」近似，
        # 精确实现应传入交易日历取下一交易日
        sellable = d if t0_ok else d + pd.Timedelta(days=1)
        self.lots.append(_Lot(code, d, float(price), float(qty), sellable))

    def sellable_qty(self, code: str, date) -> float:
        d = pd.Timestamp(date)
        return sum(l.qty for l in self.lots
                   if l.stock_code == code and l.sellable_from <= d)

    def can_sell(self, code: str, date) -> bool:
        return self.sellable_qty(code, date) > 0

    def sell(self, code: str, date, price: float, qty: float) -> float:
        """卖出，返回实现盈亏。不可卖数量会被拒绝。"""
        d = pd.Timestamp(date)
        avail = self.sellable_qty(code, date)
        if qty > avail + 1e-9:
            raise ValueError(
                f"{code} 在 {d.date()} 可卖 {avail:.0f} 股，试图卖出 {qty:.0f} 股。"
                "T+1 约束被违反。")
        pnl = 0.0
        left = qty
        for lot in self.lots:
            if left <= 0:
                break
            if lot.stock_code != code or lot.sellable_from > d:
                continue
            take = min(lot.qty, left)
            pnl += (float(price) - lot.price) * take
            lot.qty -= take
            left -= take
        self.lots = [l for l in self.lots if l.qty > 1e-9]
        self.realized += pnl
        return pnl

    def holdings(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for l in self.lots:
            out[l.stock_code] = out.get(l.stock_code, 0.0) + l.qty
        return out
