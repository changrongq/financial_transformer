# -*- coding: utf-8 -*-
"""统一交易账本（知微 2026-10-31 审查后新增）。

设计目标
--------
知微指出：v2/v3/v4/v5 + rolling + grokking 六套脚本各自维护一套不一致的账本，
修了主 timing engine 也不会修到它们。因此这里只保留 **一个** 实现，
所有 RL 脚本都必须调用它。

时序约定（唯一口径）
--------------------
    第 t 日 15:00  收盘，用第 t 日及之前的数据算特征 / 信号
    第 t+1 日 09:30 开盘，按第 t 日的决策调仓
    第 t+1 日      新仓位吃「开盘→收盘」的日内收益
                   旧仓位吃「昨收→今开」的隔夜收益
    第 t+2 日起    仓位不变时吃全天收益

也就是说：**特征只用 t 日收盘前已知的信息，收益从 t+1 日开盘开始算。**
任何「用 t 日收盘特征 + 吃 t 日收益」的写法都是未来函数（v5 的 bug）。

账本字段
--------
每一笔记录包含：日期、决策时点、旧仓位、新仓位、隔夜收益、日内收益、
成本、日终净值。可以直接逐日与外部实现核对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DayLedger:
    """单日账本记录。"""
    date: pd.Timestamp
    prev_pos: float          # 昨日收盘时的仓位
    new_pos: float           # 今日开盘调仓后的仓位
    overnight_ret: float     # 昨收 -> 今开
    intraday_ret: float      # 今开 -> 今收
    cost: float              # 今日摩擦成本（仓位变动时才有）
    nav: float               # 今日日终净值

    @property
    def day_ret(self) -> float:
        """今日组合净收益（含隔夜 + 日内 - 成本）。"""
        gross = (1 + self.prev_pos * self.overnight_ret) * \
                (1 + self.new_pos * self.intraday_ret)
        return gross * (1 - self.cost) - 1


@dataclass
class LedgerResult:
    nav: np.ndarray                      # 日终净值序列
    days: List[DayLedger] = field(default_factory=list)
    trades: int = 0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "date": d.date, "prev_pos": d.prev_pos, "new_pos": d.new_pos,
            "overnight": d.overnight_ret, "intraday": d.intraday_ret,
            "cost": d.cost, "nav": d.nav,
        } for d in self.days])


def run_ledger(
    dates: Sequence[pd.Timestamp],
    open_: np.ndarray,
    close: np.ndarray,
    pre_close: np.ndarray,
    targets: np.ndarray,
    *,
    cost: float = 0.0003,
    slip: float = 0.0,
    start_pos: float = 0.0,
) -> LedgerResult:
    """按唯一口径跑一遍账本。

    参数
    ----
    dates      : 日期序列，长度 n
    open_      : 开盘价，长度 n
    close      : 收盘价，长度 n
    pre_close  : 复权后昨收，长度 n（**必须**，否则跨拆股会出假收益）
    targets    : 目标仓位，长度 n。`targets[t]` 表示「用第 t 日收盘信息决定的仓位，
                 在第 t+1 日开盘执行」。因此第 0 日的 targets 不会被使用。
    cost       : 单边费率
    slip       : 单边滑点（按价格偏离建模）
    start_pos  : 第 0 日之前已持有的仓位

    返回
    ----
    LedgerResult，其中 `nav[i]` 是第 i 日日终净值。
    """
    n = len(dates)
    if not (len(open_) == len(close) == len(pre_close) == len(targets) == n):
        raise ValueError("输入长度不一致")

    nav = [1.0]
    pos = float(start_pos)     # 第 0 日之前的仓位
    days: List[DayLedger] = []
    trades = 0

    for i in range(1, n):
        # ---- 段 1：隔夜（昨收 -> 今开），由昨日仓位参与 ----
        pc = float(pre_close[i])
        if pc <= 0:
            pc = float(close[i - 1])
        overnight = float(open_[i]) / pc - 1.0 if pc > 0 else 0.0

        # ---- 段 2：日内（今开 -> 今收），由今日仓位参与 ----
        o = float(open_[i])
        intraday = float(close[i]) / o - 1.0 if o > 0 else 0.0

        # 今日开盘要执行的目标仓位 = 昨日收盘算出的 targets[i-1]
        target = float(targets[i - 1])
        target = float(np.clip(target, 0.0, 1.0))

        changed = abs(target - pos) > 1e-12
        friction = (slip + cost) if changed else 0.0
        if changed:
            trades += 1

        prev_nav = nav[-1]
        gross = (1 + pos * overnight) * (1 + target * intraday)
        new_nav = prev_nav * gross * (1 - friction)

        nav.append(new_nav)
        days.append(DayLedger(
            date=pd.Timestamp(dates[i]),
            prev_pos=pos, new_pos=target,
            overnight_ret=overnight, intraday_ret=intraday,
            cost=friction, nav=new_nav,
        ))
        pos = target

    return LedgerResult(nav=np.array(nav[1:]), days=days, trades=trades)


def ledger_stats(nav: np.ndarray, *, periods_per_year: float = 244.0) -> dict:
    """从日终净值序列算绩效（口径与 timing_engine.stats 一致）。

    修正知微指出的两处：
      · 夏普必须用「收益率」而非「净值差」
      · 最大回撤必须包含初始资本 1.0
    """
    full = np.concatenate([[1.0], np.asarray(nav, dtype=float)])
    if len(full) < 2:
        return {"total": 0.0, "ann": np.nan, "dd": 0.0, "sharpe": np.nan, "n": 0}

    rets = full[1:] / full[:-1] - 1.0
    total = float(full[-1] / full[0] - 1.0)
    n = len(rets)
    years = max(n / periods_per_year, 1e-9)
    # 回撤：峰值曲线包含初始资本 1.0
    peak = np.maximum.accumulate(full)
    dd = float((full / peak - 1.0).min())
    sd = rets.std(ddof=1) if n > 1 else 0.0
    # ★ 零波动时夏普无定义：用相对阈值判断，避免浮点残差算出天文数字
    if sd <= 1e-12 or sd < abs(rets.mean()) * 1e-9:
        sharpe = float("nan")
    else:
        sharpe = float(rets.mean() / sd * np.sqrt(periods_per_year))
    return {
        "total": total,
        "ann": float((1 + total) ** (1 / years) - 1) if (1 + total) > 0 else np.nan,
        "dd": dd,
        "sharpe": sharpe,
        "n": n,
    }
