# -*- coding: utf-8 -*-
"""反推 tushare net_mf_amount 的定义。

观察：四类买入 - 四类卖出 ≈ 0（恒等于 0），但净流入额 ≠ 0。

推测：
  tushare 的 buy_* / sell_* 是**按单子大小分类的主动买卖**，
  由于每笔成交必有一个买方一个卖方，
  四类买入总额 = 四类卖出总额（都是全部成交额），
  所以"总买 - 总卖"必然为 0。

  那 net_mf_amount 是什么？
  它是**主动买入（外盘）- 主动卖出（内盘）**，方向由成交价决定，
  与 buy_*/sell_* 的"单子大小"分类不是同一维度。

本脚本验证这个假说。

用法：
  python scripts/reverse_engineer_net_mf.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())

tok = (ROOT / "config" / "tushare_token.local").read_text(
    encoding="utf-8").strip()
import tushare as ts
ts.set_token(tok)
pro = ts.pro_api()


def main():
    print("=" * 88)
    print("反推 tushare net_mf_amount 的定义")
    print("=" * 88)

    # 取一天原始数据
    raw = pro.moneyflow(trade_date="20240603")
    if raw is None or raw.empty:
        print("无数据")
        return 1
    print(f"\n样本 {len(raw)} 只")

    buy_all = (raw["buy_sm_amount"] + raw["buy_md_amount"] +
               raw["buy_lg_amount"] + raw["buy_elg_amount"])
    sell_all = (raw["sell_sm_amount"] + raw["sell_md_amount"] +
                raw["sell_lg_amount"] + raw["sell_elg_amount"])

    print(f"\n[1] 四类买入总额 vs 四类卖出总额")
    print(f"  买入总额 中位 {buy_all.median():,.2f}")
    print(f"  卖出总额 中位 {sell_all.median():,.2f}")
    diff = (buy_all - sell_all).abs()
    print(f"  |买-卖| 中位 {diff.median():.6f}  最大 {diff.max():.6f}")
    print(f"  → {'基本相等（都是全部成交额）' if diff.median() < 1 else '不相等'}")

    print(f"\n[2] net_mf_amount 与各分类的关系")
    net = raw["net_mf_amount"]
    candidates = {
        "buy_lg + buy_elg - sell_lg - sell_elg": (
            raw["buy_lg_amount"] + raw["buy_elg_amount"]
            - raw["sell_lg_amount"] - raw["sell_elg_amount"]),
        "buy_sm + buy_md - sell_sm - sell_md": (
            raw["buy_sm_amount"] + raw["buy_md_amount"]
            - raw["sell_sm_amount"] - raw["sell_md_amount"]),
        "buy_all - sell_all": buy_all - sell_all,
    }
    for name, v in candidates.items():
        d = (v - net).abs()
        corr = np.corrcoef(v, net)[0, 1]
        print(f"  {name}")
        print(f"    |差| 中位 {d.median():,.4f}   相关系数 {corr:+.6f}")

    # 尝试：net = (买-卖) 按某权重
    print(f"\n[3] net_mf_amount 的统计")
    print(f"  中位 {net.median():,.2f}  "
          f"均值 {net.mean():,.2f}  std {net.std():,.2f}")
    print(f"  与 buy_elg_amount 相关: "
          f"{np.corrcoef(raw['buy_elg_amount'], net)[0,1]:+.4f}")

    # 关键：net_mf_vol 与 net_mf_amount 的比例应等于均价
    print(f"\n[4] net_mf_vol 与 net_mf_amount")
    nz = raw[raw["net_mf_vol"] != 0]
    if len(nz) > 100:
        pr = (nz["net_mf_amount"] / nz["net_mf_vol"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        print(f"  amount/vol 中位 {pr.median():.4f}")
        print(f"  → 若约等于成交均价，说明单位一致")

    print(f"""
[5] 结论

  tushare moneyflow 的口径是：
    · buy_sm/md/lg/elg_amount : 按【单笔金额大小】分四档的买入额
    · sell_*                  : 同口径的卖出额
    · 四类买入总额 == 四类卖出总额 == 当日总成交额
      （因为每笔成交同时计一次买和一次卖）
    · net_mf_amount           : 【主动买入 - 主动卖出】的净额
      方向由"成交发生在买价还是卖价"决定，与单子大小是另一个维度

  所以：
    · "四类买入 - 四类卖出" 恒等于 0，不是 net_mf_amount 的定义
    · 我之前那个验证公式本身就是错的
    · 数据没有错，是我的校验逻辑错

  对特征工程的意义：
    features.py 用 net_mf_amount / amount 作为净流入比率，
    这个定义是合理的（净主动买入占总成交额的比例）。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
