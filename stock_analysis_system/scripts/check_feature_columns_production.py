# -*- coding: utf-8 -*-
"""走生产路径，确认 8 个"缺失列"的真实取值。

修正上一版审计的错误：
  上一版我手工构造 df 直接喂给 build_layer_features，
  但没有 merge stock_moneyflow，导致资金流原始列不存在，
  于是误判为"配置声明 32 个特征但只有 24 个有数据"。

  实际上 stock_moneyflow 有 216 万行数据（2025-01-02 起）。
  正确的做法是走 _load_trade_data_with_concepts 生产路径。

本脚本验证：
  1. 生产路径下 32 个特征列的**真实**非空率与唯一值数
  2. 资金流数据的时间覆盖 vs 训练窗口的缺口
  3. 哪些列真的是常数（零信息）

用法：
  python scripts/check_feature_columns_production.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "financial_transformer" / "src"))
sys.path.insert(0, str(ROOT / "src"))

FEATURE_COLS = [
    "open", "close", "high", "low", "volume", "amount", "turnover_rate",
    "return_1d", "return_5d", "return_20d", "volatility_20d",
    "amount_ma5_ma20", "volume_ma5_ma20", "rsi_14", "macd", "macd_signal",
    "trend_strength", "close_to_high", "close_to_low", "gap_open",
    "market_cap_ratio", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos", "net_mf_amount_ratio", "net_lg_amount_ratio",
    "net_elg_amount_ratio", "net_mf_ratio_5d", "net_lg_ratio_5d",
    "net_elg_ratio_5d", "mf_acceleration",
]


def main():
    from financial_transformer_experiment.data import (
        _load_trade_data_with_concepts)
    from financial_transformer_experiment.config import load_config
    from hierarchical_system.features import build_layer_features

    cfg_path = (ROOT / "experiments" / "financial_transformer"
                / "config_moneyflow_v1.yaml")
    cfg = load_config(str(cfg_path))
    trade_db = str((ROOT / cfg.databases.trade_db).resolve()) \
        if not Path(cfg.databases.trade_db).is_absolute() \
        else cfg.databases.trade_db

    print("=" * 92)
    print("生产路径下 32 个特征列的真实情况")
    print("=" * 92)
    print(f"trade_db: {trade_db}")

    # 取几只股票，走生产路径
    import sqlite3
    con = sqlite3.connect(trade_db)
    codes = [r[0] for r in con.execute(
        "SELECT DISTINCT stock_code FROM stock_data "
        "ORDER BY stock_code LIMIT 6").fetchall()]
    con.close()

    frames = []
    for code in codes:
        try:
            td = _load_trade_data_with_concepts(
                trade_db, code, manual_concept_map={},
                concept_label_source=None)
            if td.empty:
                continue
            f = build_layer_features(td, 30.0)
            # 加时间特征（生产路径里也有这步）
            from financial_transformer_experiment.data import _add_time_features
            f = _add_time_features(f)
            frames.append(f)
            print(f"  {code}: {len(f):,} 行, "
                  f"资金流列={'净流入额' in td.columns}")
        except Exception as e:
            print(f"  {code}: {type(e).__name__}: {e}")

    if not frames:
        print("\n未能构造任何特征帧")
        return 1

    out = pd.concat(frames, ignore_index=True)
    print(f"\n合并后 {len(out):,} 行, {len(out.columns)} 列")

    # ============================================================
    # 1. 逐列真实情况
    # ============================================================
    print("\n" + "=" * 92)
    print("[1] 32 个配置特征的实际情况")
    print("=" * 92)
    print(f"  {'列':<22}{'存在':>6}{'非空率':>9}{'唯一值':>9}"
          f"{'均值':>13}{'标准差':>13}  判定")

    rows = []
    for c in FEATURE_COLS:
        if c not in out.columns:
            print(f"  {c:<22}{'N':>6}{'-':>9}{'-':>9}{'-':>13}"
                  f"{'-':>13}  **列不存在**")
            rows.append((c, False, 0, 0, np.nan, np.nan, "列不存在"))
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        n = int(s.notna().sum())
        rate = n / len(out) * 100 if len(out) else 0
        u = int(s.nunique(dropna=True))
        mu = float(s.mean()) if n else np.nan
        sd = float(s.std()) if n else np.nan
        if n == 0:
            v = "**全 NaN**"
        elif u <= 1:
            v = f"**常数 {mu:.6g}**"
        else:
            v = "有变化"
        rows.append((c, True, rate, u, mu, sd, v))
        print(f"  {c:<22}{'Y':>6}{rate:>8.1f}%{u:>9}{mu:>13.5g}"
              f"{sd:>13.5g}  {v}")

    # ============================================================
    # 2. 汇总
    # ============================================================
    print("\n" + "=" * 92)
    print("[2] 汇总")
    print("=" * 92)
    missing = [r for r in rows if not r[1]]
    allnan = [r for r in rows if r[1] and r[2] == 0]
    const = [r for r in rows if r[1] and 0 < r[2] and r[3] <= 1]
    ok = [r for r in rows if r[1] and r[3] > 1]

    print(f"  列不存在:  {len(missing)} 个 {[r[0] for r in missing]}")
    print(f"  全 NaN:    {len(allnan)} 个 {[r[0] for r in allnan]}")
    print(f"  常数列:    {len(const)} 个 {[(r[0], r[4]) for r in const]}")
    print(f"  有变化:    {len(ok)} 个")

    # ============================================================
    # 3. 资金流时间覆盖
    # ============================================================
    print("\n" + "=" * 92)
    print("[3] 资金流数据 vs 训练窗口的时间覆盖")
    print("=" * 92)
    import sqlite3
    con = sqlite3.connect(trade_db)
    r = con.execute("SELECT MIN(日期), MAX(日期), COUNT(*) "
                    "FROM stock_moneyflow").fetchone()
    mf_min, mf_max, mf_n = r
    r = con.execute("SELECT MIN(日期), MAX(日期), COUNT(*) "
                    "FROM stock_data").fetchone()
    sd_min, sd_max, sd_n = r
    con.close()

    print(f"  stock_data     : {sd_min} ~ {sd_max}  ({sd_n:,} 行)")
    print(f"  stock_moneyflow: {mf_min} ~ {mf_max}  ({mf_n:,} 行)")
    if mf_min > sd_min:
        n_before = con_days = None
        import sqlite3
        con = sqlite3.connect(trade_db)
        n_before = con.execute(
            "SELECT COUNT(DISTINCT 日期) FROM stock_data WHERE 日期 < ?",
            (mf_min,)).fetchone()[0]
        con.close()
        print(f"\n  ★ 训练数据比资金流多 {n_before} 个交易日")
        print(f"    这些天的资金流特征会是 NaN → 被 mask 成 0.0")
        print(f"    也就是说：{n_before} 个交易日的 7 个资金流特征是常数 0")

    # ============================================================
    # 4. 结论
    # ============================================================
    print("\n" + "=" * 92)
    print("[4] 结论")
    print("=" * 92)
    print(f"""
  与上一版审计的差异：
    上一版我说"32 个特征只有 24 个有数据"——**这是错的**。
    原因是我手工构造 df 时没有 merge stock_moneyflow，
    误把"我的脚本没加载"当成"数据不存在"。

    走生产路径后，资金流列是存在的（stock_moneyflow 有 216 万行）。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
