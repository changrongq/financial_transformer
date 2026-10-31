# -*- coding: utf-8 -*-
"""验证 market_cap_ratio 与 close 是否完全共线。

代码（features.py:153）：
    market_cap_ratio = _safe_div(df["close"], market_cap_yi)

而 market_cap_yi 来自 _latest_market_caps()，是**最新一期**市值（常数）。
所以 market_cap_ratio = close / 常数 → 与 close 完全共线。

这既是尺度问题（两列携带同一信息），
也是知微第 6 项（用最新市值选历史池）的同一个根因。

用法：
  python scripts/verify_market_cap_collinearity.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "financial_transformer" / "src"))
sys.path.insert(0, str(ROOT / "src"))


def main():
    from financial_transformer_experiment.data import (
        _load_trade_data_with_concepts)
    from hierarchical_system.features import build_layer_features

    db = str((ROOT.parent / "stock_history.db").resolve())
    con = sqlite3.connect(db)
    codes = [r[0] for r in con.execute(
        "SELECT DISTINCT stock_code FROM stock_data "
        "ORDER BY stock_code LIMIT 5").fetchall()]
    con.close()

    print("=" * 88)
    print("market_cap_ratio vs close 共线性验证")
    print("=" * 88)

    frames = []
    for c in codes:
        try:
            td = _load_trade_data_with_concepts(
                db, c, manual_concept_map={}, concept_label_source=None)
            if td.empty:
                continue
            f = build_layer_features(td, 30.0)
            frames.append(f)
        except Exception as e:
            print(f"  {c}: {type(e).__name__}: {e}")

    if not frames:
        print("未能构造特征帧")
        return 1

    out = pd.concat(frames, ignore_index=True)
    print(f"\n样本 {len(out):,} 行")

    if "market_cap_ratio" not in out.columns or "close" not in out.columns:
        print("缺少所需列")
        return 1

    a = pd.to_numeric(out["market_cap_ratio"], errors="coerce")
    b = pd.to_numeric(out["close"], errors="coerce")
    m = a.notna() & b.notna()
    print(f"有效样本 {int(m.sum()):,}")

    if m.sum() < 100:
        print("有效样本不足")
        return 1

    corr = float(np.corrcoef(a[m], b[m])[0, 1])
    ratio = (a[m] / b[m]).replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\n[1] 线性关系")
    print(f"  相关系数           : {corr:.12f}")
    print(f"  ratio 唯一值数     : {ratio.nunique()}")
    print(f"  ratio 中位数       : {ratio.median():.10g}")
    print(f"  ratio 标准差       : {ratio.std():.3g}")

    print(f"\n[2] 判定")
    if abs(corr) > 0.999999 and ratio.nunique() <= 3:
        print(f"  ★ 完全共线：market_cap_ratio = close × {ratio.median():.10g}")
        print(f"    （market_cap_yi 是常数 30.0，因为我们传的是固定值）")
        print(f"  → 这一列不提供任何 close 之外的信息")
    elif abs(corr) > 0.999:
        print(f"  ★ 高度共线（r={corr:.8f}），基本是 close 的线性变换")
    else:
        print(f"  不共线（r={corr:.6f}）")

    # 各列的"唯一信息量"
    print(f"\n[3] 相关列的唯一值数对比")
    for c in ["close", "market_cap_ratio", "amount",
              "amount_market_cap_ratio"]:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce").dropna()
            print(f"  {c:<26} 唯一值 {s.nunique():>6}  "
                  f"std {s.std():.6g}")

    print(f"""
[4] 结论

  market_cap_ratio = close / market_cap_yi
  由于 market_cap_yi 是**最新一期常数**（知微第 6 项）：
    · 对单只股票，它只是 close 的线性缩放
    · 对**跨股票**，它携带"谁市值小"的信息（这正是 micro 层的定义）

  所以在 micro 层内部（同一市值档），这一列几乎无额外信息。
  但在跨层训练时它是有意义的。

  真正的问题是：market_cap_yi 用**最新市值**回填到整段历史，
  违反了 as-of 原则（知微第 6 项）。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
