"""验证 offset 修正后，择时和买入持有是否起止完全对齐。

这是钳星审查的第二处发现：offset off-by-one。
本脚本做正式的回归测试，应长期保留。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_engine as T


def main() -> None:
    db = T.resolve_db()
    df = T.load_etf(db, "510300.SH")
    print(f"原始 df: {len(df)} 天, {df['date'].min().date()} ~ {df['date'].max().date()}")

    print()
    print("=" * 96)
    print("验证：择时 vs 买入持有 的起点是否对齐（不同 window/threshold）")
    print("=" * 96)
    print(f"{'window':<8}{'阈值':<10}{'择时天数':>10}{'BH天数':>10}"
          f"{'择时起点':>14}{'BH起点':>14}{'对齐':>8}")
    print("-" * 96)

    all_ok = True
    for w in [5, 10, 20, 30, 60]:
        for th in [0.001, 0.0, -0.001]:
            r = T.run_strict_t1(df, cost=0.0005, slip=0.0, window=w, threshold=th)
            d = r["df"]
            curve = r["curve"]
            off = T.align_offset(df, len(curve))
            bh = T.buy_hold_curve(df, off, cost=0.0005, slip=0.0)

            t_start = d.loc[1, "date"]
            b_start = df.loc[off, "date"]

            ok_days = len(curve) == len(bh)
            ok_start = t_start == b_start
            ok = ok_days and ok_start
            if not ok:
                all_ok = False

            print(f"{w:<8}{th*100:>+7.1f}%{len(curve):>10}{len(bh):>10}"
                  f"{str(t_start.date()):>14}{str(b_start.date()):>14}"
                  f"{'OK' if ok else 'FAIL':>8}")

    print("-" * 96)
    print(f"\n总体: {'全部对齐 PASS' if all_ok else '存在不对齐 FAIL'}")

    # 影响量化
    print()
    print("=" * 96)
    print("offset off-by-one 的影响量化（沪深300, window=10）")
    print("=" * 96)
    r = T.run_strict_t1(df, cost=0.0005, slip=0.0, window=10, threshold=0.0)
    off = T.align_offset(df, len(r["curve"]))
    bh_fix = T.buy_hold_curve(df, off, cost=0.0005, slip=0.0)
    bh_old = T.buy_hold_curve(df, off + 1, cost=0.0005, slip=0.0)

    s_fix, s_old = T.stats(bh_fix), T.stats(bh_old)
    print(f"  旧实现（买在 off+1）: 累计 {s_old['total']*100:+.2f}%")
    print(f"  修正后（买在 off）:   累计 {s_fix['total']*100:+.2f}%")
    print(f"  差异: {(s_fix['total'] - s_old['total'])*100:+.2f}pp")

    # 最终数字
    st = T.stats(r["curve"])
    print()
    print("=" * 96)
    print("最终数字（沪深300ETF, window=10, threshold=0, 单边万5）")
    print("=" * 96)
    print(f"  择时:     累计 {st['total']*100:+8.2f}%  年化 {st['ann']*100:+6.2f}%  "
          f"回撤 {st['dd']*100:6.1f}%  夏普 {st['sharpe']:.2f}")
    print(f"  买入持有: 累计 {s_fix['total']*100:+8.2f}%  年化 {s_fix['ann']*100:+6.2f}%  "
          f"回撤 {s_fix['dd']*100:6.1f}%  夏普 {s_fix['sharpe']:.2f}")
    print(f"  优势: {(st['total'] - s_fix['total'])*100:+.2f}pp")
    print(f"  天数: 择时={len(r['curve'])}  BH={len(bh_fix)}  交易={r['trades']}")


if __name__ == "__main__":
    main()
