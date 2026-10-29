"""多标的择时验证 v2（修正口径：严格 T+1）。

修正说明：
  前一版 validate_timing_multi_assets.py 用"只吃日内"口径，T+1 下不可实现。
  本版用 timing_engine.run_strict_t1（T日信号 → T+1开盘执行 → 吃全天收益）。

对比三种口径：
  A 理想化（违规）— 仅供参考，展示虚高幅度
  B 严格 T+1（真实）— 这是可实现的
  买入持有

输出：
  · 各标的三口径对比
  · 严格 T+1 下的参数稳健性
  · 严格 T+1 下的分年度一致性
  · 滑点敏感性

用法：
  python scripts/validate_timing_multi_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_engine as T

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

ASSETS = [
    ("510050.SH", "上证50ETF", "大盘蓝筹"),
    ("510300.SH", "沪深300ETF", "大盘"),
    ("510500.SH", "中证500ETF", "中盘"),
    ("512100.SH", "中证1000ETF", "小盘"),
    ("159915.SZ", "创业板ETF", "成长"),
    ("588000.SH", "科创50ETF", "科技"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    db = T.resolve_db()

    # ========== 第一部分：三口径对比 ==========
    print("=" * 108)
    print("【核心结果】三种口径对比（A=理想化违规，B=严格T+1真实）")
    print("=" * 108)
    print(f"{'标的':<14}{'风格':<9}{'年数':>5}{'买入持有':>11}{'A理想化':>11}{'B严格T+1':>12}"
          f"{'B优势':>11}{'B回撤/买入':>14}{'B夏普/买入':>14}")
    print("-" * 108)

    rows = []
    for code, name, style in ASSETS:
        df = T.load_etf(db, code)
        if df is None:
            print(f"{name:<14}{style:<9} 数据不足")
            continue

        strict = T.run_strict_t1(df, args.cost, 0.0, args.window, args.threshold)
        ideal = T.run_intraday_idealized(df, args.cost, 0.0, args.window, args.threshold)
        offset = len(df) - len(strict["curve"])
        bh = T.buy_hold_curve(df, offset)

        m = min(len(strict["curve"]), len(ideal["curve"]), len(bh))
        s_s, s_i, s_b = (T.stats(strict["curve"][:m]), T.stats(ideal["curve"][:m]),
                         T.stats(bh[:m]))
        adv = s_s["total"] - s_b["total"]
        flag = "WIN " if adv > 0 else "LOSE"

        print(f"{name:<14}{style:<9}{len(df)/244:>5.1f}{s_b['total']*100:>+10.1f}%"
              f"{s_i['total']*100:>+10.1f}%{s_s['total']*100:>+11.1f}%"
              f"{flag}{adv*100:>+7.1f}pp"
              f"{s_s['dd']*100:>8.1f}%/{s_b['dd']*100:>6.1f}%"
              f"{s_s['sharpe']:>8.2f}/{s_b['sharpe']:>6.2f}")

        rows.append({
            "code": code, "name": name, "style": style,
            "years": round(len(df) / 244, 1), "days": int(m),
            "buy_hold": round(s_b["total"], 6),
            "idealized": round(s_i["total"], 6),
            "strict_t1": round(s_s["total"], 6),
            "advantage": round(adv, 6),
            "strict_dd": round(s_s["dd"], 6), "bh_dd": round(s_b["dd"], 6),
            "strict_sharpe": round(s_s["sharpe"], 4), "bh_sharpe": round(s_b["sharpe"], 4),
            "trades": strict["trades"],
        })

    print("-" * 108)
    if rows:
        n_win = sum(1 for r in rows if r["advantage"] > 0)
        print(f"\n严格 T+1 下跑赢买入持有: {n_win}/{len(rows)}")
        print(f"优势中位数: {np.median([r['advantage'] for r in rows])*100:+.1f}pp")
        dd_improve = [r["strict_dd"] - r["bh_dd"] for r in rows]
        print(f"回撤改善中位数: {np.median(dd_improve)*100:+.1f}pp  "
              f"（改善的标的: {sum(1 for x in dd_improve if x>0)}/{len(dd_improve)}）")
        sh_improve = [r["strict_sharpe"] - r["bh_sharpe"] for r in rows]
        print(f"夏普改善中位数: {np.median(sh_improve):+.2f}  "
              f"（改善的标的: {sum(1 for x in sh_improve if x>0)}/{len(sh_improve)}）")
        print(f"理想化口径虚高中位数: "
              f"{np.median([r['idealized']-r['strict_t1'] for r in rows])*100:+.1f}pp")

    # ========== 第二部分：参数稳健性（严格T+1）==========
    print("\n" + "=" * 108)
    print("【参数稳健性】严格 T+1 下，20 组参数 × 6 标的")
    print("=" * 108)
    print(f"{'窗口':<8}" + "".join(f"{f'th{t*100:+.1f}%':>13}" for t in [0.001, 0.0, -0.001, -0.002]))
    print("-" * 108)

    grid = {}
    for w in [5, 10, 20, 30, 60]:
        line = f"{w:<8}"
        for th in [0.001, 0.0, -0.001, -0.002]:
            wins = tot = 0
            advs = []
            for code, name, style in ASSETS:
                df = T.load_etf(db, code)
                if df is None:
                    continue
                r = T.run_strict_t1(df, args.cost, 0.0, w, th)
                off = len(df) - len(r["curve"])
                bh = T.buy_hold_curve(df, off)
                m = min(len(r["curve"]), len(bh))
                s_s, s_b = T.stats(r["curve"][:m]), T.stats(bh[:m])
                a = s_s["total"] - s_b["total"]
                advs.append(a)
                tot += 1
                wins += int(a > 0)
            grid[f"w{w}_th{th}"] = {"wins": wins, "total": tot,
                                    "median_adv": round(float(np.median(advs)), 6) if advs else 0}
            line += f"{wins:>4}/{tot:<8}"
        print(line)
    print("-" * 108)

    all_w = sum(v["wins"] for v in grid.values())
    all_t = sum(v["total"] for v in grid.values())
    print(f"\n  总胜场: {all_w}/{all_t} = {all_w/all_t*100:.1f}%")
    print(f"  每组参数最少胜场: {min(v['wins'] for v in grid.values())}/{max(v['total'] for v in grid.values())}")

    # ========== 第三部分：滑点敏感性（严格T+1）==========
    print("\n" + "=" * 108)
    print("【滑点敏感性】严格 T+1 下加不同滑点")
    print("=" * 108)
    print(f"{'标的':<14}{'买入持有':>11}" + "".join(f"{f'slip{s*100:.2f}%':>13}"
                                                for s in [0.0, 0.0005, 0.001, 0.002]))
    print("-" * 108)

    slip_rows = []
    for code, name, style in ASSETS:
        df = T.load_etf(db, code)
        if df is None:
            continue
        bh = T.buy_hold_curve(df, len(df) - (len(df) - args.window - 1))
        s_b = T.stats(bh)
        line = f"{name:<14}{s_b['total']*100:>+10.1f}%"
        row = {"code": code, "name": name, "buy_hold": round(s_b["total"], 6)}
        for slip in [0.0, 0.0005, 0.001, 0.002]:
            r = T.run_strict_t1(df, args.cost, slip, args.window, args.threshold)
            off = len(df) - len(r["curve"])
            bh2 = T.buy_hold_curve(df, off)
            m = min(len(r["curve"]), len(bh2))
            s_s = T.stats(r["curve"][:m])
            s_b2 = T.stats(bh2[:m])
            adv = s_s["total"] - s_b2["total"]
            flag = "" if adv > 0 else "*"
            line += f"{s_s['total']*100:>+9.1f}%{flag:<3}"
            row[f"slip_{slip}"] = round(s_s["total"], 6)
            row[f"adv_{slip}"] = round(adv, 6)
        print(line)
        slip_rows.append(row)
    print("-" * 108)
    print("  （* 表示该滑点下已跑输买入持有）")
    print("\n  各滑点下跑赢标的数:")
    for slip in [0.0, 0.0005, 0.001, 0.002]:
        n = sum(1 for r in slip_rows if r.get(f"adv_{slip}", -1) > 0)
        print(f"    滑点 {slip*100:>4.2f}%: {n}/{len(slip_rows)}")

    # ========== 第四部分：分年度（严格T+1）==========
    print("\n" + "=" * 108)
    print("【分年度一致性】严格 T+1 下（默认参数）")
    print("=" * 108)
    for code, name in [("510050.SH", "上证50ETF"), ("512100.SH", "中证1000ETF")]:
        df = T.load_etf(db, code)
        if df is None:
            continue
        r = T.run_strict_t1(df, args.cost, 0.0, args.window, args.threshold)
        off = len(df) - len(r["curve"])
        bh = T.buy_hold_curve(df, off)
        m = min(len(r["curve"]), len(bh))
        eq_t, eq_b = r["curve"][:m], bh[:m]
        dd = df.iloc[off:off + m].reset_index(drop=True)
        dd["year"] = dd["date"].dt.year

        print(f"\n  {name}（{m/244:.1f}年，{r['trades']}次交易）")
        print(f"  {'年份':<8}{'买入持有':>12}{'严格T+1':>12}{'差额':>12}")
        print("  " + "-" * 46)
        wins = tot = 0
        for y in sorted(dd["year"].unique()):
            idx = np.where((dd["year"] == y).values)[0]
            if len(idx) < 60:
                continue
            pt = eq_t[idx[0] - 1] if idx[0] > 0 else 1.0
            pb = eq_b[idx[0] - 1] if idx[0] > 0 else 1.0
            rb = eq_b[idx[-1]] / pb - 1
            rt = eq_t[idx[-1]] / pt - 1
            wins += int(rt > rb)
            tot += 1
            print(f"  {y:<8}{rb*100:>+11.1f}%{rt*100:>+11.1f}%{(rt-rb)*100:>+11.1f}pp")
        print("  " + "-" * 46)
        if tot:
            print(f"  逐年胜率: {wins}/{tot} = {wins/tot*100:.0f}%")

    # ========== 结论 ==========
    print("\n" + "=" * 108)
    print("结论（严格 T+1 口径）")
    print("=" * 108)
    if rows:
        n_win = sum(1 for r in rows if r["advantage"] > 0)
        dd_ok = sum(1 for r in rows if r["strict_dd"] > r["bh_dd"])
        print(f"  1. 收益: {n_win}/{len(rows)} 跑赢买入持有，"
              f"优势中位数 {np.median([r['advantage'] for r in rows])*100:+.1f}pp")
        print(f"  2. 回撤: {dd_ok}/{len(rows)} 优于买入持有，"
              f"改善中位数 {np.median([r['strict_dd']-r['bh_dd'] for r in rows])*100:+.1f}pp")
        print(f"  3. 参数: {all_w}/{all_t} = {all_w/all_t*100:.1f}% 胜率")
        print(f"  4. 滑点 0.1%: {sum(1 for r in slip_rows if r.get('adv_0.001',-1)>0)}/{len(slip_rows)} 仍赢")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "timing_multi_v2_strict.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "assets": rows,
        "param_grid": grid,
        "slippage": slip_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
