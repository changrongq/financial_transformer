"""每日「择时 + 个股动量」信号（实盘可用版）。

对应研究成果（EXPERIMENT_FULL_RECORD_FOR_DOUBAO.md 第七章）：
  - 择时（时间维度）：大盘10日均收益 < 0 → CASH（空仓）
  - 个股动量（截面维度）：HOLD 时，买近20日涨幅最强的 30%

输出：
  - 今日择时状态（HOLD / CASH）
  - 若 HOLD：给出动量选股清单（含24日涨幅、成交额、是否涨停等可执行信息）
  - 落盘 JSON 供其他程序读取

用法：
  python scripts/daily_momentum_signal.py
  python scripts/daily_momentum_signal.py --top-pct 0.30 --mom-window 20 --min-amount 10000

⚠️ 重要提示：
  1. 本脚本基于历史回测逻辑，实盘前必须小仓位验证
  2. 涨停股（涨幅≥9.5%）会被标记为「买不进」，需注意
  3. 回测存在幸存者偏差，实盘收益预期需打折
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _resolve_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def load_data(db: str, prefer: str = "auto"):
    """加载数据。

    prefer:
      - "auto"             : 优先 historical_daily（8年，277只），缺失时回退 stock_data
      - "stock_data"       : 强制用 stock_data（全市场，最新日期更近）
      - "historical_daily" : 强制用 historical_daily（8年历史，回测用）
    """
    conn = sqlite3.connect(db)
    price = pd.DataFrame()
    source = ""

    if prefer in ("auto", "historical_daily"):
        try:
            price = pd.read_sql_query(
                "SELECT stock_code, date, close, amount FROM historical_daily", conn
            )
            source = "historical_daily(行业股277只, 2019-2026)"
        except Exception:
            price = pd.DataFrame()

    if price.empty:
        price = pd.read_sql_query(
            "SELECT stock_code, [日期] AS date, [收盘] AS close, [成交额] AS amount FROM stock_data",
            conn,
        )
        source = "stock_data(全市场, 2024-2026)"

    idx = pd.read_sql_query("SELECT date, pct_chg, close FROM index_daily_sh", conn)
    bi = pd.read_sql_query("SELECT stock_code, 名称, 行业 FROM stock_basic_info", conn)
    conn.close()

    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["stock_code"] = price["stock_code"].astype(str).str.zfill(6)
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price["amount"] = pd.to_numeric(price["amount"], errors="coerce")
    price = price.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    price["ret"] = price.groupby("stock_code")["close"].pct_change()

    # 择时输入：全市场等权日收益（与 daily_regime_timing_signal.py 口径一致）
    market_eq = price.groupby("date")["ret"].mean().rename("market_ret").reset_index()

    bi["stock_code"] = bi["stock_code"].astype(str).str.zfill(6)
    return price, market_eq, bi, source


def main() -> None:
    parser = argparse.ArgumentParser(description="每日择时+动量选股信号")
    parser.add_argument("--mom-window", type=int, default=20, help="动量回看窗口（交易日）")
    parser.add_argument("--top-pct", type=float, default=0.30, help="选动量最强的前百分之多少")
    parser.add_argument("--min-amount", type=float, default=10000, help="最低成交额（万元），过滤流动性")
    parser.add_argument("--timing-window", type=int, default=10, help="择时均收益窗口")
    parser.add_argument("--max-picks", type=int, default=20, help="最多选几只（0=不限制）。实盘建议20只以内")
    parser.add_argument("--include-st", action="store_true", help="保留 ST 股（默认剔除）")
    parser.add_argument("--as-of", default=None, help="指定日期 YYYY-MM-DD（默认最新交易日），用于历史复盘")
    parser.add_argument(
        "--source",
        choices=["auto", "stock_data", "historical_daily"],
        default="stock_data",
        help="数据源：stock_data=全市场最新（实盘用，默认）；historical_daily=8年行业股（回测用）",
    )
    args = parser.parse_args()

    db = _resolve_db()
    price, market_eq, bi, source = load_data(db, prefer=args.source)
    print(f"数据源: {source}")
    print(f"数据范围: {price['date'].min().date()} ~ {price['date'].max().date()}")

    latest_date = price["date"].max()
    if args.as_of:
        as_of = pd.to_datetime(args.as_of, errors="coerce")
        if pd.isna(as_of):
            raise SystemExit(f"--as-of 日期格式错误: {args.as_of}")
        avail = price.loc[price["date"] <= as_of, "date"]
        if avail.empty:
            raise SystemExit(f"--as-of {args.as_of} 之前无数据")
        latest_date = avail.max()
    print(f"\n信号日期: {latest_date.date()}")

    # ===== 1. 择时判断（全市场等权，与 daily_regime_timing_signal.py 口径一致）=====
    me = market_eq[market_eq["date"] <= latest_date].sort_values("date")
    regime_mean = float(me["market_ret"].tail(args.timing_window).mean())
    day_ret = float(me["market_ret"].iloc[-1])
    signal = "HOLD" if regime_mean >= 0 else "CASH"

    print("\n" + "=" * 70)
    print("【1】择时信号")
    print("=" * 70)
    print(f"  大盘近 {args.timing_window} 日均收益: {regime_mean*100:+.3f}%")
    print(f"  当日全市场等权收益: {day_ret*100:+.3f}%")
    if signal == "CASH":
        print(f"  信号: CASH（空仓，不买入任何股票）")
    else:
        print(f"  信号: HOLD（可以持仓）")

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": str(latest_date.date()),
        "data_source": source,
        "timing": {
            "signal": signal,
            "regime_mean": round(regime_mean, 6),
            "day_return": round(day_ret, 6),
            "window": args.timing_window,
        },
        "picks": [],
    }

    if signal == "CASH":
        print("\n  → 今日空仓，不选股。")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "daily_momentum_signal.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入: {out}")
        return

    # ===== 2. 动量选股 =====
    print("\n" + "=" * 70)
    print(f"【2】动量选股（近{args.mom_window}日涨幅最强的前{int(args.top_pct*100)}%）")
    print("=" * 70)

    # 计算动量
    recent = price[price["date"] <= latest_date]
    pivot = recent.pivot_table(index="date", columns="stock_code", values="close", aggfunc="first")
    if len(pivot) < args.mom_window + 1:
        print("  数据不足，无法计算动量")
        return
    mom = pivot.iloc[-1] / pivot.iloc[-(args.mom_window + 1)] - 1.0

    # 最新一日成交额
    amt_latest = recent[recent["date"] == latest_date].set_index("stock_code")["amount"]
    # 最新一日涨跌幅（判断涨停）
    ret_latest = recent[recent["date"] == latest_date].set_index("stock_code")["ret"]

    candidates = pd.DataFrame({"mom": mom, "amount": amt_latest, "ret_today": ret_latest}).dropna(subset=["mom"])
    # 流动性过滤
    if args.min_amount > 0:
        candidates = candidates[candidates["amount"] >= args.min_amount]

    # ST 股过滤（回测未排除，属于乐观偏差，实盘必须剔除）
    if not args.include_st:
        name_map = bi.set_index("stock_code")["名称"].astype(str)
        is_st = candidates.index.map(lambda c: "ST" in name_map.get(c, "") or "退" in name_map.get(c, ""))
        n_st = int(np.sum(is_st))
        candidates = candidates[~is_st]
        if n_st > 0:
            print(f"  已剔除 {n_st} 只 ST/退市风险股")

    if candidates.empty:
        print("  无满足流动性条件的股票")
        return

    n_pick = max(int(len(candidates) * args.top_pct), 3)
    if args.max_picks > 0:
        n_pick = min(n_pick, args.max_picks)
    top = candidates.sort_values("mom", ascending=False).head(n_pick).copy()

    # 标注是否涨停
    top["limit_up"] = top["ret_today"] >= 0.095
    top = top.reset_index().rename(columns={"index": "stock_code", "mom": "mom_20d"})
    top = top.merge(bi[["stock_code", "名称", "行业"]], on="stock_code", how="left")

    print(f"  候选池: {len(candidates)} 只 → 选出 {len(top)} 只\n")
    print(f"{'代码':<8}{'名称':<10}{'行业':<12}{'20日涨幅':>10}{'今日涨幅':>10}{'成交额(万)':>12}{'可买':>6}")
    for _, r in top.iterrows():
        buyable = "否(涨停)" if r["limit_up"] else "是"
        name = str(r.get("名称") or "")[:8]
        ind = str(r.get("行业") or "")[:10]
        print(f"{r['stock_code']:<8}{name:<10}{ind:<12}{r['mom_20d']*100:>9.1f}%{r['ret_today']*100:>9.2f}%{r['amount']:>12.0f}{buyable:>6}")

    n_limit = int(top["limit_up"].sum())
    if n_limit > 0:
        print(f"\n  [注意] {n_limit} 只今日涨停（买不进），实际可买 {len(top)-n_limit} 只")

    result["picks"] = [
        {
            "stock_code": r["stock_code"],
            "name": str(r.get("名称") or ""),
            "industry": str(r.get("行业") or ""),
            "momentum_20d": round(float(r["mom_20d"]), 6),
            "ret_today": round(float(r["ret_today"]), 6),
            "amount_wan": round(float(r["amount"]), 2),
            "buyable": not bool(r["limit_up"]),
        }
        for _, r in top.iterrows()
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "daily_momentum_signal.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("【提醒】")
    print("=" * 70)
    print("  1. 这是回测逻辑的实盘信号，未经真实交易验证，建议小仓位试")
    print("  2. 等权买入选出的股票，每 5 个交易日重新调仓")
    print("  3. 回测存在幸存者偏差，实际收益预期需打折")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
