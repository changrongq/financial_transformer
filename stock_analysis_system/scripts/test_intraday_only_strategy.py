"""挖：既然"日内赚钱"，能不能直接"每天开盘买、收盘卖"？

发现：
  11/11 标的都是「隔夜负、日内正」
  沪深300指数 20.8年：隔夜 -14.93%，日内 +26.17%

诱人的想法：
  如果只在日内持仓（每日开盘买、收盘卖），不就能拿到日内收益？

但这里有个致命陷阱：
  每天买卖 2 次，一年 244 天 = 488 次交易
  即使单边万5，一年成本 = 488 × 0.05% = 24.4%
  → 日内收益 26.17% 几乎全被吃掉！

本脚本要回答：
  1. 纯"日内策略"（每日开盘买收盘卖）在不同成本下的表现？
  2. 成本临界点是多少？（多少成本以内能盈利）
  3. 相比"择时买入持有"，哪个更划算？

用法：
  python scripts/test_intraday_only_strategy.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
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


def load_prices(db: str, table: str, code: str) -> pd.DataFrame | None:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            f"SELECT trade_date, open, close FROM {table} WHERE ts_code=? ORDER BY trade_date",
            conn, params=(code,))
    except Exception:
        conn.close()
        return None
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 500:
        return None
    df["ret"] = df["close"].pct_change(fill_method=None)
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def metrics(eq: np.ndarray) -> dict:
    r = np.diff(np.concatenate([[1.0], eq]))
    total = float(eq[-1] - 1)
    years = len(eq) / 244
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe}


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    db = _resolve_db()
    targets = [
        ("510300.SH", "etf_daily", "沪深300ETF(14.4年)"),
        ("510050.SH", "etf_daily", "上证50ETF(20.5年)"),
        ("000300.SH", "index_daily", "沪深300指数(20.8年)"),
    ]

    print("=" * 96)
    print("「每天开盘买、收盘卖」策略测试 —— 检验日内收益能否被吃掉")
    print("=" * 96)

    all_out = {}
    for code, table, name in targets:
        df = load_prices(db, table, code)
        if df is None:
            print(f"\n{name}: 无数据")
            continue

        print(f"\n{'='*96}")
        print(f"{name}   {df['date'].min().date()} ~ {df['date'].max().date()}  {len(df)} 天")
        print(f"{'='*96}")

        intraday = df["intraday"].values
        ret = df["ret"].values

        print(f"\n  {'成本(单边)':<14}{'累计收益':>14}{'年化':>11}{'最大回撤':>12}{'夏普':>9}{'说明':>16}")
        print("  " + "-" * 76)

        rows = []
        for cost in [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]:
            # 每天开盘买、收盘卖：吃 intraday，每天付 2 次成本（买+卖）
            net = intraday - 2 * cost
            eq = np.cumprod(1 + net)
            m = metrics(eq)
            note = "理论值" if cost == 0 else ""
            if cost > 0 and m["ann"] < 0:
                note = "亏钱"
            print(f"  {cost*100:>6.2f}%       {m['total']*100:>+13.1f}%{m['ann']*100:>+10.2f}%"
                  f"{m['dd']*100:>11.1f}%{m['sharpe']:>9.2f}{note:>16}")
            rows.append({"cost_one_way": cost, **{k: round(v, 6) for k, v in m.items()}})

        # 成本临界点：让年化 = 0
        ann0 = metrics(np.cumprod(1 + intraday))["ann"]
        breakeven = ann0 / (2 * 244) if ann0 > 0 else 0
        print(f"\n  成本临界点（年化归零）: 单边 ≈ {breakeven*100:.3f}%  "
              f"（每日 {breakeven*200*100:.3f}%）")
        print(f"  ETF 实际成本约 0.03%~0.05%（含买卖价差+佣金），"
              f"{'低于临界点，理论可行' if breakeven > 0.0005 else '可能高于临界点'}")

        # 对比：买入持有
        eq_bh = np.cumprod(1 + ret)
        mb = metrics(eq_bh)
        print(f"\n  对比 买入持有: 累计 {mb['total']*100:+.1f}%  年化 {mb['ann']*100:+.2f}%  "
              f"回撤 {mb['dd']*100:.1f}%  夏普 {mb['sharpe']:.2f}")

        # 对比：择时 —— 用标准引擎的【严格 T+1】口径
        # （GUIDE 审查问题5：原实现用 close/open-1 只吃日内，与 v5.0 口径不一致）
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            import timing_engine as _TE

            _etf_df = _TE.load_etf(db, code) if table == "etf_daily" else None
            if _etf_df is not None:
                _strict = _TE.run_strict_t1(_etf_df, cost=0.0005, slip=0.0,
                                            window=10, threshold=0.0)
                _off = len(_etf_df) - len(_strict["curve"])
                _bh_curve = _TE.buy_hold_curve(_etf_df, _off)
                _m = min(len(_strict["curve"]), len(_bh_curve))
                _s_strict = _TE.stats(_strict["curve"][:_m])
                _s_bh = _TE.stats(_bh_curve[:_m])
                print(f"\n  对比 择时(严格T+1, 万5成本): 累计 {_s_strict['total']*100:+.1f}%  "
                      f"年化 {_s_strict['ann']*100:+.2f}%  回撤 {_s_strict['dd']*100:.1f}%  "
                      f"夏普 {_s_strict['sharpe']:.2f}")
                print(f"  对比 买入持有(严格T+1):      累计 {_s_bh['total']*100:+.1f}%  "
                      f"年化 {_s_bh['ann']*100:+.2f}%  回撤 {_s_bh['dd']*100:.1f}%  "
                      f"夏普 {_s_bh['sharpe']:.2f}")
                print(f"       （严格 T+1：T日信号→T+1开盘执行→持仓吃全天收益，含隔夜）")
                mt = _s_strict
                mb = _s_bh
            else:
                raise RuntimeError("非 ETF 标的，无严格T+1对照")
        except Exception as _e:
            print(f"\n  对比 择时: 无法用标准引擎计算（{type(_e).__name__}: {_e}）")
            mt = {"total": np.nan, "ann": np.nan, "dd": np.nan, "sharpe": np.nan}

        all_out[name] = {
            "days": len(df), "intraday_ann": round(ann0, 6),
            "breakeven_cost": round(breakeven, 6),
            "grid": rows,
            "buy_hold": {k: (round(v, 6) if isinstance(v, (int, float)) and not pd.isna(v) else None)
                         for k, v in mb.items()},
            "timing_strict_t1": {k: (round(v, 6) if isinstance(v, (int, float)) and not pd.isna(v) else None)
                                 for k, v in mt.items()},
        }

    # ===== 总结 =====
    print("\n" + "=" * 96)
    print("总结")
    print("=" * 96)
    print("  1. 「每天日内交易」理论上能吃到大额日内收益，但每日2次成本极高")
    print("  2. 成本临界点约 0.03%~0.08% 单边（ETF 实际成本 0.03-0.05%，正好卡在死亡线）")
    print("  3. 更优方案：择时（T日信号→T+1开盘执行，吃全天收益）")
    print("     交易次数从 488次/年 降到约 34次/年，成本降低 14 倍")
    print()
    print("  ⚠️ 口径说明（GUIDE 审查问题5）：")
    print("     本脚本主结论（日内交易成本太高）用『每天开盘买收盘卖』口径，")
    print("     但『对比择时』一格已改用标准引擎的【严格 T+1】口径，")
    print("     与 validate_timing_multi_v2.py 的最终数字一致。")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "intraday_only_strategy.json"
    out.write_text(json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
