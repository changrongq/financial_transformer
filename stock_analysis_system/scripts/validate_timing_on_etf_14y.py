"""用 14 年真实 ETF 数据验证择时规则（消除 2 年小样本偏差）。

背景：
  之前所有择时结论都基于 stock_data（全市场等权，478天，2024-10~2026-10）。
  现在有了 510300 沪深300ETF 的 14.4 年数据（3503天），可以真正验证。

关键设计（吸取未来函数教训）：
  所有信号都用 T 日收盘价计算，T+1 日开盘执行。

测试内容：
  1. 不同窗口（5/10/20/30/60日）× 不同阈值（0/-0.1%/-0.2%/+0.1%）
  2. 每种组合都对比：买入持有 vs 择时（次日开盘）
  3. 分年度一致性（这是最关键的检验）
  4. 全区间 + 多个子区间

用法：
  python scripts/validate_timing_on_etf_14y.py
  python scripts/validate_timing_on_etf_14y.py --code 510300.SH --cost 0.0005
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


def load_etf(db: str, code: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, vol, amount FROM etf_daily WHERE ts_code = ? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    if df.empty:
        raise SystemExit(f"{code} 无数据，先跑 fetch_etf_history.py")
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close", "open"]).sort_values("date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change(fill_method=None)
    # 隔夜跳空（昨收 -> 今开）
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    # 日内（今开 -> 今收）
    df["intraday"] = df["close"] / df["open"] - 1.0
    return df


def stats(eq: np.ndarray) -> dict:
    r = np.diff(np.concatenate([[1.0], eq]))
    total = float(eq[-1] - 1)
    years = len(eq) / 244
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan
    calmar = float(ann / abs(dd)) if dd < 0 else np.nan
    return {"total": total, "ann": ann, "dd": dd, "sharpe": sharpe, "calmar": calmar, "days": len(eq)}


def run_timing(df: pd.DataFrame, window: int, threshold: float, cost: float,
               exec_mode: str = "next_open") -> tuple[np.ndarray, np.ndarray]:
    """返回 (择时净值, 买入持有净值)。

    exec_mode:
      - "next_open"  : T日收盘算信号 → T+1 开盘买/卖，T+1 吃 开盘→收盘 的收益
                       （最真实：开盘成交，当天只吃日内部分）
      - "next_close" : T日收盘算信号 → T+1 收盘成交，T+2 才持仓
      - "same_close" : T日收盘算信号 → T日收盘成交（未来函数，仅作对照）
    """
    d = df.copy()
    d["mean"] = d["ret"].rolling(window, min_periods=3).mean()
    d["signal"] = d["mean"] >= threshold
    d = d.dropna(subset=["mean"]).reset_index(drop=True)

    n = len(d)
    eq_t, eq_b = [1.0], [1.0]
    pos = 0.0
    prev_held = False

    for i in range(n):
        if i == 0:
            eq_t.append(1.0)
            eq_b.append(1.0)
            continue

        if exec_mode == "same_close":
            # 未来函数：今天收盘信号决定今天持仓，吃今天全天收益
            target = 1.0 if d.loc[i, "signal"] else 0.0
            r = float(d.loc[i, "ret"]) if not np.isnan(d.loc[i, "ret"]) else 0.0
            trade = abs(target - pos)
            eq_t.append(eq_t[-1] * (1 + target * r - trade * cost))
            pos = target
        elif exec_mode == "next_open":
            # T-1 收盘信号决定今天开盘的动作
            held = bool(d.loc[i - 1, "signal"])
            target = 1.0 if held else 0.0
            trade = abs(target - pos)
            # 今天开盘成交后，吃「开盘→收盘」收益（若持有）
            r_intraday = float(d.loc[i, "intraday"]) if not np.isnan(d.loc[i, "intraday"]) else 0.0
            # 若从空仓变持仓：开盘买入，吃日内；若从持仓变空仓：开盘卖出，日内不参与
            eq_t.append(eq_t[-1] * (1 + target * r_intraday - trade * cost))
            pos = target
        elif exec_mode == "next_close":
            # T-2 收盘信号决定今天持仓（最保守）
            held = bool(d.loc[i - 2, "signal"]) if i >= 2 else False
            target = 1.0 if held else 0.0
            trade = abs(target - pos)
            r = float(d.loc[i, "ret"]) if not np.isnan(d.loc[i, "ret"]) else 0.0
            eq_t.append(eq_t[-1] * (1 + target * r - trade * cost))
            pos = target

        r_all = float(d.loc[i, "ret"]) if not np.isnan(d.loc[i, "ret"]) else 0.0
        eq_b.append(eq_b[-1] * (1 + r_all))

    return np.array(eq_t[1:]), np.array(eq_b[1:]), d


def main() -> None:
    parser = argparse.ArgumentParser(description="14年 ETF 择时验证")
    parser.add_argument("--code", default="510300.SH")
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument("--start", default=None)
    args = parser.parse_args()

    df = load_etf(_resolve_db(), args.code)
    if args.start:
        df = df[df["date"] >= args.start].reset_index(drop=True)
    print(f"标的: {args.code}  {df['date'].min().date()} ~ {df['date'].max().date()}  "
          f"（{len(df)} 天 ≈ {len(df)/244:.1f} 年）")
    print(f"单边成本: {args.cost*100:.3f}%\n")

    windows = [5, 10, 20, 30, 60]
    thresholds = [0.001, 0.0, -0.001, -0.002]

    print("=" * 104)
    print(f"{'窗口':>5}{'阈值':>8} | {'买入持有':>10}{'择时(次日开盘)':>15}{'优势':>11}{'回撤(择时/买入)':>18}{'夏普(择时/买入)':>18}")
    print("=" * 104)

    rows = []
    for w in windows:
        for th in thresholds:
            eq_t, eq_b, d = run_timing(df, w, th, args.cost, "next_open")
            st, sb = stats(eq_t), stats(eq_b)
            adv = st["total"] - sb["total"]
            flag = "WIN " if adv > 0 else "LOSE"
            print(f"{w:>5}{th*100:>7.1f}% | {sb['total']*100:>+9.1f}%{st['total']*100:>+14.1f}%"
                  f"{flag}{adv*100:>+7.1f}pp"
                  f"{st['dd']*100:>9.1f}%/{sb['dd']*100:>6.1f}%"
                  f"{st['sharpe']:>9.2f}/{sb['sharpe']:>6.2f}")
            rows.append({
                "window": w, "threshold": th,
                "bh_total": sb["total"], "tim_total": st["total"], "adv": adv,
                "tim_dd": st["dd"], "bh_dd": sb["dd"],
                "tim_sharpe": st["sharpe"], "bh_sharpe": sb["sharpe"],
                "tim_ann": st["ann"], "bh_ann": sb["ann"],
            })
    print("=" * 104)

    rdf = pd.DataFrame(rows)
    n_win = int((rdf["adv"] > 0).sum())
    print(f"\n跑赢买入持有: {n_win}/{len(rdf)} 组")
    print(f"优势中位数: {rdf['adv'].median()*100:+.1f}pp   平均: {rdf['adv'].mean()*100:+.1f}pp")
    print(f"回撤改善中位数: {(rdf['tim_dd'] - rdf['bh_dd']).median()*100:+.1f}pp")
    print(f"夏普改善中位数: {(rdf['tim_sharpe'] - rdf['bh_sharpe']).median():+.2f}")

    # ===== 默认参数分年度 =====
    print("\n" + "=" * 104)
    print("默认参数（窗口10, 阈值0）分年度一致性")
    print("=" * 104)
    eq_t, eq_b, d = run_timing(df, 10, 0.0, args.cost, "next_open")
    yr = d["date"].dt.year.values
    print(f"{'年份':<8}{'买入持有':>12}{'择时':>12}{'差额':>12}{'当年是否赢':>12}")
    year_rows = []
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() < 20:
            continue
        bt = eq_b[m][-1] / (eq_b[m][0] / (1 + 0)) if m.sum() else 1
        # 用当年首尾比（相对前一年末）
        idx = np.where(m)[0]
        prev_b = eq_b[idx[0] - 1] if idx[0] > 0 else 1.0
        prev_t = eq_t[idx[0] - 1] if idx[0] > 0 else 1.0
        rb = eq_b[idx[-1]] / prev_b - 1
        rt = eq_t[idx[-1]] / prev_t - 1
        win = "WIN" if rt > rb else "LOSE"
        print(f"{y:<8}{rb*100:>+11.1f}%{rt*100:>+11.1f}%{(rt-rb)*100:>+11.1f}pp{win:>12}")
        year_rows.append({"year": int(y), "bh": float(rb), "tim": float(rt), "adv": float(rt - rb)})

    yd = pd.DataFrame(year_rows)
    if not yd.empty:
        print(f"\n逐年胜率: {int((yd['adv']>0).sum())}/{len(yd)} 年")

    # ===== 各执行方式对比 =====
    print("\n" + "=" * 104)
    print("执行方式对比（窗口10, 阈值0）")
    print("=" * 104)
    print(f"{'执行方式':<24}{'累计收益':>13}{'年化':>10}{'最大回撤':>12}{'夏普':>9}")
    exec_rows = []
    for mode, label in [("same_close", "T日收盘成交（未来函数）"),
                        ("next_open", "T+1开盘成交（真实）"),
                        ("next_close", "T+1收盘成交（保守）")]:
        e, b, _ = run_timing(df, 10, 0.0, args.cost, mode)
        s = stats(e)
        print(f"{label:<24}{s['total']*100:>+12.1f}%{s['ann']*100:>+9.1f}%"
              f"{s['dd']*100:>11.1f}%{s['sharpe']:>9.2f}")
        exec_rows.append({"mode": mode, **{k: round(v, 6) for k, v in s.items()}})
    s = stats(eq_b)
    print(f"{'买入持有':<24}{s['total']*100:>+12.1f}%{s['ann']*100:>+9.1f}%"
          f"{s['dd']*100:>11.1f}%{s['sharpe']:>9.2f}")

    # ===== 子区间稳健性 =====
    print("\n" + "=" * 104)
    print("子区间检验（窗口10, 阈值0, 次日开盘）")
    print("=" * 104)
    print(f"{'区间':<26}{'买入持有':>12}{'择时':>12}{'差额':>12}")
    splits = [("2012-05", "2016-12"), ("2017-01", "2020-12"), ("2021-01", "2026-10")]
    sub_rows = []
    for a, b in splits:
        sub = df[(df["date"] >= a) & (df["date"] <= b)].reset_index(drop=True)
        if len(sub) < 100:
            continue
        e, bb, _ = run_timing(sub, 10, 0.0, args.cost, "next_open")
        se, sb2 = stats(e), stats(bb)
        print(f"{a} ~ {b:<14}{sb2['total']*100:>+11.1f}%{se['total']*100:>+11.1f}%"
              f"{(se['total']-sb2['total'])*100:>+11.1f}pp")
        sub_rows.append({"start": a, "end": b, "bh": sb2["total"], "tim": se["total"],
                         "adv": se["total"] - sb2["total"]})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"timing_etf14y_{args.code.replace('.', '_')}.json"
    out.write_text(json.dumps({
        "params": vars(args),
        "period": {"start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
                   "days": len(df)},
        "grid": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        "n_win": n_win, "n_total": len(rows),
        "yearly": year_rows,
        "exec_modes": exec_rows,
        "sub_periods": sub_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
