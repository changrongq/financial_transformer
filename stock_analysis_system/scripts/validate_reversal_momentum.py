"""验证「反向动量」假设：买动量最弱的股票是否有效。

依据：
  正向动量组跑赢全市场只有 37%（即 63% 的时候跑输），
  那么反过来买"动量最弱"的，理论上应该跑赢 63%。

风险（必须同时检验）：
  反转策略"赚小钱亏大钱"——跌多了可能继续跌到退市/暴雷。
  所以除了平均收益，必须看：
    1. 尾部风险（最差 5% 的调仓期）
    2. 是否依赖少数极端反弹

用法：
  python scripts/validate_reversal_momentum.py
  python scripts/validate_reversal_momentum.py --hold-days 5 --reverse-pct 0.10
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


def load_data():
    conn = sqlite3.connect(_resolve_db())
    price = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [收盘] AS close, "
        "[成交额] AS amount, [成交量] AS volume FROM stock_data", conn
    )
    bi = pd.read_sql_query("SELECT stock_code, 名称 FROM stock_basic_info", conn)
    conn.close()

    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price["stock_code"] = price["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    for c in ["close", "amount", "volume"]:
        price[c] = pd.to_numeric(price[c], errors="coerce")
    price = price.dropna(subset=["close"])
    price = price[price["stock_code"].str.len() == 6]
    price = price.sort_values(["stock_code", "date"]).reset_index(drop=True)

    miss = int(price["amount"].isna().sum())
    if miss:
        price["amount"] = price["amount"].fillna(price["close"] * price["volume"])
        print(f"  [数据修正] 成交额缺失 {miss:,} 行，用 收盘价×成交量 估算")
    price["amount"] = price["amount"].fillna(0.0)

    bi["stock_code"] = bi["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    return price, bi


def main() -> None:
    parser = argparse.ArgumentParser(description="验证反向动量（买最弱）")
    parser.add_argument("--mom-window", type=int, default=20)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--sel-pct", type=float, default=0.10, help="选多少比例")
    parser.add_argument("--min-amount", type=float, default=10000)
    args = parser.parse_args()

    price, bi = load_data()
    print(f"数据范围: {price['date'].min().date()} ~ {price['date'].max().date()}")

    pivot = price.pivot_table(index="date", columns="stock_code", values="close", aggfunc="first").sort_index()
    amt = price.pivot_table(index="date", columns="stock_code", values="amount", aggfunc="first").reindex(index=pivot.index)
    mom = pivot / pivot.shift(args.mom_window) - 1.0

    st = set(bi.loc[bi["名称"].astype(str).str.contains("ST|退", na=False), "stock_code"])
    valid = [c for c in pivot.columns if c not in st]
    dates = pivot.index

    rows = []
    for i in range(args.mom_window + 1, len(dates) - args.hold_days, args.hold_days):
        d, d2 = dates[i], dates[i + args.hold_days]
        m = mom.loc[d, valid].dropna()
        a = amt.loc[d].reindex(m.index)
        m = m[a >= args.min_amount]
        if len(m) < 50:
            continue

        n_sel = max(int(len(m) * args.sel_pct), 5)
        srt = m.sort_values(ascending=False)
        strong = srt.head(n_sel).index       # 动量最强
        weak = srt.tail(n_sel).index         # 动量最弱

        fwd_all = (pivot.loc[d2, valid] / pivot.loc[d, valid] - 1.0)
        mkt = float(fwd_all.mean())

        f_strong = fwd_all.reindex(strong).dropna()
        f_weak = fwd_all.reindex(weak).dropna()

        rows.append({
            "date": d,
            "mkt": mkt,
            "strong_ret": float(f_strong.mean()) if len(f_strong) else np.nan,
            "weak_ret": float(f_weak.mean()) if len(f_weak) else np.nan,
            "weak_mom": float(m[weak].mean()),
            "strong_mom": float(m[strong].mean()),
        })

    df = pd.DataFrame(rows).dropna()
    print(f"调仓次数: {len(df)}")

    print("\n" + "=" * 80)
    print(f"{'方案':<20}{'平均5日收益':>14}{'超额(减市场)':>15}{'跑赢率':>10}{'最差5%':>12}{'最好5%':>12}")
    print("=" * 80)

    out = {"params": vars(args), "n_periods": len(df)}
    for name, col in [("市场平均", None), ("动量最强(正向)", "strong_ret"), ("动量最弱(反向)", "weak_ret")]:
        if col is None:
            print(f"{name:<20}{df['mkt'].mean()*100:>+13.2f}%{'—':>15}{'—':>10}"
                  f"{df['mkt'].quantile(0.05)*100:>+11.2f}%{df['mkt'].quantile(0.95)*100:>+11.2f}%")
            continue
        excess = df[col] - df["mkt"]
        win = float((excess > 0).mean())
        print(
            f"{name:<20}{df[col].mean()*100:>+13.2f}%{excess.mean()*100:>+14.2f}%"
            f"{win*100:>9.1f}%{excess.quantile(0.05)*100:>+11.2f}%{excess.quantile(0.95)*100:>+11.2f}%"
        )
        out[name] = {
            "avg_ret": round(float(df[col].mean()), 6),
            "excess": round(float(excess.mean()), 6),
            "win_rate": round(win, 4),
            "worst_5pct": round(float(excess.quantile(0.05)), 6),
            "best_5pct": round(float(excess.quantile(0.95)), 6),
        }

    print("=" * 80)

    strong_ex = out["动量最强(正向)"]["excess"]
    weak_ex = out["动量最弱(反向)"]["excess"]
    print("\n判断：")
    print(f"  正向动量超额: {strong_ex*100:+.2f}%  {'无效' if strong_ex < 0 else '有效'}")
    print(f"  反向动量超额: {weak_ex*100:+.2f}%  {'有效' if weak_ex > 0 else '无效'}")

    if weak_ex > 0:
        print(f"\n  反向有效！但注意尾部风险：")
        print(f"    最差5%的调仓期超额: {out['动量最弱(反向)']['worst_5pct']*100:+.2f}%（赚小钱亏大钱的迹象）")

    # 相关性
    corr = df["strong_mom"].corr(df["weak_ret"] - df["mkt"])
    print(f"\n  动量强度 vs 反向超额的相关性: {corr:+.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "reversal_momentum_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入: {RESULTS_DIR / 'reversal_momentum_validation.json'}")


if __name__ == "__main__":
    main()
