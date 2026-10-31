# -*- coding: utf-8 -*-
"""扩展特征集检验：量价 + 换手 + 市场宽度（知微第三步的延伸）。

第一批检验（`test_feature_value_strict.py`）用了 12 个**纯价格**特征，
结论是「无可靠增量信息」。

本脚本换的是**特征种类**，不是模型：
  · 量价：成交额相对水平、量价相关性、量能变化
  · 换手：真实自由流通换手率（ETF 用 vol 近似）
  · 宽度：全市场涨跌家数比、涨停占比（需要 stock_data）
  · 波动结构：高低价振幅、日内位置、隔夜/日内分解

严格沿用第一批的方法：
  · Ridge(alpha=10) + 置换检验 + 分块 bootstrap + 滚动窗口
  · 6 只标的独立
  · 时间切分 60/20/20
  · 三条判据全满足才算「有信息」

用法：
  python scripts/test_feature_value_extended.py
  python scripts/test_feature_value_extended.py --perms 100 --horizons 5 10 20
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_feature_value_strict import (CODES, NAMES, spearman,
                                       permutation_null, block_bootstrap_ci,
                                       rolling_ic)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def resolve_db() -> str:
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            with cfg.open("r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            raw = c.get("databases", {}).get("trade_db")
            if raw:
                p = Path(raw)
                if not p.is_absolute():
                    p = ROOT / p
                if p.exists():
                    return str(p)
        except Exception:
            pass
    return str((ROOT.parent / "stock_history.db").resolve())


# ============================================================
# 市场宽度（全市场，非单标的）
# ============================================================
def load_market_breadth(db: str) -> pd.DataFrame:
    """从 stock_data 算每日市场宽度指标。

    ⚠️ stock_data 只有 2024-10 起的数据，所以宽度特征只能覆盖近 2 年。
       这意味着用宽度特征的那一组样本会大幅缩短，检验力度降低。
    """
    conn = sqlite3.connect(db)
    try:
        sql = """
        SELECT 日期 AS date, stock_code,
               收盘, 开盘, 成交额, 成交量
        FROM stock_data
        WHERE 收盘 IS NOT NULL AND 收盘 > 0
        """
        df = pd.read_sql_query(sql, conn)
    except Exception:
        conn.close()
        return pd.DataFrame()
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["收盘", "开盘", "成交额", "成交量"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["stock_code", "date"])
    df["ret"] = df.groupby("stock_code")["收盘"].pct_change(fill_method=None)

    g = df.groupby("date")
    out = pd.DataFrame({
        "breadth_up": g["ret"].apply(lambda s: float((s > 0).mean())),
        "breadth_disp": g["ret"].std(),
        "breadth_amt": g["成交额"].sum(),
        "breadth_n": g["ret"].count(),
    }).reset_index()

    # 涨停近似：涨幅 > 9.8%（不区分 ST/科创）
    def limit_up_ratio(s):
        return float((s > 0.098).mean())

    lu = g["ret"].apply(limit_up_ratio).rename("breadth_limitup").reset_index()
    out = out.merge(lu, on="date", how="left")

    # 量价配合
    amt = out["breadth_amt"]
    out["breadth_amt_ratio"] = amt / amt.rolling(20).mean()
    out["breadth_amt_chg"] = amt.pct_change(5)

    return out


# ============================================================
# 扩展特征
# ============================================================
def load_etf_ext(db: str, code: str, breadth: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, pre_close, vol, amount "
        "FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)

    df["pre_close"] = df["pre_close"].replace(0, np.nan)
    df["pre_close"] = df["pre_close"].fillna(df["close"].shift(1))
    df["ret"] = df["close"] / df["pre_close"] - 1.0
    df["gap"] = df["open"] / df["pre_close"] - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0

    # ---------- 价格类（对照用，few 个）----------
    c = df["close"]
    for n in [5, 20]:
        df[f"ret{n}"] = c / df["pre_close"].shift(n - 1) - 1.0
    df["vol20"] = df["ret"].rolling(20).std()
    df["regime10"] = df["ret"].rolling(10).mean()

    # ---------- 量价类（新增）----------
    amt = df["amount"].replace(0, np.nan)
    vol = df["vol"].replace(0, np.nan)

    # 成交额相对 20/60 日均值
    df["amt_ratio20"] = amt / amt.rolling(20).mean()
    df["amt_ratio60"] = amt / amt.rolling(60).mean()
    # 成交额变化率
    df["amt_chg5"] = amt.pct_change(5)
    df["amt_chg20"] = amt.pct_change(20)
    # 量能趋势：短期均量 / 长期均量
    df["vol_ma5_20"] = vol.rolling(5).mean() / vol.rolling(20).mean()
    # 量价相关性（20日）：涨时放量还是缩量
    df["pv_corr20"] = df["ret"].rolling(20).corr(
        np.log(amt).diff())
    # 换手近似：成交量 / 20日均量
    df["turnover_proxy"] = vol / vol.rolling(20).mean()
    # 量价背离：价格新高但量没跟上
    price_hi = c >= c.rolling(20).max()
    vol_hi = vol >= vol.rolling(20).max() * 0.9
    df["pv_diverge"] = ((price_hi) & (~vol_hi)).astype(float)

    # ---------- 波动结构（新增）----------
    df["hl_range"] = (df["high"] - df["low"]) / df["pre_close"]
    df["close_pos"] = (df["close"] - df["low"]) / (
        (df["high"] - df["low"]).replace(0, np.nan))
    df["gap_abs"] = df["gap"].abs()
    df["intra_strength"] = df["intraday"]
    df["range_ratio"] = df["hl_range"] / df["hl_range"].rolling(20).mean()

    # ---------- 市场宽度（合并，仅近期可用）----------
    if not breadth.empty:
        df = df.merge(breadth, on="date", how="left")

    return df.replace([np.inf, -np.inf], np.nan)


PRICE_FEATS = ["ret5", "ret20", "vol20", "regime10"]
PV_FEATS = ["amt_ratio20", "amt_ratio60", "amt_chg5", "amt_chg20",
            "vol_ma5_20", "pv_corr20", "turnover_proxy", "pv_diverge"]
VOLSTRUCT_FEATS = ["hl_range", "close_pos", "gap_abs", "intra_strength",
                   "range_ratio"]
BREADTH_FEATS = ["breadth_up", "breadth_disp", "breadth_amt_ratio",
                 "breadth_amt_chg", "breadth_limitup"]


def make_label(df: pd.DataFrame, h: int) -> np.ndarray:
    return (df["close"].shift(-h) / df["close"] - 1.0).values.astype(np.float64)


def fit_ic(X, y, tr, te) -> float:
    if len(tr) < 200 or len(te) < 80:
        return np.nan
    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=10.0)
    m.fit(sc.transform(X[tr]), y[tr])
    return spearman(m.predict(sc.transform(X[te])), y[te])


def permutation_p(X, y, tr, te, n_perm: int, seed: int = 0) -> dict:
    if len(tr) < 200 or len(te) < 80:
        return {"p": np.nan, "null_std": np.nan}
    rng = np.random.RandomState(seed)
    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr])
    Xte = sc.transform(X[te])
    real = spearman(Ridge(alpha=10.0).fit(Xtr, y[tr]).predict(Xte), y[te])
    null = []
    for _ in range(n_perm):
        yp = y[tr].copy()
        rng.shuffle(yp)
        v = spearman(Ridge(alpha=10.0).fit(Xtr, yp).predict(Xte), y[te])
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)
    if len(null) == 0 or not np.isfinite(real):
        return {"p": np.nan, "null_std": np.nan, "ic": real}
    p = float((np.abs(null) >= abs(real)).mean())
    return {"p": p, "null_std": float(null.std()), "ic": real,
            "pctile": float((np.abs(null) < abs(real)).mean() * 100)}


def analyze_set(name: str, feats: list, df: pd.DataFrame,
                horizons: list, n_perm: int) -> dict:
    """对一组特征做完整检验。"""
    sub = df.dropna(subset=feats).reset_index(drop=True)
    n = len(sub)
    out = {"feature_set": name, "n_feats": len(feats), "n_rows": n,
           "features": feats, "horizons": {}}
    if n < 600:
        out["error"] = f"too few rows {n}"
        return out

    X = sub[feats].values.astype(np.float64)
    i_tr, i_va = int(n * 0.60), int(n * 0.80)

    for h in horizons:
        y = make_label(sub, h)
        ok = np.isfinite(y)
        idx = np.where(ok)[0]
        tr = idx[idx < i_tr]
        te = idx[idx >= i_va]
        if len(tr) < 200 or len(te) < 80:
            out["horizons"][str(h)] = {"error": "sample too small"}
            continue

        d = permutation_p(X, y, tr, te, n_perm)
        sc = StandardScaler().fit(X[tr])
        pred = Ridge(alpha=10.0).fit(
            sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        lo, hi = block_bootstrap_ci(pred, y[te])
        roll = rolling_ic(X, y, n_splits=5)
        same = (sum(1 for r in roll if np.isfinite(r["ic"])
                    and np.sign(r["ic"]) == np.sign(d["ic"]))
                if np.isfinite(d.get("ic", np.nan)) else 0)
        tot = sum(1 for r in roll if np.isfinite(r["ic"]))

        out["horizons"][str(h)] = {
            "ic": d.get("ic", np.nan),
            "p_value": d.get("p", np.nan),
            "null_std": d.get("null_std", np.nan),
            "ic_ci95": [lo, hi],
            "n_test": int(len(te)),
            "rolling_same_sign": same,
            "rolling_total": tot,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--out", default="results/feature_value_extended.json")
    args = ap.parse_args()

    db = resolve_db()
    print("=" * 90)
    print("扩展特征集检验：量价 / 波动结构 / 市场宽度")
    print(f"标的 {len(CODES)} 只   持有期 {args.horizons}   置换 {args.perms}")
    print("=" * 90)

    print("\n加载市场宽度...")
    breadth = load_market_breadth(db)
    if breadth.empty:
        print("  市场宽度不可用（stock_data 为空或列名不符）")
    else:
        print(f"  宽度数据 {len(breadth)} 天："
              f"{breadth['date'].min().date()} ~ {breadth['date'].max().date()}")

    FEATURE_SETS = {
        "仅价格(基线)": PRICE_FEATS,
        "价格+量价": PRICE_FEATS + PV_FEATS,
        "价格+量价+波动结构": PRICE_FEATS + PV_FEATS + VOLSTRUCT_FEATS,
    }
    if not breadth.empty:
        FEATURE_SETS["价格+量价+波动+宽度"] = (PRICE_FEATS + PV_FEATS +
                                              VOLSTRUCT_FEATS + BREADTH_FEATS)

    report = {"codes": CODES, "horizons": args.horizons,
              "n_perm": args.perms,
              "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
              "breadth_available": not breadth.empty,
              "results": {}}

    for code in CODES:
        print(f"\n{'='*90}")
        print(f"{code} {NAMES.get(code, '')}")
        print("=" * 90)
        df = load_etf_ext(db, code, breadth)
        if df.empty:
            print("  无数据")
            continue
        print(f"  区间 {df['date'].min().date()} ~ {df['date'].max().date()} "
              f"原始 {len(df)} 天")

        code_res = {}
        for sname, feats in FEATURE_SETS.items():
            r = analyze_set(sname, feats, df, args.horizons, args.perms)
            code_res[sname] = r
            if "error" in r:
                print(f"  {sname:<22} {r['error']}")
                continue
            parts = []
            for h in args.horizons:
                d = r["horizons"].get(str(h), {})
                if "error" in d:
                    continue
                parts.append(f"H{h}: IC={d['ic']:+.3f} p={d['p_value']:.2f}")
            print(f"  {sname:<22} n={r['n_rows']:>5}  " + "  ".join(parts))
        report["results"][code] = code_res

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 90)
    print("汇总：各特征集在 6 标的上的表现")
    print("=" * 90)

    summary = {}
    for sname in FEATURE_SETS:
        per_h = {}
        for h in args.horizons:
            ics, ps, rolls = [], [], []
            for code in CODES:
                d = report["results"].get(code, {}).get(sname, {}).get(
                    "horizons", {}).get(str(h), {})
                if "ic" not in d or not np.isfinite(d["ic"]):
                    continue
                ics.append(d["ic"])
                ps.append(d["p_value"])
                rolls.append((d["rolling_same_sign"], d["rolling_total"]))

            if not ics:
                continue
            ics = np.array(ics)
            ps = np.array(ps)
            n_sig = int((ps < 0.05).sum())
            n_pos = int((ics > 0).sum())
            n_neg = int((ics < 0).sum())
            stable = sum(1 for s, t in rolls if t > 0 and s / t >= 0.6)
            informative = (n_sig >= max(3, len(ics) // 2)
                           and max(n_pos, n_neg) >= max(4, int(len(ics) * 0.75))
                           and stable >= max(3, len(ics) // 2))
            per_h[str(h)] = {
                "median_ic": float(np.median(ics)),
                "n_significant": n_sig, "n_total": len(ics),
                "n_pos": n_pos, "n_neg": n_neg,
                "n_stable": stable,
                "informative": bool(informative),
            }
        summary[sname] = per_h

    print(f"\n{'特征集':<24}{'持有期':<8}{'中位IC':>9}{'显著':>7}{'正/负':>9}{'稳定':>7}  判定")
    print("-" * 90)
    for sname, per_h in summary.items():
        for h, d in sorted(per_h.items(), key=lambda x: int(x[0])):
            verdict = "有信息 <<<" if d["informative"] else "无"
            print(f"{sname:<24}H={h:<6}{d['median_ic']:>+9.3f}"
                  f"{d['n_significant']:>4}/{d['n_total']}"
                  f"{d['n_pos']:>4}/{d['n_neg']:<4}"
                  f"{d['n_stable']:>4}/{d['n_total']:<3}  {verdict}")

    any_info = any(d["informative"]
                   for per_h in summary.values() for d in per_h.values())
    report["summary"] = summary
    report["verdict"] = "有增量信息" if any_info else "无增量信息"

    print("\n" + "=" * 90)
    print(f"总体结论: {report['verdict']}")
    print("=" * 90)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
