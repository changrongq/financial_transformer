# -*- coding: utf-8 -*-
"""特征增量价值检验（知微 RL 复核第六节第三步）。

知微原话：
  「用简单正则化模型和固定规则作为基线，按时间划分训练、验证与最终测试，
    考虑标签/持有期跨边界造成的重叠。比较相同日期、相同费用及风险口径下的
    现金、持有和简单仓位规则。既然原择时策略已失去证据支持，
    不宜再将它当作已确认有效的骨架。」

所以在做任何新的 RL 之前，先回答一个更基础的问题：

    **我们手上这些特征，到底有没有样本外的增量信息？**

如果答案是没有，那么无论 RL 多复杂，都只是在拟合噪声。

================================================================
设计
================================================================

【标签】
  用未来的「多日累计收益」而不是单日收益。理由：
    · 单日收益信噪比约 0.05，几乎全是噪声
    · 持有期 H 天时，决策的实际收益就是 H 日累计
  测试多个 H ∈ {1, 5, 10, 20}，逐个看是否有信息。

【特征】
  只用 t 日收盘及之前可得的量（严格无未来函数）：
    · 趋势类：ret5 / ret10 / ret20 / ret60
    · 波动类：vol10 / vol60 / vol_ratio
    · 择时类：regime5 / regime10 / regime20（当前策略用的就是 regime10）
    · 状态类：down_run（连续下跌天数）/ drawdown_from_high

【模型】
  三个按复杂度递增的模型，全部带正则化：
    · Ridge（线性，最强正则）
    · Logistic / 线性分类（符号预测）
    · HistGradientBoosting（非线性，但限深、限轮数）
  如果**最简单**的模型都没有样本外信息，那更复杂的模型也不该先试。

【时间划分】
  train / valid / test 严格按时间切，不随机打乱：
    前 60% train，中间 20% valid，最后 20% test
  并在 test 上**只评估一次**。

【持有期重叠】
  H 日标签导致相邻样本的标签区间重叠。用两种方式处理：
    · 评估时按「每 H 天取一个样本」做非重叠子采样
    · 同时报告未去重叠的结果，两者对比

【基线】
  全部在同一时间区间、同一成本口径下比较：
    · 现金（恒定 0 仓位）
    · 买入持有（恒定 1 仓位）
    · regime10 规则（现有策略骨架）
    · 随机仓位（对照，验证评估流程本身没有偏差）

【评价指标】
  · 方向准确率（涨/跌预测对不对）
  · IC（预测值与实际收益的秩相关）
  · 多空分层收益（预测最高 20% vs 最低 20%）
  · 扣成本后的净收益与回撤（用 rl_ledger 统一账本）

用法：
  python scripts/test_feature_value.py
  python scripts/test_feature_value.py --code 510300.SH --horizons 1 5 10 20
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
from rl_ledger import run_ledger, ledger_stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# 数据
# ============================================================
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


def load_etf(db: str, code: str) -> pd.DataFrame:
    """加载 ETF，全部特征只用 t 日收盘及之前的信息。"""
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, pre_close "
        "FROM etf_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,))
    conn.close()
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close", "pre_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)].sort_values("date").reset_index(drop=True)

    df["pre_close"] = df["pre_close"].replace(0, np.nan)
    df["pre_close"] = df["pre_close"].fillna(df["close"].shift(1))
    df["ret"] = df["close"] / df["pre_close"] - 1.0

    # ---- 特征（全部 t 日可得）----
    # 用 pre_close 累计，跨拆股连续
    c = df["close"]
    for n in [5, 10, 20, 60]:
        df[f"ret{n}"] = c / df["pre_close"].shift(n - 1) - 1.0

    for n in [10, 60]:
        df[f"vol{n}"] = df["ret"].rolling(n).std()

    df["vol_ratio"] = df["vol10"] / df["vol60"].replace(0, np.nan)

    for n in [5, 10, 20]:
        df[f"regime{n}"] = df["ret"].rolling(n).mean()

    neg = (df["ret"] < 0).astype(int)
    run = neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1
    df["down_run"] = np.where(neg == 1, run, 0).astype(float)

    roll_max = c.rolling(120, min_periods=20).max()
    df["dd_from_high"] = c / roll_max - 1.0

    return df.replace([np.inf, -np.inf], np.nan)


FEATURES = ["ret5", "ret10", "ret20", "ret60",
            "vol10", "vol60", "vol_ratio",
            "regime5", "regime10", "regime20",
            "down_run", "dd_from_high"]


# ============================================================
# 标签：未来 H 日累计收益（t 日收盘 → t+H 日收盘）
# ============================================================
def make_label(df: pd.DataFrame, h: int) -> pd.Series:
    return df["close"].shift(-h) / df["close"] - 1.0


# ============================================================
# 评估
# ============================================================
def ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman 秩相关。常数输入返回 nan。"""
    if len(pred) < 10:
        return np.nan
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if np.nanstd(pred) < 1e-12 or np.nanstd(actual) < 1e-12:
        return np.nan
    import warnings
    from scipy.stats import spearmanr, ConstantInputWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            v = spearmanr(pred, actual).correlation
            return float(v) if np.isfinite(v) else np.nan
        except Exception:
            return np.nan


def evaluate_predictions(pred: np.ndarray, actual: np.ndarray,
                         *, top_q: float = 0.2) -> dict:
    """方向准确率 + IC + 分层收益。"""
    if len(pred) < 20:
        return {"n": len(pred), "hit": np.nan, "ic": np.nan,
                "top_mean": np.nan, "bottom_mean": np.nan, "spread": np.nan}

    # 方向：预测多头（pred>中位数）时实际是否为正
    thr = np.median(pred)
    long_mask = pred > thr
    hit = float(np.mean(actual[long_mask] > 0)) if long_mask.sum() > 0 else np.nan

    # 分层
    lo = np.quantile(pred, top_q)
    hi = np.quantile(pred, 1 - top_q)
    top_mean = float(actual[pred >= hi].mean()) if (pred >= hi).sum() > 0 else np.nan
    bot_mean = float(actual[pred <= lo].mean()) if (pred <= lo).sum() > 0 else np.nan

    return {
        "n": int(len(pred)),
        "hit": hit,
        "ic": ic(pred, actual),
        "top_mean": top_mean,
        "bottom_mean": bot_mean,
        "spread": (top_mean - bot_mean) if np.isfinite(top_mean) and np.isfinite(bot_mean) else np.nan,
    }


def run_model(name: str, Xtr, ytr, Xva, yva, Xte, yte,
              *, classification: bool = False) -> dict:
    """训练一个模型并返回 train/valid/test 指标。"""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    if classification:
        # 标签转成涨/跌
        m = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
        m.fit(Xtr_s, (ytr > 0).astype(int))
        pred_va = m.predict_proba(Xva_s)[:, 1] - 0.5
        pred_te = m.predict_proba(Xte_s)[:, 1] - 0.5
        pred_tr = m.predict_proba(Xtr_s)[:, 1] - 0.5
    elif name == "ridge":
        m = Ridge(alpha=10.0)
        m.fit(Xtr_s, ytr)
        pred_tr, pred_va, pred_te = m.predict(Xtr_s), m.predict(Xva_s), m.predict(Xte_s)
    elif name == "gbm":
        m = HistGradientBoostingRegressor(
            max_depth=3, max_iter=120, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=1.0, random_state=0)
        m.fit(Xtr_s, ytr)
        pred_tr = m.predict(Xtr_s)
        pred_va = m.predict(Xva_s)
        pred_te = m.predict(Xte_s)
    else:
        raise ValueError(name)

    return {
        "model": name,
        "train": evaluate_predictions(pred_tr, ytr),
        "valid": evaluate_predictions(pred_va, yva),
        "test": evaluate_predictions(pred_te, yte),
    }


# ============================================================
# 策略回测（同一账本，同一费用）
# ============================================================
def backtest_positions(df: pd.DataFrame, targets: np.ndarray,
                       mask: np.ndarray, cost: float) -> dict:
    """在 mask 指定的区间上用统一账本跑。

    targets[t] 是「t 日收盘决定的仓位」，在第 t+1 日开盘执行。
    mask 之外的 target 强制为 0 且不参与统计。
    """
    n = len(df)
    tgt = np.zeros(n)
    idx = np.where(mask)[0]
    # ★ 只把 mask 范围内的仓位写入（targets 是全长数组，取同样的下标）
    tgt[idx] = np.asarray(targets, dtype=float)[idx]

    sub_o = df["open"].values.astype(float)
    sub_c = df["close"].values.astype(float)
    sub_pc = df["pre_close"].values.astype(float)

    # 只在 mask 的第一个位置开始记账
    if len(idx) == 0:
        return {"total": 0.0, "dd": 0.0, "sharpe": np.nan, "trades": 0}

    s = int(idx[0])
    end = int(idx[-1]) + 1
    r = run_ledger(df["date"].values[s:end],
                   sub_o[s:end], sub_c[s:end], sub_pc[s:end],
                   tgt[s:end], cost=cost, slip=0.0, start_pos=0.0)
    st = ledger_stats(r.nav)
    st["trades"] = r.trades
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="510300.SH")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--cost", type=float, default=0.0003)
    ap.add_argument("--out", default="results/feature_value.json")
    args = ap.parse_args()

    db = resolve_db()
    df = load_etf(db, args.code)
    if df.empty:
        print(f"没有数据: {args.code}")
        return 1

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    n = len(df)
    print("=" * 78)
    print(f"特征增量价值检验   {args.code}")
    print(f"区间 {df['date'].min().date()} ~ {df['date'].max().date()}  共 {n} 天")
    print(f"特征 {len(FEATURES)} 个  成本 {args.cost*100:.3f}%/边")
    print("=" * 78)

    # 时间划分
    i_tr = int(n * 0.60)
    i_va = int(n * 0.80)
    print(f"train 0~{i_tr}   valid {i_tr}~{i_va}   test {i_va}~{n}")

    # ---- 基线策略（不依赖特征的固定规则）----
    print("\n" + "-" * 78)
    print("基线（同一区间、同一成本）")
    print("-" * 78)

    test_mask = np.zeros(n, dtype=bool)
    test_mask[i_va:n] = True

    baselines = {}
    # 现金
    baselines["现金"] = backtest_positions(df, np.zeros(n), test_mask, args.cost)
    # 买入持有
    baselines["买入持有"] = backtest_positions(df, np.ones(n), test_mask, args.cost)
    # regime10 规则
    rg = (df["regime10"] >= 0).astype(float).values
    baselines["regime10规则"] = backtest_positions(df, rg, test_mask, args.cost)

    for k, v in baselines.items():
        print(f"  {k:<14} 总收益 {v['total']*100:+8.1f}%   回撤 {v['dd']*100:7.1f}%   "
              f"夏普 {v['sharpe']:6.2f}   换手 {v['trades']:4d}")

    report = {"code": args.code, "n": n,
              "period": {"start": str(df["date"].min().date()),
                         "end": str(df["date"].max().date())},
              "split": {"train": i_tr, "valid": i_va, "test": n},
              "features": FEATURES,
              "baselines": baselines,
              "horizons": {}}

    # ---- 逐持有期检验 ----
    for h in args.horizons:
        print("\n" + "-" * 78)
        print(f"持有期 H = {h} 日")
        print("-" * 78)

        y = make_label(df, h)
        ok = y.notna().values
        idx_ok = np.where(ok)[0]
        tr = idx_ok[idx_ok < i_tr]
        va = idx_ok[(idx_ok >= i_tr) & (idx_ok < i_va)]
        te = idx_ok[idx_ok >= i_va]

        X = df[FEATURES].values.astype(np.float64)

        res_h = {"n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                 "models": {}, "overlap_check": {}}

        if len(te) < 50:
            print("  测试样本不足，跳过")
            report["horizons"][str(h)] = res_h
            continue

        # 三个模型
        for mname in ["ridge", "logistic", "gbm"]:
            cls = (mname == "logistic")
            r = run_model(mname, X[tr], y.values[tr], X[va], y.values[va],
                          X[te], y.values[te], classification=cls)
            res_h["models"][mname] = r
            t = r["test"]
            print(f"  {mname:<9} test: IC {t['ic']:+.4f}  方向命中 {t['hit']:.3f}  "
                  f"分层差 {t['spread']*100:+.2f}%")

        # ---- 持有期重叠处理：按 H 天非重叠采样 ----
        te_ns = te[::h]
        if len(te_ns) >= 30:
            for mname in ["ridge"]:
                r_ns = run_model(mname, X[tr], y.values[tr], X[va], y.values[va],
                                 X[te_ns], y.values[te_ns])
                res_h["overlap_check"][mname] = r_ns["test"]
                t = r_ns["test"]
                print(f"  [非重叠] {mname:<6} test: IC {t['ic']:+.4f}  "
                      f"方向命中 {t['hit']:.3f}  分层差 {t['spread']*100:+.2f}%  (n={t['n']})")

        # ---- 零假设检验：打乱标签后模型还能不能"预测" ----
        rng = np.random.RandomState(42)
        y_perm = y.values.copy()
        y_perm[tr] = y_perm[tr][rng.permutation(len(tr))]
        r_perm = run_model("ridge", X[tr], y_perm[tr], X[va], y_perm[va],
                           X[te], y_perm[te])
        res_h["permutation"] = r_perm["test"]
        print(f"  [打乱标签] ridge test: IC {r_perm['test']['ic']:+.4f}  "
              f"方向命中 {r_perm['test']['hit']:.3f}   ← 应在 0.5 附近")

        report["horizons"][str(h)] = res_h

    # ============================================================
    # 判定
    # ============================================================
    print("\n" + "=" * 78)
    print("判定")
    print("=" * 78)

    checks = []
    for h_str, rh in report["horizons"].items():
        if "models" not in rh or not rh["models"]:
            continue
        ridge_ic = rh["models"]["ridge"]["test"]["ic"]
        gbm_ic = rh["models"]["gbm"]["test"]["ic"]
        perm_ic = rh.get("permutation", {}).get("ic", np.nan)

        # 判据：真实 IC 显著大于打乱标签的 IC
        informative = (np.isfinite(ridge_ic) and np.isfinite(perm_ic)
                       and ridge_ic > perm_ic + 0.03)
        checks.append({
            "horizon": h_str,
            "ridge_ic": ridge_ic,
            "gbm_ic": gbm_ic,
            "perm_ic": perm_ic,
            "informative": bool(informative),
        })
        print(f"  H={h_str:>2}日  ridge IC={ridge_ic:+.4f}  "
              f"gbm IC={gbm_ic:+.4f}  打乱 IC={perm_ic:+.4f}  → "
              f"{'有信息' if informative else '无显著信息'}")

    any_info = any(c["informative"] for c in checks)
    print(f"\n  结论: {'至少一个持有期存在增量信息' if any_info else '所有持有期都无显著增量信息'}")

    report["checks"] = checks
    report["verdict"] = "有增量信息" if any_info else "无增量信息"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
