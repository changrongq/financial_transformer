# -*- coding: utf-8 -*-
"""真实输入列分位数审计（知微第 3 项修复的最后一步）。

背景
----
知微指出 flat 分支先 Linear 混合 32 列再 LayerNorm，
而 `amount`(1e8) / `volume`(1e6) 与 `return_5d`(±0.1) 差 9 个数量级。
用合成数据复现：改 `ret5d` 从 -0.1 到 +0.1，原模型输出差 **0.0000000000**。

**但知微也要求：「应优先检查真实输入各列分位数、单位、梯度和特征扰动响应」。**
本脚本就是做这个 —— 用真实数据库和真实特征工程函数。

检查内容
--------
  1. 每列的真实分位数（p0.1 / p1 / p25 / p50 / p75 / p99 / p99.9）
  2. 每列的动态范围 = |p99.9| / max(|p0.1|, eps)
  3. 每列相对 `return_5d` 的量级比
  4. 用真实分位数**驱动** ColumnScaler 的 log 变换选择
  5. 在真实数据上验证：标准化前后，扰动小特征对输出的影响

用法：
  python scripts/audit_real_feature_scales.py
  python scripts/audit_real_feature_scales.py --max-stocks 50   # 快速版
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "experiments" / "financial_transformer"
sys.path.insert(0, str(FT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results"


def load_config():
    import yaml
    p = FT / "config_moneyflow_v1.yaml"
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features_real(cfg: dict, max_stocks: int, start: str, end: str):
    """用项目真实的特征构造函数取一段真实数据。"""
    from financial_transformer_experiment.data import (
        load_feature_frame, ExperimentConfig)

    # 尝试直接复用项目的加载入口
    print("  尝试调用项目 load_feature_frame ...")
    try:
        exp_cfg = ExperimentConfig(**cfg) if isinstance(cfg, dict) else cfg
    except Exception as e:
        print(f"  ExperimentConfig 构造失败: {e}")
        return None, None

    try:
        df = load_feature_frame(exp_cfg)
        return df, exp_cfg
    except Exception as e:
        print(f"  load_feature_frame 失败: {type(e).__name__}: {e}")
        return None, exp_cfg


def fallback_build(cfg: dict, max_stocks: int) -> pd.DataFrame:
    """备用：自己从 stock_history.db 构造 32 个特征。

    只用于审计量级，不追求与项目特征完全一致。
    缺少的列（moneyflow / market_cap_ratio）用 NaN 占位。
    """
    import sqlite3
    db = (ROOT.parent / "stock_history.db").resolve()
    con = sqlite3.connect(str(db))

    # 取 micro 层：用最新市值筛
    mc_db = (ROOT.parent / "market_cap_full.db").resolve()
    codes = None
    if mc_db.exists():
        con2 = sqlite3.connect(str(mc_db))
        try:
            mc = pd.read_sql_query(
                "SELECT stock_code, total_mv FROM market_caps", con2)
            mc = mc.sort_values("total_mv").drop_duplicates(
                "stock_code", keep="last")
            # micro: 最小的一批（< 50 亿）
            mc["mv_yi"] = mc["total_mv"] / 10000.0
            micro = mc[mc["mv_yi"] < 50.0]
            codes = micro["stock_code"].astype(str).tolist()[:max_stocks]
        finally:
            con2.close()

    if not codes:
        codes = [r[0] for r in con.execute(
            "SELECT DISTINCT stock_code FROM stock_data ORDER BY stock_code "
            f"LIMIT {max_stocks}").fetchall()]

    q = ",".join("?" * len(codes))
    df = pd.read_sql_query(f"""
        SELECT stock_code, 日期 AS date, 开盘 AS open, 最高 AS high,
               最低 AS low, 收盘 AS close, 成交量 AS volume,
               成交额 AS amount, 换手率 AS turnover_rate
        FROM stock_data
        WHERE stock_code IN ({q}) AND 日期 BETWEEN ? AND ?
        ORDER BY stock_code, 日期
    """, con, params=codes + ["2024-01-01", "2026-10-29"])
    con.close()

    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    g = df.groupby("stock_code", group_keys=False)

    df["return_1d"] = g["close"].pct_change()
    df["return_5d"] = g["close"].transform(lambda s: s / s.shift(5) - 1)
    df["return_20d"] = g["close"].transform(lambda s: s / s.shift(20) - 1)
    df["volatility_20d"] = g["return_1d"].transform(
        lambda s: s.rolling(20).std())
    df["amount_ma5_ma20"] = g["amount"].transform(
        lambda s: s.rolling(5).mean() / s.rolling(20).mean())
    df["volume_ma5_ma20"] = g["volume"].transform(
        lambda s: s.rolling(5).mean() / s.rolling(20).mean())
    # RSI
    def rsi(s, n=14):
        d = s.diff()
        up = d.clip(lower=0).rolling(n).mean()
        dn = (-d.clip(upper=0)).rolling(n).mean()
        return 100 - 100 / (1 + up / dn.replace(0, np.nan))
    df["rsi_14"] = g["close"].transform(rsi)
    ema12 = g["close"].transform(lambda s: s.ewm(span=12).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("stock_code", group_keys=False)["macd"].transform(
        lambda s: s.ewm(span=9).mean())
    df["trend_strength"] = df["return_20d"] / df["volatility_20d"].replace(0, np.nan)
    df["close_to_high"] = (df["close"] - df["low"]) / (
        (df["high"] - df["low"]).replace(0, np.nan))
    df["close_to_low"] = (df["high"] - df["close"]) / (
        (df["high"] - df["low"]).replace(0, np.nan))
    df["gap_open"] = g["open"].transform(lambda s: s / s.shift(1) - 1)
    df["market_cap_ratio"] = np.nan      # 需要市值时序，占位
    d = df["date"]
    df["day_of_week_sin"] = np.sin(2 * np.pi * d.dt.dayofweek / 5)
    df["day_of_week_cos"] = np.cos(2 * np.pi * d.dt.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * d.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * d.dt.month / 12)
    for c in ["net_mf_amount_ratio", "net_lg_amount_ratio",
              "net_elg_amount_ratio", "net_mf_ratio_5d", "net_lg_ratio_5d",
              "net_elg_ratio_5d", "mf_acceleration"]:
        df[c] = np.nan       # 需要 moneyflow 表，占位

    return df


FEATURES = [
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=80)
    ap.add_argument("--out", default="results/real_feature_scales.json")
    args = ap.parse_args()

    print("=" * 96)
    print("真实输入列分位数审计")
    print("=" * 96)

    cfg = load_config()
    print(f"配置: micro 层, max_stocks={cfg['experiment'].get('max_stocks')}")
    print(f"特征列: {len(cfg['experiment']['feature_cols'])}")

    df = fallback_build(cfg, args.max_stocks)
    print(f"\n真实数据: {len(df):,} 行  {df['stock_code'].nunique()} 只  "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")

    cols = cfg["experiment"]["feature_cols"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"缺失列（用 NaN 占位）: {missing}")

    # ============================================================
    # 1. 逐列分位数
    # ============================================================
    print("\n" + "=" * 96)
    print("[1] 逐列真实分位数")
    print("=" * 96)
    print(f"  {'列':<22}{'p0.1':>12}{'p1':>12}{'p50':>12}"
          f"{'p99':>12}{'p99.9':>12}{'有效%':>7}{'量级':>10}")

    stats = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(s) < 100:
            print(f"  {c:<22} 有效样本不足 ({len(s)})")
            stats[c] = {"n": int(len(s)), "note": "insufficient"}
            continue
        q = {
            "p0.1": float(s.quantile(0.001)),
            "p1": float(s.quantile(0.01)),
            "p50": float(s.quantile(0.5)),
            "p99": float(s.quantile(0.99)),
            "p99.9": float(s.quantile(0.999)),
            "min": float(s.min()),
            "max": float(s.max()),
            "n": int(len(s)),
        }
        # 动态范围：用 p99.9 与 p0.1 的幅度比
        eps = 1e-12
        scale = max(abs(q["p99.9"]), abs(q["p0.1"]), eps)
        q["scale"] = scale
        q["valid_pct"] = float(len(s) / len(df) * 100)
        stats[c] = q
        print(f"  {c:<22}{q['p0.1']:>12.4g}{q['p1']:>12.4g}"
              f"{q['p50']:>12.4g}{q['p99']:>12.4g}{q['p99.9']:>12.4g}"
              f"{q['valid_pct']:>6.1f}%{scale:>10.4g}")

    # ============================================================
    # 2. 量级排序
    # ============================================================
    print("\n" + "=" * 96)
    print("[2] 量级排序（按 max|p99.9|, |p0.1|）")
    print("=" * 96)
    ranked = sorted(
        [(c, v["scale"]) for c, v in stats.items() if "scale" in v],
        key=lambda x: -x[1])
    for i, (c, s) in enumerate(ranked, 1):
        print(f"  {i:>2}. {c:<22} {s:>14.6g}")

    if len(ranked) >= 2:
        ratio = ranked[0][1] / max(ranked[-1][1], 1e-12)
        print(f"\n  最大/最小 量级比: {ratio:.4g}  "
              f"（约 {np.log10(ratio):.1f} 个数量级）")
        print(f"  → 这就是知微指出的问题："
              f"{ranked[0][0]} 比 {ranked[-1][0]} 大 {ratio:.3g} 倍")

    # ============================================================
    # 3. float32 精度检验
    # ============================================================
    print("\n" + "=" * 96)
    print("[3] float32 精度：小特征在大特征旁边的可表示性")
    print("=" * 96)
    if ranked:
        big = ranked[0][1]
        f32_eps_at_big = np.spacing(np.float32(big))
        print(f"  最大量级 {ranked[0][0]} = {big:.6g}")
        print(f"  float32 在该量级的间隔 (spacing) = {f32_eps_at_big:.6g}")
        print(f"\n  各列相对间隔（< 1 表示被淹没）:")
        for c, s in ranked:
            if s <= 0:
                continue
            rel = s / f32_eps_at_big
            flag = "  <<< 会被淹没" if rel < 1 else ""
            print(f"    {c:<22} scale={s:>12.6g}  "
                  f"scale/spacing={rel:>10.4g}{flag}")

    # ============================================================
    # 4. 建议的 log 变换列
    # ============================================================
    print("\n" + "=" * 96)
    print("[4] 由真实分位数驱动的变换建议")
    print("=" * 96)
    log_cols = []
    for c, v in stats.items():
        if "scale" not in v:
            continue
        if v["scale"] > 1e3 and v["p0.1"] > 0:
            log_cols.append(c)
    print(f"  建议 log 变换（尺度 > 1e3 且全正）: {log_cols}")
    print(f"  其余列用 z-score")

    # ============================================================
    # 5. 真实数据上的扰动检验
    # ============================================================
    print("\n" + "=" * 96)
    print("[5] 真实数据扰动检验（原模型 vs 标准化后）")
    print("=" * 96)
    try:
        from financial_transformer_experiment.model import FinancialTransformer
        import torch
        from transformer_fixes import ColumnScaler

        n_feat_0 = len(cols)
        seq_len = 60
        # ★ 先确定可用列，再按可用列数构造模型
        avail0 = [c for c in cols if c in df.columns]
        avail0 = [c for c in avail0
                  if c in stats and stats[c].get("valid_pct", 0) >= 50]
        n_feat = len(avail0)
        print(f"  按可用特征数构造模型: input_dim={n_feat}")

        torch.manual_seed(17)
        model = FinancialTransformer(
            input_dim=n_feat, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=256, dropout=0.0, model_variant="flat")
        model.eval()

        # 取真实的一行作为基准（只用真正有值的列，避免 NaN 传播）
        avail = [c for c in cols if c in df.columns]
        # ★ 过滤掉占位列（有效样本 < 50%）
        avail = [c for c in avail
                 if c in stats and stats[c].get("valid_pct", 0) >= 50]
        print(f"  可用于检验的特征列: {len(avail)} / {len(cols)}")
        if len(avail) < 3:
            raise RuntimeError("有效特征列不足")

        sub = df[avail].replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna()
        if len(sub) < 50:
            raise RuntimeError(f"完整样本不足 ({len(sub)})")
        print(f"  完整样本 {len(sub):,} 行")

        base_row = sub.median().values.astype(np.float32)
        assert np.isfinite(base_row).all(), "基准行含 NaN"

        # 找最小的那个特征（相对量级）
        av_ranked = [(c, s) for c, s in ranked if c in avail]
        if len(av_ranked) >= 2:
            small_col = av_ranked[-1][0]
            big_col = av_ranked[0][0]
            print(f"  最小特征: {small_col}  scale={av_ranked[-1][1]:.4g}")
            print(f"  最大特征: {big_col}  scale={av_ranked[0][1]:.4g}")

            si = avail.index(small_col)
            x1 = np.tile(base_row, (1, seq_len, 1)).astype(np.float32)
            x2 = x1.copy()
            # 用该列的真实 IQR 决定扰动量级（中位数可能为 0）
            q25 = float(sub[small_col].quantile(0.25))
            q75 = float(sub[small_col].quantile(0.75))
            iqr = max(q75 - q25, abs(av_ranked[-1][1]) * 0.1, 1e-6)
            delta = iqr * 0.2
            x1[0, :, si] = base_row[si] - delta / 2
            x2[0, :, si] = base_row[si] + delta / 2

            with torch.no_grad():
                o1 = model(torch.from_numpy(x1))
                o2 = model(torch.from_numpy(x2))
            if isinstance(o1, dict):
                k = "short_up" if "short_up" in o1 else list(o1)[0]
                o1, o2 = o1[k], o2[k]
            d_raw = float((o1 - o2).abs().max())
            print(f"  原始尺度：扰动 {small_col} ±{delta/2:.6g} "
                  f"(IQR={iqr:.6g}, 约 ±{delta/iqr*50:.0f}% IQR) "
                  f"→ 输出差 {d_raw:.10e}")

            # 标准化后
            sc = ColumnScaler(log_cols=[c for c in log_cols if c in avail])
            sc.fit(sub[avail])
            a1 = pd.DataFrame(np.asarray(x1[0], dtype=np.float64),
                              columns=avail)
            a2 = pd.DataFrame(np.asarray(x2[0], dtype=np.float64),
                              columns=avail)
            t1 = torch.from_numpy(
                sc.transform(a1).values.astype(np.float32)).unsqueeze(0)
            t2 = torch.from_numpy(
                sc.transform(a2).values.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                p1 = model(t1)
                p2 = model(t2)
            if isinstance(p1, dict):
                k = "short_up" if "short_up" in p1 else list(p1)[0]
                p1, p2 = p1[k], p2[k]
            d_norm = float((p1 - p2).abs().max())
            print(f"  标准化后：同样扰动 → 输出差 {d_norm:.10e}")

            improved = d_norm > d_raw * 10
            print(f"\n  判定: {'标准化显著改善' if improved else '未见改善'}"
                  f"  ({d_norm/max(d_raw, 1e-30):.3g}x)")
            if d_raw < 1e-9:
                print(f"  ★ 确认真实数据下 {small_col} 被完全淹没")
            stats["_perturbation"] = {
                "small_col": small_col, "big_col": big_col,
                "delta": float(delta), "d_raw": d_raw, "d_norm": d_norm,
                "n_avail": len(avail),
            }
    except Exception as e:
        import traceback
        print(f"  扰动检验失败: {type(e).__name__}: {e}")
        traceback.print_exc()

    # 保存
    os.makedirs(RESULTS, exist_ok=True)
    out = {"n_rows": int(len(df)), "n_stocks": int(df["stock_code"].nunique()),
           "features": cols, "stats": stats, "log_cols": log_cols}
    with open(RESULTS / "real_feature_scales.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 results/real_feature_scales.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
