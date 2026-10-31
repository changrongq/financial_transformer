# -*- coding: utf-8 -*-
"""用生产路径做列分位数审计（补齐资金流之后）。

修正 audit_real_feature_scales.py 的问题：
  那个脚本用 fallback_build 手工构造 df，没有 merge stock_moneyflow，
  所以永远只有 24 列可用。

本脚本直接走 _load_trade_data_with_concepts + build_layer_features，
得到与训练时完全一致的 32 列。

用法：
  python scripts/audit_scales_production.py --max-stocks 30
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "financial_transformer" / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FEATURE_COLS = [
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
    ap.add_argument("--max-stocks", type=int, default=30)
    ap.add_argument("--out", default="results/scales_production.json")
    args = ap.parse_args()

    from financial_transformer_experiment.data import (
        _load_trade_data_with_concepts, _add_time_features)
    from hierarchical_system.features import build_layer_features

    db = str((ROOT.parent / "stock_history.db").resolve())
    con = sqlite3.connect(db)
    codes = [r[0] for r in con.execute(
        "SELECT DISTINCT stock_code FROM stock_data "
        f"ORDER BY stock_code LIMIT {args.max_stocks}").fetchall()]
    con.close()

    print("=" * 92)
    print("生产路径列分位数审计（资金流已补齐）")
    print("=" * 92)

    frames = []
    fails = 0
    for c in codes:
        try:
            td = _load_trade_data_with_concepts(
                db, c, manual_concept_map={}, concept_label_source=None)
            if td.empty:
                continue
            f = build_layer_features(td, 30.0)
            f = _add_time_features(f)
            frames.append(f)
        except Exception as e:
            fails += 1
            if fails <= 3:
                print(f"  {c}: {type(e).__name__}: {e}")

    out = pd.concat(frames, ignore_index=True)
    print(f"\n{len(frames)} 只股票, {len(out):,} 行, {fails} 只失败")

    print("\n" + "=" * 92)
    print("[1] 逐列分位数（生产路径）")
    print("=" * 92)
    print(f"  {'列':<22}{'非空率':>9}{'唯一值':>9}{'p0.1':>12}"
          f"{'p50':>12}{'p99.9':>12}{'量级':>11}")

    stats = {}
    avail = []
    for c in FEATURE_COLS:
        if c not in out.columns:
            print(f"  {c:<22} 列不存在")
            continue
        s = pd.to_numeric(out[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
        n = int(s.notna().sum())
        if n < 50:
            print(f"  {c:<22} 有效 {n}")
            continue
        rate = n / len(out) * 100
        u = int(s.nunique())
        p_lo = float(s.quantile(0.001))
        p50 = float(s.quantile(0.5))
        p_hi = float(s.quantile(0.999))
        scale = max(abs(p_hi), abs(p_lo), 1e-12)
        stats[c] = {"rate": rate, "uniq": u, "p_lo": p_lo,
                    "p50": p50, "p_hi": p_hi, "scale": scale}
        if u > 1:
            avail.append(c)
        print(f"  {c:<22}{rate:>8.1f}%{u:>9}{p_lo:>12.4g}"
              f"{p50:>12.4g}{p_hi:>12.4g}{scale:>11.4g}")

    print(f"\n  有变化且可用的列: {len(avail)} 个")

    ranked = sorted([(c, v["scale"]) for c, v in stats.items() if c in avail],
                    key=lambda x: -x[1])
    print("\n" + "=" * 92)
    print("[2] 量级排序")
    print("=" * 92)
    for i, (c, s) in enumerate(ranked, 1):
        print(f"  {i:>2}. {c:<22} {s:>14.6g}")

    if len(ranked) >= 2:
        ratio = ranked[0][1] / max(ranked[-1][1], 1e-12)
        print(f"\n  最大/最小 = {ratio:.4g}（约 {np.log10(ratio):.1f} 个数量级）")

    print("\n" + "=" * 92)
    print("[3] float32 精度判定（scale/spacing < 1 表示被淹没）")
    print("=" * 92)
    big = ranked[0][1]
    spacing = float(np.spacing(np.float32(big)))
    print(f"  最大量级 {ranked[0][0]} = {big:.6g}")
    print(f"  float32 spacing = {spacing:.6g}")
    n_drown = 0
    for c, s in ranked:
        rel = s / spacing
        flag = "  淹没" if rel < 1 else ""
        if rel < 1:
            n_drown += 1
        print(f"  {c:<22} {rel:>12.4g}{flag}")
    print(f"\n  被淹没: {n_drown}/{len(ranked)}")

    print("\n" + "=" * 92)
    print("[4] 真实扰动检验")
    print("=" * 92)
    try:
        from financial_transformer_experiment.model import FinancialTransformer
        from transformer_fixes import ColumnScaler

        sub = out[avail].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  完整样本 {len(sub):,} 行")
        if len(sub) < 100:
            raise RuntimeError("样本不足")

        n_feat = len(avail)
        seq_len = 60
        torch.manual_seed(17)
        model = FinancialTransformer(
            input_dim=n_feat, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=256, dropout=0.0, model_variant="flat")
        model.eval()

        base = sub.median().values.astype(np.float32)
        av_ranked = [(c, s) for c, s in ranked if c in avail]
        small = av_ranked[-1][0]
        si = avail.index(small)

        q25 = float(sub[small].quantile(0.25))
        q75 = float(sub[small].quantile(0.75))
        iqr = max(q75 - q25, abs(av_ranked[-1][1]) * 0.1, 1e-9)
        delta = iqr * 0.2

        x1 = np.tile(base, (1, seq_len, 1)).astype(np.float32)
        x2 = x1.copy()
        x1[0, :, si] = base[si] - delta / 2
        x2[0, :, si] = base[si] + delta / 2

        with torch.no_grad():
            o1, o2 = model(torch.from_numpy(x1)), model(torch.from_numpy(x2))
        if isinstance(o1, dict):
            k = "short_up" if "short_up" in o1 else list(o1)[0]
            o1, o2 = o1[k], o2[k]
        d_raw = float((o1 - o2).abs().max())
        print(f"  最小特征: {small}  scale={av_ranked[-1][1]:.4g}")
        print(f"  扰动 ±{delta/2:.6g}（10% IQR）")
        print(f"  原始尺度输出差: {d_raw:.10e}")

        log_cols = [c for c in ["volume", "amount"] if c in avail]
        sc = ColumnScaler(log_cols=log_cols).fit(sub[avail])
        a1 = pd.DataFrame(np.asarray(x1[0], dtype=np.float64), columns=avail)
        a2 = pd.DataFrame(np.asarray(x2[0], dtype=np.float64), columns=avail)
        t1 = torch.from_numpy(sc.transform(a1).values.astype(np.float32)).unsqueeze(0)
        t2 = torch.from_numpy(sc.transform(a2).values.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            p1, p2 = model(t1), model(t2)
        if isinstance(p1, dict):
            k = "short_up" if "short_up" in p1 else list(p1)[0]
            p1, p2 = p1[k], p2[k]
        d_norm = float((p1 - p2).abs().max())
        print(f"  log_cols: {log_cols}")
        print(f"  标准化后输出差: {d_norm:.10e}")
        print(f"  改善: {d_norm/max(d_raw,1e-30):.3g}x")

        if d_raw < 1e-9:
            print(f"  ★ {small} 在原始尺度下被完全淹没")
    except Exception as e:
        import traceback
        print(f"  扰动检验失败: {type(e).__name__}: {e}")
        traceback.print_exc()

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"n_rows": int(len(out)), "n_avail": len(avail),
                   "avail": avail, "stats": stats},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
