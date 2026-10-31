# -*- coding: utf-8 -*-
"""第一次正式重训：单一任务 + 逐列尺度 + 标签成熟度（知微第 4/5 步）。

设计原则（逐条对应知微的要求）
------------------------------
1. **单一任务**：只预测 `short_up`（未来 5 日累计收益 ≥ 3%）。
   其他 4 个任务暂不训练，避免多目标混杂。

2. **逐列尺度**：用 `ColumnScaler`（来自 `transformer_fixes.py`），
   由真实分位数驱动 —— `volume` / `amount` 用 log1p + z-score，
   其余用 z-score。scaler 随 checkpoint 保存。

3. **标签成熟度**：训练集只保留标签在 train_end 之前已成熟的样本。
   5 日标签意味着 train_end 前 5 个交易日的样本要剔除。

4. **同口径对照**：同时跑三个基线
   · 常数概率（预测 = 训练集正例率）
   · Ridge（线性，单一任务）
   · 小 Transformer（d_model=64, 2 层）

5. **只评价一次**：test 集只用一次，用于最终报告。

用法：
  python scripts/train_single_task.py
  python scripts/train_single_task.py --max-stocks 200 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "financial_transformer" / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from transformer_fixes import ColumnScaler, snapshot_state_dict  # noqa: E402

RESULTS = ROOT / "results"
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
LOG_COLS = ["volume", "amount"]


def load_panel(max_stocks: int, db: str) -> tuple:
    """走生产路径构造面板。"""
    from financial_transformer_experiment.data import (
        _load_trade_data_with_concepts, _add_time_features,
        add_multitask_targets)
    from hierarchical_system.features import build_layer_features

    con = sqlite3.connect(db)
    codes = [r[0] for r in con.execute(
        "SELECT DISTINCT stock_code FROM stock_data "
        f"ORDER BY stock_code LIMIT {max_stocks}").fetchall()]
    con.close()

    frames = []
    fails = 0
    t0 = time.time()
    for i, c in enumerate(codes, 1):
        try:
            td = _load_trade_data_with_concepts(
                db, c, manual_concept_map={}, concept_label_source=None)
            if td.empty:
                continue
            f = build_layer_features(td, 30.0)
            f = _add_time_features(f)
            f = add_multitask_targets(
                f, short_horizon=5, short_threshold=0.03, hold_days=45,
                take_profit_threshold=0.12, stop_loss_threshold=-0.06,
                exit_label_mode="window_any")
            frames.append(f)
        except Exception as e:
            fails += 1
            if fails <= 3:
                print(f"  {c}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            print(f"  加载 {i}/{len(codes)}  成功 {len(frames)}  "
                  f"{time.time()-t0:.0f}s")

    df = pd.concat(frames, ignore_index=True)
    print(f"  面板: {len(df):,} 行  {df['stock_code'].nunique()} 只  "
          f"{fails} 只失败")
    return df


def add_label_end(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """为每行加上标签观察结束日（按交易日历）。"""
    dates = sorted(pd.to_datetime(df["date"]).unique())
    cal = pd.DatetimeIndex(dates)
    pos = cal.searchsorted(pd.to_datetime(df["date"]), side="left")
    end = []
    for p in pos:
        j = p + horizon
        end.append(cal[j] if j < len(cal) else pd.NaT)
    df = df.copy()
    df["label_end_date"] = end
    return df


def build_sequences(df: pd.DataFrame, feat_cols: list, seq_len: int) -> dict:
    """按股票构造序列。返回 {split: (X, y, meta)}。"""
    out = {}
    train_end = pd.Timestamp("2025-09-30")
    val_end = pd.Timestamp("2025-12-31")
    test_end = pd.Timestamp("2026-03-04")

    rows = {"train": [], "val": [], "test": []}
    for code, g in df.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[feat_cols].to_numpy(dtype=np.float32)
        y = g["target_up_5d"].to_numpy(dtype=np.float32)
        fr = g["future_return_5d"].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g["date"]).to_list()
        ends = g["label_end_date"].to_list()

        for i in range(seq_len - 1, len(g)):
            d = pd.Timestamp(dates[i])
            if d <= train_end:
                split = "train"
                boundary = train_end
            elif d <= val_end:
                split = "val"
                boundary = val_end
            elif d <= test_end:
                split = "test"
                boundary = test_end
            else:
                continue

            # ★ 标签成熟度：标签结束日必须在边界内
            le = ends[i]
            if pd.isna(le) or pd.Timestamp(le) > boundary:
                continue
            if not np.isfinite(y[i]):
                continue

            w = vals[i - seq_len + 1: i + 1]
            if not np.isfinite(w).all():
                continue

            rows[split].append((w, y[i], fr[i], code, str(d.date())))

    for s, lst in rows.items():
        if not lst:
            out[s] = (np.zeros((0, seq_len, len(feat_cols)), dtype=np.float32),
                      np.zeros(0), np.zeros(0), [])
            continue
        X = np.stack([r[0] for r in lst])
        yy = np.array([r[1] for r in lst], dtype=np.float32)
        ff = np.array([r[2] for r in lst], dtype=np.float32)
        meta = [(r[3], r[4]) for r in lst]
        out[s] = (X, yy, ff, meta)
    return out


class SmallTransformer(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, n_layers=2,
                 ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.norm(self.proj(x))
        h = self.enc(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def auc(y, p):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # 处理并列
    s = pd.Series(p)
    ranks = s.rank().values
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    if len(a) < 20:
        return np.nan
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="results/single_task_train.json")
    args = ap.parse_args()

    db = str((ROOT.parent / "stock_history.db").resolve())
    print("=" * 92)
    print("第一次正式重训：单一任务 + ColumnScaler + 标签成熟度")
    print("=" * 92)

    # ---- 1. 数据 ----
    print("\n[1] 加载面板（生产路径）...")
    df = load_panel(args.max_stocks, db)
    df = df.dropna(subset=["target_up_5d"]).reset_index(drop=True)
    df = add_label_end(df, horizon=5)

    avail = [c for c in FEATURE_COLS if c in df.columns]
    print(f"  特征 {len(avail)}/{len(FEATURE_COLS)}")
    before = len(df)
    df = df.dropna(subset=avail).reset_index(drop=True)
    print(f"  完整样本 {len(df):,}（原 {before:,}）")

    # ---- 2. 序列 ----
    print(f"\n[2] 构造序列（seq_len={args.seq_len}，含标签成熟度过滤）...")
    ds = build_sequences(df, avail, args.seq_len)
    for s in ["train", "val", "test"]:
        X, y, fr, meta = ds[s]
        if len(X):
            print(f"  {s:<6} {len(X):>7,} 样本  正例率 {y.mean()*100:5.1f}%  "
                  f"日期 {meta[0][1]} ~ {meta[-1][1]}")
        else:
            print(f"  {s:<6} 空")

    Xtr, ytr, frtr, _ = ds["train"]
    Xva, yva, frva, _ = ds["val"]
    Xte, yte, frte, _ = ds["test"]
    if len(Xtr) < 1000 or len(Xte) < 100:
        print("样本不足，退出")
        return 1

    # ---- 3. 尺度 ----
    print(f"\n[3] 逐列尺度（log_cols={LOG_COLS}）...")
    tr_df = pd.DataFrame(Xtr[:, -1, :], columns=avail)
    scaler = ColumnScaler(log_cols=[c for c in LOG_COLS if c in avail])
    scaler.fit(tr_df)

    def scale(X3):
        n, t, f = X3.shape
        flat = pd.DataFrame(X3.reshape(-1, f), columns=avail)
        return scaler.transform(flat).values.astype(np.float32).reshape(n, t, f)

    Xtr_s, Xva_s, Xte_s = scale(Xtr), scale(Xva), scale(Xte)

    # ---- 4. 基线 1：常数概率 ----
    print("\n[4] 基线")
    print("-" * 92)
    base_p = float(ytr.mean())
    auc_const = auc(yte, np.full(len(yte), base_p))
    print(f"  常数概率({base_p:.4f})  test AUC = {auc_const:.4f}（应为 0.5）")

    # ---- 5. 基线 2：Ridge ----
    from sklearn.linear_model import Ridge
    print(f"\n  Ridge（用最后一帧特征）:")
    Ztr = Xtr_s[:, -1, :]
    Zva = Xva_s[:, -1, :]
    Zte = Xte_s[:, -1, :]
    ridge_aucs = []
    for a in [1.0, 10.0, 100.0]:
        m = Ridge(alpha=a).fit(Ztr, ytr)
        pv = m.predict(Zva)
        pt = m.predict(Zte)
        av, at = auc(yva, pv), auc(yte, pt)
        ridge_aucs.append((a, av, at))
        print(f"    alpha={a:<6} val AUC {av:.4f}  test AUC {at:.4f}")
    best_ridge = max(ridge_aucs, key=lambda x: x[1])
    print(f"    最佳(按 val): alpha={best_ridge[0]}  test AUC {best_ridge[2]:.4f}")

    # ---- 6. Transformer ----
    print(f"\n[5] 小 Transformer（{len(args.seeds)} 个种子）")
    print("-" * 92)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SmallTransformer(len(avail)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01)
        lossf = nn.BCEWithLogitsLoss()

        Xtr_t = torch.from_numpy(Xtr_s)
        ytr_t = torch.from_numpy(ytr)
        Xva_t = torch.from_numpy(Xva_s).to(device)
        best_state = None
        best_val = -1
        bad = 0

        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(Xtr_t))
            tot = 0.0
            nb = 0
            for i in range(0, len(perm), args.batch):
                idx = perm[i:i + args.batch]
                xb = Xtr_t[idx].to(device)
                yb = ytr_t[idx].to(device)
                logits = model(xb)
                loss = lossf(logits, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss)
                nb += 1

            model.eval()
            with torch.no_grad():
                pv = model(Xva_t).cpu().numpy()
            va = auc(yva, pv)
            if np.isfinite(va) and va > best_val:
                best_val = va
                best_state = snapshot_state_dict(model)
                bad = 0
            else:
                bad += 1
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"    seed={seed} ep={ep:>3} loss={tot/max(nb,1):.4f} "
                      f"valAUC={va:.4f}" + ("  *" if bad == 0 else ""))

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pt = model(torch.from_numpy(Xte_s).to(device)).cpu().numpy()
        at = auc(yte, pt)
        ic = spearman(pt, frte)
        print(f"    seed={seed}: best val AUC {best_val:.4f}  "
              f"test AUC {at:.4f}  test IC {ic:.4f}")
        results.append({"seed": seed, "best_val": best_val,
                        "test_auc": at, "test_ic": ic,
                        "pred_std": float(np.std(pt))})

    # ---- 7. 汇总 ----
    print("\n" + "=" * 92)
    print("[6] 汇总")
    print("=" * 92)
    tests = np.array([r["test_auc"] for r in results])
    print(f"  {'模型':<24}{'test AUC':>12}{'说明':>30}")
    print(f"  {'常数概率':<24}{auc_const:>12.4f}{'理论 0.5':>30}")
    print(f"  {'Ridge (best by val)':<24}{best_ridge[2]:>12.4f}"
          f"{f'alpha={best_ridge[0]}':>30}")
    print(f"  {'Transformer (mean)':<24}{np.nanmean(tests):>12.4f}"
          f"{f'std {np.nanstd(tests):.4f}':>30}")
    print(f"  {'Transformer (best)':<24}{np.nanmax(tests):>12.4f}"
          f"{results[int(np.nanargmax(tests))]['seed']:>28}")

    checks = [
        ("Transformer 优于常数基线", np.nanmean(tests) > 0.52,
         f"{np.nanmean(tests):.4f} vs 0.5000"),
        ("Transformer 优于 Ridge", np.nanmean(tests) > best_ridge[2],
         f"{np.nanmean(tests):.4f} vs {best_ridge[2]:.4f}"),
        ("多种子稳定（std < 0.03）", np.nanstd(tests) < 0.03,
         f"std {np.nanstd(tests):.4f}"),
        ("预测不塌缩（std > 1e-3）", np.mean([r["pred_std"] for r in results]) > 1e-3,
         f"pred_std {np.mean([r['pred_std'] for r in results]):.6f}"),
    ]
    all_ok = True
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_ok &= ok

    if all_ok:
        verdict = "修复后 Transformer 有增量，值得继续"
    elif np.nanmean(tests) > 0.52:
        verdict = "有信号但不稳定，需扩大样本"
    else:
        verdict = "修复后仍无增量"
    print(f"\n  结论: {verdict}")

    out = {
        "params": vars(args),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "pos_rate_train": float(ytr.mean()),
        "auc_const": auc_const,
        "ridge": [{"alpha": a, "val": v, "test": t}
                  for a, v, t in ridge_aucs],
        "ridge_best": {"alpha": best_ridge[0], "test_auc": best_ridge[2]},
        "transformer": results,
        "transformer_mean_test_auc": float(np.nanmean(tests)),
        "checks": [{"name": n, "ok": bool(o), "detail": d}
                   for n, o, d in checks],
        "verdict": verdict,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
