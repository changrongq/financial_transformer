from __future__ import annotations

import argparse
import json
import joblib
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKER_DIR = EXPERIMENT_ROOT / "results" / "typhoon_tracker"
DEFAULT_MODEL_DIR = EXPERIMENT_ROOT / "results" / "typhoon_persistence_model"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "results" / "typhoon_persistence_model"
DEFAULT_TRADE_DB = ROOT.parent / "stock_history.db"


@dataclass(frozen=True)
class PersistenceParams:
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 5
    lgb_learning_rate: float = 0.03
    lgb_num_leaves: int = 31
    lgb_min_child_samples: int = 20
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 0.1
    test_split: float = 0.2
    random_state: int = 42


# ---- v1.0 特征集 ----
FEATURE_COLS_V0 = [
    "organization_score", "o_slope", "thermo_score",
    "herd_to_slow_ratio",
    "dominant_role_herd", "dominant_role_slow", "dominant_role_trend",
    "resonance_count", "structure_stability_days",
    "return_1d", "volume_acceleration",
    "o_lag1", "o_lag2", "o_lag3",
]

FEATURE_COLS_V1_MONEYFLOW = [
    "elg_net_amount",       # 特大单净流入
    "elg_in_out_ratio",     # 特大单流入/流出比
    "lg_net_amount",        # 大单净流入
    "lg_in_out_ratio",
    "md_net_amount",        # 中单净流入
    "sm_net_amount",        # 小单净流入
    "sm_consecutive_out",   # 小单连续净流出天数
    "lg_elg_ratio",         # 大单 vs 特大单分歧
]

FEATURE_COLS_V1_MOMENTUM = [
    "volume_ratio_5d_20d",  # 成交量突变率
]

FEATURE_COLS_V1_TURNOVER = [
    "real_turnover_rate",       # 真实换手率（修正股权结构污染）
    "turnover_amplify_ratio",   # 真实/名义换手率放大倍数（羊群效应增强信号）
]

FEATURE_COLS_V1 = FEATURE_COLS_V0 + FEATURE_COLS_V1_MONEYFLOW + FEATURE_COLS_V1_MOMENTUM + FEATURE_COLS_V1_TURNOVER

# 实际使用：先尝试 v1，moneyflow 不可用时退到 v0
FEATURE_COLS = FEATURE_COLS_V1

FEATURE_DISPLAY_NAMES = {
    "organization_score": "O值",
    "o_slope": "dO/dt",
    "thermo_score": "热力评分",
    "herd_to_slow_ratio": "herding比率",
    "dominant_role_herd": "角色:herd",
    "dominant_role_slow": "角色:slow",
    "dominant_role_trend": "角色:trend",
    "resonance_count": "共振数",
    "structure_stability_days": "稳定天数",
    "return_1d": "日涨跌幅",
    "volume_acceleration": "量加速度",
    "o_lag1": "O值T-1",
    "o_lag2": "O值T-2",
    "o_lag3": "O值T-3",
    "elg_net_amount": "特大单净流入",
    "elg_in_out_ratio": "特大单流入流出比",
    "lg_net_amount": "大单净流入",
    "lg_in_out_ratio": "大单流入流出比",
    "md_net_amount": "中单净流入",
    "sm_net_amount": "小单净流入",
    "sm_consecutive_out": "小单连续流出",
    "lg_elg_ratio": "大小单分歧",
    "volume_ratio_5d_20d": "量突变率",
    "real_turnover_rate": "真实换手率",
    "turnover_amplify_ratio": "换手率放大倍数",
}


def _load_moneyflow_frame() -> pd.DataFrame:
    """从 stock_moneyflow 表加载全量资金流向数据。"""
    db = DEFAULT_TRADE_DB
    if not db.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db))
    try:
        frame = pd.read_sql_query(
            "SELECT * FROM stock_moneyflow WHERE stock_code IS NOT NULL", conn
        )
    finally:
        conn.close()
    if frame.empty or len(frame.columns) < 20:
        return pd.DataFrame()

    working = pd.DataFrame({
        "stock_code": frame.iloc[:, 0].astype(str).str.zfill(6),
        "date": pd.to_datetime(frame.iloc[:, 1]),
    })
    numeric = frame.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # col 2-3: 小单买卖量额, 6-7: 中单, 10-11: 大单, 14-15: 特大单
    working["sm_buy"] = numeric.iloc[:, 3]   # 小单买入额
    working["sm_sell"] = numeric.iloc[:, 5]  # 小单卖出额
    working["md_buy"] = numeric.iloc[:, 7]   # 中单买入额
    working["md_sell"] = numeric.iloc[:, 9]  # 中单卖出额
    working["lg_buy"] = numeric.iloc[:, 11]  # 大单买入额
    working["lg_sell"] = numeric.iloc[:, 13] # 大单卖出额
    working["elg_buy"] = numeric.iloc[:, 15] # 特大单买入额
    working["elg_sell"] = numeric.iloc[:, 17] # 特大单卖出额

    working["elg_net_amount"] = working["elg_buy"] - working["elg_sell"]
    working["lg_net_amount"] = working["lg_buy"] - working["lg_sell"]
    working["md_net_amount"] = working["md_buy"] - working["md_sell"]
    working["sm_net_amount"] = working["sm_buy"] - working["sm_sell"]

    working["elg_in_out_ratio"] = np.where(
        working["elg_sell"].abs() < 1, 1.0,
        working["elg_buy"] / working["elg_sell"].abs()
    ).clip(0, 10)
    working["lg_in_out_ratio"] = np.where(
        working["lg_sell"].abs() < 1, 1.0,
        working["lg_buy"] / working["lg_sell"].abs()
    ).clip(0, 10)
    working["lg_elg_ratio"] = np.where(
        working["elg_net_amount"].abs() < 1, 0.0,
        working["lg_net_amount"] / working["elg_net_amount"].abs()
    ).clip(-5, 5)

    # 小单连续净流出天数
    working = working.sort_values(["stock_code", "date"])
    working["sm_net_sign"] = (working["sm_net_amount"] < 0).astype(int)
    working["sm_consecutive_out"] = (
        working.groupby("stock_code")["sm_net_sign"]
        .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        .where(working["sm_net_sign"] == 1, 0)
    )
    working = working.drop(columns=["sm_net_sign"])

    keep = ["stock_code", "date"] + FEATURE_COLS_V1_MONEYFLOW
    return working[[c for c in keep if c in working.columns]]


def _load_real_turnover_frame() -> pd.DataFrame:
    """从 real_turnover_rate 表加载真实换手率（含名义换手率用于放大倍数）。"""
    db = DEFAULT_TRADE_DB
    if not db.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db))
    try:
        frame = pd.read_sql_query(
            "SELECT stock_code, date, real_turnover_rate FROM real_turnover_rate",
            conn,
        )
        # 名义换手率来自 stock_data，用于计算放大倍数
        nominal = pd.read_sql_query(
            "SELECT stock_code, [日期] AS date, [换手率] AS nominal_turnover_rate FROM stock_data",
            conn,
        )
    finally:
        conn.close()
    if frame.empty:
        return pd.DataFrame()
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["real_turnover_rate"] = pd.to_numeric(frame["real_turnover_rate"], errors="coerce")

    if not nominal.empty:
        nominal["stock_code"] = nominal["stock_code"].astype(str).str.zfill(6)
        nominal["date"] = pd.to_datetime(nominal["date"], errors="coerce")
        nominal["nominal_turnover_rate"] = pd.to_numeric(nominal["nominal_turnover_rate"], errors="coerce")
        frame = frame.merge(nominal, on=["stock_code", "date"], how="left")
        frame["turnover_amplify_ratio"] = np.where(
            frame["nominal_turnover_rate"].fillna(0) > 0.01,
            frame["real_turnover_rate"] / frame["nominal_turnover_rate"].clip(lower=0.01),
            1.0,
        ).clip(0, 50)
        frame = frame[["stock_code", "date", "real_turnover_rate", "turnover_amplify_ratio"]]
    else:
        frame["turnover_amplify_ratio"] = 1.0
    return frame


def _load_training_data(tracker_dir: Path) -> pd.DataFrame:
    embryo_files = sorted(tracker_dir.glob("typhoon_embryo_tracker_*.csv"))
    mature_files = sorted(tracker_dir.glob("typhoon_tracker_20*.csv"))
    mature_files = [f for f in mature_files if "embryo" not in f.name]
    all_files = sorted(set(embryo_files + mature_files), key=lambda x: x.stem.split("_")[-1])
    if not all_files:
        raise FileNotFoundError(f"没有找到追踪 CSV 文件: {tracker_dir}")

    frames = []
    for f in all_files:
        df = pd.read_csv(f, dtype={"stock_code": str})
        # 丢弃全空列，避免 concat 产生 all-NA 列的 FutureWarning
        df = df.dropna(axis=1, how="all")
        date_tag = f.stem
        for prefix in ["typhoon_embryo_tracker_", "typhoon_tracker_"]:
            date_tag = date_tag.replace(prefix, "")
        df["report_date"] = pd.to_datetime(date_tag, errors="coerce")
        df["source"] = "embryo" if "embryo" in f.name else "mature"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["stock_code"] = combined["stock_code"].astype(str).str.zfill(6)
    combined = combined.sort_values(["stock_code", "report_date"]).reset_index(drop=True)
    return combined


def _build_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    for col in ["organization_score", "o_slope", "thermo_score", "herd_to_slow_ratio", "return_1d", "volume_acceleration"]:
        working[col] = pd.to_numeric(working.get(col, 0.0), errors="coerce").fillna(0.0)

    for col in ["resonance_count", "structure_stability_days"]:
        working[col] = pd.to_numeric(working.get(col, 0), errors="coerce").fillna(0).astype(int)

    role_series = working.get("dominant_role", pd.Series(["unknown"] * len(working))).fillna("unknown").astype(str)
    working["dominant_role_herd"] = (role_series == "herd").astype(int)
    working["dominant_role_slow"] = (role_series == "slow").astype(int)
    working["dominant_role_trend"] = (role_series == "trend").astype(int)

    grouped = working.groupby("stock_code", group_keys=False)
    working["o_lag1"] = grouped["organization_score"].shift(1).fillna(0.0)
    working["o_lag2"] = grouped["organization_score"].shift(2).fillna(0.0)
    working["o_lag3"] = grouped["organization_score"].shift(3).fillna(0.0)

    # ---- v1.0: 成交量突变率 ----
    if "volume" in working.columns:
        working["volume_5d"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
        working["volume_20d"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        working["volume_ratio_5d_20d"] = np.where(
            working["volume_20d"].fillna(1) > 0,
            working["volume_5d"] / working["volume_20d"].fillna(1),
            1.0
        ).clip(0, 10)
    else:
        working["volume_ratio_5d_20d"] = 1.0

    # ---- v1.0: moneyflow 资金结构因子 ----
    mf = _load_moneyflow_frame()
    if not mf.empty:
        working["date"] = pd.to_datetime(working.get("report_date", pd.NaT), errors="coerce")
        mf["date"] = pd.to_datetime(mf["date"])
        working = working.merge(mf, on=["stock_code", "date"], how="left")
        for col in FEATURE_COLS_V1_MONEYFLOW:
            if col in working.columns:
                working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
            else:
                working[col] = 0.0
    else:
        for col in FEATURE_COLS_V1_MONEYFLOW:
            working[col] = 0.0

    # ---- v1.0: 真实换手率（修正股权结构污染） ----
    rto = _load_real_turnover_frame()
    if "date" not in working.columns:
        working["date"] = pd.to_datetime(working.get("report_date", pd.NaT), errors="coerce")
    if not rto.empty:
        rto["date"] = pd.to_datetime(rto["date"], errors="coerce")
        working = working.merge(rto, on=["stock_code", "date"], how="left")
    for col in FEATURE_COLS_V1_TURNOVER:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
        else:
            working[col] = 0.0

    # ---- 标签 ----
    working["o_lead3"] = grouped["organization_score"].shift(-3)
    working["target_upgrade_3d"] = (
        (working["organization_score"] >= 0.55)
        & (working["organization_score"] < 0.78)
        & (working["o_lead3"] >= 0.78)
    ).astype(int)

    working["return_lead1"] = grouped["return_1d"].shift(-1)
    working["target_next_day_up"] = working["return_lead1"].gt(0.03).fillna(False).astype(int)
    working["target_next_day_down"] = working["return_lead1"].lt(-0.03).fillna(False).astype(int)

    # 确定实际可用的特征列
    available = [c for c in FEATURE_COLS_V1 if c in working.columns]
    if not available:
        available = [c for c in FEATURE_COLS_V0 if c in working.columns]
    return working.dropna(subset=available).reset_index(drop=True), available


def _train_model(df: pd.DataFrame, target_col: str, params: PersistenceParams, model_name: str) -> dict:
    feature_cols_used = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols_used].copy()
    y = df[target_col].copy()

    if y.nunique() < 2:
        return {"trained": False, "message": f"标签 {target_col} 只有一个类别", "model_name": model_name}

    pos_ratio = float(y.mean())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.test_split, random_state=params.random_state,
        stratify=y if pos_ratio > 0.05 else None,
    )

    scale_pos_weight = float(max(1.0, (1.0 - pos_ratio) / max(pos_ratio, 0.01)))
    model = HistGradientBoostingClassifier(
        max_iter=params.lgb_n_estimators,
        max_depth=params.lgb_max_depth,
        learning_rate=params.lgb_learning_rate,
        l2_regularization=params.lgb_reg_lambda,
        min_samples_leaf=params.lgb_min_child_samples,
        random_state=params.random_state,
    )
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob) if y_test.nunique() >= 2 else 0.5

    # permutation importance
    importances = {}
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=params.random_state, n_jobs=1)
        importances = dict(zip(feature_cols_used, perm.importances_mean.tolist()))
    except Exception:
        pass

    return {
        "trained": True,
        "model_name": model_name,
        "target": target_col,
        "samples": int(len(df)),
        "positive_ratio": float(pos_ratio),
        "test_auc": float(auc),
        "feature_importances": {k: float(v) for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]},
        "model": model,
    }


def train_persistence_models(df: pd.DataFrame, params: PersistenceParams, model_dir: Path) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for target, name in [
        ("target_upgrade_3d", "survival_3d"),
        ("target_next_day_up", "persistence_next_day"),
        ("target_next_day_down", "risk_next_day"),
    ]:
        r = _train_model(df, target, params, name)
        model = r.pop("model", None)
        if model is not None:
            model_path = model_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
        r["model_path"] = str(model_dir / f"{name}.joblib") if model is not None else None
        results[name] = r

    return results


def _attribution(row: pd.Series, model, feature_cols: list[str]) -> list[dict]:
    """用特征绝对值排名做简单归因（Top3 特征值最大的因子）。"""
    items = []
    for col in feature_cols:
        val = float(row.get(col, 0.0) or 0.0)
        display = FEATURE_DISPLAY_NAMES.get(col, col)
        items.append({"feature": col, "display": display, "value": round(val, 4), "score": round(abs(val), 4)})
    items.sort(key=lambda x: abs(x["score"]), reverse=True)
    top = items[:3]
    if all(abs(it["score"]) < 1e-6 for it in top):
        return []
    return top


def _star_rating(prob: float, risk: float | None) -> str:
    if prob is None or risk is None:
        return "N/A"
    if prob > 0.85 and (risk or 1.0) < 0.05:
        return "=== 强信号"
    if prob > 0.70:
        return "== 中等信号"
    return "= 弱信号"


def _v1_warnings(row: pd.Series) -> list[str]:
    warns = []
    try:
        herding = float(row.get("herd_to_slow_ratio", 0.0) or 0.0)
    except (ValueError, TypeError):
        herding = 0.0
    if herding > 15:
        warns.append("资金过热 (herding>15)")
    try:
        elg_ratio = float(row.get("elg_in_out_ratio", 1.0) or 1.0)
    except (ValueError, TypeError):
        elg_ratio = 1.0
    if 0 < elg_ratio < 0.6:
        warns.append("特大单流入流出比<0.6: 机构反手")
    try:
        sm_out = int(row.get("sm_consecutive_out", 0) or 0)
    except (ValueError, TypeError):
        sm_out = 0
    if sm_out >= 2:
        warns.append(f"小单连续{sm_out}日净流出: 散户割肉")
    return warns


def predict_persistence(row: pd.Series, model_dir: Path) -> dict:
    features = {}
    for col in FEATURE_COLS_V0:
        val = row.get(col)
        features[col] = float(val) if val is not None and not pd.isna(val) else 0.0
    for col in FEATURE_COLS_V1_MONEYFLOW + FEATURE_COLS_V1_MOMENTUM + FEATURE_COLS_V1_TURNOVER:
        val = row.get(col)
        features[col] = float(val) if val is not None and not pd.isna(val) else 0.0

    role = str(row.get("dominant_role", "") or "")
    features["dominant_role_herd"] = 1.0 if role == "herd" else 0.0
    features["dominant_role_slow"] = 1.0 if role == "slow" else 0.0
    features["dominant_role_trend"] = 1.0 if role == "trend" else 0.0

    # 确定可用列
    use_cols = [c for c in FEATURE_COLS_V1 if c in features]
    if not use_cols:
        use_cols = FEATURE_COLS_V0
    X = pd.DataFrame([features])[use_cols]

    probs = {}
    attributions = {}
    for name in ["persistence_next_day", "risk_next_day", "survival_3d"]:
        model_path = model_dir / f"{name}.joblib"
        if not model_path.exists():
            probs[name] = None
            continue
        model = joblib.load(model_path)
        probs[name] = float(model.predict_proba(X)[0][1])
        attributions[name] = _attribution(pd.Series(features), model, use_cols)

    persistence = probs.get("persistence_next_day")
    risk = probs.get("risk_next_day")

    return {
        "survival_prob": probs.get("survival_3d"),
        "persistence_prob": persistence,
        "risk_prob": risk,
        "star_rating": _star_rating(persistence, risk),
        "persistence_attribution": attributions.get("persistence_next_day", []),
        "risk_attribution": attributions.get("risk_next_day", []),
        "warnings": _v1_warnings(row),
    }


def predict_batch(embryo_candidates: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    # 先合并 moneyflow 数据
    mf = _load_moneyflow_frame()
    date_col = "report_date" if "report_date" in embryo_candidates.columns else "date"
    if not mf.empty and date_col in embryo_candidates.columns:
        embryo_candidates["_join_date"] = pd.to_datetime(embryo_candidates[date_col], errors="coerce")
        mf["_join_date"] = pd.to_datetime(mf["date"])
        embryo_candidates = embryo_candidates.merge(
            mf, left_on=["stock_code", "_join_date"], right_on=["stock_code", "_join_date"], how="left", suffixes=("", "_mf")
        )
        for col in FEATURE_COLS_V1_MONEYFLOW:
            target_col = col if col in embryo_candidates.columns else f"{col}_mf" if f"{col}_mf" in embryo_candidates.columns else col
            if target_col in embryo_candidates.columns:
                embryo_candidates[col] = pd.to_numeric(embryo_candidates[target_col], errors="coerce").fillna(0.0)
            else:
                embryo_candidates[col] = 0.0
    if "volume_ratio_5d_20d" not in embryo_candidates.columns:
        embryo_candidates["volume_ratio_5d_20d"] = 1.0

    # 合并真实换手率
    rto = _load_real_turnover_frame()
    if not rto.empty:
        if "_join_date" not in embryo_candidates.columns:
            embryo_candidates["_join_date"] = pd.to_datetime(
                embryo_candidates[date_col], errors="coerce"
            ) if date_col in embryo_candidates.columns else pd.NaT
        # 胚胎 CSV 无日期列时，退而用每只股票最新一天的真实换手率
        rto = rto.sort_values(["stock_code", "date"]).drop_duplicates("stock_code", keep="last")
        embryo_candidates = embryo_candidates.merge(
            rto[["stock_code", "real_turnover_rate", "turnover_amplify_ratio"]],
            on="stock_code", how="left",
        )
    for col in FEATURE_COLS_V1_TURNOVER:
        if col not in embryo_candidates.columns:
            embryo_candidates[col] = 0.0
        embryo_candidates[col] = pd.to_numeric(embryo_candidates[col], errors="coerce").fillna(0.0)

    results = []
    for _, row in embryo_candidates.iterrows():
        probs = predict_persistence(row, model_dir)
        results.append(probs)
    prob_df = pd.DataFrame(results)
    return pd.concat([embryo_candidates.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="台风胚胎持续性预测模型")
    parser.add_argument("--tracker-dir", default=str(DEFAULT_TRACKER_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--predict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker_dir = Path(args.tracker_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_training_data(tracker_dir)
    print(f"加载历史数据: {len(df)} 行, {df['stock_code'].nunique()} 只股票, {df['report_date'].nunique()} 个交易日")

    df, used_cols = _build_features_and_labels(df)
    print(f"特征工程后: {len(df)} 行, 特征列={len(used_cols)}, upgrade_3d={df['target_upgrade_3d'].mean():.4f}, up_1d={df['target_next_day_up'].mean():.4f}")

    if args.train:
        params = PersistenceParams()
        results = train_persistence_models(df, params, model_dir)

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "samples": int(len(df)),
            "params": asdict(params),
            "models": results,
        }
        summary_path = output_dir / "persistence_model_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n模型训练结果:")
        for name, r in results.items():
            print(f"  {name}: auc={r.get('test_auc', 'N/A')}, pos_ratio={r.get('positive_ratio', 'N/A'):.4f}")

    if args.predict:
        latest_embryo = sorted(tracker_dir.glob("typhoon_embryo_tracker_*.csv"))[-1]
        candidates = pd.read_csv(latest_embryo, dtype={"stock_code": str})
        preds_df = predict_batch(candidates, model_dir)
        date_tag = latest_embryo.stem.split("typhoon_embryo_tracker_")[-1]

        # v1.0: 用 JSON 输出，原生支持嵌套字段（归因/评级/风控）
        json_path = output_dir / f"persistence_predictions_{date_tag}.json"
        keep_cols = [
            "stock_code", "stock_name", "organization_score", "o_slope", "o_rising",
            "thermo_score", "dominant_role", "herd_to_slow_ratio", "embryo_score",
            "primary_concept_name", "resonance_count", "structure_stability_days",
            "survival_prob", "persistence_prob", "risk_prob",
            "star_rating", "persistence_attribution", "risk_attribution", "warnings",
            "close", "return_1d", "real_turnover_rate", "turnover_amplify_ratio",
        ]
        keep_cols = [c for c in keep_cols if c in preds_df.columns]
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "date": date_tag,
            "source_file": str(latest_embryo),
            "count": int(len(preds_df)),
            "predictions": preds_df[keep_cols].to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n预测结果已写入: {json_path}")

        # 同时保留 CSV 导入（简单列）
        csv_path = output_dir / f"persistence_predictions_{date_tag}.csv"
        csv_cols = ["stock_code", "stock_name", "organization_score", "survival_prob", "persistence_prob", "risk_prob", "star_rating"]
        csv_cols = [c for c in csv_cols if c in preds_df.columns]
        preds_df[csv_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

        show_cols = ["stock_code", "organization_score", "persistence_prob", "risk_prob", "star_rating"]
        show_cols = [c for c in show_cols if c in preds_df.columns]
        print(preds_df[show_cols].head(10).to_string())


if __name__ == "__main__":
    main()
