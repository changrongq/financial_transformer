from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hierarchical_system.features import build_layer_features
from hierarchical_system.features import layer_from_market_cap_yi
from hierarchical_system.logging_utils import setup_logger
from hierarchical_system.multitask_data import add_multitask_targets

from .config import ExperimentConfig


TARGET_COLUMNS = {
    "short_up": "target_up_5d",
    "take_profit": "target_take_profit_45d",
    "stop_loss": "target_stop_loss_45d",
    "max_gain": "max_gain_45d",
    "max_drawdown": "max_drawdown_45d",
}
ROLE_VECTOR_COLUMNS = ["herd_act", "slow_act", "trend_act"]


logger = setup_logger("financial_transformer.data")


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    input_dim: int
    selected_stock_codes: list[str]
    diagnostics: dict[str, object] | None = None


class SequenceDataset(Dataset):
    def __init__(self, x: list[np.ndarray], targets: dict[str, np.ndarray], meta: pd.DataFrame):
        self.x = [torch.tensor(row, dtype=torch.float32) for row in x]
        self.lengths = torch.tensor([row.shape[0] for row in x], dtype=torch.long) if x else torch.empty((0,), dtype=torch.long)
        self.targets = {name: torch.tensor(values, dtype=torch.float32) for name, values in targets.items()}
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        sample_targets = {name: values[index] for name, values in self.targets.items()}
        return self.x[index], sample_targets, int(self.lengths[index].item()), self.meta.iloc[index].to_dict()


def collate_sequence_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], int, dict]]) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict]]:
    sequences, targets, lengths, metas = zip(*batch)
    batch_size = len(sequences)
    max_len = max(lengths)
    input_dim = sequences[0].shape[1]
    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    for index, (sequence, length) in enumerate(zip(sequences, lengths)):
        x[index, :length] = sequence
        padding_mask[index, :length] = False
    stacked_targets = {
        name: torch.stack([sample[name] for sample in targets])
        for name in TARGET_COLUMNS
    }
    return x, stacked_targets, torch.tensor(lengths, dtype=torch.long), padding_mask, list(metas)


def normalize_stock_code(stock_code: str) -> str:
    code = str(stock_code).strip()
    if "." in code:
        code = code.split(".", 1)[0]
    return code


def _empty_role_vector_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["stock_code", "date", *ROLE_VECTOR_COLUMNS])


def _build_role_vectors_from_simulation(simulation: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    demand_cols = {
        "herd_act": "herd_demand",
        "slow_act": "slow_demand",
        "trend_act": "trend_demand",
    }
    demand_matrix = pd.DataFrame(
        {
            output_col: pd.to_numeric(simulation[input_col], errors="coerce").abs()
            for output_col, input_col in demand_cols.items()
        }
    )
    total_activity = demand_matrix.sum(axis=1)
    normalized = demand_matrix.div(total_activity.replace(0.0, np.nan), axis=0).fillna(0.0)
    return pd.DataFrame(
        {
            "stock_code": normalize_stock_code(stock_code),
            "date": pd.to_datetime(simulation["date"]),
            "herd_act": normalized["herd_act"],
            "slow_act": normalized["slow_act"],
            "trend_act": normalized["trend_act"],
        }
    )


def _load_role_vector_frame(config: ExperimentConfig) -> pd.DataFrame:
    root_dir_value = config.experiment.role_vector_root_dir
    if not root_dir_value:
        return _empty_role_vector_frame()

    root_dir = Path(root_dir_value)
    csv_paths = sorted(root_dir.glob(config.experiment.role_vector_glob))
    frames: list[pd.DataFrame] = []
    loaded_parents: set[Path] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, dtype={"stock_code": str})
        required_columns = {"date", *ROLE_VECTOR_COLUMNS}
        if not required_columns.issubset(frame.columns):
            logger.warning("skipping role vector file missing columns path=%s", csv_path)
            continue
        working = frame[[column_name for column_name in ["stock_code", "date", *ROLE_VECTOR_COLUMNS] if column_name in frame.columns]].copy()
        if "stock_code" not in working.columns:
            working["stock_code"] = csv_path.parent.name
        working["stock_code"] = working["stock_code"].astype(str).map(normalize_stock_code)
        working["date"] = pd.to_datetime(working["date"])
        for column_name in ROLE_VECTOR_COLUMNS:
            working[column_name] = pd.to_numeric(working[column_name], errors="coerce")
        working["source_mtime"] = csv_path.stat().st_mtime
        frames.append(working)
        loaded_parents.add(csv_path.parent.resolve())

    fallback_pattern = config.experiment.role_vector_glob.replace("role_activity_vectors.csv", "simulated_path.csv")
    fallback_count = 0
    if fallback_pattern != config.experiment.role_vector_glob:
        for simulation_path in sorted(root_dir.glob(fallback_pattern)):
            if simulation_path.parent.resolve() in loaded_parents:
                continue
            simulation = pd.read_csv(simulation_path)
            required_sim_cols = {"date", "herd_demand", "slow_demand", "trend_demand"}
            if not required_sim_cols.issubset(simulation.columns):
                logger.warning("skipping simulation fallback missing columns path=%s", simulation_path)
                continue
            stock_code = str(simulation_path.parent.name).split("_")[-2]
            working = _build_role_vectors_from_simulation(simulation, stock_code)
            working["source_mtime"] = simulation_path.stat().st_mtime
            frames.append(working)
            fallback_count += 1

    if not frames:
        logger.warning("role vector root configured but no files matched root=%s glob=%s", root_dir, config.experiment.role_vector_glob)
        return _empty_role_vector_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["stock_code", "date", "source_mtime"]).drop_duplicates(["stock_code", "date"], keep="last")
    combined = combined.drop(columns=["source_mtime"]).reset_index(drop=True)
    logger.info(
        "loaded role vectors files=%s fallback_simulations=%s rows=%s stocks=%s",
        len(csv_paths),
        fallback_count,
        len(combined),
        combined["stock_code"].nunique(),
    )
    return combined


def _merge_role_vector_features(feature_df: pd.DataFrame, role_vector_df: pd.DataFrame) -> pd.DataFrame:
    merged = feature_df.copy()
    for column_name in ROLE_VECTOR_COLUMNS:
        if column_name not in merged.columns:
            merged[column_name] = 0.0
    if merged.empty or role_vector_df.empty or "stock_code" not in merged.columns or "date" not in merged.columns:
        return merged

    working = merged.drop(columns=ROLE_VECTOR_COLUMNS, errors="ignore")
    working["stock_code"] = working["stock_code"].astype(str).map(normalize_stock_code)
    working["date"] = pd.to_datetime(working["date"])
    working = working.merge(role_vector_df, on=["stock_code", "date"], how="left")
    for column_name in ROLE_VECTOR_COLUMNS:
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce").fillna(0.0)
    return working


def _add_requested_role_interaction_features(feature_df: pd.DataFrame, requested_feature_cols: list[str]) -> pd.DataFrame:
    requested = {str(column_name) for column_name in requested_feature_cols}
    if not requested:
        return feature_df

    working = feature_df.copy()
    for column_name in ROLE_VECTOR_COLUMNS:
        if column_name not in working.columns:
            working[column_name] = 0.0
        working[column_name] = pd.to_numeric(working[column_name], errors="coerce").fillna(0.0)

    total_role_activity = working[ROLE_VECTOR_COLUMNS].sum(axis=1)
    slow_dominant = (
        (working["slow_act"] >= working["herd_act"])
        & (working["slow_act"] >= working["trend_act"])
        & (total_role_activity > 1e-9)
    ).astype(float)

    if "slow_dominant" in requested:
        working["slow_dominant"] = slow_dominant

    for feature_name in requested_feature_cols:
        if feature_name.startswith("slow_act_x_"):
            base_col = feature_name[len("slow_act_x_"):]
            if base_col in working.columns:
                base_series = pd.to_numeric(working[base_col], errors="coerce").fillna(0.0)
                working[feature_name] = working["slow_act"] * base_series
        elif feature_name.startswith("slow_dom_x_"):
            base_col = feature_name[len("slow_dom_x_"):]
            if base_col in working.columns:
                base_series = pd.to_numeric(working[base_col], errors="coerce").fillna(0.0)
                working[feature_name] = slow_dominant * base_series
    return working


def _stock_fold_id(stock_code: str, total_folds: int) -> int:
    normalized = normalize_stock_code(stock_code).encode("utf-8")
    digest = hashlib.md5(normalized).hexdigest()
    return int(digest[:8], 16) % total_folds


def _split_allows_stock(split: str, stock_code: str, config: ExperimentConfig) -> bool:
    total_folds = config.experiment.stock_cv_total_folds
    if total_folds is None:
        return True
    allowed_folds = {
        "train": config.experiment.train_stock_folds,
        "val": config.experiment.val_stock_folds,
        "test": config.experiment.test_stock_folds,
    }.get(split, [])
    if not allowed_folds:
        return True
    return _stock_fold_id(stock_code, int(total_folds)) in {int(fold) for fold in allowed_folds}


def _latest_market_caps(config: ExperimentConfig) -> pd.DataFrame:
    conn = sqlite3.connect(config.databases.market_cap_db)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()]
        date_col = "trade_date" if "trade_date" in columns else "update_date" if "update_date" in columns else None
        select_cols = ["stock_code", "total_mv"]
        if date_col:
            select_cols.append(f"{date_col} AS trade_date")
        df = pd.read_sql_query(f"SELECT {', '.join(select_cols)} FROM market_caps", conn)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError("市值库中没有可用记录。")
    sort_cols = ["stock_code"] + (["trade_date"] if "trade_date" in df.columns else [])
    df = df.sort_values(sort_cols).drop_duplicates("stock_code", keep="last")
    df["market_cap_yi"] = df["total_mv"].astype(float) / 10000.0
    if config.experiment.stock_pool:
        normalized_pool = {normalize_stock_code(code) for code in config.experiment.stock_pool}
        df = df[df["stock_code"].astype(str).isin(normalized_pool)].copy()
    if config.experiment.layer_filter:
        df["layer"] = df["market_cap_yi"].apply(layer_from_market_cap_yi)
        df = df[df["layer"] == config.experiment.layer_filter].copy()
    if config.experiment.max_stocks:
        df = df.sort_values("market_cap_yi", ascending=False).head(int(config.experiment.max_stocks)).copy()
    if df.empty:
        raise RuntimeError("按当前实验配置没有筛出任何股票，请检查 stock_pool、layer_filter 或 max_stocks。")
    return df[["stock_code", "market_cap_yi"]]


def _load_trade_data(trade_db: str, stock_code: str) -> pd.DataFrame:
    return _load_trade_data_with_concepts(trade_db, stock_code)


def _labels_from_concept_df(concept_df: pd.DataFrame) -> dict[str, str]:
    if concept_df.empty:
        return {"concept_names": "", "concept_ids": ""}
    concept_names = sorted({str(value).strip() for value in concept_df["concept_name"].dropna().tolist() if str(value).strip()})
    concept_ids = sorted({str(value).strip() for value in concept_df["concept_id"].dropna().tolist() if str(value).strip()})
    return {
        "concept_names": "|".join(concept_names),
        "concept_ids": "|".join(concept_ids),
    }


def _load_manual_concept_labels(config: ExperimentConfig) -> dict[str, dict[str, str]]:
    labels_path_value = config.experiment.manual_concept_labels_path
    if not labels_path_value:
        return {}

    labels_path = Path(labels_path_value)
    if not labels_path.exists():
        raise RuntimeError(f"未找到人工概念标签文件: {labels_path}")

    labels_df = pd.read_csv(labels_path, dtype={"stock_code": str})
    if "stock_code" not in labels_df.columns:
        raise RuntimeError("人工概念标签文件至少需要 stock_code 列。")

    concept_names_col = "manual_concept_names" if "manual_concept_names" in labels_df.columns else "concept_names" if "concept_names" in labels_df.columns else None
    concept_ids_col = "manual_concept_ids" if "manual_concept_ids" in labels_df.columns else "concept_ids" if "concept_ids" in labels_df.columns else None
    if concept_names_col is None:
        raise RuntimeError("人工概念标签文件至少需要 manual_concept_names 或 concept_names 列。")

    labels_df["stock_code"] = labels_df["stock_code"].astype(str).map(normalize_stock_code)
    labels_df[concept_names_col] = labels_df[concept_names_col].fillna("").astype(str)
    if concept_ids_col is not None:
        labels_df[concept_ids_col] = labels_df[concept_ids_col].fillna("").astype(str)

    concept_map: dict[str, dict[str, str]] = {}
    for stock_code, group in labels_df.groupby("stock_code"):
        concept_names = sorted({item for raw in group[concept_names_col].tolist() for item in _split_pipe_values(raw)})
        concept_ids = sorted({item for raw in group[concept_ids_col].tolist() for item in _split_pipe_values(raw)}) if concept_ids_col is not None else []
        concept_map[stock_code] = {
            "concept_names": "|".join(concept_names),
            "concept_ids": "|".join(concept_ids),
        }

    logger.info("loaded manual concept labels path=%s stock_count=%s", labels_path, len(concept_map))
    return concept_map


def _resolve_concept_labels(
    normalized_code: str,
    concept_df: pd.DataFrame,
    manual_concept_map: dict[str, dict[str, str]] | None,
    concept_label_source: str,
) -> dict[str, str]:
    manual_concept_map = manual_concept_map or {}
    manual_entry_exists = normalized_code in manual_concept_map
    if concept_label_source == "manual_only":
        return manual_concept_map.get(normalized_code, {"concept_names": "", "concept_ids": ""})
    if concept_label_source == "manual_first" and manual_entry_exists:
        return manual_concept_map[normalized_code]
    if not concept_df.empty:
        return _labels_from_concept_df(concept_df)
    if concept_label_source == "manual_first" and manual_entry_exists:
        return manual_concept_map[normalized_code]
    return {"concept_names": "", "concept_ids": ""}


def _load_trade_data_with_concepts(
    trade_db: str,
    stock_code: str,
    *,
    manual_concept_map: dict[str, dict[str, str]] | None = None,
    concept_label_source: str = "db_only",
) -> pd.DataFrame:
    normalized_code = normalize_stock_code(stock_code)
    conn = sqlite3.connect(trade_db)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM stock_data WHERE stock_code = ? ORDER BY 日期",
            conn,
            params=[normalized_code],
        )
        # 真实换手率（修正股权结构污染，见 free_float.py）
        try:
            rto_df = pd.read_sql_query(
                "SELECT stock_code, date, real_turnover_rate "
                "FROM real_turnover_rate WHERE stock_code = ?",
                conn,
                params=[normalized_code],
            )
        except Exception:
            rto_df = pd.DataFrame()
        try:
            moneyflow_df = pd.read_sql_query(
                "SELECT * FROM stock_moneyflow WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[normalized_code],
            )
        except Exception:
            moneyflow_df = pd.DataFrame()
        try:
            top_inst_df = pd.read_sql_query(
                "SELECT * FROM stock_top_inst WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[normalized_code],
            )
        except Exception:
            top_inst_df = pd.DataFrame()
        try:
            margin_df = pd.read_sql_query(
                "SELECT * FROM stock_margin_detail WHERE stock_code = ? ORDER BY 日期",
                conn,
                params=[normalized_code],
            )
        except Exception:
            margin_df = pd.DataFrame()
        try:
            basic_info_df = pd.read_sql_query(
                "SELECT * FROM stock_basic_info WHERE stock_code = ?",
                conn,
                params=[normalized_code],
            )
        except Exception:
            basic_info_df = pd.DataFrame()
        concept_df = pd.DataFrame()
        if concept_label_source != "manual_only":
            try:
                concept_df = pd.read_sql_query(
                    "SELECT concept_id, concept_name FROM stock_concept_members WHERE stock_code = ? ORDER BY concept_id",
                    conn,
                    params=[normalized_code],
                )
            except Exception:
                concept_df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns=rename_map)
    keep_cols = [col for col in ["stock_code", "date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"] if col in df.columns]
    df = df[keep_cols].copy()
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_turnover_mask = pd.Series(False, index=df.index)
    if "volume" in df.columns:
        missing_turnover_mask = df["volume"].isna()
        if "amount" in df.columns:
            missing_turnover_mask = missing_turnover_mask & df["amount"].isna()
        df.loc[missing_turnover_mask, "volume"] = 0.0
    if "amount" in df.columns:
        df["amount"] = df["amount"].fillna(df["close"] * df["volume"])
    else:
        df["amount"] = df["close"] * df["volume"]
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = np.nan
    df.loc[df["turnover_rate"].isna() & missing_turnover_mask, "turnover_rate"] = 0.0
    df["date"] = pd.to_datetime(df["date"])
    # 真实换手率：用修正后的真实换手率替代名义换手率（缺失回退名义）
    if not rto_df.empty:
        rto_df = rto_df.rename(columns={"date": "_rto_date"})
        rto_df["_rto_date"] = pd.to_datetime(rto_df["_rto_date"], errors="coerce")
        rto_df["real_turnover_rate"] = pd.to_numeric(rto_df["real_turnover_rate"], errors="coerce")
        df = df.merge(
            rto_df[["stock_code", "_rto_date", "real_turnover_rate"]],
            left_on=["stock_code", "date"],
            right_on=["stock_code", "_rto_date"],
            how="left",
        ).drop(columns=["_rto_date"])
        df["turnover_rate"] = df["real_turnover_rate"].fillna(df["turnover_rate"])
    if not moneyflow_df.empty:
        moneyflow_df = moneyflow_df.rename(columns={"日期": "date"})
        moneyflow_df["date"] = pd.to_datetime(moneyflow_df["date"])
        for col in [c for c in moneyflow_df.columns if c not in {"stock_code", "date"}]:
            moneyflow_df[col] = pd.to_numeric(moneyflow_df[col], errors="coerce")
        df = df.merge(moneyflow_df, on=["stock_code", "date"], how="left")
    if not top_inst_df.empty:
        top_inst_df = top_inst_df.rename(columns={"日期": "date"})
        top_inst_df["date"] = pd.to_datetime(top_inst_df["date"])
        for col in [c for c in top_inst_df.columns if c not in {"stock_code", "date"}]:
            top_inst_df[col] = pd.to_numeric(top_inst_df[col], errors="coerce")
        df = df.merge(top_inst_df, on=["stock_code", "date"], how="left")
    if not margin_df.empty:
        margin_df = margin_df.rename(columns={"日期": "date"})
        margin_df["date"] = pd.to_datetime(margin_df["date"])
        for col in [c for c in margin_df.columns if c not in {"stock_code", "date"}]:
            margin_df[col] = pd.to_numeric(margin_df[col], errors="coerce")
        df = df.merge(margin_df, on=["stock_code", "date"], how="left")
    if not basic_info_df.empty:
        basic_info_df = basic_info_df.copy()
        for column_name in ["名称", "地域", "行业", "市场", "上市日期"]:
            if column_name in basic_info_df.columns:
                df[column_name] = basic_info_df.iloc[0][column_name]
    concept_labels = _resolve_concept_labels(normalized_code, concept_df, manual_concept_map, concept_label_source)
    resolved_concept_names = _split_pipe_values(concept_labels.get("concept_names", ""))
    df["concept_count"] = float(len(resolved_concept_names))
    df["concept_names"] = "|".join(resolved_concept_names)
    df["concept_ids"] = concept_labels.get("concept_ids", "")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    day_of_week = df["date"].dt.dayofweek
    month = df["date"].dt.month
    time_features = pd.DataFrame(
        {
            "day_of_week_sin": np.sin(2 * math.pi * day_of_week / 7.0),
            "day_of_week_cos": np.cos(2 * math.pi * day_of_week / 7.0),
            "month_sin": np.sin(2 * math.pi * (month - 1) / 12.0),
            "month_cos": np.cos(2 * math.pi * (month - 1) / 12.0),
        },
        index=df.index,
    )
    existing_columns = [column_name for column_name in time_features.columns if column_name in df.columns]
    if existing_columns:
        time_features = time_features.drop(columns=existing_columns)
    if time_features.empty:
        return df
    return pd.concat([df, time_features], axis=1)


def _finalize_feature_frame(feature_df: pd.DataFrame, stock_code: str, market_cap_yi: float, config: ExperimentConfig) -> pd.DataFrame:
    finalized = feature_df.assign(
        stock_code=normalize_stock_code(stock_code),
        layer=feature_df.get("layer", config.experiment.layer_filter or "unknown"),
        market_cap_yi=feature_df.get("market_cap_yi", market_cap_yi),
        future_return=feature_df["future_return_5d"],
        target_up=feature_df["target_up_5d"],
    )
    return finalized.replace([np.inf, -np.inf], np.nan)


def _add_regime_context_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or "date" not in feature_df.columns:
        return feature_df
    regime_feature_names = [
        "regime_breadth_10d",
        "regime_breadth_20d",
        "regime_return_mean_10d",
        "regime_return_vol_10d",
        "regime_turnover_ratio_10_20",
        "regime_net_mf_mean_10d",
    ]
    if all(feature_name in feature_df.columns for feature_name in regime_feature_names):
        return feature_df

    working = feature_df.copy()
    working["date"] = pd.to_datetime(working["date"])
    daily = working.groupby("date", as_index=False).agg(
        cross_section_return_mean=("return_1d", "mean"),
        cross_section_turnover_mean=("turnover_rate", "mean"),
        cross_section_net_mf_mean=("net_mf_amount_ratio", "mean"),
        cross_section_up_ratio=("target_up", "mean"),
    ).sort_values("date").reset_index(drop=True)
    daily["cross_section_turnover_mean"] = daily["cross_section_turnover_mean"].ffill()
    daily["cross_section_net_mf_mean"] = daily["cross_section_net_mf_mean"].fillna(0.0)
    daily["regime_breadth_10d"] = daily["cross_section_up_ratio"].rolling(10, min_periods=3).mean()
    daily["regime_breadth_20d"] = daily["cross_section_up_ratio"].rolling(20, min_periods=5).mean()
    daily["regime_return_mean_10d"] = daily["cross_section_return_mean"].rolling(10, min_periods=3).mean()
    daily["regime_return_vol_10d"] = daily["cross_section_return_mean"].rolling(10, min_periods=3).std()
    turnover_mean_10d = daily["cross_section_turnover_mean"].rolling(10, min_periods=3).mean()
    turnover_mean_20d = daily["cross_section_turnover_mean"].rolling(20, min_periods=5).mean()
    daily["regime_turnover_ratio_10_20"] = turnover_mean_10d / turnover_mean_20d.replace(0.0, np.nan)
    daily["regime_net_mf_mean_10d"] = daily["cross_section_net_mf_mean"].rolling(10, min_periods=3).mean()
    merge_columns = ["date", *regime_feature_names]
    merged = working.merge(daily[merge_columns], on="date", how="left")
    return merged


def _add_industry_sentiment_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or "date" not in feature_df.columns:
        return feature_df

    industry_feature_names = [
        "industry_breadth_pos_1d",
        "industry_breadth_pos_5d",
        "industry_return_mean_1d",
        "industry_return_std_1d",
        "industry_return_mean_5d",
        "industry_net_mf_mean_1d",
        "industry_net_mf_std_1d",
        "industry_net_mf_mean_5d",
        "industry_leader_return_1d",
        "stock_vs_industry_leader_1d",
        "stock_vs_industry_mean_1d",
        "stock_vs_industry_mean_5d",
        "stock_vs_industry_net_mf_1d",
        "stock_vs_industry_net_mf_5d",
        "industry_return_zscore_1d",
        "industry_net_mf_zscore_1d",
        "industry_return_rank_pct_1d",
        "industry_net_mf_rank_pct_1d",
        "industry_return_rank_pct_5d",
        "industry_net_mf_rank_pct_5d",
        "industry_return_rank_drift_5d",
        "industry_net_mf_rank_drift_5d",
    ]
    if all(feature_name in feature_df.columns for feature_name in industry_feature_names):
        return feature_df

    working = feature_df.copy()
    working["date"] = pd.to_datetime(working["date"])
    industry_name = working.get("行业")
    if industry_name is None:
        working["industry_name"] = "unknown"
    else:
        working["industry_name"] = industry_name.fillna("unknown").astype(str).str.strip().replace({"": "unknown"})

    daily = working.groupby(["industry_name", "date"], as_index=False).agg(
        industry_return_mean_1d=("return_1d", "mean"),
        industry_return_std_1d=("return_1d", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) or 0.0)),
        industry_breadth_pos_1d=("return_1d", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        industry_net_mf_mean_1d=("net_mf_amount_ratio", "mean"),
        industry_net_mf_std_1d=("net_mf_amount_ratio", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) or 0.0)),
        industry_leader_return_1d=("return_1d", "max"),
    )
    daily = daily.sort_values(["industry_name", "date"]).reset_index(drop=True)
    daily["industry_net_mf_mean_1d"] = daily["industry_net_mf_mean_1d"].fillna(0.0)
    daily["industry_return_std_1d"] = daily["industry_return_std_1d"].fillna(0.0)
    daily["industry_net_mf_std_1d"] = daily["industry_net_mf_std_1d"].fillna(0.0)

    daily["industry_breadth_pos_5d"] = (
        daily.groupby("industry_name")["industry_breadth_pos_1d"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )
    daily["industry_return_mean_5d"] = (
        daily.groupby("industry_name")["industry_return_mean_1d"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )
    daily["industry_net_mf_mean_5d"] = (
        daily.groupby("industry_name")["industry_net_mf_mean_1d"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )

    merge_columns = [
        "industry_name",
        "date",
        "industry_breadth_pos_1d",
        "industry_breadth_pos_5d",
        "industry_return_mean_1d",
        "industry_return_std_1d",
        "industry_return_mean_5d",
        "industry_net_mf_mean_1d",
        "industry_net_mf_std_1d",
        "industry_net_mf_mean_5d",
        "industry_leader_return_1d",
    ]
    merged = working.merge(daily[merge_columns], on=["industry_name", "date"], how="left")
    merged["stock_vs_industry_leader_1d"] = merged["return_1d"] - merged["industry_leader_return_1d"]
    merged["stock_vs_industry_mean_1d"] = merged["return_1d"] - merged["industry_return_mean_1d"]
    merged["stock_vs_industry_mean_5d"] = merged["return_5d"] - merged["industry_return_mean_5d"]
    merged["stock_vs_industry_net_mf_1d"] = merged["net_mf_amount_ratio"] - merged["industry_net_mf_mean_1d"]
    merged["stock_vs_industry_net_mf_5d"] = merged["net_mf_ratio_5d"] - merged["industry_net_mf_mean_5d"]

    return_std = pd.to_numeric(merged["industry_return_std_1d"], errors="coerce").replace(0.0, np.nan)
    net_mf_std = pd.to_numeric(merged["industry_net_mf_std_1d"], errors="coerce").replace(0.0, np.nan)
    merged["industry_return_zscore_1d"] = (merged["return_1d"] - merged["industry_return_mean_1d"]) / return_std
    merged["industry_net_mf_zscore_1d"] = (merged["net_mf_amount_ratio"] - merged["industry_net_mf_mean_1d"]) / net_mf_std
    merged["industry_return_zscore_1d"] = pd.to_numeric(merged["industry_return_zscore_1d"], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    merged["industry_net_mf_zscore_1d"] = pd.to_numeric(merged["industry_net_mf_zscore_1d"], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    merged["industry_return_rank_pct_1d"] = merged.groupby(["industry_name", "date"])["return_1d"].transform(_rank_pct_with_singleton)
    merged["industry_net_mf_rank_pct_1d"] = merged.groupby(["industry_name", "date"])["net_mf_amount_ratio"].transform(_rank_pct_with_singleton)
    merged["industry_return_rank_pct_5d"] = merged.groupby(["industry_name", "date"])["return_5d"].transform(_rank_pct_with_singleton)
    merged["industry_net_mf_rank_pct_5d"] = merged.groupby(["industry_name", "date"])["net_mf_ratio_5d"].transform(_rank_pct_with_singleton)
    merged = merged.sort_values(["stock_code", "date"]).reset_index(drop=True)
    merged["industry_return_rank_drift_5d"] = (
        merged.groupby("stock_code")["industry_return_rank_pct_1d"].transform(lambda s: s - s.shift(5))
    )
    merged["industry_net_mf_rank_drift_5d"] = (
        merged.groupby("stock_code")["industry_net_mf_rank_pct_1d"].transform(lambda s: s - s.shift(5))
    )
    merged["industry_return_rank_drift_5d"] = pd.to_numeric(merged["industry_return_rank_drift_5d"], errors="coerce").fillna(0.0)
    merged["industry_net_mf_rank_drift_5d"] = pd.to_numeric(merged["industry_net_mf_rank_drift_5d"], errors="coerce").fillna(0.0)
    return merged


def _split_pipe_values(raw_value: object) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    return [item for item in (part.strip() for part in text.split("|")) if item]


def _rank_pct_with_singleton(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid_count = int(numeric.notna().sum())
    if valid_count <= 0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    if valid_count == 1:
        return pd.Series(np.ones(len(series), dtype=float), index=series.index)
    ranked = numeric.rank(method="average", pct=True)
    return ranked.fillna(0.0)


def _weighted_group_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid_mask = numeric_values.notna() & np.isfinite(numeric_values.to_numpy(dtype=float)) & (numeric_weights > 0)
    if not bool(valid_mask.any()):
        return 0.0
    valid_values = numeric_values[valid_mask].to_numpy(dtype=float)
    valid_weights = numeric_weights[valid_mask].to_numpy(dtype=float)
    weight_sum = float(valid_weights.sum())
    if weight_sum <= 0:
        return 0.0
    return float(np.dot(valid_values, valid_weights) / weight_sum)


def _rolling_corr_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str, reference_col: str, window: int, min_periods: int) -> pd.Series:
    pieces: list[pd.Series] = []
    for _, group in df.groupby(group_cols, sort=False):
        value_series = pd.to_numeric(group[value_col], errors="coerce")
        reference_series = pd.to_numeric(group[reference_col], errors="coerce")
        corr_series = value_series.rolling(window=window, min_periods=min_periods).corr(reference_series)
        pieces.append(pd.Series(corr_series.to_numpy(dtype=float), index=group.index))
    if not pieces:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return pd.concat(pieces).sort_index()


def _ewm_mean_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str, span: int, min_periods: int) -> pd.Series:
    pieces: list[pd.Series] = []
    for _, group in df.groupby(group_cols, sort=False):
        numeric = pd.to_numeric(group[value_col], errors="coerce")
        ewm_series = numeric.ewm(span=span, adjust=False, min_periods=min_periods).mean()
        pieces.append(pd.Series(ewm_series.to_numpy(dtype=float), index=group.index))
    if not pieces:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return pd.concat(pieces).sort_index()


def _add_concept_sentiment_features(feature_df: pd.DataFrame, config: ExperimentConfig | None = None) -> pd.DataFrame:
    if feature_df.empty or "date" not in feature_df.columns:
        return feature_df

    concept_feature_names = [
        "concept_member_count",
        "concept_exposure_score",
        "concept_breadth_pos_1d",
        "concept_breadth_pos_5d",
        "concept_return_mean_1d",
        "concept_return_mean_5d",
        "concept_net_mf_mean_1d",
        "concept_net_mf_mean_5d",
        "concept_leader_return_1d",
        "stock_vs_concept_leader_1d",
    ]
    if all(feature_name in feature_df.columns for feature_name in concept_feature_names):
        return feature_df

    concept_top_k = None
    concept_weighting = "uniform"
    concept_exposure_lookback_days = 20
    if config is not None:
        concept_top_k = config.experiment.concept_sentiment_top_k
        concept_weighting = config.experiment.concept_sentiment_weighting
        concept_exposure_lookback_days = int(config.experiment.concept_exposure_lookback_days)

    working = feature_df.copy()
    working["date"] = pd.to_datetime(working["date"])
    if "concept_count" not in working.columns:
        working["concept_count"] = 0.0
    if "concept_names" not in working.columns:
        working["concept_names"] = ""

    concept_membership = working[["stock_code", "date", "return_1d", "net_mf_amount_ratio", "concept_names"]].copy()
    concept_membership["concept_name"] = concept_membership["concept_names"].apply(_split_pipe_values)
    concept_membership = concept_membership.explode("concept_name")
    concept_membership["concept_name"] = concept_membership["concept_name"].fillna("").astype(str).str.strip()
    concept_membership = concept_membership[concept_membership["concept_name"] != ""].reset_index(drop=True)

    if concept_membership.empty:
        working["concept_member_count"] = pd.to_numeric(working["concept_count"], errors="coerce").fillna(0.0)
        for feature_name in concept_feature_names:
            if feature_name == "concept_member_count":
                continue
            working[feature_name] = 0.0
        return working

    daily = concept_membership.groupby(["concept_name", "date"], as_index=False).agg(
        concept_return_mean_1d=("return_1d", "mean"),
        concept_breadth_pos_1d=("return_1d", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        concept_net_mf_mean_1d=("net_mf_amount_ratio", "mean"),
        concept_leader_return_1d=("return_1d", "max"),
    )
    daily = daily.sort_values(["concept_name", "date"]).reset_index(drop=True)
    daily["concept_net_mf_mean_1d"] = daily["concept_net_mf_mean_1d"].fillna(0.0)
    daily["concept_breadth_pos_5d"] = (
        daily.groupby("concept_name")["concept_breadth_pos_1d"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )
    daily["concept_return_mean_5d"] = (
        daily.groupby("concept_name")["concept_return_mean_1d"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )
    daily["concept_net_mf_mean_5d"] = (
        daily.groupby("concept_name")["concept_net_mf_mean_1d"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )

    concept_with_daily = concept_membership.merge(
        daily,
        on=["concept_name", "date"],
        how="left",
    )

    concept_with_daily["concept_strength_score"] = (
        concept_with_daily.groupby(["stock_code", "date"])["concept_breadth_pos_5d"].transform(_rank_pct_with_singleton)
        + concept_with_daily.groupby(["stock_code", "date"])["concept_return_mean_5d"].transform(_rank_pct_with_singleton)
        + concept_with_daily.groupby(["stock_code", "date"])["concept_net_mf_mean_5d"].transform(_rank_pct_with_singleton)
    ) / 3.0
    concept_with_daily["concept_strength_score"] = concept_with_daily["concept_strength_score"].fillna(0.0)

    concept_with_daily = concept_with_daily.sort_values(["stock_code", "concept_name", "date"]).reset_index(drop=True)
    corr_min_periods = max(4, concept_exposure_lookback_days // 3)
    ewm_min_periods = max(2, concept_exposure_lookback_days // 4)
    concept_with_daily["concept_return_excess_1d"] = (
        pd.to_numeric(concept_with_daily["return_1d"], errors="coerce")
        - pd.to_numeric(concept_with_daily["concept_return_mean_1d"], errors="coerce")
    ).fillna(0.0)
    concept_with_daily["concept_net_mf_excess_1d"] = (
        pd.to_numeric(concept_with_daily["net_mf_amount_ratio"], errors="coerce")
        - pd.to_numeric(concept_with_daily["concept_net_mf_mean_1d"], errors="coerce")
    ).fillna(0.0)
    concept_with_daily["concept_strength_decay_score"] = _ewm_mean_by_group(
        concept_with_daily,
        ["stock_code", "concept_name"],
        "concept_strength_score",
        span=concept_exposure_lookback_days,
        min_periods=ewm_min_periods,
    ).fillna(0.0)
    concept_with_daily["concept_return_excess_decay"] = _ewm_mean_by_group(
        concept_with_daily,
        ["stock_code", "concept_name"],
        "concept_return_excess_1d",
        span=concept_exposure_lookback_days,
        min_periods=ewm_min_periods,
    ).fillna(0.0)
    concept_with_daily["concept_net_mf_excess_decay"] = _ewm_mean_by_group(
        concept_with_daily,
        ["stock_code", "concept_name"],
        "concept_net_mf_excess_1d",
        span=concept_exposure_lookback_days,
        min_periods=ewm_min_periods,
    ).fillna(0.0)
    concept_with_daily["concept_relative_excess_score"] = (
        concept_with_daily.groupby(["stock_code", "date"])["concept_return_excess_decay"].transform(_rank_pct_with_singleton)
        + concept_with_daily.groupby(["stock_code", "date"])["concept_net_mf_excess_decay"].transform(_rank_pct_with_singleton)
    ) / 2.0
    concept_with_daily["concept_exposure_score"] = (
        concept_with_daily.groupby(["stock_code", "date"])["concept_strength_score"].transform(_rank_pct_with_singleton)
        + concept_with_daily.groupby(["stock_code", "date"])["concept_strength_decay_score"].transform(_rank_pct_with_singleton)
        + concept_with_daily.groupby(["stock_code", "date"])["concept_relative_excess_score"].transform(_rank_pct_with_singleton)
    ) / 3.0
    concept_with_daily["concept_exposure_score"] = concept_with_daily["concept_exposure_score"].fillna(0.0)
    if concept_weighting == "strength":
        concept_with_daily["concept_selection_score"] = concept_with_daily["concept_strength_score"]
    elif concept_weighting == "exposure":
        concept_with_daily["concept_selection_score"] = concept_with_daily["concept_exposure_score"]
    elif concept_weighting == "strength_exposure":
        concept_with_daily["concept_selection_score"] = (
            0.5 * concept_with_daily["concept_strength_score"]
            + 0.5 * concept_with_daily["concept_exposure_score"]
        )
    else:
        concept_with_daily["concept_selection_score"] = 1.0
    concept_with_daily = concept_with_daily.sort_values(
        ["stock_code", "date", "concept_selection_score", "concept_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    if concept_top_k is not None:
        concept_with_daily = (
            concept_with_daily.groupby(["stock_code", "date"], group_keys=False)
            .head(int(concept_top_k))
            .reset_index(drop=True)
        )

    if concept_weighting == "strength":
        concept_with_daily["concept_weight"] = concept_with_daily["concept_strength_score"].clip(lower=1e-6)
    elif concept_weighting == "exposure":
        concept_with_daily["concept_weight"] = concept_with_daily["concept_exposure_score"].clip(lower=1e-6)
    elif concept_weighting == "strength_exposure":
        concept_with_daily["concept_weight"] = (
            0.5 * concept_with_daily["concept_strength_score"]
            + 0.5 * concept_with_daily["concept_exposure_score"]
        ).clip(lower=1e-6)
    else:
        concept_with_daily["concept_weight"] = 1.0
    concept_with_daily["concept_weight"] = concept_with_daily.groupby(["stock_code", "date"])["concept_weight"].transform(
        lambda s: s / float(s.sum()) if float(s.sum()) > 0 else np.ones(len(s), dtype=float) / max(len(s), 1)
    )

    stock_daily_concepts = concept_with_daily.groupby(["stock_code", "date"])[
        [
            "concept_name",
            "concept_exposure_score",
            "concept_breadth_pos_1d",
            "concept_breadth_pos_5d",
            "concept_return_mean_1d",
            "concept_return_mean_5d",
            "concept_net_mf_mean_1d",
            "concept_net_mf_mean_5d",
            "concept_leader_return_1d",
            "concept_weight",
        ]
    ].apply(
        lambda group: pd.Series(
            {
                "concept_member_count": float(group["concept_name"].nunique()),
                "concept_exposure_score": _weighted_group_mean(group["concept_exposure_score"], group["concept_weight"]),
                "concept_breadth_pos_1d": _weighted_group_mean(group["concept_breadth_pos_1d"], group["concept_weight"]),
                "concept_breadth_pos_5d": _weighted_group_mean(group["concept_breadth_pos_5d"], group["concept_weight"]),
                "concept_return_mean_1d": _weighted_group_mean(group["concept_return_mean_1d"], group["concept_weight"]),
                "concept_return_mean_5d": _weighted_group_mean(group["concept_return_mean_5d"], group["concept_weight"]),
                "concept_net_mf_mean_1d": _weighted_group_mean(group["concept_net_mf_mean_1d"], group["concept_weight"]),
                "concept_net_mf_mean_5d": _weighted_group_mean(group["concept_net_mf_mean_5d"], group["concept_weight"]),
                "concept_leader_return_1d": float(pd.to_numeric(group["concept_leader_return_1d"], errors="coerce").fillna(0.0).max()),
            }
        )
    ).reset_index()
    if "level_2" in stock_daily_concepts.columns:
        stock_daily_concepts = stock_daily_concepts.drop(columns=["level_2"])

    merged = working.merge(stock_daily_concepts, on=["stock_code", "date"], how="left")
    merged["concept_member_count"] = merged["concept_member_count"].fillna(pd.to_numeric(merged["concept_count"], errors="coerce")).fillna(0.0)
    merged["concept_leader_return_1d"] = merged["concept_leader_return_1d"].fillna(0.0)
    merged["stock_vs_concept_leader_1d"] = np.where(
        merged["concept_member_count"] > 0,
        pd.to_numeric(merged["return_1d"], errors="coerce") - pd.to_numeric(merged["concept_leader_return_1d"], errors="coerce"),
        0.0,
    )
    for feature_name in [
        "concept_exposure_score",
        "concept_breadth_pos_1d",
        "concept_breadth_pos_5d",
        "concept_return_mean_1d",
        "concept_return_mean_5d",
        "concept_net_mf_mean_1d",
        "concept_net_mf_mean_5d",
    ]:
        merged[feature_name] = merged[feature_name].fillna(0.0)
    return merged


def _missing_counts_for_columns(feature_df: pd.DataFrame, required_columns: list[str], *, count_absent_as_missing: bool = True) -> dict[str, int]:
    missing_counts: dict[str, int] = {}
    total_rows = int(len(feature_df))
    for column_name in required_columns:
        if column_name not in feature_df.columns:
            missing_counts[column_name] = total_rows if count_absent_as_missing else 0
            continue
        numeric_values = pd.to_numeric(feature_df[column_name], errors="coerce")
        missing_counts[column_name] = int((~np.isfinite(numeric_values.to_numpy(dtype=float))).sum())
    return missing_counts


def _top_missing_columns(missing_counts: dict[str, int], limit: int = 5) -> list[dict[str, int | str]]:
    ranked = [
        {"column": column_name, "missing_count": int(missing_count)}
        for column_name, missing_count in sorted(missing_counts.items(), key=lambda item: (-item[1], item[0]))
        if missing_count > 0
    ]
    return ranked[:limit]


def _zero_turnover_fill_count(df: pd.DataFrame) -> int:
    if df.empty or "volume" not in df.columns:
        return 0
    volume_missing = pd.to_numeric(df["volume"], errors="coerce").isna()
    if "amount" in df.columns:
        amount_missing = pd.to_numeric(df["amount"], errors="coerce").isna()
    else:
        amount_missing = pd.Series(True, index=df.index)
    return int((volume_missing & amount_missing).sum())


def _assign_split(date_value: pd.Timestamp, config: ExperimentConfig) -> str | None:
    train_end = pd.Timestamp(config.splits.train_end)
    val_end = pd.Timestamp(config.splits.val_end)
    test_end = pd.Timestamp(config.splits.test_end)
    if date_value <= train_end:
        return "train"
    if date_value <= val_end:
        return "val"
    if date_value <= test_end:
        return "test"
    return None


def _build_sequences(feature_df: pd.DataFrame, config: ExperimentConfig, diagnostics: dict[str, object] | None = None) -> DatasetBundle:
    feature_cols = config.experiment.feature_cols
    seq_len = config.model.seq_len
    min_seq_len = int(config.model.min_seq_len or seq_len)
    selected_stock_codes = sorted(feature_df["stock_code"].astype(str).dropna().unique().tolist())
    sequence_filtering = Counter()
    rows: dict[str, list] = {
        "train_x": [], "train_y": [], "train_meta": [],
        "val_x": [], "val_y": [], "val_meta": [],
        "test_x": [], "test_y": [], "test_meta": [],
    }

    for stock_code, group in feature_df.groupby("stock_code"):
        group = group.sort_values("date").reset_index(drop=True)
        stock_fold_id = _stock_fold_id(stock_code, int(config.experiment.stock_cv_total_folds)) if config.experiment.stock_cv_total_folds is not None else None
        values = group[feature_cols].to_numpy(dtype=np.float32)
        target_arrays = {
            name: group[column].to_numpy(dtype=np.float32)
            for name, column in TARGET_COLUMNS.items()
        }
        future_returns = group["future_return_5d"].to_numpy(dtype=np.float32)
        dates = group["date"].to_list()

        for idx in range(min_seq_len - 1, len(group)):
            sequence_filtering["candidate_windows"] += 1
            current_date = pd.Timestamp(dates[idx])
            split = _assign_split(current_date, config)
            if split is None:
                sequence_filtering["skipped_outside_split"] += 1
                continue
            if not _split_allows_stock(split, stock_code, config):
                sequence_filtering["skipped_stock_fold_filter"] += 1
                continue
            start_idx = max(0, idx - seq_len + 1)
            window = values[start_idx: idx + 1]
            if not np.isfinite(window).all():
                sequence_filtering["skipped_nonfinite_window"] += 1
                continue
            target_values = {name: float(array[idx]) for name, array in target_arrays.items()}
            if not all(np.isfinite(value) for value in target_values.values()):
                sequence_filtering["skipped_nonfinite_target"] += 1
                continue
            rows[f"{split}_x"].append(window)
            rows[f"{split}_y"].append(target_values)
            rows[f"{split}_meta"].append({
                "stock_code": stock_code,
                "date": str(current_date.date()),
                "layer": group.loc[idx, "layer"] if "layer" in group.columns else None,
                "future_return": float(future_returns[idx]),
                "stock_fold_id": stock_fold_id,
            })
            sequence_filtering["accepted_windows"] += 1

    def make_dataset(split: str) -> SequenceDataset:
        x_rows = rows[f"{split}_x"]
        y_rows = rows[f"{split}_y"]
        meta_rows = rows[f"{split}_meta"]
        if not x_rows:
            empty_targets = {name: np.empty((0,), dtype=np.float32) for name in TARGET_COLUMNS}
            return SequenceDataset(
                [],
                empty_targets,
                pd.DataFrame(columns=["stock_code", "date", "layer", "future_return", "stock_fold_id"]),
            )
        stacked_targets = {
            name: np.array([row[name] for row in y_rows], dtype=np.float32)
            for name in TARGET_COLUMNS
        }
        return SequenceDataset(x_rows, stacked_targets, pd.DataFrame(meta_rows))

    diagnostics_payload = dict(diagnostics or {})
    diagnostics_payload["sequence_filtering"] = {
        key: int(value)
        for key, value in {
            **sequence_filtering,
            "train_samples": len(rows["train_x"]),
            "val_samples": len(rows["val_x"]),
            "test_samples": len(rows["test_x"]),
        }.items()
    }

    return DatasetBundle(
        train=make_dataset("train"),
        val=make_dataset("val"),
        test=make_dataset("test"),
        input_dim=len(feature_cols),
        selected_stock_codes=selected_stock_codes,
        diagnostics=diagnostics_payload,
    )


def build_datasets(config: ExperimentConfig) -> DatasetBundle:
    market_caps = _latest_market_caps(config)
    manual_concept_map = _load_manual_concept_labels(config)
    role_vector_df = _load_role_vector_frame(config)
    frames: list[pd.DataFrame] = []
    required_columns = [
        *config.experiment.feature_cols,
        "future_return_5d",
        "target_up_5d",
        "target_take_profit_45d",
        "target_stop_loss_45d",
        "max_gain_45d",
        "max_drawdown_45d",
    ]
    pre_regime_required_columns = [
        column_name for column_name in required_columns
        if column_name not in {
            "regime_breadth_10d",
            "regime_breadth_20d",
            "regime_return_mean_10d",
            "regime_return_vol_10d",
            "regime_turnover_ratio_10_20",
            "regime_net_mf_mean_10d",
            "industry_breadth_pos_1d",
            "industry_breadth_pos_5d",
            "industry_return_mean_1d",
            "industry_return_std_1d",
            "industry_return_mean_5d",
            "industry_net_mf_mean_1d",
            "industry_net_mf_std_1d",
            "industry_net_mf_mean_5d",
            "industry_leader_return_1d",
            "stock_vs_industry_leader_1d",
            "stock_vs_industry_mean_1d",
            "stock_vs_industry_mean_5d",
            "stock_vs_industry_net_mf_1d",
            "stock_vs_industry_net_mf_5d",
            "industry_return_zscore_1d",
            "industry_net_mf_zscore_1d",
            "industry_return_rank_pct_1d",
            "industry_net_mf_rank_pct_1d",
            "industry_return_rank_pct_5d",
            "industry_net_mf_rank_pct_5d",
            "industry_return_rank_drift_5d",
            "industry_net_mf_rank_drift_5d",
            "concept_member_count",
            "concept_exposure_score",
            "concept_breadth_pos_1d",
            "concept_breadth_pos_5d",
            "concept_return_mean_1d",
            "concept_return_mean_5d",
            "concept_net_mf_mean_1d",
            "concept_net_mf_mean_5d",
            "concept_leader_return_1d",
            "stock_vs_concept_leader_1d",
        }
    ]
    missing_column_totals: Counter[str] = Counter()
    per_stock_quality: list[dict[str, object]] = []
    raw_row_count = 0
    cleaned_row_count = 0
    zero_turnover_fill_count = 0

    for _, row in market_caps.iterrows():
        stock_code = str(row["stock_code"])
        market_cap_yi = float(row["market_cap_yi"])
        trade_df = _load_trade_data_with_concepts(
            config.databases.trade_db,
            stock_code,
            manual_concept_map=manual_concept_map,
            concept_label_source=config.experiment.concept_label_source,
        )
        if trade_df.empty:
            continue
        feature_df = build_layer_features(trade_df, market_cap_yi)
        feature_df = _add_time_features(feature_df)
        feature_df = add_multitask_targets(
            feature_df,
            short_horizon=config.multitask.short_horizon,
            short_threshold=config.multitask.short_threshold,
            hold_days=config.multitask.hold_days,
            take_profit_threshold=config.multitask.take_profit_threshold,
            stop_loss_threshold=config.multitask.stop_loss_threshold,
            exit_label_mode=config.multitask.exit_label_mode,
        )
        feature_df = _finalize_feature_frame(feature_df, stock_code, market_cap_yi, config)
        feature_df = _merge_role_vector_features(feature_df, role_vector_df)
        feature_df = _add_requested_role_interaction_features(feature_df, config.experiment.feature_cols)
        before_drop_count = int(len(feature_df))
        stock_zero_turnover_fill_count = _zero_turnover_fill_count(trade_df)
        missing_counts = _missing_counts_for_columns(feature_df, pre_regime_required_columns, count_absent_as_missing=False)
        cleaned_feature_df = feature_df.dropna(subset=pre_regime_required_columns).reset_index(drop=True)
        after_drop_count = int(len(cleaned_feature_df))
        raw_row_count += before_drop_count
        cleaned_row_count += after_drop_count
        zero_turnover_fill_count += stock_zero_turnover_fill_count
        missing_column_totals.update(missing_counts)
        per_stock_quality.append(
            {
                "stock_code": normalize_stock_code(stock_code),
                "rows_before_dropna": before_drop_count,
                "rows_after_dropna": after_drop_count,
                "dropped_rows": before_drop_count - after_drop_count,
                "retained_ratio": float(after_drop_count / before_drop_count) if before_drop_count > 0 else 0.0,
                "zero_turnover_fill_count": stock_zero_turnover_fill_count,
                "zero_turnover_fill_ratio": float(stock_zero_turnover_fill_count / before_drop_count) if before_drop_count > 0 else 0.0,
                "top_missing_columns": ", ".join(
                    f"{item['column']}:{item['missing_count']}" for item in _top_missing_columns(missing_counts)
                ),
            }
        )
        frames.append(cleaned_feature_df)

    if not frames:
        raise RuntimeError("未能从股票池构造任何有效特征。")

    combined = pd.concat(frames, ignore_index=True)
    combined_before_regime_drop_count = int(len(combined))
    combined = _add_regime_context_features(combined)
    combined = _add_industry_sentiment_features(combined)
    combined = _add_concept_sentiment_features(combined, config)
    post_regime_missing_counts = _missing_counts_for_columns(combined, required_columns)
    combined = combined.dropna(
        subset=required_columns
    ).reset_index(drop=True)
    combined_after_regime_drop_count = int(len(combined))

    top_missing = _top_missing_columns(dict(missing_column_totals), limit=8)
    worst_stocks = sorted(per_stock_quality, key=lambda item: (-int(item["dropped_rows"]), str(item["stock_code"])))[:8]
    diagnostics = {
        "raw_row_count": int(raw_row_count),
        "cleaned_row_count": int(cleaned_row_count),
        "combined_before_regime_drop_count": combined_before_regime_drop_count,
        "combined_after_regime_drop_count": combined_after_regime_drop_count,
        "retained_ratio": float(cleaned_row_count / raw_row_count) if raw_row_count > 0 else 0.0,
        "zero_turnover_fill_count": int(zero_turnover_fill_count),
        "zero_turnover_fill_ratio": float(zero_turnover_fill_count / raw_row_count) if raw_row_count > 0 else 0.0,
        "top_missing_columns": top_missing,
        "post_regime_top_missing_columns": _top_missing_columns(post_regime_missing_counts, limit=8),
        "worst_stocks": worst_stocks,
        "per_stock": per_stock_quality,
    }
    logger.info(
        "data quality raw_rows=%s cleaned_rows=%s retained_ratio=%.4f zero_turnover_fill_ratio=%.4f top_missing=%s",
        raw_row_count,
        cleaned_row_count,
        diagnostics["retained_ratio"],
        diagnostics["zero_turnover_fill_ratio"],
        ", ".join(f"{item['column']}:{item['missing_count']}" for item in top_missing) or "none",
    )
    if worst_stocks:
        logger.warning(
            "worst stock retention=%s",
            "; ".join(
                f"{item['stock_code']} kept={item['rows_after_dropna']}/{item['rows_before_dropna']} zero_turnover_fill={item['zero_turnover_fill_count']} top_missing={item['top_missing_columns']}"
                for item in worst_stocks[:5]
            ),
        )
    return _build_sequences(combined, config, diagnostics=diagnostics)


def build_feature_frame(config: ExperimentConfig) -> pd.DataFrame:
    market_caps = _latest_market_caps(config)
    manual_concept_map = _load_manual_concept_labels(config)
    role_vector_df = _load_role_vector_frame(config)
    frames: list[pd.DataFrame] = []

    for _, row in market_caps.iterrows():
        stock_code = str(row["stock_code"])
        market_cap_yi = float(row["market_cap_yi"])
        trade_df = _load_trade_data_with_concepts(
            config.databases.trade_db,
            stock_code,
            manual_concept_map=manual_concept_map,
            concept_label_source=config.experiment.concept_label_source,
        )
        if trade_df.empty:
            continue
        feature_df = build_layer_features(trade_df, market_cap_yi)
        feature_df = _add_time_features(feature_df)
        feature_df = add_multitask_targets(
            feature_df,
            short_horizon=config.multitask.short_horizon,
            short_threshold=config.multitask.short_threshold,
            hold_days=config.multitask.hold_days,
            take_profit_threshold=config.multitask.take_profit_threshold,
            stop_loss_threshold=config.multitask.stop_loss_threshold,
            exit_label_mode=config.multitask.exit_label_mode,
        )
        feature_df = _finalize_feature_frame(feature_df, stock_code, market_cap_yi, config)
        feature_df = _merge_role_vector_features(feature_df, role_vector_df)
        feature_df = _add_requested_role_interaction_features(feature_df, config.experiment.feature_cols)
        feature_df = feature_df.reset_index(drop=True)
        frames.append(feature_df)

    if not frames:
        raise RuntimeError("未能从股票池构造任何有效特征。")

    combined = _add_regime_context_features(pd.concat(frames, ignore_index=True))
    combined = _add_industry_sentiment_features(combined)
    return _add_concept_sentiment_features(combined, config)