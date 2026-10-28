"""日历因子验证：旅游↔节假日、农业↔春耕、电力/家电↔夏季。

方法：
  - 读 historical_daily（目标行业股票 2019-2026 日线）
  - 读 index_daily_sh（上证指数基准）
  - 统计各行业组在「目标月份」vs「其他月份」的超额收益（相对指数）
  - 检查季节性规律是否跨年稳定

用法：
  python scripts/review_calendar_factors.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 每个行业组的"目标月份"（季节效应假设）
# 旅游：国庆(10月)前是9月底；春节前是1-2月 → 关注1、2、9、10月
# 农业：春耕/两会 → 2、3、4月
# 电力/家电：夏季用电/空调 → 6、7、8月
GROUP_TARGET_MONTHS = {
    "旅游": [1, 2, 9, 10],
    "农业": [2, 3, 4],
    "电力": [6, 7, 8],
    "家电": [6, 7, 8],
}

GROUP_KEYWORDS = {
    "旅游": ["旅游景点", "旅游服务", "酒店餐饮"],
    "农业": ["农业综合", "农药化肥"],
    "电力": ["火力发电", "水力发电", "其他发电"],
    "家电": ["家用电器", "白色家电"],
}


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


def load_group_codes(db_path: str) -> dict[str, set[str]]:
    conn = sqlite3.connect(db_path)
    bi = pd.read_sql_query("SELECT stock_code, 行业 FROM stock_basic_info", conn)
    conn.close()
    bi["行业"] = bi["行业"].astype(str)
    result = {}
    for group, kws in GROUP_KEYWORDS.items():
        mask = bi["行业"].str.contains("|".join(kws), na=False)
        result[group] = set(bi[mask]["stock_code"].astype(str).str.zfill(6).tolist())
    return result


def main() -> None:
    db_path = _resolve_db()
    group_codes = load_group_codes(db_path)

    # 读历史日线
    conn = sqlite3.connect(db_path)
    daily = pd.read_sql_query("SELECT stock_code, date, close FROM historical_daily", conn)
    index = pd.read_sql_query("SELECT date, pct_chg FROM index_daily_sh", conn)
    conn.close()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    index["date"] = pd.to_datetime(index["date"])
    index["pct_chg"] = pd.to_numeric(index["pct_chg"], errors="coerce")

    # 每只股票每日收益
    daily = daily.dropna(subset=["close"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    daily["ret"] = daily.groupby("stock_code")["close"].pct_change()
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year

    # 合并指数收益
    daily = daily.merge(index.rename(columns={"pct_chg": "idx_pct"}), on="date", how="left")
    daily["excess"] = daily["ret"] - daily["idx_pct"] / 100.0  # 个股日收益 - 指数日收益

    daily = daily.dropna(subset=["ret", "excess"])

    print("=" * 70)
    print("日历因子验证：目标月份 vs 其他月份 的行业超额收益（相对上证指数）")
    print("=" * 70)

    for group, codes in group_codes.items():
        sub = daily[daily["stock_code"].isin(codes)]
        if sub.empty:
            print(f"\n【{group}】无数据")
            continue
        target_months = GROUP_TARGET_MONTHS[group]

        # 按月平均超额收益（跨年汇总）
        monthly = sub.groupby("month")["excess"].mean() * 100  # 转为百分比
        target_avg = monthly[monthly.index.isin(target_months)].mean()
        other_avg = monthly[~monthly.index.isin(target_months)].mean()
        n_stocks = len(codes)

        print(f"\n【{group}】{n_stocks} 只，目标月份 {target_months}")
        # 打印每月超额
        monthly_series = monthly.round(3)
        print("  各月超额收益%(1-12月):")
        print("  " + " ".join(f"{m}月:{monthly_series.get(m, 0):+.3f}" for m in range(1, 13)))
        print(f"  目标月份均值: {target_avg:+.3f}% / 其他月份均值: {other_avg:+.3f}%")
        print(f"  → 季节效应差异: {target_avg - other_avg:+.3f}%")

        # 按每年看目标月份是否稳定跑赢
        annual = sub[sub["month"].isin(target_months)].groupby("year")["excess"].mean() * 100
        other_annual = sub[~sub["month"].isin(target_months)].groupby("year")["excess"].mean() * 100
        print("  各年目标月份超额%(vs 其他月):")
        for y in sorted(set(annual.index) | set(other_annual.index)):
            t = annual.get(y, 0)
            o = other_annual.get(y, 0)
            print(f"    {y}: 目标月 {t:+.3f}% / 其他月 {o:+.3f}% / 差 {t-o:+.3f}%")


if __name__ == "__main__":
    main()
