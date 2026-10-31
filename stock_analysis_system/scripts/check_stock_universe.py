# -*- coding: utf-8 -*-
"""确认股票清单：含退市股的完整列表。

问题：
  只拉 stock_data 现有的 5879 只，会漏掉 2024-10 之前退市的股票，
  造成幸存者偏差。

tushare stock_basic 支持 list_status:
  L = 上市
  D = 退市
  P = 暂停上市

用法：
  python scripts/check_stock_universe.py
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


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


def load_token():
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return os.environ.get("TUSHARE_TOKEN")


def main():
    db = resolve_db()
    con = sqlite3.connect(db)

    token = load_token()
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    print("=" * 88)
    print("股票清单确认")
    print("=" * 88)

    frames = {}
    for st in ["L", "D", "P"]:
        try:
            df = pro.stock_basic(
                exchange="", list_status=st,
                fields="ts_code,symbol,name,area,industry,market,"
                       "list_date,delist_date,list_status")
            frames[st] = df
            print(f"\nlist_status={st}: {len(df)} 只")
            if len(df):
                print(f"  板块分布: {df['market'].value_counts().to_dict()}")
        except Exception as e:
            print(f"\nlist_status={st}: 失败 {e}")
            frames[st] = pd.DataFrame()

    all_df = pd.concat([f for f in frames.values() if len(f)],
                       ignore_index=True)
    print(f"\n合计 {len(all_df)} 只")

    # 与本地对比
    local = set(r[0] for r in con.execute(
        "SELECT DISTINCT stock_code FROM stock_data").fetchall())
    local = {c for c in local if c is not None}
    all_df["code6"] = all_df["symbol"].astype(str).str.zfill(6)
    remote = set(all_df["code6"])

    print(f"\n本地有   : {len(local):,}")
    print(f"tushare  : {len(remote):,}")
    print(f"本地有 tushare 无: {len(local - remote)}")
    if local - remote:
        print(f"  样例: {sorted(local - remote)[:10]}")
    print(f"tushare 有 本地无: {len(remote - local):,}")

    missing = all_df[~all_df["code6"].isin(local)]
    if len(missing):
        print(f"\n本地缺失的 {len(missing)} 只按状态:")
        print(f"  {missing['list_status'].value_counts().to_dict()}")
        print(f"  按市场: {missing['market'].value_counts().to_dict()}")
        # 退市股
        d = missing[missing["list_status"] == "D"]
        if len(d):
            print(f"\n  退市股 {len(d)} 只，退市日范围: "
                  f"{d['delist_date'].min()} ~ {d['delist_date'].max()}")
            print(f"  样例:\n{d[['code6','name','market','list_date','delist_date']].head(8).to_string(index=False)}")

    # 2021-11 之后上市的
    all_df["list_date"] = pd.to_numeric(all_df["list_date"], errors="coerce")
    recent = all_df[all_df["list_date"] >= 20211101]
    print(f"\n2021-11 之后上市: {len(recent)} 只")

    con.close()

    # 保存清单供后续使用
    out = ROOT / "results" / "stock_universe.csv"
    import os as _os
    _os.makedirs(out.parent, exist_ok=True)
    all_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n清单已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
