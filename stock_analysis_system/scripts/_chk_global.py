import sqlite3

import pandas as pd

c = sqlite3.connect(r"D:\qcr\Stock\stock_history.db")
try:
    df = pd.read_sql_query(
        "SELECT ts_code, market, COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b "
        "FROM global_daily GROUP BY ts_code ORDER BY market, n DESC", c)
    if df.empty:
        print("global_daily 表为空（拉取尚未完成）")
    else:
        df["years"] = (df["n"] / 244).round(1)
        print(df.to_string(index=False))
except Exception as e:
    print("没有 global_daily 表:", e)
c.close()
