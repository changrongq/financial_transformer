from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
import pandas as pd


@contextmanager
def sqlite_conn(db_path: str | Path):
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def ensure_parent_dir(db_path: str | Path) -> None:
    Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)


def read_sql(query: str, db_path: str | Path, params: tuple | list | None = None) -> pd.DataFrame:
    with sqlite_conn(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def write_frame(df: pd.DataFrame, table_name: str, db_path: str | Path, if_exists: str = "append") -> None:
    ensure_parent_dir(db_path)
    with sqlite_conn(db_path) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
