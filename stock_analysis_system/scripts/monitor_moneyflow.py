# -*- coding: utf-8 -*-
"""监控资金流补齐进度。"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())

con = sqlite3.connect(DB, timeout=30)
try:
    r = con.execute("SELECT COUNT(*) FROM moneyflow_backfill_progress").fetchone()
    n = r[0] if r else 0
    prog = con.execute("""
        SELECT status, COUNT(*) FROM moneyflow_backfill_progress
        GROUP BY status
    """).fetchall()
    tot = con.execute("SELECT COUNT(*) FROM stock_moneyflow").fetchone()[0]
    days = con.execute(
        "SELECT COUNT(DISTINCT 日期) FROM stock_moneyflow "
        "WHERE 日期 < '2025-01-02'").fetchone()[0]
    print(f"进度表记录: {n} 天  状态: {dict(prog)}")
    print(f"stock_moneyflow 总行: {tot:,}")
    print(f"2025 前已补: {days} / 771 天")
finally:
    con.close()
