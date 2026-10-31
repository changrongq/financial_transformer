# -*- coding: utf-8 -*-
"""监控补充进度（可反复运行）。"""
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = str((ROOT.parent / "stock_history.db").resolve())


def snapshot():
    con = sqlite3.connect(DB, timeout=30)
    try:
        n_ok = con.execute(
            "SELECT COUNT(*) FROM stock_pre_close_progress WHERE status='ok'"
        ).fetchone()[0]
        n_bad = con.execute(
            "SELECT COUNT(*) FROM stock_pre_close_progress WHERE status!='ok'"
        ).fetchone()[0]
        tot, filled = con.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pre_close IS NOT NULL THEN 1 ELSE 0 END)
            FROM stock_data""").fetchone()
        return n_ok, n_bad, tot, (filled or 0)
    finally:
        con.close()


if len(sys.argv) > 1 and sys.argv[1] == "--watch":
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    last = None
    for _ in range(40):
        n_ok, n_bad, tot, filled = snapshot()
        cov = filled / tot * 100
        if last is not None:
            delta = n_ok - last
            speed = delta / interval
            left = (5879 - n_ok) / speed / 60 if speed > 0 else float("inf")
            print(f"  ok={n_ok} bad={n_bad} 覆盖 {cov:.2f}%  "
                  f"速度 {speed*60:.0f}只/分  剩余约 {left:.1f}分", flush=True)
        else:
            print(f"  ok={n_ok} bad={n_bad} 覆盖 {cov:.2f}%", flush=True)
        last = n_ok
        if n_ok >= 5860:
            print("  接近完成", flush=True)
            break
        time.sleep(interval)
else:
    n_ok, n_bad, tot, filled = snapshot()
    print(f"已完成 {n_ok:,} 只  非ok {n_bad}  "
          f"覆盖 {filled/tot*100:.2f}% ({filled:,}/{tot:,})")
