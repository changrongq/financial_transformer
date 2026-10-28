import sqlite3
from pathlib import Path

db = Path("d:/qcr/Stock/stock_analysis_system/stock_history.db")
c = sqlite3.connect(str(db))

rows = c.execute("PRAGMA table_list").fetchall()
print("Tables via PRAGMA:")
for r in rows:
    print(f"  {r}")

# Also try info
try:
    rows2 = c.execute("SELECT * FROM sqlite_master WHERE type='table'").fetchall()
    print("\nTables via sqlite_master:")
    for r in rows2:
        print(f"  {r}")
except Exception as e:
    print(f"sqlite_master failed: {e}")

c.close()
