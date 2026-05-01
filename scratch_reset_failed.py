import sqlite3
from pathlib import Path

db_path = Path("experiment_system/experiments.db")

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute("UPDATE experiments SET status = 'pending' WHERE status = 'failed'")
    updated = cursor.rowcount
    conn.commit()
    print(f"✅ Se han cambiado {updated} experimentos de estado 'failed' a 'pending'.")
