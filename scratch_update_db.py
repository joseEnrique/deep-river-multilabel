import sqlite3
import json
import sys
from pathlib import Path

# Add project root to sys.path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from experiment_system.db import make_exp_name, DB_PATH, _connect

def update_db():
    print(f"Updating DB at {DB_PATH}")
    with _connect() as conn:
        rows = conn.execute("SELECT exp_id, config FROM experiments").fetchall()
        
        updates = []
        for row in rows:
            exp_id = row["exp_id"]
            config = json.loads(row["config"])
            new_name = make_exp_name(config)
            updates.append((new_name, exp_id))
            
        print(f"Found {len(updates)} experiments to update.")
        
        conn.executemany("UPDATE experiments SET exp_name = ? WHERE exp_id = ?", updates)
        conn.commit()
    print("Database updated successfully.")

if __name__ == "__main__":
    update_db()
