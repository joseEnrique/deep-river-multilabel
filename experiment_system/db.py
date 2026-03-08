"""
db.py — SQLite state management for the experiment tracking system.

Each experiment has a unique exp_id (SHA256 of its JSON config).
Status lifecycle: pending → running → done / failed

On restart: done experiments are skipped; failed/interrupted ones are retried.
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd


DB_PATH = Path(__file__).parent / "experiments.db"


def _connect(db_path=None):
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create the experiments table if it does not exist."""
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id       TEXT PRIMARY KEY,
                exp_name     TEXT NOT NULL,
                config       TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                started_at   TEXT,
                finished_at  TEXT,
                result_json  TEXT,
                error        TEXT
            )
        """)
        conn.commit()


def make_exp_id(config: dict) -> str:
    """Deterministic SHA256 ID from a config dict."""
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def make_exp_name(config: dict) -> str:
    """
    Build a human-readable experiment name, e.g.:
        LSTM_ai4i_ph1_h128_nl2_w200_lr1e-3_adam_BCE
    """
    arch      = config.get("architecture", "LSTM")
    dataset   = config.get("dataset", "ai4i")
    loss_type = config.get("loss", {}).get("type", "?")
    ph        = config.get("past_history", "?")
    h         = config.get("hidden_dim", "?")
    nl        = config.get("num_layers", "?")
    lr        = config.get("lr", "?")
    w         = config.get("window_size", "?")
    opt       = config.get("optimizer", "adam")
    norm      = config.get("normalization", "none")
    bidir     = "_bidir" if config.get("bidirectional") else ""
    parts = [arch, dataset, f"ph{ph}", f"h{h}", f"nl{nl}", f"w{w}", f"lr{lr}", opt, norm, loss_type]
    return "".join(str(p) for p in ["_".join(parts), bidir])


def register_experiments(configs: list[dict], db_path=None):
    """
    Insert experiments as 'pending'. Ignores if exp_id already exists
    (idempotent — safe to call on every run).
    Returns list of (exp_id, exp_name, config) tuples.
    """
    with _connect(db_path) as conn:
        registered = []
        for cfg in configs:
            exp_id   = make_exp_id(cfg)
            exp_name = make_exp_name(cfg)
            conn.execute(
                "INSERT OR IGNORE INTO experiments (exp_id, exp_name, config, status) VALUES (?, ?, ?, 'pending')",
                (exp_id, exp_name, json.dumps(cfg, sort_keys=True))
            )
            registered.append((exp_id, exp_name, cfg))
        conn.commit()
    return registered


def get_pending(db_path=None) -> list[tuple[str, str, dict]]:
    """Return (exp_id, exp_name, config) for all pending or failed experiments."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT exp_id, exp_name, config FROM experiments WHERE status IN ('pending', 'failed') ORDER BY rowid"
        ).fetchall()
    return [(r["exp_id"], r["exp_name"], json.loads(r["config"])) for r in rows]


def get_summary(db_path=None) -> dict:
    """Return counts per status."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as n FROM experiments GROUP BY status"
        ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def claim(exp_id: str, db_path=None):
    """Mark experiment as running."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE experiments SET status='running', started_at=? WHERE exp_id=?",
            (datetime.now().isoformat(), exp_id)
        )
        conn.commit()


def mark_done(exp_id: str, result: dict, db_path=None):
    """Mark experiment as done and store its result JSON."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE experiments SET status='done', finished_at=?, result_json=? WHERE exp_id=?",
            (datetime.now().isoformat(), json.dumps(result), exp_id)
        )
        conn.commit()


def mark_failed(exp_id: str, error: str, db_path=None):
    """Mark experiment as failed with error message."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE experiments SET status='failed', finished_at=?, error=? WHERE exp_id=?",
            (datetime.now().isoformat(), str(error)[:2000], exp_id)
        )
        conn.commit()


def export_results(output_path: str, db_path=None) -> pd.DataFrame:
    """Export all done experiments to a CSV and return as DataFrame."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT exp_id, exp_name, config, result_json, started_at, finished_at FROM experiments WHERE status='done'"
        ).fetchall()

    records = []
    for r in rows:
        cfg = json.loads(r["config"])
        result = json.loads(r["result_json"]) if r["result_json"] else {}
        record = {"exp_name": r["exp_name"], "exp_id": r["exp_id"],
                  **cfg, **result,
                  "started_at": r["started_at"], "finished_at": r["finished_at"]}
        records.append(record)

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
