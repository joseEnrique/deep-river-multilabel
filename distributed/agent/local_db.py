"""
local_db.py — local SQLite mirror of everything the agent reports to the backend.

Acts as a safety net: if the backend dies or the network drops, the agent still
has a complete record (configs, status transitions, every checkpoint, final
metrics, errors). Schema is intentionally close to experiment_system/db.py so
analysis scripts that already read SQLite keep working.

The backend is authoritative for cross-machine state (claim / pending lists);
the local DB is authoritative for "what did THIS agent observe".
"""

from __future__ import annotations
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    exp_name        TEXT PRIMARY KEY,
    architecture    TEXT,
    dataset         TEXT,
    config          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    agent_id        TEXT,
    device          TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    final_metrics   TEXT,
    duration_s      REAL,
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    exp_name     TEXT NOT NULL,
    step         INTEGER NOT NULL,
    elapsed_s    REAL NOT NULL,
    metrics_json TEXT NOT NULL,
    timestamp    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_exp ON checkpoints(exp_name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
"""


class LocalDB:
    """Thread-safe wrapper. Every agent process opens its own LocalDB instance."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as c:
            c.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    # ── Writes ───────────────────────────────────────────────────────────────

    def upsert_pending(self, exp_name: str, architecture: str, dataset: str,
                       config: dict[str, Any]) -> None:
        """Insert as pending if it doesn't exist; otherwise leave it alone."""
        now = self._now()
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT OR IGNORE INTO experiments
                   (exp_name, architecture, dataset, config, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (exp_name, architecture, dataset, json.dumps(config, sort_keys=True),
                 now, now),
            )

    def bulk_upsert_pending(self, items: list[tuple[str, str, str, dict[str, Any]]]) -> int:
        """Bulk INSERT OR IGNORE in a single transaction (executemany).
        items: list of (exp_name, architecture, dataset, config_dict).
        Returns the rowcount reported by SQLite (number of new rows)."""
        if not items:
            return 0
        now = self._now()
        rows = [
            (n, a, d, json.dumps(cfg, sort_keys=True), now, now)
            for (n, a, d, cfg) in items
        ]
        with self._lock, self._connect() as c:
            c.execute("BEGIN")
            cur = c.executemany(
                """INSERT OR IGNORE INTO experiments
                   (exp_name, architecture, dataset, config, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                rows,
            )
            inserted = cur.rowcount
            c.execute("COMMIT")
        return inserted

    def mark_running(self, exp_name: str, agent_id: str, device: str) -> None:
        now = self._now()
        with self._lock, self._connect() as c:
            c.execute(
                """UPDATE experiments
                   SET status='running', agent_id=?, device=?, started_at=?, updated_at=?, error=NULL
                   WHERE exp_name=?""",
                (agent_id, device, now, now, exp_name),
            )

    def append_checkpoint(self, exp_name: str, step: int, elapsed_s: float,
                          metrics: dict[str, float]) -> None:
        """Single-checkpoint write. Kept for compatibility but not used by the
        normal flow — the agent batches checkpoints and persists them all at
        finish() to avoid hitting the disk on every step."""
        now = self._now()
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT INTO checkpoints (exp_name, step, elapsed_s, metrics_json, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (exp_name, step, elapsed_s, json.dumps(metrics), now),
            )
            c.execute("UPDATE experiments SET updated_at=? WHERE exp_name=?",
                      (now, exp_name))

    def save_completed(self, exp_name: str, final_metrics: dict[str, float],
                       duration_s: float, checkpoints: list[dict]) -> None:
        """Single-transaction commit at the end of an experiment:
        bulk-insert all the checkpoints AND mark the experiment as done.
        One disk write per experiment instead of one per checkpoint."""
        now = self._now()
        rows = [
            (exp_name, int(cp["step"]), float(cp["elapsed_s"]),
             json.dumps(cp["metrics"]), now)
            for cp in (checkpoints or [])
        ]
        with self._lock, self._connect() as c:
            c.execute("BEGIN")
            if rows:
                c.executemany(
                    """INSERT INTO checkpoints
                       (exp_name, step, elapsed_s, metrics_json, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )
            c.execute(
                """UPDATE experiments
                   SET status='done', final_metrics=?, duration_s=?,
                       finished_at=?, updated_at=?, error=NULL
                   WHERE exp_name=?""",
                (json.dumps(final_metrics), duration_s, now, now, exp_name),
            )
            c.execute("COMMIT")

    def mark_done(self, exp_name: str, final_metrics: dict[str, float],
                  duration_s: float) -> None:
        now = self._now()
        with self._lock, self._connect() as c:
            c.execute(
                """UPDATE experiments
                   SET status='done', final_metrics=?, duration_s=?, finished_at=?, updated_at=?, error=NULL
                   WHERE exp_name=?""",
                (json.dumps(final_metrics), duration_s, now, now, exp_name),
            )

    def mark_failed(self, exp_name: str, error: str) -> None:
        now = self._now()
        with self._lock, self._connect() as c:
            c.execute(
                """UPDATE experiments
                   SET status='failed', error=?, finished_at=?, updated_at=?
                   WHERE exp_name=?""",
                (str(error)[:4000], now, now, exp_name),
            )

    # ── Reads ────────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, int]:
        with self._lock, self._connect() as c:
            rows = c.execute(
                "SELECT status, COUNT(*) AS n FROM experiments GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def get(self, exp_name: str) -> dict | None:
        with self._lock, self._connect() as c:
            row = c.execute("SELECT * FROM experiments WHERE exp_name=?",
                            (exp_name,)).fetchone()
        return dict(row) if row else None

    def list_status(self, status: str) -> list[dict]:
        with self._lock, self._connect() as c:
            rows = c.execute("SELECT * FROM experiments WHERE status=? ORDER BY exp_name",
                             (status,)).fetchall()
        return [dict(r) for r in rows]
