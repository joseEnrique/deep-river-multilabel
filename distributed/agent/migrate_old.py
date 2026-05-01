#!/usr/bin/env python
"""
migrate_old.py — import legacy experiment_system data into the centralized backend.

The old SQLite (experiment_system/experiments.db) only stores final metrics in
`result_json`; per-step checkpoints live in CSV files at
`experiment_system/results/<exp_name>_checkpoints.csv`. Either or both may be
missing — this script handles all three cases:

    1. SQLite + CSV checkpoints  → full import (final metrics + curve)
    2. SQLite only               → final metrics only, empty checkpoint list
    3. SQLite missing some rows  → just imports what exists

It's idempotent: 409 from the backend or "already done" rows are skipped.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from naming import make_exp_name
from api_client import BackendClient, BackendError
from local_db import LocalDB

# Final-metric keys produced by the old runner (see experiment_system/runner.py).
METRIC_KEYS = (
    "subset_acc", "hamm_loss",
    "examp_f1", "examp_prec", "examp_rec",
    "micro_f1", "micro_prec", "micro_rec",
    "macro_f1", "macro_prec", "macro_rec",
)


def read_old_db(path: Path, status: str) -> list[dict]:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT exp_name, config, status, started_at, finished_at, result_json, error "
        "FROM experiments WHERE status=?",
        (status,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def read_csv_checkpoints(csv_path: Path) -> list[dict]:
    """Parse experiment_system/results/<exp>_checkpoints.csv into our shape.
    Returns [] if the file doesn't exist or is malformed."""
    if not csv_path.exists():
        return []
    out = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    step = int(row["step"])
                except (KeyError, ValueError):
                    continue
                try:
                    elapsed = float(row.get("elapsed_s") or 0)
                except ValueError:
                    elapsed = 0.0
                metrics = {}
                for k in METRIC_KEYS:
                    v = row.get(k)
                    if v is None or v == "":
                        continue
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        continue
                out.append({"step": step, "elapsed_s": elapsed, "metrics": metrics})
    except Exception as e:
        print(f"  warn: could not parse {csv_path.name}: {e}", file=sys.stderr)
        return []
    return out


def find_checkpoints(ckpt_dir: Path | None, *candidate_names: str) -> list[dict]:
    """Try several name conventions; return the first non-empty match."""
    if ckpt_dir is None:
        return []
    for name in candidate_names:
        if not name:
            continue
        ckpts = read_csv_checkpoints(ckpt_dir / f"{name}_checkpoints.csv")
        if ckpts:
            return ckpts
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Migrate legacy experiment_system data into the centralized backend.")
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="Agent config (uses backend_url, api_key, dataset).")
    ap.add_argument("--source-db", required=True,
                    help="Path to the old experiment_system/experiments.db.")
    ap.add_argument("--checkpoints-dir", default=None,
                    help="Optional: path to old results/ folder. If omitted, "
                         "experiments are imported with no per-step checkpoints.")
    ap.add_argument("--dataset", default=None,
                    help="Override target dataset (default: from config).")
    ap.add_argument("--status", default="done",
                    choices=("done", "pending", "failed", "running"),
                    help="Which old-status rows to import (default: done).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just print what would be imported, no HTTP calls.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = args.dataset or cfg["dataset"]
    backend_url = cfg["backend_url"]
    api_key = cfg.get("api_key") or os.environ.get("BACKEND_API_KEY") or None
    local_db_path = str(cfg.get("local_db_path") or (HERE / f"local_{dataset}.db"))

    client = BackendClient(base_url=backend_url, dataset=dataset,
                           agent_id="migrator", api_key=api_key)
    local = LocalDB(local_db_path)

    src = Path(args.source_db).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: source DB not found: {src}", file=sys.stderr)
        sys.exit(1)

    ckpt_dir = Path(args.checkpoints_dir).expanduser().resolve() if args.checkpoints_dir else None
    if ckpt_dir and not ckpt_dir.exists():
        print(f"ERROR: checkpoints dir not found: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    rows = read_old_db(src, args.status)
    print(f"source: {src}")
    print(f"  → {len(rows)} experiments with status='{args.status}'")
    print(f"  → checkpoints dir: {ckpt_dir if ckpt_dir else '(none — final metrics only)'}")
    print(f"target: {backend_url} dataset={dataset}")
    print()

    created = existed_remote = finished = skipped_done = with_ckpts = errors = 0

    for r in rows:
        cfg_dict = json.loads(r["config"])
        old_name = r.get("exp_name") or ""
        new_name = make_exp_name(cfg_dict)
        arch = cfg_dict.get("architecture", "LSTM")

        result = {}
        if r.get("result_json"):
            try:
                result = json.loads(r["result_json"])
            except json.JSONDecodeError:
                pass
        final_metrics = {k: float(result[k]) for k in METRIC_KEYS
                         if k in result and result[k] is not None}
        duration = float(result.get("duration_s") or 0)

        ckpts = find_checkpoints(ckpt_dir, old_name, new_name)
        if ckpts:
            with_ckpts += 1

        if args.dry_run:
            print(f"[dry] {new_name}  metrics={len(final_metrics)} ckpts={len(ckpts)}")
            continue

        # Skip if already done in our local mirror — avoids inserting duplicate
        # checkpoint rows on a re-run of this script.
        existing_local = local.get(new_name)
        if existing_local and existing_local.get("status") == "done":
            skipped_done += 1
            continue

        # 1) create on backend (idempotent: 409 → already existed, that's fine).
        local.upsert_pending(new_name, arch, dataset, cfg_dict)
        try:
            ok, _ = client.create(new_name, arch, cfg_dict)
            if ok:
                created += 1
            else:
                existed_remote += 1
        except BackendError as e:
            print(f"  ERR create {new_name}: {e}", file=sys.stderr)
            errors += 1
            continue

        # 2) finish — only metrics (and optionally checkpoints) go in the request.
        if args.status != "done":
            # Importing non-done rows: just registered, don't finish.
            continue
        try:
            client.finish(new_name, final_metrics, duration, checkpoints=ckpts or None)
            local.save_completed(new_name, final_metrics, duration, ckpts)
            finished += 1
        except BackendError as e:
            if e.status == 409:
                # Already done remotely — bring local up to speed and move on.
                local.save_completed(new_name, final_metrics, duration, ckpts)
                skipped_done += 1
            else:
                print(f"  ERR finish {new_name}: {e}", file=sys.stderr)
                errors += 1

    print()
    print(f"created={created}  existed_remote={existed_remote}  "
          f"finished={finished}  already_done={skipped_done}  "
          f"with_checkpoints={with_ckpts}  errors={errors}")


if __name__ == "__main__":
    main()
