#!/usr/bin/env python
"""
download_results.py — fetch the per-dataset results.csv from the backend.

Equivalent to the old `final_results.csv` export, but pulled remotely so the
output is always consistent with the centralized store.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yaml
from api_client import BackendClient


def main():
    ap = argparse.ArgumentParser(description="Download per-dataset results.csv")
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="Agent config YAML (uses backend_url, api_key, dataset)")
    ap.add_argument("--dataset", default=None,
                    help="Override dataset from config")
    ap.add_argument("--out", default=None,
                    help="Output path (default: ./results_<dataset>_<status>.csv)")
    ap.add_argument("--status", default="done",
                    help="status filter: done|running|failed|pending|all (default: done)")
    ap.add_argument("--architecture", default=None)
    ap.add_argument("--agent-id", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--loss-type", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = args.dataset or cfg["dataset"]
    backend_url = cfg["backend_url"]
    api_key = cfg.get("api_key") or os.environ.get("BACKEND_API_KEY") or None
    out = args.out or str(HERE / f"results_{dataset}_{args.status}.csv")

    client = BackendClient(base_url=backend_url, dataset=dataset,
                           agent_id="cli-downloader", api_key=api_key)

    filters = {
        "status": args.status,
        "architecture": args.architecture,
        "agent_id": args.agent_id,
        "device": args.device,
        "loss_type": args.loss_type,
        "limit": args.limit,
        "offset": args.offset,
    }
    saved = client.download_results_csv(out, **filters)
    size = Path(saved).stat().st_size
    print(f"saved {saved} ({size} bytes) [dataset={dataset} status={args.status}]")


if __name__ == "__main__":
    main()
