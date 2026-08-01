#!/usr/bin/env python
"""fetch_checkpoints.py — dump the embedded checkpoints of ONE experiment.

Los checkpoints van embebidos en el documento del experimento (no hay endpoint
propio): se leen con GET /api/v1/datasets/{ds}/experiments/{name}/ -> campo
`checkpoints`, cada uno con `step` y su dict `metrics` completo.

Uso:
    python fetch_checkpoints.py --config <agent.yaml> \
        --exp-config '{"architecture":"LSTM", ..., "loss":{...}}' \
        --out /path/out.json

`--exp-config` es el config PLANO del experimento (mismos campos que usa el
agente); de ahí se reconstruye el exp_name canónico con naming.make_exp_name.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaml
from api_client import BackendClient
from naming import make_exp_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--exp-config", required=True, help="flat experiment config as JSON")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    exp_cfg = json.loads(args.exp_config)
    dataset = args.dataset or exp_cfg.get("dataset") or cfg["dataset"]
    exp_cfg.setdefault("dataset", dataset)

    cl = BackendClient(base_url=cfg["backend_url"], dataset=dataset,
                       agent_id="fetch-cp", api_key=cfg.get("api_key"))
    name = make_exp_name(exp_cfg)
    doc = cl.get(name)
    if doc is None:
        print(f"NOT FOUND: {name}", file=sys.stderr)
        sys.exit(2)
    cps = doc.get("checkpoints") or []
    out = {
        "exp_name": name,
        "dataset": dataset,
        "status": doc.get("status"),
        "final_metrics": doc.get("final_metrics") or {},
        "duration_s": doc.get("duration_s"),
        "config": doc.get("config") or exp_cfg,
        "checkpoints": [{"step": c.get("step"), "elapsed_s": c.get("elapsed_s"),
                         "metrics": c.get("metrics") or {}} for c in cps],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"saved {args.out}  exp={name}  checkpoints={len(cps)}")


if __name__ == "__main__":
    main()
