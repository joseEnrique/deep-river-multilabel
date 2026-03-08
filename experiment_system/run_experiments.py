#!/usr/bin/env python
"""
run_experiments.py — Fault-tolerant experiment runner.

Usage:
    python run_experiments.py                      # uses config.yaml
    python run_experiments.py --config myconf.yaml # custom config
    python run_experiments.py --status             # show DB status, no run

Resume behaviour:
    - 'done' experiments are always skipped.
    - 'failed' and 'pending' experiments (including interrupted ones
      that were left in 'running' state) are retried.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
import yaml

# ── Project paths ─────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from experiment_system import db, runner

# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid(cfg: dict) -> list[dict]:
    """
    Generate the full cartesian product of model parameters × loss configs.

    Config schema:
      model:          # dict of lists — all combinations taken
        past_history: [1, 2, 5]
        ...
      loss:           # list of dicts — each is a separate loss config
        - type: BCE
        - type: FullAdaptive
          ...
    """
    model_grid = cfg.get("model", {})
    loss_list  = cfg.get("loss", [{"type": "BCE"}])

    # Cartesian product of model params
    keys   = list(model_grid.keys())
    values = [model_grid[k] if isinstance(model_grid[k], list) else [model_grid[k]]
              for k in keys]

    experiments = []
    for combo in itertools.product(*values):
        model_cfg = dict(zip(keys, combo))
        for loss_cfg in loss_list:
            exp = {**model_cfg, "loss": loss_cfg}
            experiments.append(exp)

    return experiments


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fault-tolerant experiment runner")
    parser.add_argument("--config", default=str(HERE / "config.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--status", action="store_true",
                        help="Print DB status and exit without running")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    db_path  = HERE / cfg.get("db_path", "experiments.db")
    res_dir  = HERE / cfg.get("results_dir", "results")
    final_csv = HERE / cfg.get("final_results_file", "results/final_results.csv")
    ckpt_every = cfg.get("checkpoint_every", 500)

    # Init DB
    db.init_db(db_path)

    # Build and register all experiments (idempotent)
    grid = build_grid(cfg)
    all_exps = db.register_experiments(grid, db_path)

    summary = db.get_summary(db_path)
    total = sum(summary.values())
    done  = summary.get("done", 0)
    print(f"\n{'='*60}")
    print(f"Experiment grid:  {len(grid)} configurations")
    print(f"DB status:        {json.dumps(summary)}")
    print(f"Progress:         {done}/{total} done")
    print(f"{'='*60}\n")

    if args.status:
        return

    # Device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}\n")

    # ── Run pending experiments ────────────────────────────────────────────
    pending = db.get_pending(db_path)
    print(f"Experiments to run: {len(pending)}\n")

    for i, (exp_id, exp_name, exp_cfg) in enumerate(pending, 1):
        loss_type = exp_cfg.get("loss", {}).get("type", "?")
        print(f"\n{'#'*60}")
        print(f"[{i}/{len(pending)}] {exp_name}")
        print(f"{'#'*60}")

        db.claim(exp_id, db_path)

        try:
            result = runner.run(
                exp_id=exp_id,
                exp_name=exp_name,
                config=exp_cfg,
                results_dir=res_dir,
                checkpoint_every=ckpt_every,
                device_str=device_str,
            )
            db.mark_done(exp_id, result, db_path)
            print(f"  ✅ DONE → MacroF1={result.get('macro_f1', '?')}%  "
                  f"MicroF1={result.get('micro_f1', '?')}%  "
                  f"({result.get('duration_s', '?')}s)")

        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted during {exp_id}. Marking as failed so it retries next run.")
            db.mark_failed(exp_id, "KeyboardInterrupt", db_path)
            raise  # propagate so the process exits cleanly

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  ❌ FAILED: {e}")
            print(tb)
            db.mark_failed(exp_id, f"{e}\n{tb[:1500]}", db_path)

    # ── Export final results ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("All experiments finished. Exporting final_results.csv …")
    df = db.export_results(str(final_csv), db_path)
    print(f"✅ Saved {len(df)} rows → {final_csv}")

    final_summary = db.get_summary(db_path)
    print(f"Final DB status: {json.dumps(final_summary)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
