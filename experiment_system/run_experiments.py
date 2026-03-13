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
import concurrent.futures
import multiprocessing

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
    Generate the cartesian product of model parameters × loss configs.
    Supports defining `model` as a list of dicts to avoid illogical combinations
    (e.g., bidirectional=True for MLP).
    """
    model_cfg_raw = cfg.get("model", {})
    if isinstance(model_cfg_raw, dict):
        models_configs = [model_cfg_raw]
    elif isinstance(model_cfg_raw, list):
        models_configs = model_cfg_raw
    else:
        models_configs = []

    loss_list  = cfg.get("loss", [{"type": "BCE"}])

    experiments = []
    
    for model_grid in models_configs:
        keys   = list(model_grid.keys())
        values = [model_grid[k] if isinstance(model_grid[k], list) else [model_grid[k]]
                  for k in keys]
        
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

    # Device baseline (overridden by config if specified)
    default_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Default Device: {default_device_str}\n")

    # ── Helpers for parallel execution ─────────────────────────────────────
    def run_single_experiment(worker_args):
        i, (exp_id, exp_name, exp_cfg) = worker_args
        loss_type = exp_cfg.get("loss", {}).get("type", "?")
        
        # We don't print massive banners here to avoid messy console output, 
        # but you can log the start.
        print(f"[{i}/{len(pending)}] STARTED: {exp_name} on {exp_cfg.get('device', default_device_str)}")

        db.claim(exp_id, db_path)
        current_device_str = exp_cfg.get("device", default_device_str)

        try:
            result = runner.run(
                exp_id=exp_id,
                exp_name=exp_name,
                config=exp_cfg,
                results_dir=res_dir,
                checkpoint_every=ckpt_every,
                device_str=current_device_str,
            )
            db.mark_done(exp_id, result, db_path)
            print(f"[{i}/{len(pending)}] ✅ DONE → {exp_name} | MacroF1={result.get('macro_f1', '?')}% "
                  f"({result.get('duration_s', '?')}s)")
            return True

        except KeyboardInterrupt:
            # En procesos paralelos el KeyboardInterrupt es delicado, pero marcamos como fallido
            db.mark_failed(exp_id, "KeyboardInterrupt", db_path)
            return False

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[{i}/{len(pending)}] ❌ FAILED: {exp_name} -> {e}")
            db.mark_failed(exp_id, f"{e}\n{tb[:1500]}", db_path)
            return False

    # ── Run pending experiments (PARALLEL) ─────────────────────────────────
    pending = db.get_pending(db_path)
    print(f"Experiments to run: {len(pending)}\n")
    
    # We calculate how many workers to use. You can tweak this.
    # A good default is trying to launch 1-2 workers per available GPU max.
    # But for a hardcoded max, let's use a conservative number.
    max_workers = min(len(pending), 6) # Up to 6 simultaneous experiments
    print(f"Launching ProcessPool with {max_workers} parallel workers...\n")

    # Prepare arguments for the workers
    worker_tasks = [(i, exp) for i, exp in enumerate(pending, 1)]

    # Use robust start method for PyTorch multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # Already set

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # We don't strictly need to wait for `.result()` unless we want to handle exceptions 
            # in the main thread, but it's good to consume the iterator so the main thread blocks.
            for _ in executor.map(run_single_experiment, worker_tasks):
                pass
    except KeyboardInterrupt:
        print("\n\n⚠️ Global KeyboardInterrupt detected! Shutting down workers...")
        # Note: Executor shutdown takes care of child processes in standard Python, 
        # but PyTorch processes might need forced killing in extreme cases.
        sys.exit(1)

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
