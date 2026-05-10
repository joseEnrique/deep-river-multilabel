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

def run_single_experiment(worker_args):
    """
    Helper for parallel execution
    """
    i, total_pending, exp_id, exp_name, exp_cfg, db_path, res_dir, ckpt_every, default_device_str = worker_args
    loss_type = exp_cfg.get("loss", {}).get("type", "?")
    
    current_device_str = exp_cfg.get("device", default_device_str)
    
    # We don't print massive banners here to avoid messy console output, 
    # but you can log the start.
    print(f"[{i}/{total_pending}] STARTED: {exp_name} on {current_device_str}")

    if not db.claim(exp_id, db_path):
        print(f"[{i}/{total_pending}] ⏭️ SKIPPED (Already claimed): {exp_name}")
        return True

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
        print(f"[{i}/{total_pending}] ✅ DONE → {exp_name} on {current_device_str} | MacroF1={result.get('macro_f1', '?')}% "
              f"({result.get('duration_s', '?')}s)")
        return True

    except KeyboardInterrupt:
        # En procesos paralelos el KeyboardInterrupt es delicado, pero marcamos como fallido
        db.mark_failed(exp_id, "KeyboardInterrupt", db_path)
        return False

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[{i}/{total_pending}] ❌ FAILED: {exp_name} on {current_device_str} -> {e}")
        db.mark_failed(exp_id, f"{e}\n{tb[:1500]}", db_path)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fault-tolerant experiment runner")
    parser.add_argument("--config", default=str(HERE / "config.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--status", action="store_true",
                        help="Print DB status and exit without running")
    parser.add_argument("--device", type=str, default=None,
                        help="OVERRIDE: Force ALL selected experiments to run on this device")
    parser.add_argument("--filter-device", type=str, default=None,
                        help="FILTER: Only run experiments that were originally assigned to this device in config.yaml")
    parser.add_argument("--reverse", action="store_true",
                        help="Process the pending experiments list in reverse order (useful for cleanly splitting load with another GPU)")
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
    print(f"Experiment grid:  {total} configurations in DB")
    print(f"DB status:        {json.dumps(summary)}")
    if args.device:
        print(f"Device Override:  All pending tasks will be forced to run on '{args.device}'")
    print(f"Progress:         {done}/{total} done")
    print(f"{'='*60}\n")

    if args.status:
        return

    # Device baseline (overridden by config if specified)
    default_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Default Device: {default_device_str}\n")

    # ── Run pending experiments (PARALLEL, STRICT GPU CONCURRENCY) ─────────
    all_pending = db.get_pending(db_path)
    
    # Filter only those that are in the current config grid
    grid_ids = {db.make_exp_id(c) for c in grid}
    pending = [p for p in all_pending if p[0] in grid_ids]
    
    # Filter by original device if requested
    if args.filter_device:
        pending = [p for p in pending if p[2].get("device") == args.filter_device]
    
    print(f"Experiments to run (from current config): {len(pending)}")
    if args.filter_device:
        print(f"  └ (Filtered only experiments originally for '{args.filter_device}')")
    elif len(pending) != len(all_pending):
        print(f"  └ (Skipped {len(all_pending) - len(pending)} pending experiments from older configs)")
    print("\n")
    
    if not pending:
        print("Nothing to run.")
        return

    if args.reverse:
        pending = list(reversed(pending))
        print("  └ (Processing list in REVERSE order)")

    # To strictly ensure exactly 1 process per GPU, we map the available targets.
    # We will submit experiments to a specific GPU executor one by one as they finish.
    
    # Filter tasks by their configured target device
    target_devices = ["cuda:1", "cuda:2"]
    if args.device:
        target_devices = [args.device]
        
    tasks_by_device = {dev: [] for dev in target_devices}
    other_tasks = []
    
    for i, (exp_id, exp_name, exp_cfg) in enumerate(pending, 1):
        target_dev = args.device if args.device else exp_cfg.get("device", default_device_str)
        
        # Override inside the config we pass to the runner
        if args.device:
            exp_cfg["device"] = args.device
            
        worker_args = (i, len(pending), exp_id, exp_name, exp_cfg, db_path, res_dir, ckpt_every, default_device_str)
        
        if target_dev in tasks_by_device:
            tasks_by_device[target_dev].append(worker_args)
        else:
            other_tasks.append(worker_args)
            
    # We use robust start method for PyTorch multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # Already set

    # Process tasks per device.
    # Since we want exactly 1 process running on cuda:1 and 1 process running on cuda:2,
    # we can use a ProcessPoolExecutor with max_workers=2, but to guarantee they don't
    # step on each other's GPU, it's safer to have one logical queue per device.
    # An elegant way in Python is to use one executor with 2 workers, and just submit all of them,
    # but that doesn't natively guarantee "worker A only does cuda 1, worker B only does cuda 2".
    
    # We'll use a custom dispatcher loop with exactly 1 ThreadPool worker PER target GPU,
    # which in turn spawns exactly 1 Process.
    
    def process_queue(queue_name, tasks_list):
        if not tasks_list:
            return
            
        import threading
        
        max_slots = 4
        available_slots = max_slots
        condition = threading.Condition()

        def get_slots_needed(cfg):
            arch = cfg.get("arch", cfg.get("architecture", ""))
            ws = cfg.get("window_size", 1)

            hd = cfg.get("hidden_dim", 32)
            if isinstance(hd, list): hd = max(hd)
            elif "hidden_dims" in cfg: hd = max(cfg["hidden_dims"])

            nl = cfg.get("num_layers", 1)

            # Base compute
            compute_score = ws * (hd ** 2) * nl

            # Penalizaciones
            if arch == "Transformer":
                compute_score += (ws ** 2) * hd * nl * 0.2
            elif arch == "LSTM":
                compute_score *= 1.5
            elif arch in ("MLP", "CNN"):
                compute_score *= 0.3   # MLP/CNN son mucho más ligeros que un LSTM equivalente

            # Tiers: SMALL (1 slot, ~25% GPU) / MEDIUM (2, ~50%) / LARGE (4, 100%)
            SMALL_CEILING = 1_500_000
            MEDIUM_CEILING = 30_000_000

            if compute_score <= SMALL_CEILING:
                return 1  # SMALL
            if compute_score <= MEDIUM_CEILING:
                return 2  # MEDIUM
            return 4      # LARGE
        
        futures = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_slots) as executor:
            remaining_tasks = list(tasks_list)
            
            while remaining_tasks:
                with condition:
                    # Esperar si no hay NINGÚN slot libre
                    while available_slots == 0:
                        condition.wait()
                        
                    # Buscar el primer experimento que quepa en los slots actuales
                    selected_idx = -1
                    slots_needed = 0
                    
                    for idx, task in enumerate(remaining_tasks):
                        cfg = task[4]
                        slots = get_slots_needed(cfg)
                        if slots <= available_slots:
                            selected_idx = idx
                            slots_needed = slots
                            break
                            
                    if selected_idx != -1:
                        # Hemos encontrado uno que cabe. Lo extraemos de la lista.
                        selected_task = remaining_tasks.pop(selected_idx)
                        available_slots -= slots_needed
                    else:
                        # Quedan slots libres (ej. 1) pero todos los pendientes son demasiado grandes (ej. necesitan 2).
                        # Tenemos que esperar obligatoriamente a que acabe el experimento actual.
                        condition.wait()
                        continue
                
                future = executor.submit(run_single_experiment, selected_task)
                futures.append(future)
                
                def make_release_cb(slots):
                    def release_callback(fut):
                        nonlocal available_slots
                        with condition:
                            available_slots += slots
                            condition.notify_all()
                    return release_callback
                
                future.add_done_callback(make_release_cb(slots_needed))
                
            # Wait for all experiments on this GPU to finish
            concurrent.futures.wait(futures)

    print(f"Launching dispatchers with dynamic slots (up to 2 per GPU based on Compute Score)...\n")
    if args.device:
        print(f" - {args.device} queue: {len(tasks_by_device[args.device])} tasks")
    else:
        print(f" - cuda:1 queue: {len(tasks_by_device['cuda:1'])} tasks")
        print(f" - cuda:2 queue: {len(tasks_by_device['cuda:2'])} tasks")
        
    if other_tasks:
        print(f" - Other devices queue: {len(other_tasks)} tasks")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_devices) + (1 if other_tasks else 0)) as main_executor:
            # Dispatch threads. Each thread processes one specific GPU queue strictly sequentially.
            if args.device:
                main_executor.submit(process_queue, args.device, tasks_by_device[args.device])
            else:
                main_executor.submit(process_queue, "cuda:1", tasks_by_device["cuda:1"])
                main_executor.submit(process_queue, "cuda:2", tasks_by_device["cuda:2"])
                
            if other_tasks:
                main_executor.submit(process_queue, "other", other_tasks)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Global KeyboardInterrupt detected! Shutting down...")
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
