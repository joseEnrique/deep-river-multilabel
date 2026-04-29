#!/usr/bin/env python
"""
show_missing.py — Lists experiments that are pending, failed, or running based on config.yaml.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from experiment_system import db
from experiment_system.run_experiments import load_config, build_grid

def main():
    parser = argparse.ArgumentParser(description="Show missing (pending/failed/running) experiments.")
    parser.add_argument("--config", default=str(HERE / "config.yaml"), help="Path to YAML config file")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    db_path = HERE / cfg.get("db_path", "experiments.db")
    
    # Initialize and sync DB with current config
    db.init_db(db_path)
    grid = build_grid(cfg)
    db.register_experiments(grid, db_path)

    import json
    
    grid_ids = {db.make_exp_id(c) for c in grid}

    with db._connect(db_path) as conn:
        # Fetch ALL rows so we can count done vs non-done for the current grid
        all_rows = conn.execute(
            "SELECT exp_id, exp_name, status, config FROM experiments"
        ).fetchall()

    grid_rows = [r for r in all_rows if r["exp_id"] in grid_ids]
    
    grid_total = len(grid_ids)
    grid_done = sum(1 for r in grid_rows if r["status"] == "done")
    grid_pending = sum(1 for r in grid_rows if r["status"] == "pending")
    grid_failed = sum(1 for r in grid_rows if r["status"] == "failed")
    grid_running = sum(1 for r in grid_rows if r["status"] == "running")

    print(f"\n{'='*60}")
    print(f"Experiment grid (del config {args.config}): {grid_total} configuraciones")
    print(f"  • Ejecutados (done): {grid_done}/{grid_total} ({(grid_done/grid_total*100) if grid_total else 0:.1f}%)")
    print(f"  • Pendientes:        {grid_pending}")
    print(f"  • Fallados:          {grid_failed}")
    print(f"  • Ejecutándose:      {grid_running}")
    print(f"{'='*60}\n")
    
    # Filter only missing ones for the detailed list
    rows = [r for r in grid_rows if r["status"] != "done"]
    rows.sort(key=lambda x: (x["status"], x["exp_name"]))
        
    if not rows:
        print("✨ All experiments are done!")
        return

    missing_by_device = {}
    time_by_device = {}
    
    def get_slots_needed(cfg):
        arch = cfg.get("arch", cfg.get("architecture", ""))
        ws = cfg.get("window_size", 1)
        hd = cfg.get("hidden_dim", 32)
        if isinstance(hd, list): hd = max(hd)
        elif "hidden_dims" in cfg: hd = max(cfg["hidden_dims"])
        nl = cfg.get("num_layers", 1)
        compute_score = ws * (hd ** 2) * nl
        if arch == "Transformer": compute_score += (ws ** 2) * hd * nl
        elif arch == "LSTM": compute_score *= 1.5
        return 2 if compute_score > 6800000 else 1

    def estimate_time_seconds(cfg):
        epochs = cfg.get("epochs", 1)
        # Aproximación lineal (regla de 3) entre 1 época (50s) y 20 épocas (700s)
        base_time = 50 if epochs <= 1 else 50 + (epochs - 1) * (700 - 50) / (20 - 1)
        
        # Con la nueva casuística (Out-of-Order Execution y Slots):
        # Los experimentos LIGHT (1 slot) corren de 2 en 2, por lo que su impacto
        # de tiempo efectivo en la cola de la gráfica se divide a la mitad.
        slots = get_slots_needed(cfg)
        return base_time if slots == 2 else base_time / 2
    
    for row in rows:
        cfg_dict = json.loads(row["config"])
        device = cfg_dict.get("device", "default (cuda)")
        missing_by_device[device] = missing_by_device.get(device, 0) + 1
        time_by_device[device] = time_by_device.get(device, 0) + estimate_time_seconds(cfg_dict)

    def format_time(seconds):
        seconds = int(seconds)
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        parts = []
        if days > 0: parts.append(f"{days}d")
        if hours > 0: parts.append(f"{hours}h")
        if minutes > 0: parts.append(f"{minutes}m")
        if not parts: return "< 1m"
        return " ".join(parts)

    print(f"Summary of missing experiments by device:")
    for device, count in sorted(missing_by_device.items()):
        est_time = format_time(time_by_device.get(device, 0))
        print(f"  • {device}: {count} experiment(s) [~ {est_time}]")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
