import sqlite3
import json
import yaml
from experiment_system.run_experiments import build_grid

def get_slots_needed(cfg):
    arch = cfg.get('arch', cfg.get('architecture', ''))
    ws = cfg.get('window_size', 1)
    
    hd = cfg.get('hidden_dim', 32)
    if isinstance(hd, list): hd = max(hd)
    elif 'hidden_dims' in cfg: hd = max(cfg['hidden_dims'])
    
    nl = cfg.get('num_layers', 1)

    compute_score = ws * (hd ** 2) * nl
    
    if arch == 'Transformer':
        compute_score += (ws ** 2) * hd * nl
    elif arch == 'LSTM':
        compute_score *= 1.5
        
    MEDIUM_CEILING = 6800000
    
    if compute_score > MEDIUM_CEILING:
        return 2, compute_score
    return 1, compute_score

with open('experiment_system/config.yaml') as f:
    cfg = yaml.safe_load(f)

grid = build_grid(cfg)

import hashlib
def make_exp_id(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]

grid_ids = {make_exp_id(c): c for c in grid}

with sqlite3.connect('experiment_system/experiments.db') as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM experiments WHERE status != 'done'").fetchall()

cuda2_tasks = []
for r in rows:
    if r['exp_id'] in grid_ids:
        c = grid_ids[r['exp_id']]
        if c.get('device') == 'cuda:2':
            cuda2_tasks.append(c)

heavy_count = 0
light_count = 0
for c in cuda2_tasks:
    slots, _ = get_slots_needed(c)
    if slots == 2: heavy_count += 1
    else: light_count += 1

print(f"Active grid pending on cuda:2: {len(cuda2_tasks)}")
print(f"HEAVY pending on cuda:2: {heavy_count}")
print(f"LIGHT pending on cuda:2: {light_count}")
