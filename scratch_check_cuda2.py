import sqlite3
import json

db_path = "experiment_system/experiments.db"

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

with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM experiments WHERE status = 'pending'").fetchall()

cuda2_tasks = []
for r in rows:
    cfg = json.loads(r['config'])
    if cfg.get('device') == 'cuda:2':
        cuda2_tasks.append(cfg)

print(f"Total pending on cuda:2: {len(cuda2_tasks)}")
heavy_count = 0
light_count = 0
for cfg in cuda2_tasks:
    slots, score = get_slots_needed(cfg)
    if slots == 2: heavy_count += 1
    else: light_count += 1

print(f"HEAVY pending on cuda:2: {heavy_count}")
print(f"LIGHT pending on cuda:2: {light_count}")
