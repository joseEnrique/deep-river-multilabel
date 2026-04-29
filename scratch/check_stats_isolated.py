import yaml
import itertools

def build_grid(cfg: dict) -> list[dict]:
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

stats = {}
total_heavy = 0
total_light = 0

for exp in grid:
    arch = exp.get('arch', exp.get('architecture', 'Unknown'))
    slots, score = get_slots_needed(exp)
    if arch not in stats:
        stats[arch] = {'HEAVY': 0, 'LIGHT': 0, 'MAX_SCORE': 0}
    
    if slots == 2:
        stats[arch]['HEAVY'] += 1
        total_heavy += 1
    else:
        stats[arch]['LIGHT'] += 1
        total_light += 1
        
    stats[arch]['MAX_SCORE'] = max(stats[arch]['MAX_SCORE'], score)

print(f'Total HEAVY (2 slots): {total_heavy}')
print(f'Total LIGHT (1 slot): {total_light}')
print('\nBreakdown by Architecture:')
for arch, data in stats.items():
    print(f' - {arch}: {data["HEAVY"]} HEAVY, {data["LIGHT"]} LIGHT (Max Score seen: {data["MAX_SCORE"]})')
