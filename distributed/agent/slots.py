"""
Slot accounting per GPU — calcado de experiment_system/run_experiments.py.

Cada GPU expone un número de slots (capacidad de concurrencia). Cada
experimento consume 1 slot (LIGHT) o 2 (HEAVY) según `compute_score`.
Esto permite co-ubicar dos experimentos pequeños en la misma GPU.
"""

# Techo empírico (mismo valor que el scheduler local).
MEDIUM_CEILING = 6_800_000

# Default si el config del agent no define `max_slots_per_device`.
DEFAULT_MAX_SLOTS_PER_GPU = 2


def get_slots_needed(cfg: dict) -> int:
    """Retorna 1 (LIGHT/MEDIUM) o 2 (HEAVY) según el compute_score del config."""
    arch = cfg.get("arch", cfg.get("architecture", ""))
    ws = cfg.get("window_size", 1)

    hd = cfg.get("hidden_dim", 32)
    if isinstance(hd, list):
        hd = max(hd)
    elif "hidden_dims" in cfg:
        hd = max(cfg["hidden_dims"])

    nl = cfg.get("num_layers", 1)

    score = ws * (hd ** 2) * nl
    if arch == "Transformer":
        score += (ws ** 2) * hd * nl
    elif arch == "LSTM":
        score *= 1.5

    return 2 if score > MEDIUM_CEILING else 1
