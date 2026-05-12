"""
Slot accounting per GPU — tres tiers (SMALL/MEDIUM/LARGE).

Cada GPU expone un número de slots (capacidad de concurrencia) y cada
experimento consume 1 (SMALL ~25% GPU), 2 (MEDIUM ~50%) o 4 (LARGE 100%)
según `compute_score`. Esto permite, p.ej., 4 SMALL en paralelo en la
misma GPU, o 1 MEDIUM + 2 SMALL.
"""

# Boundary scores (compute_score = ws · hd² · nl, ×1.5 si LSTM,
# + ws² · hd · nl si Transformer, ×0.3 si MLP/CNN).
SMALL_CEILING = 1_500_000    # ≤ → SMALL (1 slot)
MEDIUM_CEILING = 30_000_000  # ≤ → MEDIUM (2 slots), > → LARGE (4 slots)

# Default si el config del agent no define `max_slots_per_device`.
DEFAULT_MAX_SLOTS_PER_GPU = 4

SMALL_SLOTS = 1
MEDIUM_SLOTS = 2
LARGE_SLOTS = 4


def get_slots_needed(cfg: dict) -> int:
    """Retorna 1 (SMALL), 2 (MEDIUM) o 4 (LARGE) según compute_score."""
    arch = cfg.get("arch", cfg.get("architecture", ""))
    ws = cfg.get("window_size", 1)

    hd = cfg.get("hidden_dim", 32)
    if isinstance(hd, list):
        hd = max(hd)
    elif "hidden_dims" in cfg:
        hd = max(cfg["hidden_dims"])

    nl = cfg.get("num_layers", 1)
    ph = max(1, int(cfg.get("past_history", 1) or 1))
    ep = max(1, int(cfg.get("epochs", 1) or 1))

    score = ws * (hd ** 2) * nl
    if arch == "Transformer":
        # Self-attention escala O(ws²) en FLOPs, pero las RTX 3090 paralelizan
        # bien matmuls grandes y el % real de GPU usado es bastante menor que
        # lo que sugeriría el conteo teórico → factor 0.2.
        score += (ws ** 2) * hd * nl * 0.2
    elif arch == "LSTM":
        score *= 1.5
    elif arch in ("MLP", "CNN"):
        score *= 0.3   # MLP/CNN son mucho más ligeros que un LSTM equivalente

    # past_history y epochs aumentan el coste lineal con cada uno para todos
    # los modelos (más timesteps por forward · más pasadas del optimizador).
    score *= ph * ep

    if score <= SMALL_CEILING:
        return SMALL_SLOTS
    if score <= MEDIUM_CEILING:
        return MEDIUM_SLOTS
    return LARGE_SLOTS
