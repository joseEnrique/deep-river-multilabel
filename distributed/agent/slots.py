"""
Slot accounting per GPU — tres tiers (SMALL/MEDIUM/LARGE).

Cada GPU expone 4 slots y cada experimento consume 2 (SMALL ~50% GPU),
2 (MEDIUM ~50%) o 4 (LARGE 100%) según `compute_score`. Esto permite,
en una GPU de 4 slots, hasta 2 SMALL, 2 MEDIUM o 1 LARGE en paralelo
(válido para MLP, LSTM y CNN — misma fórmula de coste).
"""

# Boundary scores. Misma fórmula para todas las arquitecturas (MLP/LSTM/CNN):
#   compute_score = (ws · hd² · nl + ws² · hd · nl · 0.2) · ph · ep
SMALL_CEILING = 1_500_000    # ≤ → SMALL (2 slots)
MEDIUM_CEILING = 30_000_000  # ≤ → MEDIUM (2 slots), > → LARGE (4 slots)

# Default si el config del agent no define `max_slots_per_device`.
DEFAULT_MAX_SLOTS_PER_GPU = 4

SMALL_SLOTS = 2
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
    # Mismo perfil de coste para todas las arquitecturas: término cuadrático
    # en ws con factor 0.2 (las RTX 3090 paralelizan bien matmuls grandes).
    score += (ws ** 2) * hd * nl * 0.2

    # past_history y epochs aumentan el coste lineal con cada uno para todos
    # los modelos (más timesteps por forward · más pasadas del optimizador).
    score *= ph * ep

    if score <= SMALL_CEILING:
        return SMALL_SLOTS
    if score <= MEDIUM_CEILING:
        return MEDIUM_SLOTS
    return LARGE_SLOTS
