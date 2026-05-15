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


def _int(v, default: int = 1) -> int:
    """Coerciona v a int; si falla devuelve default."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _flatten_to_ints(x):
    """Aplana listas/strings/escalares a una lista de ints (descarta lo no parseable)."""
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten_to_ints(y)
    elif isinstance(x, str):
        for tok in x.replace(",", " ").split():
            try:
                yield int(tok)
            except ValueError:
                pass
    elif x is not None:
        try:
            yield int(x)
        except (TypeError, ValueError):
            pass


def _max_hidden(cfg: dict, default: int = 32) -> int:
    """Extrae max hidden dim, robusto a strings y listas anidadas (MLP hidden_dims)."""
    raw = cfg.get("hidden_dim")
    if raw is None:
        raw = cfg.get("hidden_dims", default)
    values = list(_flatten_to_ints(raw))
    return max(values) if values else default


def get_slots_needed(cfg: dict) -> int:
    """Retorna 2 (SMALL), 2 (MEDIUM) o 4 (LARGE) según compute_score."""
    ws = _int(cfg.get("window_size", 1), 1)
    hd = _max_hidden(cfg, 32)
    nl = _int(cfg.get("num_layers", 1), 1)
    ph = max(1, _int(cfg.get("past_history", 1), 1))
    ep = max(1, _int(cfg.get("epochs", 1), 1))

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
