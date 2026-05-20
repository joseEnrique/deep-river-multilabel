"""
Slot accounting per GPU — tres tiers (SMALL/MEDIUM/LARGE).

Cada GPU expone 4 slots y cada experimento consume 2 (SMALL ~50% GPU),
2 (MEDIUM ~50%) o 4 (LARGE 100%) según `compute_score`. Esto permite,
en una GPU de 4 slots, hasta 2 SMALL, 2 MEDIUM o 1 LARGE en paralelo
(válido para MLP, LSTM y CNN — misma fórmula de coste).
"""

# Boundary scores. Mide SM utilization concurrente (no tiempo total), por eso
# `epochs` y `past_history` no entran: afectan duración, no compute simultáneo.
#   compute_score = hd² · nl                       (matmuls densos: MLP/CNN/LSTM)
#                 + ws² · 0.5      (solo Transformer, captura attention O(L²))
# Modelos con hd<=32 son launch-bound (kernels CUDA pequeños), la GPU pasa la
# mitad esperando → tier SMALL/MEDIUM aunque haya muchas epochs.
SMALL_CEILING = 5_000   # hd≤64 nl=1 sin attention → kernels triviales.

MEDIUM_CEILING_DEFAULT = 50_000      # hd=128 nl≤2 cabe como MEDIUM.
MEDIUM_CEILING_TRANSFORMER = 50_000  # Transformer hd≤64 con ws=200 → MEDIUM.


def _medium_ceiling(arch: str) -> int:
    return MEDIUM_CEILING_TRANSFORMER if arch == "Transformer" else MEDIUM_CEILING_DEFAULT

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


def get_tier(cfg: dict) -> str:
    """Retorna 'SMALL', 'MEDIUM' o 'LARGE' según compute_score y arch."""
    arch = str(cfg.get("architecture") or cfg.get("arch") or "")
    ws = _int(cfg.get("window_size", 1), 1)
    hd = _max_hidden(cfg, 32)
    nl = _int(cfg.get("num_layers", 1), 1)

    # Matmuls densos (FFN, conv, gates LSTM): coste cuadrático en hidden_dim.
    score = (hd ** 2) * nl
    # Attention solo en Transformer: matriz L×L domina con ws grande.
    if arch == "Transformer":
        score += (ws ** 2) * 0.5

    if score <= SMALL_CEILING:
        return "SMALL"
    if score <= _medium_ceiling(arch):
        return "MEDIUM"
    return "LARGE"


_TIER_TO_SLOTS = {"SMALL": SMALL_SLOTS, "MEDIUM": MEDIUM_SLOTS, "LARGE": LARGE_SLOTS}


def get_slots_needed(cfg: dict) -> int:
    """Retorna 2 (SMALL), 2 (MEDIUM) o 4 (LARGE) según compute_score y arch."""
    return _TIER_TO_SLOTS[get_tier(cfg)]
