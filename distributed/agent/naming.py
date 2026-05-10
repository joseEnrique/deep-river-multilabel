"""
naming.py — exhaustive experiment-name builder.

The exp_name is the primary key in the backend, so we encode every config field
that could change the result. Two configs that differ in any value MUST produce
different exp_names — otherwise we'd silently overwrite results.

Example:
    Transformer_ai4i_ph10_h64_nl1_w200_lr0.0001_d0.2_s42_ep1_sgd_layernorm_StaticFocal_a0.25_g2.0
"""

from __future__ import annotations
from typing import Any


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        s = repr(v).rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def _hidden_token(cfg: dict) -> str:
    if "hidden_dims" in cfg and cfg["hidden_dims"] is not None:
        dims = cfg["hidden_dims"]
        return "h" + "-".join(str(d) for d in dims)
    return f"h{cfg.get('hidden_dim', '?')}"


def _num_layers_token(cfg: dict) -> str:
    if "hidden_dims" in cfg and cfg["hidden_dims"] is not None:
        return f"nl{len(cfg['hidden_dims'])}"
    return f"nl{cfg.get('num_layers', '?')}"


def _loss_token(loss_cfg: dict) -> str:
    ltype = loss_cfg.get("type", "BCE")
    parts = [ltype]
    if ltype == "StaticFocal":
        parts.append(f"a{_fmt_num(loss_cfg.get('alpha', 0.25))}")
        parts.append(f"g{_fmt_num(loss_cfg.get('gamma', 2.0))}")
    elif ltype in ("AdaptiveFocal", "FullAdaptive"):
        parts.append(f"bg{_fmt_num(loss_cfg.get('base_gamma', 2.0))}")
        parts.append(f"ba{_fmt_num(loss_cfg.get('base_alpha', 0.25))}")
        parts.append(f"dc{_fmt_num(loss_cfg.get('decay', 0.999))}")
        parts.append(f"ag{_fmt_num(loss_cfg.get('alpha_gain', 1.0))}")
        parts.append(f"gg{_fmt_num(loss_cfg.get('gamma_gain', 2.0))}")
    elif ltype == "ImprovedAdaptive":
        parts.append(f"bg{_fmt_num(loss_cfg.get('base_gamma', 2.0))}")
        parts.append(f"ba{_fmt_num(loss_cfg.get('base_alpha', 0.25))}")
        parts.append(f"dc{_fmt_num(loss_cfg.get('decay', 0.999))}")
    return "_".join(parts)


def _alpi_token(config: dict) -> str:
    """Compact token capturing the ALPI dataset parameters that change samples.
    Only emitted when dataset == 'alpi' so AI4I/NPS names stay unchanged."""
    if config.get("dataset") != "alpi":
        return ""
    parts = []
    for short, key in (("m", "machine"), ("iw", "input_win"), ("ow", "output_win"),
                       ("dl", "delta"), ("sg", "sigma"), ("mc", "min_count")):
        if key in config and config[key] is not None:
            parts.append(f"{short}{_fmt_num(config[key])}")
    return "_" + "-".join(parts) if parts else ""


def _embedding_token(config: dict) -> str:
    """Embedding hyperparams (only present for alpi-style models)."""
    if config.get("dataset") != "alpi":
        return ""
    parts = []
    if "embedding_dim" in config and config["embedding_dim"] is not None:
        parts.append(f"e{_fmt_num(config['embedding_dim'])}")
    if "num_alarms" in config and config["num_alarms"] is not None:
        parts.append(f"na{_fmt_num(config['num_alarms'])}")
    return "_" + "-".join(parts) if parts else ""


def make_exp_name(config: dict) -> str:
    """Build the canonical, exhaustive exp_name from a flat config dict."""
    arch    = config.get("architecture", "LSTM")
    dataset = config.get("dataset", "ai4i")
    ph      = config.get("past_history", "?")
    w       = config.get("window_size", "?")
    lr      = _fmt_num(config.get("lr", "?"))
    dropout = _fmt_num(config.get("dropout", 0))
    seed    = config.get("seed", "?")
    epochs  = config.get("epochs", "?")
    opt     = config.get("optimizer", "adam")
    norm    = config.get("normalization", "none")
    bidir   = "_bidir" if config.get("bidirectional") else ""
    loss    = _loss_token(config.get("loss", {"type": "BCE"}))
    alpi    = _alpi_token(config)
    emb     = _embedding_token(config)

    parts = [
        arch, dataset,
        f"ph{ph}",
        _hidden_token(config),
        _num_layers_token(config),
        f"w{w}",
        f"lr{lr}",
        f"d{dropout}",
        f"s{seed}",
        f"ep{epochs}",
        opt,
        norm,
        loss,
    ]
    return "_".join(str(p) for p in parts) + bidir + alpi + emb
