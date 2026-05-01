#!/usr/bin/env python
"""
nps_solo.py — Run a single NPS experiment alone on the GPU.

Companion to `nps_pair.py`. Run this first to get the baseline (alone) timing
and metrics, then run `nps_pair.py` with the SAME `CONFIG` dict to compare
against two-at-a-time on the same GPU.

Usage:
    python nps_solo.py
    python nps_solo.py --device cuda:2
"""

import argparse
import sys
import time
import multiprocessing
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from experiment_system import runner


# ── EDIT THIS CONFIG (must match the one in nps_pair.py) ──────────────────────

CONFIG = {
    "dataset":       "nps",
    "architecture":  "LSTM",
    "optimizer":     "adam",
    "lr":            1e-3,
    "dropout":       0.3,
    "bidirectional": True,
    "seed":          42,
    "epochs":        1,
    "past_history":  1,
    "window_size":   200,
    "hidden_dim":    128,
    "num_layers":    2,
    "loss":          {"type": "BCE"},
}

# ──────────────────────────────────────────────────────────────────────────────


def compute_score(cfg: dict) -> int:
    """Mirror of get_slots_needed's score in run_experiments.py."""
    arch = cfg.get("architecture", "")
    ws = cfg.get("window_size", 1)
    hd = cfg.get("hidden_dim", 32)
    nl = cfg.get("num_layers", 1)
    score = ws * (hd ** 2) * nl
    if arch == "Transformer":
        score += (ws ** 2) * hd * nl
    elif arch == "LSTM":
        score = score * 1.5
    return int(score)


def _run_one(args):
    tag, cfg, device, results_dir = args
    t0 = time.time()
    res = runner.run(
        exp_id=tag,
        exp_name=tag,
        config=cfg,
        results_dir=Path(results_dir),
        checkpoint_every=2_000,
        device_str=device,
    )
    return {
        "tag":        tag,
        "duration_s": time.time() - t0,
        "macro_f1":   res.get("macro_f1"),
        "micro_f1":   res.get("micro_f1"),
        "subset_acc": res.get("subset_acc"),
        "examp_f1":   res.get("examp_f1"),
    }


def main():
    parser = argparse.ArgumentParser(description="NPS solo benchmark")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    score = compute_score(CONFIG)
    tag = (f"NPS_SOLO_{CONFIG['architecture']}_W{CONFIG['window_size']}"
           f"_H{CONFIG['hidden_dim']}_L{CONFIG['num_layers']}")

    print("=" * 72)
    print(f"NPS SOLO — {tag}")
    print(f"compute_score = {score:,}     device = {args.device}")
    print(f"current MEDIUM_CEILING = 6,800,000  →  "
          f"{'HEAVY (2 slots)' if score > 6_800_000 else 'LIGHT (1 slot)'}")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp:
        with ProcessPoolExecutor(max_workers=1) as ex:
            res = list(ex.map(_run_one, [(tag, CONFIG, args.device, tmp)]))[0]

    print("\n" + "=" * 72)
    print("ALONE — RESULT")
    print("=" * 72)
    print(f"  duration   : {res['duration_s']:.1f} s")
    print(f"  subset_acc : {res['subset_acc']:.3f} %")
    print(f"  example_f1 : {res['examp_f1']:.3f} %")
    print(f"  micro_f1   : {res['micro_f1']:.3f} %")
    print(f"  macro_f1   : {res['macro_f1']:.3f} %")
    print()
    print("Now run nps_pair.py with the same CONFIG to compare.")


if __name__ == "__main__":
    main()
