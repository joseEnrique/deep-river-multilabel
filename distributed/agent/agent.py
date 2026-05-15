#!/usr/bin/env python
"""
agent.py — distributed experiment agent.

Two roles, both compatible at the same time:

  • Producer  → the agent has a `model:`/`loss:` grid in its YAML and registers
                it on the backend (idempotent: dupes are skipped via 409).

  • Consumer  → the agent claims any `pending`/`failed` experiment for the
                configured dataset and runs it on its GPUs.

By default an agent is BOTH (registers its own grid AND consumes anything
pending in the dataset, even configs registered by other agents). Set
`consume_any: false` in the config to restrict it to its own grid.

This means you can spin up "worker-only" agents with no grid: they will pick
up whatever any other agent has registered.

Usage:
    python agent.py --config config.yaml
    python agent.py --config config.yaml --register-only
    python agent.py --config config.yaml --status
    python agent.py --config config.yaml --consume-only   # skip registration
"""

from __future__ import annotations
import argparse
import itertools
import json
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from naming import make_exp_name
from api_client import BackendClient
from local_db import LocalDB


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid(cfg: dict) -> list[dict]:
    """Cartesian product of model parameters × loss configs."""
    dataset = cfg.get("dataset", "ai4i")
    model_raw = cfg.get("model")
    if model_raw is None:
        return []
    if isinstance(model_raw, dict):
        models = [model_raw]
    elif isinstance(model_raw, list):
        models = model_raw
    else:
        return []

    losses = cfg.get("loss", [{"type": "BCE"}])

    out: list[dict] = []
    for grid in models:
        keys = list(grid.keys())
        values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
        for combo in itertools.product(*values):
            mcfg = dict(zip(keys, combo))
            mcfg.setdefault("dataset", dataset)
            for loss_cfg in losses:
                exp = {**mcfg, "loss": loss_cfg}
                out.append(exp)
    return out


def register_grid(client: BackendClient, local: LocalDB, dataset: str,
                  grid: list[dict], chunk: int = 200) -> tuple[int, int]:
    """Idempotent populate of backend AND local SQLite. Bulk on both sides:
    - Backend: chunked POST /experiments/bulk (one HTTP call per chunk).
    - Local:   single executemany INSERT OR IGNORE in a transaction.
    Returns (inserted, skipped) as reported by the backend."""
    if not grid:
        return 0, 0
    items_backend: list[dict] = []
    items_local: list[tuple[str, str, str, dict]] = []
    for c in grid:
        name = make_exp_name(c)
        arch = c.get("architecture", "LSTM")
        items_backend.append({"exp_name": name, "architecture": arch, "config": c})
        items_local.append((name, arch, dataset, c))

    local.bulk_upsert_pending(items_local)

    total_ins, total_skip = 0, 0
    for i in range(0, len(items_backend), chunk):
        batch = items_backend[i:i + chunk]
        res = client.bulk_create(batch)
        total_ins += int(res.get("inserted", 0))
        total_skip += int(res.get("skipped", 0))
    return total_ins, total_skip


# ── Worker (one process per GPU) ──────────────────────────────────────────────

def _order_candidates(candidates: list[dict], device: str,
                      owned: set[str] | None) -> list[dict]:
    """
    Soft-preference ordering for what a worker on `device` should try first.

    Priority (bucket):
      1. exps in this agent's `owned` set whose registered config.device matches
      2. any exp whose registered config.device matches this worker's device
      3. exps in this agent's `owned` set
      4. anything else (so empty agents still consume foreign work)

    Dentro de cada bucket: ordena por `slots_needed` ascendente (SMALL → MEDIUM
    → LARGE) para maximizar paralelismo. P.ej. en cuda:0 (4 slots) prefiere 2
    MEDIUM (4 slots, 2 tareas) antes que 1 LARGE (4 slots, 1 tarea); los LARGE
    se coge cuando ya no quedan candidatos más pequeños o no caben más.
    """
    from slots import get_slots_needed

    def key(exp: dict) -> tuple[int, int]:
        cfg = exp.get("config") or {}
        cfg_dev = cfg.get("device")
        is_owned = owned is not None and exp["exp_name"] in owned
        if is_owned and cfg_dev == device:
            b = 0
        elif cfg_dev == device:
            b = 1
        elif is_owned:
            b = 2
        else:
            b = 3
        return (b, get_slots_needed(cfg))
    return sorted(candidates, key=key)


def _run_one_task(name: str, cfg: dict, device: str, dataset: str,
                  base_url: str, agent_id: str, api_key: str | None,
                  checkpoint_every: int, local_db_path: str) -> None:
    """Subproceso: ejecuta un experimento ya claim-eado y reporta al backend + local."""
    # Import torch inside the child to keep CUDA init isolated per process.
    import torch  # noqa: F401
    from runner_adapter import run_with_backend

    client = BackendClient(base_url=base_url, dataset=dataset, agent_id=agent_id,
                           api_key=api_key)
    local = LocalDB(local_db_path)

    print(f"[{device}] ▶  {name}", flush=True)
    t0 = time.time()
    checkpoints_buf: list[dict] = []
    try:
        def on_cp(step: int, elapsed: float, m: dict[str, float]):
            print(f"[{device}]   [{step:,}] "
                  f"SubsetAcc={m['subset_acc']:.2f} | "
                  f"MicroF1={m['micro_f1']:.2f} | "
                  f"MacroF1={m['macro_f1']:.2f} | t={elapsed:.1f}s",
                  flush=True)
            checkpoints_buf.append({"step": step, "elapsed_s": elapsed, "metrics": m})

        def on_finish(final_m: dict[str, float], duration: float):
            local.save_completed(name, final_m, duration, checkpoints_buf)
            client.finish(name, final_m, duration, checkpoints=checkpoints_buf)
            print(f"[{device}]   saved {len(checkpoints_buf)} checkpoints", flush=True)

        run_with_backend(
            exp_name=name,
            config=cfg,
            device_str=device,
            checkpoint_every=checkpoint_every,
            on_checkpoint=on_cp,
            on_finish=on_finish,
        )
        dur = time.time() - t0
        print(f"[{device}] ✓  {name}  ({dur:.1f}s)", flush=True)
    except KeyboardInterrupt:
        local.mark_failed(name, "KeyboardInterrupt")
        try:
            client.fail(name, "KeyboardInterrupt")
        except Exception:
            pass
        raise
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"{e}\n{tb[:3000]}"
        print(f"[{device}] ✗  {name}: {e}", flush=True)
        local.mark_failed(name, msg)
        try:
            client.fail(name, msg)
        except Exception as fe:
            print(f"[{device}] (failed to report failure: {fe})", flush=True)


def worker_main(device: str, agent_id: str, dataset: str, base_url: str,
                checkpoint_every: int, poll_interval: float,
                owned_filter: list[str] | None,
                local_db_path: str,
                consume_any: bool,
                api_key: str | None,
                max_slots: int) -> None:
    """Despachador con slots: hasta `max_slots` experimentos concurrentes en `device`.
    SMALL/MEDIUM=2 slots, LARGE=4 slots (slots.get_slots_needed). Calcado del scheduler local."""
    import threading
    import concurrent.futures
    from slots import get_slots_needed

    client = BackendClient(base_url=base_url, dataset=dataset, agent_id=agent_id,
                           api_key=api_key)
    local = LocalDB(local_db_path)
    pid = os.getpid()
    owned_set = set(owned_filter) if owned_filter is not None else None
    mode = "consume-any" if consume_any else "owned-only"
    print(f"[{device}] worker started (pid={pid}, agent={agent_id}, mode={mode}, "
          f"max_slots={max_slots})", flush=True)

    available_slots = max_slots
    # Contadores:
    # - smalls_running: cualquier arch en tier SMALL (tope absoluto 3).
    # - heavy_smalls_running: SMALL con past_history>=10 (secuencias largas
    #   que consumen más memoria y compute aunque caigan en tier SMALL; tope 2).
    # - transformer_smalls / transformer_others: por tier para reglas de
    #   saturación con Transformer (2 MEDIUM Tr OK, 4 SMALL Tr NO).
    smalls_running = 0
    heavy_smalls_running = 0
    transformer_smalls_running = 0
    transformer_others_running = 0
    cond = threading.Condition()

    def is_heavy_small(c: dict) -> bool:
        try:
            return int(c.get("past_history", 1)) >= 10
        except (TypeError, ValueError):
            return False

    def release_cb(slots_used: int, was_transformer: bool, was_heavy_small: bool):
        def _cb(_fut):
            nonlocal available_slots, smalls_running, heavy_smalls_running
            nonlocal transformer_smalls_running, transformer_others_running
            with cond:
                available_slots += slots_used
                if slots_used == 1:
                    smalls_running = max(0, smalls_running - 1)
                    if was_heavy_small:
                        heavy_smalls_running = max(0, heavy_smalls_running - 1)
                if was_transformer:
                    if slots_used == 1:
                        transformer_smalls_running -= 1
                    else:
                        transformer_others_running -= 1
                cond.notify_all()
        return _cb

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_slots) as executor:
        while True:
            with cond:
                while available_slots == 0:
                    cond.wait()

            pending = client.list(status="pending", limit=2000)
            failed = client.list(status="failed", limit=500)
            candidates = pending + failed

            if not consume_any and owned_set is not None:
                candidates = [c for c in candidates if c["exp_name"] in owned_set]

            if not candidates:
                # Nada pendiente. Si todos los slots están libres, salimos;
                # si hay tareas en vuelo, esperamos a que se vacíen.
                with cond:
                    if available_slots == max_slots:
                        print(f"[{device}] nothing pending, exiting worker", flush=True)
                        return
                time.sleep(poll_interval)
                continue

            candidates = _order_candidates(candidates, device, owned_set)

            launched_any = False
            for exp in candidates:
                cfg = exp["config"]
                slots_needed = get_slots_needed(cfg)

                arch = cfg.get("architecture") or cfg.get("arch") or ""
                will_be_transformer = (arch == "Transformer")

                will_be_heavy_small = (slots_needed == 1 and is_heavy_small(cfg))

                with cond:
                    if slots_needed > available_slots:
                        continue  # no cabe ahora; intenta el siguiente
                    # Regla SIEMPRE: máximo 3 SMALL concurrentes en la GPU.
                    if slots_needed == 1 and smalls_running >= 3:
                        continue
                    # Regla EXTRA para SMALLs caros (ph>=10): máx 2.
                    if will_be_heavy_small and heavy_smalls_running >= 2:
                        continue
                    # Si ya hay heavy SMALLs corriendo, no permitir que un
                    # MEDIUM/LARGE entre y sature la GPU al 100% (los heavy
                    # SMALL ya consumen suficiente; sumar más arriesga OOM/
                    # throttling). Excepción: un LARGE que ocupa exactamente
                    # toda la GPU él solo no es el caso (cabría solo si la GPU
                    # está vacía, controlado por slots_needed > available).
                    if (heavy_smalls_running > 0
                            and slots_needed > 1
                            and (available_slots - slots_needed) < 1):
                        continue
                    # Reglas SOLO cuando el que entra es Transformer:
                    #   - 2 MEDIUM Transformer (saturan al 100%, OK)
                    #   - 1 LARGE solo (caso full_gpu trivial)
                    #   - NO mezclar SMALL Transformer cuando se saturaría la GPU.
                    # LSTM/MLP/CNN no-SMALL entran sin restricción (aunque haya
                    # Transformer corriendo).
                    if will_be_transformer:
                        is_full_gpu = slots_needed == max_slots
                        slots_after = available_slots - slots_needed
                        if not is_full_gpu and slots_after < 1:
                            new_is_small = slots_needed == 1
                            if new_is_small or transformer_smalls_running > 0:
                                continue

                name = exp["exp_name"]
                claimed = client.claim(name, device=device)
                if claimed is None:
                    continue  # otro agent se lo llevó

                local.upsert_pending(name, cfg.get("architecture", "LSTM"), dataset, cfg)
                local.mark_running(name, agent_id, device)

                with cond:
                    available_slots -= slots_needed
                    if slots_needed == 1:
                        smalls_running += 1
                        if will_be_heavy_small:
                            heavy_smalls_running += 1
                    if will_be_transformer:
                        if slots_needed == 1:
                            transformer_smalls_running += 1
                        else:
                            transformer_others_running += 1

                fut = executor.submit(
                    _run_one_task,
                    name, cfg, device, dataset, base_url, agent_id,
                    api_key, checkpoint_every, local_db_path,
                )
                fut.add_done_callback(release_cb(slots_needed, will_be_transformer, will_be_heavy_small))
                launched_any = True

                with cond:
                    if available_slots == 0:
                        break

            if not launched_any:
                # Hay candidatos pero ninguno cabe en los slots libres
                # (ej. solo HEAVY pendientes con 1 slot libre): espera.
                time.sleep(poll_interval)


# ── Cube queries (CLI helpers) ────────────────────────────────────────────────

def _fmt_num(v, ndigits: int = 4) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.{ndigits}f}"


def _print_top(resp: dict, metric: str):
    metric_path = metric if metric.startswith("final_metrics.") else f"final_metrics.{metric}"
    metric_key = metric_path.split(".", 1)[1]
    exps = resp.get("experiments") or []
    print(f"[top] dataset={resp.get('dataset')} metric={metric} count={len(exps)}")
    print(f"  {'#':>3}  {metric_key:>12}  {'ARCH':<12} {'STATUS':<8}  EXP_NAME")
    for i, e in enumerate(exps, 1):
        fm = e.get("final_metrics") or {}
        val = _fmt_num(fm.get(metric_key))
        print(f"  {i:>3}  {val:>12}  {(e.get('architecture') or '-'):<12} "
              f"{(e.get('status') or '-'):<8}  {e.get('exp_name')}")


def _print_groupby(resp: dict):
    groups = resp.get("groups") or []
    aggs = resp.get("agg") or []
    keys = resp.get("by") or []
    print(f"[groupby] dataset={resp.get('dataset')} metric={resp.get('metric')} "
          f"by={','.join(keys)} order={resp.get('order')} count={len(groups)}")
    header = "  " + "  ".join(f"{k:<14}" for k in keys) + "  " + "  ".join(
        f"{a:>10}" for a in aggs)
    print(header)
    for g in groups:
        grp = g.get("group") or {}
        mtr = g.get("metrics") or {}
        cells = [f"{str(grp.get(k, '-')):<14}" for k in keys]
        vals = []
        for a in aggs:
            name = {"avg": "mean", "stddev": "std"}.get(a, a)
            vals.append(f"{_fmt_num(mtr.get(name)):>10}")
        print("  " + "  ".join(cells) + "  " + "  ".join(vals))


def _print_best_per(resp: dict):
    metric = resp.get("metric") or ""
    metric_key = metric.split(".")[-1] if metric else ""
    groups = resp.get("groups") or []
    print(f"[best-per] dataset={resp.get('dataset')} by={resp.get('by')} "
          f"metric={metric} count={len(groups)}")
    print(f"  {'VALUE':<24} {'COUNT':>6}  {metric_key:>12}  EXP_NAME")
    for g in groups:
        val = g.get("value")
        count = g.get("count")
        best = g.get("best") or {}
        fm = best.get("final_metrics") or {}
        m = _fmt_num(fm.get(metric_key))
        print(f"  {str(val):<24} {str(count):>6}  {m:>12}  {best.get('exp_name', '-')}")


def _print_distribution(resp: dict):
    bins = resp.get("bins") or []
    if not bins:
        print(f"[distribution] no data")
        return
    max_count = max((b.get("count") or 0) for b in bins) or 1
    print(f"[distribution] metric={resp.get('metric')} count={resp.get('count')} "
          f"min={_fmt_num(resp.get('min'))} max={_fmt_num(resp.get('max'))} "
          f"mean={_fmt_num(resp.get('mean'))} std={_fmt_num(resp.get('std'))}")
    for b in bins:
        lo, hi, c = b.get("lo"), b.get("hi"), b.get("count") or 0
        bar = "█" * int(40 * c / max_count)
        print(f"  [{_fmt_num(lo)}, {_fmt_num(hi)})  {c:>6}  {bar}")


def _maybe_run_cube_query(client, args) -> bool:
    """Dispatch a cube subcommand if requested. Returns True if handled."""
    triggers = [
        args.cube_metrics, args.cube_params, args.cube_param_values,
        args.cube_top, args.cube_groupby, args.cube_best_per,
        args.cube_distribution,
    ]
    if not any(triggers):
        return False

    if args.cube_metrics:
        print(json.dumps(client.cube_metrics(status=args.cube_status), indent=2))
        return True

    if args.cube_params:
        print(json.dumps(client.cube_params(status=args.cube_status), indent=2))
        return True

    if args.cube_param_values:
        resp = client.cube_param_values(
            key=args.cube_param_values, metric=args.metric,
            where=args.where, status=args.cube_status)
        print(json.dumps(resp, indent=2))
        return True

    if args.cube_top:
        if not args.metric:
            print("error: --cube-top requires --metric", file=sys.stderr)
            sys.exit(2)
        order = args.order or "desc"
        resp = client.cube_top(metric=args.metric, limit=args.limit, order=order,
                               where=args.where, status=args.cube_status)
        _print_top(resp, args.metric)
        return True

    if args.cube_groupby:
        if not args.metric:
            print("error: --cube-groupby requires --metric", file=sys.stderr)
            sys.exit(2)
        resp = client.cube_groupby(
            by=args.cube_groupby, metric=args.metric, agg=args.agg,
            order=args.order, limit=args.limit, where=args.where,
            status=args.cube_status)
        _print_groupby(resp)
        return True

    if args.cube_best_per:
        if not args.metric:
            print("error: --cube-best-per requires --metric", file=sys.stderr)
            sys.exit(2)
        order = args.order or "desc"
        resp = client.cube_best_per(
            by=args.cube_best_per, metric=args.metric, order=order,
            limit=args.limit, where=args.where, status=args.cube_status)
        _print_best_per(resp)
        return True

    if args.cube_distribution:
        if not args.metric:
            print("error: --cube-distribution requires --metric", file=sys.stderr)
            sys.exit(2)
        resp = client.cube_distribution(
            metric=args.metric, bins=args.bins, where=args.where,
            status=args.cube_status)
        _print_distribution(resp)
        return True

    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Distributed experiment agent")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--register-only", action="store_true",
                    help="Register the grid on the backend and exit.")
    ap.add_argument("--status", action="store_true",
                    help="Print backend + local stats and exit.")
    ap.add_argument("--summary", action="store_true",
                    help="Print backend summary (counts + duration + ETA) and exit.")
    ap.add_argument("--no-register", action="store_true",
                    help="Skip the bulk register step.")
    ap.add_argument("--consume-only", action="store_true",
                    help="Same as --no-register; clearer name for worker-only agents.")

    # Cube / statistical queries (read-only, all GET).
    cg = ap.add_argument_group("cube queries (read-only)")
    cg.add_argument("--cube-metrics", action="store_true",
                    help="List metric names found in final_metrics.")
    cg.add_argument("--cube-params", action="store_true",
                    help="List config keys discovered across experiments.")
    cg.add_argument("--cube-param-values", metavar="KEY",
                    help="Distinct values of KEY (with optional --metric).")
    cg.add_argument("--cube-top", action="store_true",
                    help="Top-N experiments by --metric.")
    cg.add_argument("--cube-groupby", metavar="KEYS",
                    help="Group by comma-separated keys; needs --metric.")
    cg.add_argument("--cube-best-per", metavar="KEY",
                    help="Best experiment per value of KEY (needs --metric).")
    cg.add_argument("--cube-distribution", action="store_true",
                    help="Histogram of --metric.")
    cg.add_argument("--metric", default=None,
                    help="Metric name for cube queries (e.g. exact_match).")
    cg.add_argument("--agg", default="max,mean,count",
                    help="Aggregations for --cube-groupby (default: max,mean,count).")
    cg.add_argument("--order", default=None,
                    help="Sort spec, e.g. 'max:desc' for groupby, 'desc' elsewhere.")
    cg.add_argument("--limit", type=int, default=10,
                    help="Limit for cube queries (default: 10).")
    cg.add_argument("--bins", type=int, default=10,
                    help="Bin count for --cube-distribution (default: 10).")
    cg.add_argument("--where", default=None,
                    help="Equality filter 'k=v,k2=v2' applied to the query.")
    cg.add_argument("--cube-status", default=None,
                    help="Status filter for cube queries ('done' default, 'all' to disable).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    backend_url = cfg["backend_url"]
    agent_id = cfg.get("agent_id") or socket.gethostname()
    dataset = cfg["dataset"]
    devices = cfg.get("devices") or ["cuda:0"]
    checkpoint_every = int(cfg.get("checkpoint_every", 1000))
    poll_interval = float(cfg.get("poll_interval", 5.0))
    consume_any = bool(cfg.get("consume_any", True))
    local_db_path = str(cfg.get("local_db_path") or (HERE / f"local_{dataset}.db"))
    api_key = cfg.get("api_key") or os.environ.get("BACKEND_API_KEY") or None

    # Slots per device: default global + override por device.
    from slots import DEFAULT_MAX_SLOTS_PER_GPU
    default_slots = int(cfg.get("max_slots_per_device", DEFAULT_MAX_SLOTS_PER_GPU))
    per_device_override = cfg.get("slots_per_device") or {}
    slots_for: dict[str, int] = {
        d: int(per_device_override.get(d, default_slots)) for d in devices
    }

    client = BackendClient(base_url=backend_url, dataset=dataset, agent_id=agent_id,
                           api_key=api_key)
    local = LocalDB(local_db_path)
    print(f"[agent] id={agent_id} backend={backend_url} dataset={dataset} "
          f"devices={devices} consume_any={consume_any}", flush=True)
    print(f"[agent] slots/device: {slots_for}", flush=True)
    print(f"[agent] local mirror: {local_db_path}", flush=True)

    if args.status:
        print("[agent] backend stats:")
        print(json.dumps(client.stats(), indent=2))
        print("[agent] local mirror stats:")
        print(json.dumps(local.summary(), indent=2))
        return

    if args.summary:
        import re
        from slots import get_slots_needed
        s = client.summary()

        def fmt_eta(secs: float) -> str:
            secs = int(secs)
            h, rem = divmod(secs, 3600)
            m, sec = divmod(rem, 60)
            return f"{h}h{m:02d}m{sec:02d}s"

        def parse_cfg_from_name(name: str) -> dict:
            """Extrae arch, hidden_dim(s), num_layers y window_size desde el exp_name."""
            m = re.match(
                r"^(?P<arch>[A-Za-z]+)_[^_]+_ph\d+_h(?P<hd>[\d\-]+)_nl(?P<nl>\d+)_w(?P<ws>\d+)_",
                name,
            )
            if not m:
                return {}
            hd_raw = m.group("hd")
            cfg = {
                "architecture": m.group("arch"),
                "num_layers":   int(m.group("nl")),
                "window_size":  int(m.group("ws")),
            }
            if "-" in hd_raw:
                cfg["hidden_dims"] = [int(x) for x in hd_raw.split("-")]
            else:
                cfg["hidden_dim"] = int(hd_raw)
            return cfg

        def tier_label(slots: int) -> str:
            return {1: "SMALL", 2: "MEDIUM", 4: "LARGE"}.get(slots, f"{slots}slots")

        counts = s.get("counts", {})
        print(f"[summary] dataset={s.get('dataset')} total={s.get('total')}")
        for k in ("pending", "running", "done", "failed"):
            print(f"  {k:<8}: {counts.get(k, 0)}")
        print(f"  done with duration : {s.get('done_count', 0)}")
        avg = s.get("avg_duration_s") or 0.0
        tot = s.get("total_duration_s") or 0.0
        eta = s.get("eta_s") or 0.0
        print(f"  avg duration / done: {avg:.1f}s")
        print(f"  total compute (done): {fmt_eta(tot)}")
        print(f"  ETA pending+failed  : {fmt_eta(eta)}  ({s.get('eta_method')})")

        running = s.get("running") or []
        if running:
            # Tier breakdown global y por (agent, device)
            from collections import defaultdict
            tiers = {"SMALL": 0, "MEDIUM": 0, "LARGE": 0, "?": 0}
            by_node = defaultdict(lambda: {"SMALL": 0, "MEDIUM": 0, "LARGE": 0, "?": 0})
            running_with_tier = []
            for r in running:
                cfg_inferred = parse_cfg_from_name(r.get("exp_name", ""))
                if cfg_inferred:
                    slots = get_slots_needed(cfg_inferred)
                    label = tier_label(slots)
                else:
                    slots = 0
                    label = "?"
                tiers[label] += 1
                key = (r.get("agent_id") or "-", r.get("device") or "-")
                by_node[key][label] += 1
                running_with_tier.append((r, label, slots))

            print(f"\n[running] {len(running)} experimento(s) en curso:")
            print(f"  tiers: SMALL={tiers['SMALL']}  MEDIUM={tiers['MEDIUM']}  "
                  f"LARGE={tiers['LARGE']}" + (f"  ?={tiers['?']}" if tiers['?'] else ""))

            print(f"\n[by node × device]")
            print(f"  {'AGENT':<24} {'DEVICE':<10} {'SMALL':>6} {'MEDIUM':>7} {'LARGE':>6}  TOTAL")
            for (ag, dev), c in sorted(by_node.items()):
                total = c["SMALL"] + c["MEDIUM"] + c["LARGE"] + c["?"]
                print(f"  {ag:<24} {dev:<10} {c['SMALL']:>6} {c['MEDIUM']:>7} {c['LARGE']:>6}  {total:>5}")

            print(f"\n  {'TIER':<7} {'ARCH':<12} {'AGENT':<24} {'DEVICE':<10} {'ELAPSED':>10}  EXP_NAME")
            for r, label, _ in running_with_tier:
                arch = r.get("architecture") or "-"
                ag = r.get("agent_id") or "-"
                dev = r.get("device") or "-"
                el = fmt_eta(r.get("elapsed_s") or 0.0)
                name = r.get("exp_name", "?")
                print(f"  {label:<7} {arch:<12} {ag:<24} {dev:<10} {el:>10}  {name}")
        return

    if _maybe_run_cube_query(client, args):
        return

    grid = build_grid(cfg)
    print(f"[agent] local grid: {len(grid)} configurations", flush=True)

    skip_register = args.no_register or args.consume_only
    if not skip_register and grid:
        ins, skip = register_grid(client, local, dataset, grid)
        print(f"[agent] registered: inserted={ins} skipped(existing)={skip}", flush=True)
    elif grid:
        local.bulk_upsert_pending([
            (make_exp_name(c), c.get("architecture", "LSTM"), dataset, c)
            for c in grid
        ])
    else:
        print("[agent] no local grid — running as pure consumer", flush=True)

    if args.register_only:
        return

    owned: set[str] = {make_exp_name(c) for c in grid}
    print(f"[agent] owns {len(owned)} configs (consume_any={consume_any})", flush=True)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # One worker process per device, each with its own slot budget.
    procs: list[mp.Process] = []
    for d in devices:
        p = mp.Process(target=worker_main,
                       args=(d, agent_id, dataset, backend_url,
                             checkpoint_every, poll_interval,
                             list(owned) if owned else None,
                             local_db_path, consume_any, api_key,
                             slots_for[d]),
                       name=f"worker-{d}")
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("[agent] interrupt — terminating workers...", flush=True)
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=10)
        sys.exit(1)

    print("[agent] all workers finished", flush=True)
    try:
        print(f"[agent] backend stats: {json.dumps(client.stats())}", flush=True)
    except Exception as e:
        print(f"[agent] backend stats unavailable: {e}", flush=True)
    print(f"[agent] local stats:   {json.dumps(local.summary())}", flush=True)


if __name__ == "__main__":
    main()
