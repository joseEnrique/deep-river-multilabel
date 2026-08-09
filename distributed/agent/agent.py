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
from api_client import BackendClient, BackendError


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


def register_grid(client: BackendClient, dataset: str,
                  grid: list[dict], chunk: int = 200) -> tuple[int, int]:
    """Idempotent populate of the backend: chunked POST /experiments/bulk
    (one HTTP call per chunk).

    Cada experimento lleva su `compute_score` y `size_tier` calculados aquí con
    `slots.py`, para que el backend pueda ordenar por tamaño real de modelo sin
    tener que reimplementar la fórmula ni aproximarla por `hidden_dim`.

    Returns (inserted, skipped) as reported by the backend."""
    if not grid:
        return 0, 0
    from slots import get_compute_score, get_tier

    items_backend: list[dict] = []
    for c in grid:
        name = make_exp_name(c)
        arch = c.get("architecture", "LSTM")
        items_backend.append({
            "exp_name": name,
            "architecture": arch,
            "config": c,
            "compute_score": get_compute_score(c),
            "size_tier": get_tier(c),
        })

    total_ins, total_skip = 0, 0
    for i in range(0, len(items_backend), chunk):
        batch = items_backend[i:i + chunk]
        res = client.bulk_create(batch)
        total_ins += int(res.get("inserted", 0))
        total_skip += int(res.get("skipped", 0))
    return total_ins, total_skip


# ── Worker (one process per GPU) ──────────────────────────────────────────────

PICK_ORDERS = ("speed_asc", "speed_desc", "size_asc", "size_desc", "slots")
DEFAULT_PICK_ORDER = "speed_asc"


def _resolve_pick_order(value) -> str:
    """Normaliza `pick_order` del YAML; valor desconocido → default con aviso."""
    if value is None:
        return DEFAULT_PICK_ORDER
    v = str(value).strip().lower()
    if v in PICK_ORDERS:
        return v
    print(f"[agent] pick_order='{value}' desconocido; usando "
          f"'{DEFAULT_PICK_ORDER}' (válidos: {', '.join(PICK_ORDERS)})", flush=True)
    return DEFAULT_PICK_ORDER


# pick_order → sort de servidor (BSON paths, se los pasamos a `claim-next`).
#
# El orden lo aplica Mongo sobre la cola entera, no el agente sobre una ventana.
# Dos claves:
#
#   • `config.epochs` para la velocidad — más fino que el tier FAST/MEDIUM/SLOW
#     (ordena 1 → 3 → 10 → 20 en vez de agrupar en tres cubos).
#   • `compute_score` para el tamaño — el MISMO valor que calcula
#     `slots.get_compute_score`, guardado en el documento al registrar.
#
# El score va en el documento, y no se ordena por `config.hidden_dim`, porque
# para un Transformer el término de atención (ws²·0.5) domina: un
# `hd=32, ws=500` puntúa 126.024 (LARGE) mientras que un LSTM `hd=128, nl=2`
# puntúa 32.768 (MEDIUM). Por hidden_dim el LARGE adelantaría al MEDIUM.
#
# Con `speed_asc`, el orden resultante es exactamente:
#   rápido+SMALL → rápido+MEDIUM → rápido+LARGE → lento+SMALL → ...
_SORT_BY_PICK_ORDER = {
    "speed_asc":  "config.epochs:asc,compute_score:asc",
    "speed_desc": "config.epochs:desc,compute_score:asc",
    "size_asc":   "compute_score:asc,config.epochs:asc",
    "size_desc":  "compute_score:desc,config.epochs:asc",
    "slots":      "compute_score:asc",
}


def sort_spec_for(pick_order: str) -> str:
    """Sort string que se manda a `claim-next` para un `pick_order` dado."""
    return _SORT_BY_PICK_ORDER.get(pick_order, _SORT_BY_PICK_ORDER[DEFAULT_PICK_ORDER])


# Tope del retroceso cuando no hay trabajo. Con el grid terminado, los workers
# que aún tienen tareas en vuelo pasan a preguntar cada 2 minutos en vez de
# cada `poll_interval`, que es lo que evita machacar al backend sin motivo.
MAX_IDLE_BACKOFF_S = 120.0

# Tope del retroceso cuando el backend falla (timeout, 5xx, red caída). Más
# corto que el de inactividad: aquí sí hay trabajo, solo no se puede pedir.
MAX_ERROR_BACKOFF_S = 60.0

# Retroceso tras un OOM de CUDA. Un OOM no dice que el experimento sea inválido,
# sino que la GPU está llena AHORA: se devuelve a `pending` y se espera a que
# las tareas en vuelo liberen memoria. Sin esta espera el worker vuelve a
# reclamar en milisegundos y entra en un bucle ✗/▶ que quema un slot para
# siempre.
OOM_BACKOFF_S = 30.0
MAX_OOM_BACKOFF_S = 300.0

# Sentinela que `_run_one_task` devuelve al padre para distinguir un OOM de un
# fallo normal. El proceso hijo no puede tocar el estado del worker.
OOM_SENTINEL = "__oom__"


def _is_oom(exc: BaseException) -> bool:
    """True si la excepción es una falta de memoria de CUDA.

    No se importa torch aquí (el padre nunca inicializa CUDA), así que se mira
    el nombre de la clase además del mensaje: `torch.cuda.OutOfMemoryError` no
    existe en versiones antiguas, donde el OOM llega como `RuntimeError`.
    """
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def claim_with_backoff(client: BackendClient, device: str, sort_spec: str,
                       poll_interval: float):
    """Reclama el siguiente experimento: primero con afinidad de device, luego
    sin ella.

    Son dos peticiones como mucho, y solo cuando hay un slot libre — es decir,
    como mucho una vez por lanzamiento. El reintento sin filtro solo ocurre si
    la afinidad se agotó, así que en régimen normal es una sola petición.
    """
    exp = client.claim_next(device=device, sort=sort_spec, prefer_device=device)
    if exp is None:
        exp = client.claim_next(device=device, sort=sort_spec)
    return exp


class ResultNotReported(Exception):
    """El experimento terminó bien pero no se pudo entregar el resultado.

    Distinta de un fallo de entrenamiento: el resultado existe y está a salvo
    en disco, solo falta que el backend lo acepte. Nunca debe traducirse en un
    `fail()`, que registraría como fallido algo que sí funcionó."""


def unreported_path(dataset: str) -> Path:
    """Fichero de rescate: resultados calculados que aún no aceptó el backend."""
    return HERE / f"unreported_{dataset}.jsonl"


def save_unreported(dataset: str, payload: dict) -> None:
    """Vuelca un resultado a disco en modo append.

    Es la última red de seguridad: si matan el proceso mientras reintenta, el
    resultado sigue ahí y se reenvía luego con --replay-unreported. Solo se
    escribe cuando falla la entrega, así que en marcha normal no existe."""
    try:
        with open(unreported_path(dataset), "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:  # noqa: BLE001 — el disco tampoco puede tumbarnos
        print(f"[agent] no se pudo guardar el resultado en disco: {e}", flush=True)


def report_finish(client: BackendClient, name: str, final_m: dict,
                  duration: float, checkpoints: list[dict], device: str,
                  max_backoff: float = MAX_ERROR_BACKOFF_S) -> None:
    """Entrega el resultado al backend. Reintenta indefinidamente.

    No hay número de reintentos que valga: el resultado ya costó lo que costó,
    así que se insiste con retroceso acotado hasta que el backend lo acepte.
    Un 409 significa que ya estaba `done` (por ejemplo un proceso duplicado
    tras un release) y se da por bueno: el resultado está en el backend.
    """
    payload = {"exp_name": name, "final_metrics": final_m,
               "duration_s": duration, "checkpoints": checkpoints}
    backoff = 1.0
    attempt = 0
    while True:
        attempt += 1
        try:
            client.finish(name, final_m, duration, checkpoints=checkpoints)
            if attempt > 1:
                print(f"[{device}]   resultado entregado tras {attempt} intentos",
                      flush=True)
            return
        except BackendError as e:
            if e.status == 409:
                print(f"[{device}]   ya estaba done en el backend; nada que hacer",
                      flush=True)
                return
            last = e
        except KeyboardInterrupt:
            save_unreported(client.dataset, payload)
            print(f"[{device}]   interrumpido: resultado guardado en "
                  f"{unreported_path(client.dataset).name}", flush=True)
            raise ResultNotReported(name)
        except Exception as e:  # noqa: BLE001 — red caída, DNS, lo que sea
            last = e

        # Primer fallo: asegurar el resultado en disco antes de seguir
        # insistiendo, por si matan el proceso a mitad.
        if attempt == 1:
            save_unreported(client.dataset, payload)
            print(f"[{device}]   backend no acepta el resultado ({last}); "
                  f"guardado en {unreported_path(client.dataset).name}, "
                  f"reintentando", flush=True)

        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


def replay_unreported(client: BackendClient, dataset: str) -> tuple[int, int]:
    """Reenvía los resultados que quedaron en disco. Devuelve (enviados, pendientes)."""
    path = unreported_path(dataset)
    if not path.exists():
        return 0, 0
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    sent, leftover = 0, []
    for r in rows:
        try:
            client.finish(r["exp_name"], r["final_metrics"], r["duration_s"],
                          checkpoints=r.get("checkpoints") or None)
            sent += 1
        except BackendError as e:
            if e.status == 409:
                sent += 1  # ya estaba done: nada que reenviar
            else:
                leftover.append(r)
        except Exception:  # noqa: BLE001
            leftover.append(r)

    if leftover:
        path.write_text("\n".join(json.dumps(r) for r in leftover) + "\n")
    else:
        path.unlink(missing_ok=True)
    return sent, len(leftover)


def _run_one_task(name: str, cfg: dict, device: str, dataset: str,
                  base_url: str, agent_id: str, api_key: str | None,
                  checkpoint_every: int) -> str | None:
    """Subproceso: ejecuta un experimento ya claim-eado y reporta al backend.

    Devuelve `OOM_SENTINEL` si murió por falta de memoria de CUDA (para que el
    worker padre retroceda antes de volver a reclamar), `None` en cualquier
    otro caso."""
    # Import torch inside the child to keep CUDA init isolated per process.
    import torch  # noqa: F401
    from runner_adapter import run_with_backend

    client = BackendClient(base_url=base_url, dataset=dataset, agent_id=agent_id,
                           api_key=api_key)

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
            # El resultado ya está calculado: cueste lo que cueste hay que
            # entregarlo. Reintento indefinido — perder esto es perder horas
            # de GPU, así que se insiste hasta que el backend responda.
            report_finish(client, name, final_m, duration, checkpoints_buf, device)
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
        try:
            client.fail(name, "KeyboardInterrupt")
        except Exception:
            pass
        raise
    except ResultNotReported:
        # El experimento terminó BIEN; lo que falló fue entregarlo, y ya está
        # guardado en disco para reenviarlo. Marcarlo `failed` aquí sería
        # mentir sobre el resultado, así que se deja en `running`.
        raise
    except Exception as e:
        if _is_oom(e):
            # La GPU está llena, pero el experimento es perfectamente válido:
            # marcarlo `failed` sería mentir sobre él y además lo dejaría
            # reclamable al instante (el backend re-reparte los `failed`), que
            # es justo el bucle que hay que evitar. Se devuelve a `pending` y
            # se avisa al padre para que espere.
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — liberar caché es best-effort
                pass
            print(f"[{device}] ⚠  OOM en {name}: {e}", flush=True)
            try:
                client.release(name)
            except Exception as re:  # noqa: BLE001
                print(f"[{device}] (no se pudo devolver a pending: {re})", flush=True)
            return OOM_SENTINEL

        tb = traceback.format_exc()
        msg = f"{e}\n{tb[:3000]}"
        print(f"[{device}] ✗  {name}: {e}", flush=True)
        try:
            client.fail(name, msg)
        except Exception as fe:
            print(f"[{device}] (failed to report failure: {fe})", flush=True)
    return None


def worker_main(device: str, agent_id: str, dataset: str, base_url: str,
                checkpoint_every: int, poll_interval: float,
                owned_filter: list[str] | None,
                consume_any: bool,
                api_key: str | None,
                max_slots: int,
                pick_order: str = DEFAULT_PICK_ORDER) -> None:
    """Despachador con slots: hasta `max_slots` experimentos concurrentes en `device`.
    SMALL/MEDIUM=2 slots, LARGE=4 slots (slots.get_slots_needed).

    El siguiente experimento lo elige y reclama el backend en una sola operación
    atómica (`claim-next`), así que no hay cola cacheada ni carreras entre
    agentes por el mismo documento."""
    import threading
    import concurrent.futures
    from slots import get_slots_needed

    client = BackendClient(base_url=base_url, dataset=dataset, agent_id=agent_id,
                           api_key=api_key)
    pid = os.getpid()
    sort_spec = sort_spec_for(pick_order)
    if not consume_any:
        # `claim-next` ordena y reclama en el servidor, que no sabe qué configs
        # registró este agente. El filtro owned-only ya no es aplicable.
        print(f"[{device}] aviso: consume_any=false ya no se soporta con "
              f"claim-next; consumiendo cualquier experimento pendiente",
              flush=True)
    print(f"[{device}] worker started (pid={pid}, agent={agent_id}, "
          f"max_slots={max_slots}, pick_order={pick_order}, sort={sort_spec})",
          flush=True)

    available_slots = max_slots
    cond = threading.Condition()

    # Instante hasta el que no se reclama nada nuevo por haber sufrido un OOM,
    # y retroceso actual (se dobla con cada OOM seguido, se reinicia al primer
    # experimento que termina bien).
    oom_until = 0.0
    oom_backoff = OOM_BACKOFF_S

    def release_cb(slots_used: int):
        def _cb(fut):
            nonlocal available_slots, oom_until, oom_backoff
            try:
                outcome = fut.result()
            except Exception:  # noqa: BLE001 — KeyboardInterrupt/ResultNotReported
                outcome = None
            with cond:
                available_slots += slots_used
                if outcome == OOM_SENTINEL:
                    oom_until = max(oom_until, time.time() + oom_backoff)
                    print(f"[{device}] OOM: pauso {oom_backoff:.0f}s antes de "
                          f"reclamar (slots libres: {available_slots})", flush=True)
                    oom_backoff = min(oom_backoff * 2, MAX_OOM_BACKOFF_S)
                else:
                    oom_backoff = OOM_BACKOFF_S
                cond.notify_all()
        return _cb

    # Retroceso exponencial para no martillear al backend cuando no hay nada
    # que coger. Sin esto, N workers ociosos golpearían Mongo cada
    # `poll_interval` segundos indefinidamente al terminar el grid.
    idle_backoff = poll_interval
    # Retroceso separado para fallos del backend, para no confundir "no hay
    # trabajo" con "el backend no contesta".
    error_backoff = 0.0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_slots) as executor:
        while True:
            with cond:
                while available_slots == 0:
                    cond.wait()

            # Tras un OOM se espera a que las tareas en vuelo liberen memoria
            # antes de volver a pedir trabajo: si no, el hueco que deja el
            # experimento que acaba de morir se rellena al instante con otro
            # que tampoco cabe.
            with cond:
                wait_s = oom_until - time.time()
            if wait_s > 0:
                time.sleep(min(wait_s, MAX_OOM_BACKOFF_S))
                continue

            # El backend elige Y reclama en una sola operación atómica. Primero
            # con afinidad de device (preferencia blanda: los configs registrados
            # para esta GPU), y si no hay ninguno, cualquiera.
            #
            # Un fallo transitorio del backend (timeout, 5xx tras agotar los
            # reintentos del cliente, corte de red) NO puede matar al worker:
            # eso dejaría la GPU parada hasta un reinicio manual. Se retrocede
            # y se sigue intentando.
            try:
                exp = claim_with_backoff(client, device, sort_spec, poll_interval)
            except Exception as e:  # noqa: BLE001 — cualquier fallo de red/backend
                error_backoff = min(max(error_backoff * 2, poll_interval),
                                    MAX_ERROR_BACKOFF_S)
                print(f"[{device}] backend no disponible ({type(e).__name__}: "
                      f"{e}); reintento en {error_backoff:.0f}s", flush=True)
                time.sleep(error_backoff)
                continue

            error_backoff = 0.0  # el backend responde: reinicia el retroceso

            if exp is None:
                # Nada pendiente. Si todos los slots están libres, salimos;
                # si hay tareas en vuelo, esperamos a que se vacíen.
                with cond:
                    if available_slots == max_slots:
                        print(f"[{device}] nothing pending, exiting worker", flush=True)
                        return
                time.sleep(idle_backoff)
                idle_backoff = min(idle_backoff * 2, MAX_IDLE_BACKOFF_S)
                continue

            idle_backoff = poll_interval  # hubo trabajo: vuelve al ritmo normal

            name = exp["exp_name"]
            cfg = exp["config"]
            slots_needed = get_slots_needed(cfg)

            with cond:
                fits = slots_needed <= available_slots
                if fits:
                    available_slots -= slots_needed

            if not fits:
                # Ya está reclamado, así que hay que devolverlo: si no, se
                # quedaría en `running` sin que nadie lo ejecute. Si el
                # release falla se queda huérfano, pero tumbar el worker por
                # eso sería peor: se avisa y se sigue.
                try:
                    client.release(name)
                except Exception as e:  # noqa: BLE001
                    print(f"[{device}] no se pudo devolver {name} a pending "
                          f"({e}); quedará en running hasta liberarlo a mano",
                          flush=True)
                time.sleep(poll_interval)
                continue

            fut = executor.submit(
                _run_one_task,
                name, cfg, device, dataset, base_url, agent_id,
                api_key, checkpoint_every,
            )
            fut.add_done_callback(release_cb(slots_needed))


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
    ap.add_argument("--replay-unreported", action="store_true",
                    help="Reenvía los resultados que quedaron sin entregar "
                         "(unreported_<dataset>.jsonl) y sale.")
    ap.add_argument("--backfill-scores", action="store_true",
                    help="Calcula compute_score/size_tier en los experimentos "
                         "ya registrados y sale. Necesario una vez tras "
                         "actualizar; idempotente.")
    ap.add_argument("--backfill-all", action="store_true",
                    help="Con --backfill-scores: recalcula TODOS en vez de "
                         "solo los que no lo tienen.")
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
    pick_order = _resolve_pick_order(cfg.get("pick_order"))
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
    print(f"[agent] id={agent_id} backend={backend_url} dataset={dataset} "
          f"devices={devices} consume_any={consume_any}", flush=True)
    print(f"[agent] slots/device: {slots_for}", flush=True)
    print(f"[agent] pick_order: {pick_order} -> sort={sort_spec_for(pick_order)}", flush=True)

    if args.replay_unreported:
        path = unreported_path(dataset)
        if not path.exists():
            print(f"[agent] no hay resultados pendientes ({path.name})", flush=True)
            return
        sent, left = replay_unreported(client, dataset)
        print(f"[agent] reenviados={sent} pendientes={left}", flush=True)
        if left:
            print(f"[agent] quedan {left} en {path.name}; vuelve a intentarlo "
                  f"cuando el backend esté disponible", flush=True)
        return

    if args.backfill_scores:
        scope = "TODOS" if args.backfill_all else "solo los que faltan"
        print(f"[agent] backfill de compute_score/size_tier ({scope})...", flush=True)
        t0 = time.time()
        res = client.backfill_scores(recompute_all=args.backfill_all)
        print(f"[agent] matched={res.get('matched')} "
              f"modified={res.get('modified')} "
              f"sin_score={res.get('still_missing')} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if res.get("still_missing"):
            print("[agent] AVISO: quedan documentos sin compute_score; se "
                  "colarían al principio de la cola (un campo ausente ordena "
                  "primero en ascendente).", flush=True)
        return

    if args.status:
        print("[agent] backend stats:")
        print(json.dumps(client.stats(), indent=2))
        return

    if args.summary:
        import re
        from slots import get_tier
        s = client.summary()

        def fmt_eta(secs: float) -> str:
            secs = int(secs)
            h, rem = divmod(secs, 3600)
            m, sec = divmod(rem, 60)
            return f"{h}h{m:02d}m{sec:02d}s"

        def parse_cfg_from_name(name: str) -> dict:
            """Extrae arch, hidden_dim(s), num_layers, window_size, past_history, epochs."""
            m = re.match(
                r"^(?P<arch>[A-Za-z]+)_[^_]+_ph(?P<ph>\d+)_h(?P<hd>[\d\-]+)_nl(?P<nl>\d+)_w(?P<ws>\d+)_",
                name,
            )
            if not m:
                return {}
            hd_raw = m.group("hd")
            cfg = {
                "architecture": m.group("arch"),
                "num_layers":   int(m.group("nl")),
                "window_size":  int(m.group("ws")),
                "past_history": int(m.group("ph")),
            }
            if "-" in hd_raw:
                cfg["hidden_dims"] = [int(x) for x in hd_raw.split("-")]
            else:
                cfg["hidden_dim"] = int(hd_raw)
            m_ep = re.search(r"_ep(\d+)_", name)
            if m_ep:
                cfg["epochs"] = int(m_ep.group(1))
            return cfg

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
                label = get_tier(cfg_inferred) if cfg_inferred else "?"
                tiers[label] += 1
                key = (r.get("agent_id") or "-", r.get("device") or "-")
                by_node[key][label] += 1
                running_with_tier.append((r, label))

            print(f"\n[running] {len(running)} experimento(s) en curso:")
            print(f"  tiers: SMALL={tiers['SMALL']}  MEDIUM={tiers['MEDIUM']}  "
                  f"LARGE={tiers['LARGE']}" + (f"  ?={tiers['?']}" if tiers['?'] else ""))

            print(f"\n[by node × device]")
            print(f"  {'AGENT':<24} {'DEVICE':<10} {'SMALL':>6} {'MEDIUM':>7} {'LARGE':>6}  TOTAL")
            for (ag, dev), c in sorted(by_node.items()):
                total = c["SMALL"] + c["MEDIUM"] + c["LARGE"] + c["?"]
                print(f"  {ag:<24} {dev:<10} {c['SMALL']:>6} {c['MEDIUM']:>7} {c['LARGE']:>6}  {total:>5}")

            print(f"\n  {'TIER':<7} {'ARCH':<12} {'AGENT':<24} {'DEVICE':<10} {'ELAPSED':>10}  EXP_NAME")
            for r, label in running_with_tier:
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
        ins, skip = register_grid(client, dataset, grid)
        print(f"[agent] registered: inserted={ins} skipped(existing)={skip}", flush=True)
    elif not grid:
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
                             consume_any, api_key,
                             slots_for[d], pick_order),
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


if __name__ == "__main__":
    main()
