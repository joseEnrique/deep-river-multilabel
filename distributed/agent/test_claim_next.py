#!/usr/bin/env python
"""
test_claim_next.py — cobertura del cambio a `claim-next`.

Dos bloques:
  • sort_spec_for / mapeo pick_order → sort de servidor (puro, sin red)
  • BackendClient.claim_next / release contra un backend HTTP de mentira que
    reproduce los códigos reales (200 / 404 / 409 / 5xx)

Ejecutar:  python -m pytest distributed/agent/test_claim_next.py -v
       o:  python distributed/agent/test_claim_next.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent as agent_mod
from api_client import BackendClient, BackendError
from slots import get_slots_needed


# ── pick_order → sort de servidor ────────────────────────────────────────────

def test_every_pick_order_has_a_sort_spec():
    """Ningún pick_order válido puede quedarse sin traducción."""
    for po in agent_mod.PICK_ORDERS:
        spec = agent_mod.sort_spec_for(po)
        assert spec, f"{po} no tiene sort"
        assert ":" in spec


def test_unknown_pick_order_falls_back_to_default():
    assert agent_mod.sort_spec_for("no-existe") == \
        agent_mod.sort_spec_for(agent_mod.DEFAULT_PICK_ORDER)


def test_speed_asc_orders_epochs_ascending_first():
    """speed_asc = primero lo barato: epochs asc y como clave PRIMARIA."""
    spec = agent_mod.sort_spec_for("speed_asc")
    first = spec.split(",")[0]
    assert first == "config.epochs:asc", f"clave primaria = {first}"


def test_speed_desc_inverts_only_epochs():
    """speed_desc invierte epochs pero sigue prefiriendo el modelo pequeño."""
    spec = agent_mod.sort_spec_for("speed_desc")
    parts = spec.split(",")
    assert parts[0] == "config.epochs:desc"
    assert "config.hidden_dim:asc" in parts


def test_size_orders_by_hidden_dim_first():
    assert agent_mod.sort_spec_for("size_asc").split(",")[0] == "config.hidden_dim:asc"
    assert agent_mod.sort_spec_for("size_desc").split(",")[0] == "config.hidden_dim:desc"


def test_sort_specs_only_use_safe_bson_paths():
    """El backend rechaza claves con caracteres raros: no le mandemos ninguna."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
    for po in agent_mod.PICK_ORDERS:
        for part in agent_mod.sort_spec_for(po).split(","):
            key, _, direction = part.partition(":")
            assert set(key) <= allowed, f"clave insegura: {key}"
            assert direction in ("asc", "desc"), f"dirección inválida: {direction}"


def test_sort_spec_matches_local_ordering_semantics():
    """El sort de servidor debe reproducir el orden que daba `slots.get_speed_rank`.

    Comprobamos la propiedad, no la implementación: con speed_asc, un ep=1 va
    antes que un ep=20 tanto en el ranking local como en el sort de Mongo.
    """
    from slots import get_speed_rank
    fast = {"epochs": 1, "hidden_dim": 32, "num_layers": 1}
    slow = {"epochs": 20, "hidden_dim": 32, "num_layers": 1}
    assert get_speed_rank(fast) < get_speed_rank(slow)
    assert agent_mod.sort_spec_for("speed_asc").startswith("config.epochs:asc")


# ── Backend de mentira ───────────────────────────────────────────────────────

class FakeBackend:
    """Servidor HTTP mínimo que imita las rutas que usa el agente."""

    def __init__(self):
        self.pending: list[dict] = []
        self.claimed: dict[str, dict] = {}
        self.calls: list[tuple[str, str, dict]] = []
        self.release_fail_status: int | None = None
        self.flaky_left = 0  # nº de 503 a devolver antes de responder bien
        self.throttle_left = 0  # nº de 429 (descarga de carga) antes de ceder
        self.retry_after: str | None = None

    def start(self):
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silencio
                pass

            def _send(self, code, payload, extra_headers=None):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra_headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    body = {}
                backend.calls.append(("POST", self.path, body))

                if backend.throttle_left > 0:
                    backend.throttle_left -= 1
                    hdrs = {"Retry-After": backend.retry_after} if backend.retry_after else None
                    return self._send(429, {"error": "too many requests"}, hdrs)

                if backend.flaky_left > 0:
                    backend.flaky_left -= 1
                    return self._send(503, {"error": "temporarily down"})

                if self.path.endswith("/claim-next"):
                    prefer = body.get("prefer_device")
                    for i, exp in enumerate(backend.pending):
                        if prefer and exp["config"].get("device") != prefer:
                            continue
                        exp = backend.pending.pop(i)
                        exp = {**exp, "status": "running",
                               "agent_id": body.get("agent_id"),
                               "device": body.get("device")}
                        backend.claimed[exp["exp_name"]] = exp
                        return self._send(200, exp)
                    return self._send(404, {"error": "no claimable experiment"})

                if self.path.endswith("/release"):
                    if backend.release_fail_status:
                        return self._send(backend.release_fail_status, {"error": "nope"})
                    name = self.path.split("/experiments/")[1].rsplit("/", 1)[0]
                    from urllib.parse import unquote
                    name = unquote(name)
                    exp = backend.claimed.pop(name, None)
                    if exp is None:
                        return self._send(409, {"error": "not running"})
                    backend.pending.insert(0, {k: v for k, v in exp.items()
                                               if k not in ("agent_id", "device")})
                    return self._send(200, {"status": "released"})

                return self._send(404, {"error": "unknown route"})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def backend():
    b = FakeBackend()
    url = b.start()
    b.url = url
    yield b
    b.stop()


def make_exp(name, epochs=1, hidden=32, device="cuda:0"):
    return {"exp_name": name, "status": "pending", "architecture": "LSTM",
            "config": {"epochs": epochs, "hidden_dim": hidden,
                       "num_layers": 1, "device": device,
                       "architecture": "LSTM"}}


def client_for(backend, agent_id="ag1"):
    return BackendClient(base_url=backend.url, dataset="ds",
                         agent_id=agent_id, api_key=None, retries=2, backoff=1.0)


# ── claim_next ───────────────────────────────────────────────────────────────

def test_claim_next_returns_experiment_and_sends_sort(backend):
    backend.pending = [make_exp("e1")]
    c = client_for(backend)
    got = c.claim_next(device="cuda:0", sort="config.epochs:asc")
    assert got["exp_name"] == "e1"
    assert got["status"] == "running"

    _, path, body = backend.calls[-1]
    assert path.endswith("/datasets/ds/claim-next")
    assert body["sort"] == "config.epochs:asc"
    assert body["agent_id"] == "ag1"
    assert body["device"] == "cuda:0"


def test_claim_next_returns_none_on_404(backend):
    """Cola vacía → None, no excepción: es el caso normal al terminar el grid."""
    backend.pending = []
    assert client_for(backend).claim_next(device="cuda:0") is None


def test_claim_next_omits_optional_fields_when_absent(backend):
    backend.pending = [make_exp("e1")]
    client_for(backend).claim_next(device="cuda:0")
    _, _, body = backend.calls[-1]
    assert "sort" not in body
    assert "prefer_device" not in body


def test_claim_next_prefer_device_filters(backend):
    """La afinidad de device gana al orden: solo devuelve los de esa GPU."""
    backend.pending = [make_exp("gpu1", device="cuda:1"),
                       make_exp("gpu0", device="cuda:0")]
    c = client_for(backend)
    got = c.claim_next(device="cuda:0", prefer_device="cuda:0")
    assert got["exp_name"] == "gpu0"

    # Agotada la afinidad → None, que es lo que dispara el reintento sin filtro.
    assert c.claim_next(device="cuda:0", prefer_device="cuda:0") is None

    # Sin filtro sí coge el de la otra GPU.
    assert c.claim_next(device="cuda:0")["exp_name"] == "gpu1"


def test_claim_next_never_returns_the_same_experiment_twice(backend):
    backend.pending = [make_exp(f"e{i}") for i in range(20)]
    c = client_for(backend)
    seen = []
    while (got := c.claim_next(device="cuda:0")) is not None:
        seen.append(got["exp_name"])
    assert len(seen) == 20
    assert len(set(seen)) == 20, "un experimento se entregó dos veces"


def test_claim_next_retries_on_5xx(backend):
    """Un 503 pasajero no debe perder el experimento."""
    backend.pending = [make_exp("e1")]
    backend.flaky_left = 1
    got = client_for(backend).claim_next(device="cuda:0")
    assert got is not None and got["exp_name"] == "e1"


def test_claim_next_raises_on_persistent_5xx(backend):
    backend.pending = [make_exp("e1")]
    backend.flaky_left = 99
    with pytest.raises(BackendError):
        client_for(backend).claim_next(device="cuda:0")


# ── Descarga de carga: 429 (el backend protegiéndose) ────────────────────────

def test_claim_next_retries_on_429_throttle(backend):
    """El backend descarga carga con 429 cuando llega a su tope de peticiones
    concurrentes. Eso significa 'vuelve luego', no 'ha fallado': si no se
    reintentara, proteger Mongo tumbaría a los agentes."""
    backend.pending = [make_exp("e1")]
    backend.throttle_left = 1
    got = client_for(backend).claim_next(device="cuda:0")
    assert got is not None and got["exp_name"] == "e1"


def test_429_is_in_the_retry_list():
    """Fija el contrato con el middleware del backend (chi descarga con 429)."""
    import inspect
    sig = inspect.signature(BackendClient._request)
    assert 429 in sig.parameters["retry_on"].default, \
        "el backend descarga con 429; sin reintento los agentes se caen"


def test_retry_after_header_is_honoured(backend):
    """Si el servidor dice cuánto esperar, se le hace caso."""
    backend.pending = [make_exp("e1")]
    backend.throttle_left = 1
    backend.retry_after = "0.3"
    c = client_for(backend)
    t0 = time.monotonic()
    assert c.claim_next(device="cuda:0") is not None
    assert time.monotonic() - t0 >= 0.3


def test_absurd_retry_after_is_capped(backend):
    """Una cabecera disparatada no puede dejar al agente parado media hora."""
    c = client_for(backend)

    class R:
        headers = {"Retry-After": "99999"}
    assert c._retry_delay(R(), 0) <= 60.0

    class Bad:
        headers = {"Retry-After": "no-soy-un-numero"}
    assert c._retry_delay(Bad(), 0) == c.backoff ** 0


# ── Retroceso cuando no hay trabajo ──────────────────────────────────────────

def test_idle_backoff_is_bounded_and_grows():
    """Con el grid terminado, los workers ociosos deben espaciar las consultas
    en vez de golpear al backend cada `poll_interval` indefinidamente."""
    assert agent_mod.MAX_IDLE_BACKOFF_S >= 60, "el tope es demasiado agresivo"
    assert agent_mod.MAX_IDLE_BACKOFF_S <= 600, "el tope tarda demasiado en reaccionar"

    poll = 5.0
    backoff = poll
    delays = []
    for _ in range(10):
        delays.append(backoff)
        backoff = min(backoff * 2, agent_mod.MAX_IDLE_BACKOFF_S)
    assert delays[0] == poll
    assert delays == sorted(delays), "el retroceso debe crecer, no oscilar"
    assert max(delays) <= agent_mod.MAX_IDLE_BACKOFF_S
    # En 10 vueltas ociosas se pasa de 10 consultas a bastantes menos.
    assert sum(1 for d in delays if d > poll) >= 8


def test_claim_with_backoff_makes_at_most_two_requests(backend):
    """Una petición por lanzamiento en régimen normal; dos solo si la afinidad
    de device se agotó. Es lo que acota la carga sobre Mongo."""
    backend.pending = [make_exp("e1", device="cuda:0")]
    c = client_for(backend)
    before = len(backend.calls)
    got = agent_mod.claim_with_backoff(c, "cuda:0", "config.epochs:asc", 5.0)
    assert got is not None
    assert len(backend.calls) - before == 1, "debería bastar una petición"

    # Sin nada para esta GPU: una con filtro (404) y otra sin él.
    backend.pending = [make_exp("e2", device="cuda:1")]
    before = len(backend.calls)
    got = agent_mod.claim_with_backoff(c, "cuda:0", "config.epochs:asc", 5.0)
    assert got is not None and got["exp_name"] == "e2"
    assert len(backend.calls) - before == 2

    # Cola vacía: dos peticiones y None, nunca un bucle.
    before = len(backend.calls)
    assert agent_mod.claim_with_backoff(c, "cuda:0", "config.epochs:asc", 5.0) is None
    assert len(backend.calls) - before == 2


# ── release ──────────────────────────────────────────────────────────────────

def test_release_returns_experiment_to_pending(backend):
    backend.pending = [make_exp("e1")]
    c = client_for(backend)
    c.claim_next(device="cuda:0")
    assert backend.pending == []

    assert c.release("e1") is True
    assert [e["exp_name"] for e in backend.pending] == ["e1"]
    # Y se puede volver a reclamar: no se ha perdido.
    assert c.claim_next(device="cuda:0")["exp_name"] == "e1"


def test_release_returns_false_when_not_running(backend):
    """409 → False, no excepción: el bucle no debe romperse por esto."""
    assert client_for(backend).release("no-existe") is False


def test_release_handles_names_with_slashes_and_specials(backend):
    """Los exp_name llevan puntos y guiones; deben ir URL-encoded."""
    name = "LSTM_alpi_ph1_h128_lr0.01_d0_s42_ep1_adam_none_BCE"
    backend.pending = [{**make_exp(name)}]
    c = client_for(backend)
    c.claim_next(device="cuda:0")
    assert c.release(name) is True
    assert backend.pending[0]["exp_name"] == name


# ── Interacción con el planificador de slots ─────────────────────────────────

def test_release_path_triggered_when_experiment_does_not_fit():
    """Un LARGE (4 slots) no cabe en 2 libres → hay que devolverlo.

    Reproduce la decisión del worker sin levantar el ProcessPoolExecutor.
    """
    large_cfg = {"architecture": "LSTM", "hidden_dim": 512, "num_layers": 2,
                 "epochs": 1, "window_size": 1}
    assert get_slots_needed(large_cfg) == 4

    available_slots = 2
    slots_needed = get_slots_needed(large_cfg)
    fits = slots_needed <= available_slots
    assert not fits, "un LARGE no debe caber en 2 slots"

    small_cfg = {"architecture": "LSTM", "hidden_dim": 32, "num_layers": 1,
                 "epochs": 1, "window_size": 1}
    assert get_slots_needed(small_cfg) <= 2


def test_slots_needed_is_only_2_or_4():
    """El código del worker asume esto (ya no hay ramas para 1 slot)."""
    seen = set()
    for hd in (16, 32, 64, 128, 256, 512):
        for nl in (1, 2, 3):
            for arch in ("LSTM", "CNN", "MLP", "Transformer"):
                seen.add(get_slots_needed({
                    "architecture": arch, "hidden_dim": hd,
                    "num_layers": nl, "window_size": 200}))
    assert seen <= {2, 4}, f"slots inesperados: {seen}"


# ── El mirror local ya no existe ─────────────────────────────────────────────

def test_local_db_is_gone():
    assert not (HERE / "local_db.py").exists()
    with pytest.raises(ImportError):
        import local_db  # noqa: F401


def test_agent_has_no_cache_machinery():
    """La caché de candidatos se eliminó: que no vuelva por la puerta de atrás."""
    for gone in ("_fetch_all", "_order_candidates", "CANDIDATE_PAGE",
                 "CANDIDATE_MAX", "DEFAULT_CANDIDATE_CACHE_S", "LocalDB"):
        assert not hasattr(agent_mod, gone), f"{gone} sigue existiendo"


def test_worker_main_signature_matches_spawn_call():
    """mp.Process pasa args posicionales: si la firma cambia, esto lo pilla."""
    import inspect
    params = list(inspect.signature(agent_mod.worker_main).parameters)
    assert params == [
        "device", "agent_id", "dataset", "base_url", "checkpoint_every",
        "poll_interval", "owned_filter", "consume_any", "api_key",
        "max_slots", "pick_order",
    ]


def test_run_one_task_signature_matches_submit_call():
    import inspect
    params = list(inspect.signature(agent_mod._run_one_task).parameters)
    assert params == ["name", "cfg", "device", "dataset", "base_url",
                      "agent_id", "api_key", "checkpoint_every"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
