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


def test_speed_asc_then_size_ascending():
    """La prioridad pedida: rápido+SMALL → rápido+MEDIUM → rápido+LARGE."""
    parts = agent_mod.sort_spec_for("speed_asc").split(",")
    assert parts == ["config.epochs:asc", "compute_score:asc"]


def test_speed_desc_inverts_only_epochs():
    """speed_desc invierte epochs pero sigue prefiriendo el modelo pequeño."""
    parts = agent_mod.sort_spec_for("speed_desc").split(",")
    assert parts[0] == "config.epochs:desc"
    assert "compute_score:asc" in parts


def test_size_orders_by_compute_score_first():
    assert agent_mod.sort_spec_for("size_asc").split(",")[0] == "compute_score:asc"
    assert agent_mod.sort_spec_for("size_desc").split(",")[0] == "compute_score:desc"


def test_size_key_is_compute_score_not_hidden_dim():
    """hidden_dim es una aproximación que falla con Transformer: el término de
    atención (ws²·0.5) domina, así que un hd=32/ws=500 es LARGE."""
    from slots import get_compute_score, get_tier
    tr = {"architecture": "Transformer", "hidden_dim": 32,
          "num_layers": 1, "window_size": 500}
    lstm = {"architecture": "LSTM", "hidden_dim": 128,
            "num_layers": 2, "window_size": 500}
    assert get_tier(tr) == "LARGE"
    assert get_tier(lstm) == "MEDIUM"
    assert get_compute_score(tr) > get_compute_score(lstm), \
        "el Transformer LARGE debe puntuar por encima del LSTM MEDIUM"
    assert tr["hidden_dim"] < lstm["hidden_dim"], \
        "…pero por hidden_dim iría antes: por eso se ordena por compute_score"

    for po in agent_mod.PICK_ORDERS:
        spec = agent_mod.sort_spec_for(po)
        assert "config.hidden_dim" not in spec, f"{po} sigue usando hidden_dim"
        assert "compute_score" in spec, f"{po} no usa compute_score"


def test_sort_specs_only_use_safe_bson_paths():
    """El backend rechaza claves con caracteres raros: no le mandemos ninguna."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
    for po in agent_mod.PICK_ORDERS:
        for part in agent_mod.sort_spec_for(po).split(","):
            key, _, direction = part.partition(":")
            assert set(key) <= allowed, f"clave insegura: {key}"
            assert direction in ("asc", "desc"), f"dirección inválida: {direction}"


def test_register_grid_sends_score_and_tier(backend):
    """El backend no reimplementa la fórmula: la recibe ya calculada."""
    from slots import get_compute_score, get_tier
    grid = [
        {"architecture": "Transformer", "hidden_dim": 32, "num_layers": 1,
         "window_size": 500, "epochs": 1, "dataset": "ds"},
        {"architecture": "LSTM", "hidden_dim": 128, "num_layers": 2,
         "window_size": 500, "epochs": 1, "dataset": "ds"},
    ]

    sent = {}

    class FakeClient:
        def bulk_create(self, batch):
            for item in batch:
                sent[item["exp_name"]] = item
            return {"inserted": len(batch), "skipped": 0}

    agent_mod.register_grid(FakeClient(), "ds", grid)
    assert len(sent) == 2

    by_arch = {v["architecture"]: v for v in sent.values()}
    tr, lstm = by_arch["Transformer"], by_arch["LSTM"]

    assert tr["compute_score"] == get_compute_score(grid[0])
    assert tr["size_tier"] == get_tier(grid[0]) == "LARGE"
    assert lstm["size_tier"] == "MEDIUM"
    assert tr["compute_score"] > lstm["compute_score"], \
        "el LARGE debe ir después del MEDIUM en orden ascendente"


def test_register_grid_scores_every_architecture(backend):
    """MLP usa hidden_dims (lista); no puede quedarse sin score."""
    grid = [
        {"architecture": "MLP", "hidden_dims": [128, 64, 32], "epochs": 1, "dataset": "ds"},
        {"architecture": "CNN", "hidden_dim": 64, "num_layers": 1, "epochs": 1, "dataset": "ds"},
    ]
    sent = []

    class FakeClient:
        def bulk_create(self, batch):
            sent.extend(batch)
            return {"inserted": len(batch), "skipped": 0}

    agent_mod.register_grid(FakeClient(), "ds", grid)
    for item in sent:
        assert item["compute_score"] > 0, f"sin score: {item['exp_name']}"
        assert item["size_tier"] in ("SMALL", "MEDIUM", "LARGE")


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


# ── Robustez: un fallo del backend no puede matar al worker ──────────────────

def _calls_guarded_by_try(func, call_names: set[str]) -> dict[str, bool]:
    """Para cada nombre de llamada, ¿aparece SIEMPRE dentro de un try/except
    dentro de `func`? Se comprueba sobre el AST, no por texto."""
    import ast, inspect, textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))

    found = {n: [] for n in call_names}

    def call_name(node):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    def walk(node, in_try: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Try):
                for sub in child.body:
                    walk(sub, True)
                for h in child.handlers + child.orelse + child.finalbody:
                    walk(h, in_try)
                continue
            if isinstance(child, ast.Call):
                name = call_name(child)
                if name in found:
                    found[name].append(in_try)
            walk(child, in_try)

    walk(tree, False)
    return {n: (bool(v) and all(v)) for n, v in found.items()}


def test_worker_survives_backend_failure():
    """Un ReadTimeout mató un worker entero en producción: `worker_main` no
    capturaba nada alrededor de `claim_with_backoff`, así que la GPU quedaba
    parada hasta reiniciar a mano. El bucle debe retroceder y seguir."""
    guarded = _calls_guarded_by_try(
        agent_mod.worker_main, {"claim_with_backoff", "release"})
    assert guarded["claim_with_backoff"], \
        "claim_with_backoff debe ir dentro de try/except: si no, un timeout " \
        "mata el worker y la GPU queda parada"
    import inspect
    assert "MAX_ERROR_BACKOFF_S" in inspect.getsource(agent_mod.worker_main), \
        "falta el retroceso ante errores"


def test_error_backoff_is_bounded():
    assert 0 < agent_mod.MAX_ERROR_BACKOFF_S <= 300
    # Más corto que el de inactividad: aquí sí hay trabajo, solo no se puede pedir.
    assert agent_mod.MAX_ERROR_BACKOFF_S <= agent_mod.MAX_IDLE_BACKOFF_S


def test_release_failure_does_not_propagate():
    """Si el release falla, se avisa pero no se tumba el worker."""
    guarded = _calls_guarded_by_try(agent_mod.worker_main, {"release"})
    assert guarded["release"], "client.release debe ir dentro de try/except"


def test_client_timeout_exceeds_backend_http_timeout():
    """El backend corta a los 60 s; si el cliente cortara antes, un pico de
    carga se vería como ReadTimeout en vez de como respuesta lenta."""
    import inspect
    sig = inspect.signature(BackendClient.__init__)
    assert sig.parameters["timeout"].default > 60.0, \
        "el timeout del cliente debe superar los 60 s del backend"
    assert sig.parameters["retries"].default >= 3


# ── Entrega del resultado: no se puede perder nunca ──────────────────────────

@pytest.fixture
def tmp_unreported(tmp_path, monkeypatch):
    """Redirige el fichero de rescate al tmp del test."""
    monkeypatch.setattr(agent_mod, "unreported_path",
                        lambda ds: tmp_path / f"unreported_{ds}.jsonl")
    return tmp_path


class FlakyFinishClient:
    """Cliente que falla en `finish` N veces antes de aceptar."""

    def __init__(self, fail_times, dataset="ds", exc=None, status=None):
        self.left = fail_times
        self.dataset = dataset
        self.calls = 0
        self.accepted = None
        self._exc = exc
        self._status = status

    def finish(self, name, final_metrics, duration_s, checkpoints=None):
        self.calls += 1
        if self.left > 0:
            self.left -= 1
            if self._status is not None:
                raise BackendError(self._status, "nope")
            raise (self._exc or ConnectionError("red caida"))
        self.accepted = (name, final_metrics, duration_s, checkpoints)
        return {}


def test_finish_retries_until_backend_accepts(tmp_unreported):
    """El resultado ya costó lo que costó: se insiste hasta entregarlo."""
    c = FlakyFinishClient(fail_times=4)
    agent_mod.report_finish(c, "e1", {"macro_f1": 0.7}, 12.3, [], "cuda:0",
                            max_backoff=0.01)
    assert c.calls == 5
    assert c.accepted[0] == "e1"
    assert c.accepted[1]["macro_f1"] == 0.7


def test_finish_saves_to_disk_on_first_failure(tmp_unreported):
    """Antes de ponerse a reintentar, asegura el resultado en disco por si
    matan el proceso a mitad."""
    c = FlakyFinishClient(fail_times=2)
    agent_mod.report_finish(c, "e1", {"macro_f1": 0.7}, 12.3,
                            [{"step": 1000}], "cuda:0", max_backoff=0.01)
    path = agent_mod.unreported_path("ds")
    assert path.exists(), "no guardó el resultado antes de reintentar"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows[0]["exp_name"] == "e1"
    assert rows[0]["final_metrics"]["macro_f1"] == 0.7
    assert rows[0]["checkpoints"] == [{"step": 1000}]


def test_finish_treats_409_as_done(tmp_unreported):
    """409 = ya estaba done (proceso duplicado tras un release). El resultado
    está en el backend, así que no hay que insistir."""
    c = FlakyFinishClient(fail_times=99, status=409)
    agent_mod.report_finish(c, "e1", {"macro_f1": 0.7}, 1.0, [], "cuda:0",
                            max_backoff=0.01)
    assert c.calls == 1, "un 409 no debe reintentarse"


def test_finish_never_gives_up_on_5xx(tmp_unreported):
    c = FlakyFinishClient(fail_times=6, status=503)
    agent_mod.report_finish(c, "e1", {"macro_f1": 0.7}, 1.0, [], "cuda:0",
                            max_backoff=0.01)
    assert c.calls == 7


def test_successful_experiment_is_never_marked_failed():
    """El fallo que había: si `finish` reventaba, el `except` de fuera llamaba
    a `fail()` y registraba como FALLIDO un experimento que había ido bien."""
    src = __import__("inspect").getsource(agent_mod._run_one_task)
    assert "ResultNotReported" in src, \
        "debe distinguirse 'entrenamiento falló' de 'no pude entregarlo'"
    # La rama de ResultNotReported tiene que relanzar, no llamar a fail().
    idx = src.index("except ResultNotReported")
    branch = src[idx:src.index("except Exception", idx)]
    assert "client.fail" not in branch, \
        "un fallo de entrega no puede marcar el experimento como failed"
    assert "raise" in branch


def test_replay_resends_and_clears(tmp_unreported):
    ds = "ds"
    path = agent_mod.unreported_path(ds)
    path.write_text("\n".join(json.dumps({
        "exp_name": f"e{i}", "final_metrics": {"macro_f1": i},
        "duration_s": 1.0, "checkpoints": []}) for i in range(3)) + "\n")

    class OK:
        dataset = ds
        sent = []

        def finish(self, name, final_metrics, duration_s, checkpoints=None):
            self.sent.append(name)
            return {}

    c = OK()
    sent, left = agent_mod.replay_unreported(c, ds)
    assert sent == 3 and left == 0
    assert c.sent == ["e0", "e1", "e2"]
    assert not path.exists(), "el fichero debe borrarse al vaciarse"


def test_replay_keeps_what_still_fails(tmp_unreported):
    ds = "ds"
    path = agent_mod.unreported_path(ds)
    path.write_text("\n".join(json.dumps({
        "exp_name": f"e{i}", "final_metrics": {}, "duration_s": 1.0,
        "checkpoints": []}) for i in range(3)) + "\n")

    class HalfBroken:
        dataset = ds

        def finish(self, name, final_metrics, duration_s, checkpoints=None):
            if name == "e1":
                raise ConnectionError("sigue caido")
            return {}

    sent, left = agent_mod.replay_unreported(HalfBroken(), ds)
    assert sent == 2 and left == 1
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert [r["exp_name"] for r in rows] == ["e1"], \
        "solo debe quedar el que sigue fallando"


def test_replay_counts_409_as_delivered(tmp_unreported):
    ds = "ds"
    path = agent_mod.unreported_path(ds)
    path.write_text(json.dumps({"exp_name": "e0", "final_metrics": {},
                                "duration_s": 1.0, "checkpoints": []}) + "\n")

    class AlreadyDone:
        dataset = ds

        def finish(self, *a, **k):
            raise BackendError(409, "already done")

    sent, left = agent_mod.replay_unreported(AlreadyDone(), ds)
    assert sent == 1 and left == 0
    assert not path.exists()


def test_replay_survives_corrupt_lines(tmp_unreported):
    ds = "ds"
    path = agent_mod.unreported_path(ds)
    path.write_text('{"exp_name":"ok","final_metrics":{},"duration_s":1,"checkpoints":[]}\n'
                    'esto no es json\n'
                    '\n')

    class OK:
        dataset = ds

        def finish(self, *a, **k):
            return {}

    sent, left = agent_mod.replay_unreported(OK(), ds)
    assert sent == 1 and left == 0


def test_no_file_when_everything_works(tmp_unreported):
    """En marcha normal el fichero de rescate no debe ni existir."""
    c = FlakyFinishClient(fail_times=0)
    agent_mod.report_finish(c, "e1", {"macro_f1": 0.7}, 1.0, [], "cuda:0")
    assert not agent_mod.unreported_path("ds").exists()


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
                      "agent_id", "api_key", "checkpoint_every",
                      "heartbeat", "outcome"]
    # El padre pasa los args posicionalmente a mp.Process: si se reordenan aquí
    # sin tocar allí, el experimento arranca con la config en el sitio del device.
    src = inspect.getsource(agent_mod.worker_main)
    call = src[src.index("target=_run_one_task"):src.index("name=f\"task-")]
    for p in params:
        assert p in call or p in ("cfg", "name"), f"{p} no se pasa a mp.Process"


# ── OOM: la GPU llena no es un experimento roto ───────────────────────────────

def test_is_oom_reconoce_las_dos_formas():
    """Torch moderno lanza OutOfMemoryError; el antiguo, un RuntimeError."""
    oom = type("OutOfMemoryError", (Exception,), {})
    assert agent_mod._is_oom(oom("CUDA out of memory. Tried to allocate 1.44 GiB"))
    assert agent_mod._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert agent_mod._is_oom(RuntimeError("CUDA OUT OF MEMORY"))  # sin distinguir caja


def test_is_oom_no_traga_otros_fallos():
    """Un bug real debe seguir marcándose failed, no reciclarse para siempre."""
    assert not agent_mod._is_oom(RuntimeError("shape '[2, 3]' is invalid"))
    assert not agent_mod._is_oom(ValueError("bad config"))
    assert not agent_mod._is_oom(KeyboardInterrupt())


def test_oom_devuelve_a_pending_en_vez_de_marcar_failed():
    """El bucle ✗/▶: `failed` es reclamable al instante, así que marcar un OOM
    como failed hacía que el worker lo recogiera 88 ms después, para siempre."""
    src = __import__("inspect").getsource(agent_mod._run_one_task)
    idx = src.index("if _is_oom(e):")
    branch = src[idx:src.index("tb = traceback.format_exc()", idx)]
    assert "client.release" in branch, "un OOM debe volver a pending"
    assert "client.fail" not in branch, "un OOM no es un experimento inválido"
    assert "empty_cache" in branch, "hay que soltar la caché antes de rendirse"
    assert "OOM_SENTINEL" in branch, "el padre necesita enterarse para retroceder"


def test_worker_espera_antes_de_reclamar_tras_oom():
    """Sin la espera, el hueco del que acaba de morir se rellena al instante
    con otro que tampoco cabe."""
    src = __import__("inspect").getsource(agent_mod.worker_main)
    assert "pause_until" in src and "OUTCOME_OOM" in src
    # La espera tiene que ir ANTES de reclamar, no después.
    assert src.index("pause_until - time.time()") < src.index("claim_with_backoff"), \
        "el retroceso debe aplicarse antes del claim"


def test_backoff_de_oom_crece_y_tiene_tope():
    assert agent_mod.OOM_BACKOFF_S < agent_mod.MAX_OOM_BACKOFF_S
    b = agent_mod.OOM_BACKOFF_S
    for _ in range(20):
        b = min(b * 2, agent_mod.MAX_OOM_BACKOFF_S)
    assert b == agent_mod.MAX_OOM_BACKOFF_S, "debe saturar, no crecer sin límite"


# ── Watchdog: distinguir «va lento» de «está colgado» ─────────────────────────

def test_heartbeat_late_aunque_no_haya_progreso(monkeypatch):
    """El latido no mira el entrenamiento: por eso sirve para un run de 30s y
    para uno de 38h por igual."""
    import threading
    monkeypatch.setattr(agent_mod, "HEARTBEAT_S", 0.01)
    hb = type("V", (), {"value": 0.0})()
    stop = threading.Event()
    t = threading.Thread(target=agent_mod._heartbeat_thread, args=(hb, stop),
                         daemon=True)
    t.start()
    time.sleep(0.15)
    primero = hb.value
    assert primero > 0, "el latido debe haberse emitido sin ningún checkpoint"
    time.sleep(0.1)
    assert hb.value > primero, "el latido debe repetirse"
    stop.set()
    t.join(timeout=1)
    assert not t.is_alive(), "el hilo debe morir al terminar el experimento"


def test_umbral_de_cuelgue_deja_margen_a_varios_latidos():
    """Un umbral cerca de HEARTBEAT_S mataría runs sanos por un hipo de red."""
    assert agent_mod.STALL_TIMEOUT_S >= 3 * agent_mod.HEARTBEAT_S
    assert agent_mod.WATCHDOG_POLL_S <= agent_mod.STALL_TIMEOUT_S


def test_watchdog_mata_el_proceso_y_lo_devuelve_a_pending():
    src = __import__("inspect").getsource(agent_mod.worker_main)
    sup = src[src.index("def supervise"):]
    assert "proc.terminate()" in sup
    assert "proc.kill()" in sup, "si terminate no basta, hay que rematarlo"
    assert sup.index("proc.terminate()") < sup.index(".release(name)"), \
        "primero se mata el proceso, luego se devuelve el experimento"


def test_watchdog_mide_latido_y_no_checkpoints():
    """El fallo de fondo: `checkpoint_every` cuenta pasos, así que el hueco
    entre checkpoints de un run sano llegó a 6.95h. No vale como señal."""
    src = __import__("inspect").getsource(agent_mod.worker_main)
    sup = src[src.index("def supervise"):]
    assert "heartbeat.value" in sup
    assert "checkpoint" not in sup.lower().replace("checkpoint_every", ""), \
        "el watchdog no puede depender del progreso del entrenamiento"


def test_no_queda_process_pool():
    """El pool reutiliza procesos y no deja matar una tarea concreta."""
    src = __import__("inspect").getsource(agent_mod.worker_main)
    assert "executor.submit" not in src and "concurrent.futures" not in src
    assert "mp.Process" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
