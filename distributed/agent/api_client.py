"""
api_client.py — thin HTTP client for the distributed backend.
"""

from __future__ import annotations
import time
from typing import Any
from urllib.parse import quote

import requests


class BackendError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"backend {status}: {body}")
        self.status = status
        self.body = body


class BackendClient:
    def __init__(self, base_url: str, dataset: str, agent_id: str,
                 api_key: str | None = None,
                 timeout: float = 30.0, retries: int = 3, backoff: float = 1.5):
        self.base_url = base_url.rstrip("/")
        self.dataset = dataset
        self.agent_id = agent_id
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.s = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "X-Agent-ID": agent_id,
        }
        if api_key:
            headers["X-API-Key"] = api_key
        self.s.headers.update(headers)

    def _ds_url(self, *parts: str) -> str:
        path = "/".join(quote(p, safe="") for p in parts)
        return f"{self.base_url}/api/v1/datasets/{quote(self.dataset, safe='')}/{path}".rstrip("/")

    def _request(self, method: str, url: str, *, json=None,
                 ok_codes: tuple[int, ...] = (200, 201, 204),
                 retry_on: tuple[int, ...] = (502, 503, 504)) -> dict | None:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.s.request(method, url, json=json, timeout=self.timeout)
                if r.status_code in ok_codes:
                    if r.status_code == 204 or not r.content:
                        return None
                    return r.json()
                if r.status_code in retry_on:
                    last_err = BackendError(r.status_code, r.text)
                    time.sleep(self.backoff ** attempt)
                    continue
                raise BackendError(r.status_code, r.text)
            except requests.RequestException as e:
                last_err = e
                time.sleep(self.backoff ** attempt)
        if last_err:
            raise last_err
        raise RuntimeError("unreachable")

    def health(self) -> dict:
        r = self.s.get(f"{self.base_url}/api/v1/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create(self, exp_name: str, architecture: str, config: dict[str, Any]) -> tuple[bool, dict | None]:
        """Returns (created, body). created=False if already exists (HTTP 409)."""
        url = self._ds_url("experiments")
        try:
            body = self._request("POST", url, json={
                "exp_name": exp_name,
                "architecture": architecture,
                "config": config,
            }, ok_codes=(201,))
            return True, body
        except BackendError as e:
            if e.status == 409:
                return False, None
            raise

    def bulk_create(self, items: list[dict]) -> dict:
        url = self._ds_url("experiments", "bulk")
        body = self._request("POST", url, json={"experiments": items}, ok_codes=(200,))
        return body or {}

    def list(self, status: str | None = None, limit: int = 1000, offset: int = 0) -> list[dict]:
        params = []
        if status:
            params.append(f"status={quote(status)}")
        params.append(f"limit={limit}")
        params.append(f"offset={offset}")
        url = self._ds_url("experiments") + "?" + "&".join(params)
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text)
        return r.json().get("experiments", []) or []

    def get(self, exp_name: str) -> dict | None:
        url = self._ds_url("experiments", exp_name)
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text)
        return r.json()

    def claim(self, exp_name: str, device: str) -> dict | None:
        """Returns the experiment dict if claimed; None if it was not in claimable state (409)."""
        url = self._ds_url("experiments", exp_name, "claim")
        try:
            return self._request("POST", url, json={
                "agent_id": self.agent_id,
                "device": device,
            }, ok_codes=(200,))
        except BackendError as e:
            if e.status in (404, 409):
                return None
            raise

    def checkpoint(self, exp_name: str, step: int, elapsed_s: float, metrics: dict[str, float]) -> None:
        url = self._ds_url("experiments", exp_name, "checkpoints")
        self._request("POST", url, json={
            "step": step,
            "elapsed_s": elapsed_s,
            "metrics": metrics,
        }, ok_codes=(200,))

    def finish(self, exp_name: str, final_metrics: dict[str, float], duration_s: float,
               checkpoints: list[dict] | None = None) -> dict:
        url = self._ds_url("experiments", exp_name, "finish")
        payload: dict = {
            "final_metrics": final_metrics,
            "duration_s": duration_s,
        }
        if checkpoints:
            payload["checkpoints"] = checkpoints
        return self._request("POST", url, json=payload, ok_codes=(200,)) or {}

    def fail(self, exp_name: str, error: str) -> dict:
        url = self._ds_url("experiments", exp_name, "fail")
        return self._request("POST", url, json={"error": error}, ok_codes=(200,)) or {}

    def stats(self) -> dict:
        url = self._ds_url("stats")
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text)
        return r.json()

    def summary(self) -> dict:
        """Counts por estado + estadísticas de duración + ETA."""
        url = self._ds_url("summary")
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text)
        return r.json()

    # ── Cube / statistical endpoints ─────────────────────────────────────
    def _cube_get(self, *parts: str, params: dict | None = None) -> dict:
        from urllib.parse import urlencode
        url = self._ds_url("cube", *parts)
        if params:
            qs = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
            if qs:
                url += "?" + qs
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text)
        return r.json()

    def cube_metrics(self, status: str | None = None) -> dict:
        return self._cube_get("metrics", params={"status": status})

    def cube_params(self, status: str | None = None, sample: int | None = None) -> dict:
        return self._cube_get("params", params={"status": status, "sample": sample})

    def cube_param_values(self, key: str, metric: str | None = None,
                          where: str | None = None, status: str | None = None) -> dict:
        return self._cube_get("params", "values",
                              params={"key": key, "metric": metric,
                                      "where": where, "status": status})

    def cube_top(self, metric: str, limit: int = 10, order: str = "desc",
                 where: str | None = None, status: str | None = None) -> dict:
        return self._cube_get("top",
                              params={"metric": metric, "limit": limit,
                                      "order": order, "where": where, "status": status})

    def cube_groupby(self, by: str, metric: str, agg: str = "max,mean,count",
                     order: str | None = None, limit: int = 50,
                     where: str | None = None, status: str | None = None) -> dict:
        return self._cube_get("groupby",
                              params={"by": by, "metric": metric, "agg": agg,
                                      "order": order, "limit": limit,
                                      "where": where, "status": status})

    def cube_best_per(self, by: str, metric: str, order: str = "desc",
                      limit: int = 50, where: str | None = None,
                      status: str | None = None) -> dict:
        return self._cube_get("best-per",
                              params={"by": by, "metric": metric, "order": order,
                                      "limit": limit, "where": where, "status": status})

    def cube_distribution(self, metric: str, bins: int = 10,
                          where: str | None = None, status: str | None = None) -> dict:
        return self._cube_get("distribution",
                              params={"metric": metric, "bins": bins,
                                      "where": where, "status": status})

    def download_results_csv(self, out_path: str, **filters) -> str:
        """Download the per-dataset CSV results file. `filters` are passed as
        query params (status, architecture, agent_id, device, loss_type,
        limit, offset). Returns the saved path."""
        from urllib.parse import urlencode
        qs = urlencode({k: v for k, v in filters.items() if v is not None})
        url = self._ds_url("results.csv")
        if qs:
            url += "?" + qs
        with self.s.get(url, timeout=self.timeout * 4, stream=True) as r:
            if r.status_code != 200:
                raise BackendError(r.status_code, r.text)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        return out_path
