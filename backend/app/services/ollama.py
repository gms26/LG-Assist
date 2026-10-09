"""Ollama availability checks.

Used by the health endpoint and (later) the provider layer. Never raises:
an unreachable Ollama is a normal degraded state, reported as data.
"""

import time

import httpx

from ..config import Settings


def check_ollama(settings: Settings, timeout_s: float = 3.0) -> dict:
    """Return a status dict for the health endpoint's `ollama` check."""
    started = time.perf_counter()
    try:
        # trust_env=False: Ollama is a local service and must never go through
        # an egress proxy (in proxied/sandboxed networks the proxy breaks or
        # leaks local calls; the failure mode is a confusing InvalidURL).
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout_s, trust_env=False)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return {
            "status": "ok",
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "models": models,
        }
    except Exception as exc:  # connection refused, timeout, bad response...
        return {
            "status": "unreachable",
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "detail": f"{type(exc).__name__}: {exc}",
        }
