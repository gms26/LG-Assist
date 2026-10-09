"""Model provider layer (contract v1 + Phase 3 seam).

Responsibilities:
  * resolve which provider is actually in effect, and whether a fallback
    is in effect (surfaced via GET /config/model so the frontend never
    assumes a provider);
  * stream chat completions from Ollama (local, required) and from
    OpenAI-compatible cloud endpoints (Groq, OpenRouter, Gemini).

Fallback rule: if a cloud provider is configured but no API key is present
(or the call fails at request time in Phase 3), the active provider drops
back to Ollama and the response carries a notice saying so.
"""

from collections.abc import Iterator

import httpx

from ..config import SUPPORTED_PROVIDERS, Settings

# OpenAI-compatible chat-completion endpoints (all accept a Bearer key and
# {"model","messages","stream":true}).
CLOUD_CHAT_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}

# The documented free cloud option (item 4). Groq's free tier is the default
# because it serves the same OpenAI-compatible protocol as the alternates;
# switch LGA_CLOUD_CHAT_MODEL for the others.
DEFAULT_CLOUD_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
}


class ProviderUnavailable(Exception):
    """The model provider (Ollama or cloud) could not serve the request."""


def resolve_provider(settings: Settings) -> dict:
    """Return the effective provider state for /config/model.

    ``fallback_in_effect`` is True when a cloud provider was requested but
    cannot be used (no key), so Ollama serves instead. Runtime call
    failures flip to fallback inside the chat path (Phase 3).
    """
    configured = settings.effective_provider
    if configured == "ollama":
        return {"active": "ollama", "fallback_in_effect": False}
    if not settings.cloud_configured():
        return {"active": "ollama", "fallback_in_effect": True}
    return {"active": configured, "fallback_in_effect": False}


def provider_info(settings: Settings) -> dict:
    state = resolve_provider(settings)
    active = state["active"]
    if active == "ollama":
        chat_model = settings.ollama_chat_model
    else:
        chat_model = settings.cloud_chat_model or DEFAULT_CLOUD_MODELS[active]
    return {
        "active_provider": active,
        "configured_provider": settings.effective_provider,
        "chat_model": chat_model,
        "embed_model": settings.ollama_embed_model,
        "available_providers": list(SUPPORTED_PROVIDERS),
        "fallback_in_effect": state["fallback_in_effect"],
        "cloud_configured": settings.cloud_configured(),
    }


def stream_chat_ollama(
    settings: Settings, messages: list[dict], timeout_s: float | None = None
) -> Iterator[str]:
    """Yield text chunks from Ollama's /api/chat (streaming)."""
    payload = {
        "model": settings.ollama_chat_model,
        "messages": messages,
        "stream": True,
    }
    with httpx.stream(
        "POST",
        f"{settings.ollama_base_url}/api/chat",
        json=payload,
        timeout=timeout_s or settings.ollama_timeout_s,
        trust_env=False,  # local service: never via egress proxy
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            import json

            chunk = json.loads(line)
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield text
            if chunk.get("done"):
                break


def stream_chat_cloud(
    settings: Settings, messages: list[dict], timeout_s: float | None = None
) -> Iterator[str]:
    """Yield text chunks from an OpenAI-compatible cloud endpoint (streaming)."""
    url = CLOUD_CHAT_URLS[settings.effective_provider]
    model = settings.cloud_chat_model or DEFAULT_CLOUD_MODELS[settings.effective_provider]
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {settings.cloud_api_key}"}
    with httpx.stream("POST", url, json=payload, headers=headers,
                      timeout=timeout_s or 60.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            import json

            delta = json.loads(data)["choices"][0]["delta"]
            if delta.get("content"):
                yield delta["content"]


def generate_stream(
    settings: Settings, messages: list[dict], timeout_s: float | None = None
) -> Iterator[tuple[str, str]]:
    """Yield ("token", text) chunks, optionally preceded by ("notice", msg).

    Uses the resolved active provider. If a cloud call fails mid-request,
    falls back to Ollama and emits a notice first (contract: the frontend
    shows "fell back to local model"). Raises ProviderUnavailable when
    Ollama itself cannot serve.
    """
    active = resolve_provider(settings)["active"]
    try:
        streamer = stream_chat_ollama if active == "ollama" else stream_chat_cloud
        for chunk in streamer(settings, messages, timeout_s):
            yield ("token", chunk)
    except Exception as exc:
        if active == "ollama":
            raise ProviderUnavailable(f"ollama unavailable: {type(exc).__name__}") from exc
        try:
            yield ("notice", "fell back to local model")
            for chunk in stream_chat_ollama(settings, messages, timeout_s):
                yield ("token", chunk)
        except Exception as exc2:
            raise ProviderUnavailable(
                f"cloud and ollama fallback failed: {type(exc2).__name__}"
            ) from exc2


def chat_once(settings: Settings, messages: list[dict], timeout_s: float = 20.0) -> str:
    """Non-streaming chat via the active provider (used for rewrites)."""
    return "".join(
        payload for kind, payload in generate_stream(settings, messages, timeout_s)
        if kind == "token"
    )
