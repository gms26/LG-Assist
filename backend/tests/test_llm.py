"""Provider resolution + fallback tests (item 4, no network needed)."""

import pytest

from app.config import Settings
from app.services import llm
from app.services.llm import (
    ProviderUnavailable,
    generate_stream,
    provider_info,
    resolve_provider,
)


def test_default_resolves_to_ollama_no_fallback():
    s = Settings()
    state = resolve_provider(s)
    assert state == {"active": "ollama", "fallback_in_effect": False}


def test_cloud_without_key_falls_back_to_ollama(monkeypatch):
    monkeypatch.setenv("LGA_PROVIDER", "groq")
    s = Settings()
    state = resolve_provider(s)
    assert state == {"active": "ollama", "fallback_in_effect": True}


def test_cloud_with_key_is_active(monkeypatch):
    monkeypatch.setenv("LGA_PROVIDER", "groq")
    monkeypatch.setenv("LGA_CLOUD_API_KEY", "k")
    s = Settings()
    state = resolve_provider(s)
    assert state == {"active": "groq", "fallback_in_effect": False}


def test_provider_info_shape():
    info = provider_info(Settings())
    assert info["active_provider"] == "ollama"
    assert info["available_providers"] == ["ollama", "groq", "gemini", "openrouter"]
    assert info["fallback_in_effect"] is False


def test_generate_stream_cloud_failure_falls_back_with_notice(monkeypatch):
    def boom(settings, messages, timeout_s=None):
        raise RuntimeError("cloud down")

    def ollama_ok(settings, messages, timeout_s=None):
        yield "hi "

    monkeypatch.setattr(llm, "stream_chat_cloud", boom)
    monkeypatch.setattr(llm, "stream_chat_ollama", ollama_ok)
    settings = Settings(provider="groq", cloud_api_key="k")
    events = list(generate_stream(settings, [{"role": "user", "content": "q"}]))
    assert events[0] == ("notice", "fell back to local model")
    assert events[1] == ("token", "hi ")


def test_generate_stream_ollama_failure_raises_unavailable(monkeypatch):
    def boom(settings, messages, timeout_s=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(llm, "stream_chat_ollama", boom)
    with pytest.raises(ProviderUnavailable):
        list(generate_stream(Settings(), [{"role": "user", "content": "q"}]))


def test_generate_stream_both_fail_raises_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "stream_chat_cloud", boom)
    monkeypatch.setattr(llm, "stream_chat_ollama", boom)
    settings = Settings(provider="groq", cloud_api_key="k")
    with pytest.raises(ProviderUnavailable):
        list(generate_stream(settings, [{"role": "user", "content": "q"}]))
