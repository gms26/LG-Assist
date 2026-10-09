"""Provider resolution tests (no network needed)."""

from app.config import Settings
from app.services.llm import provider_info, resolve_provider


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
