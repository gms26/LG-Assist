"""Config layer tests: safe defaults, env overrides, provider rules."""

from app.config import Settings


def test_safe_local_defaults():
    s = Settings()
    assert s.provider == "ollama"
    assert s.port == 8000
    assert s.ollama_chat_model == "qwen2.5:7b"
    assert s.ollama_embed_model == "nomic-embed-text"
    assert s.vector_dim == 768
    assert s.cloud_api_key is None
    assert not s.cloud_configured()


def test_env_override_provider(monkeypatch):
    monkeypatch.setenv("LGA_PROVIDER", "groq")
    s = Settings()
    assert s.effective_provider == "groq"
    assert not s.cloud_configured()  # key still missing -> not configured


def test_unknown_provider_falls_back_to_ollama(monkeypatch):
    monkeypatch.setenv("LGA_PROVIDER", "not-a-provider")
    assert Settings().effective_provider == "ollama"


def test_cloud_configured_only_with_key(monkeypatch):
    monkeypatch.setenv("LGA_PROVIDER", "groq")
    monkeypatch.setenv("LGA_CLOUD_API_KEY", "test-key")
    s = Settings()
    assert s.cloud_configured()
