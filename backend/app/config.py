"""Central configuration.

Every setting comes from the environment (prefix ``LGA_``) with safe,
local-first defaults. No secret is required to boot: the app runs fully on
Ollama + local Postgres. Cloud providers are opt-in via LGA_PROVIDER and
LGA_CLOUD_API_KEY, with automatic fallback to Ollama (see services/llm.py).
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_PROVIDERS = ("ollama", "groq", "gemini", "openrouter")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LGA_", env_file=".env", extra="ignore")

    # --- app ---
    app_name: str = "Lenny Growth Assistant"
    app_env: str = Field(default="local", description="local | dev | prod")
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # --- model provider ---
    provider: str = Field(default="ollama", description="ollama | groq | gemini | openrouter")
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_s: float = 60.0

    cloud_api_key: str | None = Field(default=None, description="REQUIRED when provider != ollama")
    cloud_chat_model: str = Field(default="", description="e.g. llama-3.3-70b-versatile (groq)")

    # --- database ---
    database_url: str = "postgresql+psycopg://lga:lga@localhost:5432/lga"
    db_pool_size: int = 5

    # --- retrieval ---
    vector_dim: int = 768  # nomic-embed-text output size
    retrieval_top_k: int = 6

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_provider(self) -> str:
        """Provider after validation; unknown values fall back to ollama."""
        return self.provider if self.provider in SUPPORTED_PROVIDERS else "ollama"

    def cloud_configured(self) -> bool:
        return self.effective_provider != "ollama" and bool(self.cloud_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
