"""Pydantic request/response schemas.

PROVISIONAL: endpoint paths and shapes are owned by the repo maintainer's
contract (shared with the frontend). These schemas cover what the backend
needs today (health) and will be extended/renamed to match the contract
exactly once it lands — the backend never renames endpoints on its own.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ComponentCheck(BaseModel):
    status: str = Field(description="ok | degraded | unreachable")
    latency_ms: int | None = None
    detail: str | None = None


class OllamaCheck(ComponentCheck):
    models: list[str] = Field(default_factory=list)


class ProviderInfo(BaseModel):
    active: str = Field(description="effective provider after fallback rules")
    configured: str = Field(description="provider requested via LGA_PROVIDER")
    chat_model: str
    embed_model: str
    cloud_configured: bool = False


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded")
    version: str
    provider: ProviderInfo
    checks: dict[str, ComponentCheck | OllamaCheck]
    timestamp: datetime


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list | dict | None = None
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
