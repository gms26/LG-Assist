"""Pydantic request/response schemas.

Implements the owner's API contract v1 (base path /api/v1). Endpoint paths
and shapes follow the contract exactly; anything the contract leaves vague
is flagged in the backend report for correction, never silently redefined.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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


# ---------------------------------------------------------------------------
# Contract v1: sessions, messages, artifacts, model config
# ---------------------------------------------------------------------------


class Skill(str, Enum):
    auto = "auto"
    answer = "answer"
    essay = "essay"
    artifact = "artifact"


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class SessionListOut(BaseModel):
    sessions: list[SessionOut]
    total: int


class SourceOut(BaseModel):
    """One citation attached to an assistant message (contract v1)."""

    episode: str
    guest: str | None = None
    source_url: str | None = None
    source_file: str | None = None
    excerpt: str = ""
    relevance: float | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: str
    content: str
    created_at: datetime
    sources: list[SourceOut] = Field(default_factory=list)
    artifact_id: UUID | None = None
    skill_used: str | None = None


class SessionDetailOut(BaseModel):
    session: SessionOut
    messages: list[MessageOut]


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    skill: Skill = Skill.auto


class ArtifactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    type: str  # markdown | html
    title: str
    content: str
    created_at: datetime


class ArtifactListOut(BaseModel):
    artifacts: list[ArtifactOut]


class ModelConfigOut(BaseModel):
    active_provider: str
    configured_provider: str
    chat_model: str
    embed_model: str
    available_providers: list[str]
    fallback_in_effect: bool
    cloud_configured: bool
