"""Health endpoint.

Reports database and Ollama status SEPARATELY so operators (and the UI's
model toggle) can see exactly which dependency is down:

    GET /api/v1/health ->
    {
      "status": "ok" | "degraded",
      "version": "0.1.0",
      "provider": {"active": "ollama", "configured": "ollama",
                   "chat_model": "qwen2.5:7b", "embed_model": "nomic-embed-text",
                   "cloud_configured": false},
      "checks": {
        "database": {"status": "ok", "latency_ms": 4},
        "ollama":   {"status": "ok", "latency_ms": 6, "models": ["qwen2.5:7b", ...]}
      },
      "timestamp": "..."
    }

Overall status is "degraded" if ANY check is not ok. The endpoint itself
never 500s because of a dependency — that would defeat its purpose.
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_db
from ..schemas import ComponentCheck, HealthResponse, OllamaCheck, ProviderInfo
from ..services.ollama import check_ollama

router = APIRouter(tags=["health"])


def _check_database(db: Session) -> ComponentCheck:
    started = time.perf_counter()
    try:
        db.execute(text("SELECT 1"))
        return ComponentCheck(
            status="ok", latency_ms=int((time.perf_counter() - started) * 1000)
        )
    except Exception as exc:
        return ComponentCheck(
            status="unreachable",
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=f"{type(exc).__name__}: {exc}",
        )


@router.get("/health", response_model=HealthResponse, summary="Service + dependency health")
def health(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> HealthResponse:
    db_check = _check_database(db)
    ollama_raw = check_ollama(settings)
    ollama_check = OllamaCheck(**ollama_raw)

    checks = {"database": db_check, "ollama": ollama_check}
    overall = "ok" if all(c.status == "ok" for c in checks.values()) else "degraded"

    provider = ProviderInfo(
        active=settings.effective_provider,
        configured=settings.provider,
        chat_model=settings.ollama_chat_model,
        embed_model=settings.ollama_embed_model,
        cloud_configured=settings.cloud_configured(),
    )
    return HealthResponse(
        status=overall,
        version=settings.app_version,
        provider=provider,
        checks=checks,
        timestamp=datetime.now(timezone.utc),
    )
