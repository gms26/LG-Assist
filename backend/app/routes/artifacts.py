"""Artifact read endpoints (contract v1).

Artifacts are stored exactly as generated. HTML is untrusted by definition;
the frontend renders it only inside a sandboxed iframe.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Artifact, ChatSession, User
from ..schemas import ArtifactOut
from ..services.users import get_current_user

router = APIRouter(tags=["artifacts"])


@router.get("/artifacts/{artifact_id}", response_model=ArtifactOut, summary="One artifact")
def get_artifact(
    artifact_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ArtifactOut:
    artifact = (
        db.query(Artifact)
        .join(ChatSession, Artifact.session_id == ChatSession.id)
        .filter(Artifact.id == artifact_id, ChatSession.user_id == user.id)
        .first()
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return ArtifactOut.model_validate(artifact)
