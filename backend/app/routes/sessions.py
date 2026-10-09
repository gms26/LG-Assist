"""Session endpoints (contract v1).

All endpoints are scoped to the caller from the X-User-Id header: a user
can only see their own sessions, messages, and artifacts.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Artifact, ChatSession, Message, User
from ..schemas import (
    ArtifactListOut,
    ArtifactOut,
    MessageOut,
    SessionCreate,
    SessionDetailOut,
    SessionListOut,
    SessionOut,
)
from ..services.users import get_current_user

router = APIRouter(tags=["sessions"])


def _get_owned_session(db: Session, user: User, session_id: UUID) -> ChatSession:
    chat_session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user.id)
        .first()
    )
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return chat_session


@router.post("/sessions", response_model=SessionOut, status_code=201, summary="Start a new chat")
def create_session(
    body: SessionCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatSession:
    chat_session = ChatSession(user_id=user.id, title=body.title)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


@router.get("/sessions", response_model=SessionListOut, summary="List chats, newest first")
def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionListOut:
    query = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    total = query.count()
    sessions = query.offset(offset).limit(limit).all()
    return SessionListOut(sessions=sessions, total=total)


@router.get("/sessions/{session_id}", response_model=SessionDetailOut, summary="One chat with its messages")
def get_session(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionDetailOut:
    chat_session = _get_owned_session(db, user, session_id)
    return SessionDetailOut(
        session=SessionOut.model_validate(chat_session),
        messages=[MessageOut.model_validate(m) for m in chat_session.messages],
    )


@router.delete("/sessions/{session_id}", status_code=204, summary="Remove a chat")
def delete_session(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    chat_session = _get_owned_session(db, user, session_id)
    db.delete(chat_session)
    db.commit()
    return Response(status_code=204)


@router.get(
    "/sessions/{session_id}/artifacts",
    response_model=ArtifactListOut,
    summary="Artifacts generated in a chat",
)
def list_session_artifacts(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ArtifactListOut:
    chat_session = _get_owned_session(db, user, session_id)
    artifacts = (
        db.query(Artifact)
        .filter(Artifact.session_id == chat_session.id)
        .order_by(Artifact.created_at)
        .all()
    )
    return ArtifactListOut(
        artifacts=[ArtifactOut.model_validate(a) for a in artifacts]
    )
