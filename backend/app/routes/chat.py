"""Chat endpoint with Server-Sent Events streaming (contract v1).

POST /sessions/{id}/messages
  request:  {"content": "...", "skill": "auto" | "answer" | "essay" | "artifact"}
  response: text/event-stream with events
            notice | sources | token* | artifact? | done
            (or an `error` event, after which the stream ends)

Contract behavior rules honored here:
  * the user's message is saved BEFORE generation starts;
  * the assistant message is saved when the stream finishes (its id goes
    into the final `done` event);
  * a mid-stream failure persists whatever was produced and ends with a
    structured `error` event — the stream itself never 500s.
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import ChatSession, Message, User
from ..schemas import SendMessageRequest
from ..services.agent import run_turn
from ..services.sse import format_sse
from ..services.users import get_current_user

router = APIRouter(tags=["chat"])


@router.post("/sessions/{session_id}/messages", summary="Send a message (SSE stream)")
def send_message(
    session_id: UUID,
    body: SendMessageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    chat_session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user.id)
        .first()
    )
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Rule: user message is saved before generation starts.
    db.add(Message(session_id=chat_session.id, role="user", content=body.content, sources=[]))
    chat_session.updated_at = datetime.now(timezone.utc)
    db.commit()

    def stream():
        text_parts: list[str] = []
        sources: list = []
        skill_used = "answer"
        try:
            for event, data in run_turn(db, chat_session, body.content, body.skill):
                if event == "token":
                    text_parts.append(data["text"])
                elif event == "sources":
                    sources = data["sources"]
                elif event == "done":
                    # Rule: assistant message is saved when the stream
                    # finishes; done carries its durable id.
                    skill_used = data["skill_used"]
                    assistant = Message(
                        session_id=chat_session.id,
                        role="assistant",
                        content="".join(text_parts),
                        sources=sources,
                        skill_used=skill_used,
                    )
                    db.add(assistant)
                    db.commit()
                    db.refresh(assistant)
                    data["message_id"] = str(assistant.id)
                yield format_sse(event, data)
        except Exception:
            # Persist the partial turn so it isn't lost, then end with a
            # structured error event (contract v1: error ends the stream).
            if text_parts:
                db.add(
                    Message(
                        session_id=chat_session.id,
                        role="assistant",
                        content="".join(text_parts),
                        sources=sources,
                        skill_used=skill_used,
                    )
                )
                db.commit()
            yield format_sse(
                "error",
                {"code": "internal_error", "message": "Generation failed mid-stream."},
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
