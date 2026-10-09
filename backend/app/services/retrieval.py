"""Retrieval over the transcript_chunks pgvector index (item 3).

- rewrite_followup(): turn a follow-up into a standalone question using
  recent session history (LLM rewrite, degrades to the raw question).
- retrieve(): embed the question with Ollama and run a cosine-similarity
  search; hits below LGA_RETRIEVAL_MIN_SCORE are dropped, so weak retrieval
  triggers the contract's "not supported by the transcripts" path.
"""

import httpx
from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Message, TranscriptChunk
from .llm import ProviderUnavailable, chat_once

REWRITE_SYSTEM = (
    "Rewrite the user's follow-up question as a standalone question that "
    "carries over all needed context from the conversation history. "
    "Return ONLY the rewritten question, nothing else."
)


def get_recent_history(db: Session, session_id, limit: int = 6) -> list[dict]:
    rows = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


def rewrite_followup(
    settings: Settings,
    history: list[dict],
    question: str,
    llm_fn=None,
) -> str:
    """Rewrite a follow-up into a standalone question.

    ``llm_fn(messages) -> str`` is injectable for tests. Any failure
    degrades to the raw question — retrieval still runs, just less sharp.
    """
    if not history:
        return question
    if llm_fn is None:
        def llm_fn(messages):
            return chat_once(settings, messages, timeout_s=20.0)
    messages = (
        [{"role": "system", "content": REWRITE_SYSTEM}]
        + history[-6:]
        + [{"role": "user", "content": question}]
    )
    try:
        rewritten = (llm_fn(messages) or "").strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def embed_query(settings: Settings, text: str) -> list[float]:
    """Embed one query string with Ollama. Raises ProviderUnavailable."""
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.ollama_embed_model, "input": text},
            timeout=30.0,
            trust_env=False,  # local service: never via egress proxy
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception as exc:
        raise ProviderUnavailable(f"embedding failed: {type(exc).__name__}") from exc


def hits_from_rows(rows, min_score: float) -> list[dict]:
    """Map (chunk, cosine_distance) rows to citation dicts, applying the
    min-score threshold. Pure function — unit-testable without a DB."""
    hits = []
    for chunk, distance in rows:
        relevance = max(0.0, min(1.0, 1.0 - float(distance)))
        if relevance < min_score:
            continue
        excerpt = chunk.text[:400] + ("…" if len(chunk.text) > 400 else "")
        hits.append(
            {
                "episode": chunk.episode,
                "guest": chunk.guest,
                "source_url": chunk.source_url,
                "source_file": chunk.source_file,
                "excerpt": excerpt,
                "relevance": round(relevance, 3),
                "chunk_id": str(chunk.id),
                "text": chunk.text,
            }
        )
    return hits


def retrieve(
    db: Session,
    settings: Settings,
    question: str,
    top_k: int | None = None,
) -> list[dict]:
    """Top-k grounded chunks for question, with source info + relevance."""
    query_vec = embed_query(settings, question)
    distance = TranscriptChunk.embedding.cosine_distance(query_vec)
    rows = (
        db.query(TranscriptChunk, distance.label("distance"))
        .order_by(distance)
        .limit(top_k or settings.retrieval_top_k)
        .all()
    )
    return hits_from_rows(rows, settings.retrieval_min_score)
