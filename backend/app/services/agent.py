"""Agent turn runner: skill routing + grounded-answer flow (contract v1).

Emits the SSE event sequence for one user message as (event, data) tuples;
the chat route formats them and owns persistence. Framework-agnostic on
purpose — no FastAPI imports here.

Current behavior (Phase 2/3 seams marked):
  * skill routing: ``auto`` resolves to ``answer``; ``essay``/``artifact``
    are accepted and reported via skill_used but generate through the same
    grounded path until their skills land.
  * retrieval: ``retrieve()`` is the Phase 2 seam — it returns [] until the
    transcript index exists.
  * grounding rule (contract v1): when retrieval is empty or weak, the
    assistant still returns a normal answer saying the transcripts don't
    support it, with an empty sources list and a ``notice`` event. This is
    NOT an error.

Phase 2 will fill retrieve(); Phase 3 will add real generation behind the
token events (Ollama streaming via services/llm).
"""

from collections.abc import Iterator

from ..models import ChatSession
from ..schemas import Skill

NOTICE_NOT_SUPPORTED = "not supported by the transcripts"

NOT_SUPPORTED_MESSAGE = (
    "I don't have transcript material that supports an answer to that yet. "
    "The knowledge base hasn't been ingested in this deployment, so I can't "
    "ground a reply in Lenny's episodes. Once transcripts are loaded, I'll "
    "cite the exact episodes I draw on."
)


def resolve_skill(requested: Skill) -> str:
    """Map the requested skill to the concrete skill used for this turn."""
    if requested == Skill.auto:
        return Skill.answer.value
    return requested.value


def retrieve(db, chat_session: ChatSession, content: str, top_k: int = 6) -> list[dict]:
    """Phase 2 seam: top-k transcript chunks with source info.

    Returns [] until the ingestion pipeline populates transcript_chunks.
    Each hit will look like:
      {"episode": ..., "guest": ..., "source_url": ..., "source_file": ...,
       "excerpt": ..., "relevance": 0.87}
    """
    return []


def _chunk_text(text: str, words_per_chunk: int = 12) -> Iterator[str]:
    words = text.split(" ")
    for i in range(0, len(words), words_per_chunk):
        piece = " ".join(words[i : i + words_per_chunk])
        yield piece + (" " if i + words_per_chunk < len(words) else "")


def run_turn(
    db, chat_session: ChatSession, content: str, skill: Skill
) -> Iterator[tuple[str, dict]]:
    """Yield (event, data) for the contract v1 streaming protocol.

    The route intercepts the ``done`` event to attach the persisted
    assistant message id before sending it.
    """
    skill_used = resolve_skill(skill)

    sources = retrieve(db, chat_session, content)

    if not sources:
        # Contract behavior rule: unsupported != error.
        yield ("notice", {"message": NOTICE_NOT_SUPPORTED})
    yield ("sources", {"sources": sources})

    # Generation seam (Phase 3): deterministic honest message until the
    # grounded generation path exists. Streamed as token chunks so the
    # frontend exercises the real streaming UX from day one.
    for piece in _chunk_text(NOT_SUPPORTED_MESSAGE):
        yield ("token", {"text": piece})

    # Artifact skill seam (Phase 3): an "artifact" event carrying the full
    # artifact will be yielded here when generation supports it.

    yield ("done", {"skill_used": skill_used})
