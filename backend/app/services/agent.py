"""Agent turn runner: skill routing + grounded-answer flow (contract v1).

Emits the SSE event sequence for one user message as (event, data) tuples;
the chat route formats them and owns persistence. Framework-agnostic on
purpose — no FastAPI imports here.

Turn pipeline (items 3+4):
  1. skill routing: ``auto`` resolves to ``answer`` (item 5 adds the real
     essay/artifact skills once the Ship 30 principles arrive).
  2. config-time fallback notice: if a cloud provider was requested but no
     key is configured, the turn starts with "fell back to local model".
  3. follow-up rewriting: the question is rewritten into standalone form
     from recent session history (degrades to the raw question).
  4. retrieval: top-k transcript chunks with citations; hits below
     LGA_RETRIEVAL_MIN_SCORE are dropped.
  5. grounding rule (contract v1): empty/weak retrieval -> a normal answer
     saying the transcripts don't support it, empty sources, a ``notice``
     event. NOT an error — and it works even when the LLM is down.
  6. grounded generation: the LLM answers ONLY from the retrieved excerpts,
     streamed as token events; a runtime cloud failure falls back to Ollama
     with a notice event (ProviderUnavailable -> the route's error event).
"""

from collections.abc import Iterator

from ..config import Settings, get_settings
from ..models import ChatSession
from ..schemas import Skill
from .llm import generate_stream, resolve_provider
from .retrieval import get_recent_history, retrieve, rewrite_followup
from .skills.ship30 import run_essay_skill
from .sse import chunk_text

NOTICE_NOT_SUPPORTED = "not supported by the transcripts"
NOTICE_FALLBACK = "fell back to local model"

NOT_SUPPORTED_MESSAGE = (
    "I don't have transcript material that supports an answer to that yet. "
    "I searched Lenny's podcast transcripts and nothing relevant came back, "
    "so I can't ground a reply. Try asking about product strategy, growth, "
    "pricing, hiring, or another topic covered in the episodes."
)

GROUNDED_SYSTEM = (
    "You are the Lenny Growth Assistant. Answer the user's product and growth "
    "question using ONLY the transcript excerpts below. Every substantive claim "
    "must be grounded in an excerpt — cite the episode title inline like "
    "(Episode: \"...\"). If the excerpts don't actually support an answer, say "
    "so plainly instead of inventing. Keep the answer focused and skimmable."
)


def resolve_skill(requested: Skill) -> str:
    """Map the requested skill to the concrete skill used for this turn."""
    if requested == Skill.auto:
        return Skill.answer.value
    return requested.value


def build_grounded_messages(question: str, hits: list[dict]) -> list[dict]:
    excerpts = "\n\n".join(
        f"[{i + 1}] (Episode: \"{h['episode']}\""
        + (f", Guest: {h['guest']}" if h.get("guest") else "")
        + f")\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return [
        {"role": "system", "content": GROUNDED_SYSTEM},
        {"role": "user", "content": f"Transcript excerpts:\n{excerpts}\n\nQuestion: {question}"},
    ]


def run_turn(
    db,
    chat_session: ChatSession,
    content: str,
    skill: Skill,
    settings: Settings | None = None,
) -> Iterator[tuple[str, dict]]:
    """Yield (event, data) for the contract v1 streaming protocol.

    The route intercepts the ``done`` event to attach the persisted
    assistant message id before sending it. Raises ProviderUnavailable
    when no model can serve (the route turns it into an error event).
    """
    settings = settings or get_settings()
    skill_used = resolve_skill(skill)

    # Config-time fallback: cloud requested but unusable -> notice up front.
    if resolve_provider(settings)["fallback_in_effect"]:
        yield ("notice", {"message": NOTICE_FALLBACK})

    # Follow-up rewriting from session history (the current user message was
    # saved before this runs, so drop it from the history window).
    question = content
    if settings.rewrite_enabled:
        history = get_recent_history(db, chat_session.id)
        if history and history[-1]["role"] == "user" and history[-1]["content"] == content:
            history = history[:-1]
        question = rewrite_followup(settings, history, content)

    # Item 5: Ship 30 essay skill — staged pipeline with its own retrieval.
    if skill_used == Skill.essay.value:
        yield from run_essay_skill(db, chat_session, question, settings)
        return

    # Retrieval (raises ProviderUnavailable when embeddings are unreachable).
    hits = retrieve(db, settings, question)

    if not hits:
        # Contract behavior rule: unsupported != error. Deterministic message,
        # so this path works even with the LLM down.
        yield ("notice", {"message": NOTICE_NOT_SUPPORTED})
        yield ("sources", {"sources": []})
        for piece in chunk_text(NOT_SUPPORTED_MESSAGE):
            yield ("token", {"text": piece})
        yield ("done", {"skill_used": skill_used})
        return

    yield ("sources", {"sources": [
        {k: h[k] for k in ("episode", "guest", "source_url", "source_file",
                           "excerpt", "relevance", "chunk_id")}
        for h in hits
    ]})

    # Grounded generation, streamed. Runtime cloud failure -> Ollama fallback
    # with a notice; Ollama failure -> ProviderUnavailable -> error event.
    for kind, payload in generate_stream(settings, build_grounded_messages(question, hits)):
        if kind == "notice":
            yield ("notice", {"message": payload})
        else:
            yield ("token", {"text": payload})

    # Artifact skill seam (item 5): an "artifact" event carrying the full
    # artifact will be yielded here once the skills land.

    yield ("done", {"skill_used": skill_used})
