"""Retrieval tests (item 3): rewrite degradation and hit mapping.

The pgvector query itself needs a live database; threshold math and the
rewrite fallback are pure and tested here. DB-backed retrieval tests land
with the Postgres-backed suite (item 7).
"""

from types import SimpleNamespace

from app.config import Settings
from app.services.retrieval import hits_from_rows, rewrite_followup


def _chunk(**kw):
    base = dict(episode="Ep", guest="G", source_url="u", source_file="f",
                text="x " * 500, id="00000000-0000-0000-0000-000000000000")
    base.update(kw)
    return SimpleNamespace(**base)


def test_hits_from_rows_applies_threshold_and_scores():
    rows = [(_chunk(), 0.1), (_chunk(), 0.8)]  # cosine distances
    hits = hits_from_rows(rows, min_score=0.4)
    assert len(hits) == 1  # 1-0.8=0.2 < 0.4 is dropped
    assert hits[0]["relevance"] == 0.9
    assert hits[0]["episode"] == "Ep"
    assert hits[0]["excerpt"].endswith("…")  # long text truncated
    assert "text" in hits[0]  # full text kept server-side for prompts


def test_relevance_clamped_to_unit_range():
    hits = hits_from_rows([(_chunk(), -0.5)], min_score=0.0)
    assert hits[0]["relevance"] == 1.0


def test_rewrite_no_history_returns_question_unchanged():
    called = []
    def llm_fn(messages):
        called.append(messages)
        return "rewritten?"
    out = rewrite_followup(Settings(), [], "what about pricing?", llm_fn=llm_fn)
    assert out == "what about pricing?"
    assert called == []


def test_rewrite_uses_history_and_returns_llm_result():
    history = [{"role": "user", "content": "we sell b2b saas"},
               {"role": "assistant", "content": "noted"}]
    seen = {}
    def llm_fn(messages):
        seen["n"] = len(messages)
        return "  How should we price our B2B SaaS?  "
    out = rewrite_followup(Settings(), history, "what about pricing?", llm_fn=llm_fn)
    assert out == "How should we price our B2B SaaS?"
    assert seen["n"] == 1 + 2 + 1  # system + history + question


def test_rewrite_degrades_to_raw_question_on_llm_failure():
    history = [{"role": "user", "content": "context"}]
    def boom(messages):
        raise RuntimeError("ollama down")
    out = rewrite_followup(Settings(), history, "and for enterprise?", llm_fn=boom)
    assert out == "and for enterprise?"


def test_rewrite_empty_llm_reply_falls_back():
    history = [{"role": "user", "content": "context"}]
    out = rewrite_followup(Settings(), history, "raw?", llm_fn=lambda m: "   ")
    assert out == "raw?"
