"""Agent turn-runner tests (item 3+4 pipeline, no DB/LLM needed).

Covers the contract v1 event sequence, the grounding rule, the fallback
notice, and provider-failure propagation — with fakes for DB, retrieval,
and generation. monkeypatch auto-reverts the fakes after each test.
"""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.schemas import Skill
from app.services import agent
from app.services.agent import (
    NOTICE_FALLBACK,
    NOTICE_NOT_SUPPORTED,
    resolve_skill,
    run_turn,
)
from app.services.llm import ProviderUnavailable


class _FakeQuery:
    def filter(self, *a, **k):
        return self

    def order_by(self, *a):
        return self

    def limit(self, *a):
        return self

    def all(self):
        return []


class FakeDB:
    def query(self, *a, **k):
        return _FakeQuery()


def _run(skill=Skill.auto, settings=None):
    session = SimpleNamespace(id="00000000-0000-0000-0000-000000000000")
    return list(run_turn(FakeDB(), session, "how do I price?", skill,
                         settings or Settings()))


def test_unsupported_question_event_sequence(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [])
    events = _run()
    names = [e for e, _ in events]
    assert names[0] == "notice"
    assert names[1] == "sources"
    assert "token" in names
    assert names[-1] == "done"
    assert "error" not in names


def test_unsupported_yields_notice_empty_sources_and_answer(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [])
    events = _run()
    by_event = {}
    for name, data in events:
        by_event.setdefault(name, []).append(data)
    assert NOTICE_NOT_SUPPORTED in by_event["notice"][0]["message"]
    assert by_event["sources"][0]["sources"] == []
    text = "".join(d["text"] for d in by_event["token"])
    assert "transcript" in text.lower()
    assert by_event["done"][0]["skill_used"] == "answer"


def test_config_time_fallback_emits_notice_first(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [])
    settings = Settings(provider="groq")  # no key -> fallback in effect
    events = _run(settings=settings)
    assert events[0] == ("notice", {"message": NOTICE_FALLBACK})


def test_grounded_path_streams_sources_then_tokens(monkeypatch):
    hits = [{
        "episode": "Ep", "guest": "G", "source_url": "u", "source_file": "f",
        "excerpt": "ex", "relevance": 0.9, "chunk_id": "c", "text": "full text",
    }]
    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: hits)
    monkeypatch.setattr(
        agent, "generate_stream", lambda *a, **k: iter([("token", "hello ")]))
    events = _run()
    names = [e for e, _ in events]
    assert names[0] == "sources"
    src = events[0][1]["sources"][0]
    assert src["episode"] == "Ep" and src["relevance"] == 0.9
    assert "text" not in src  # full chunk text stays server-side
    assert "".join(d["text"] for e, d in events if e == "token") == "hello "
    assert names[-1] == "done"
    assert not any(e == "notice" and NOTICE_NOT_SUPPORTED in d["message"]
                   for e, d in events)


def test_runtime_cloud_failure_notice_passes_through(monkeypatch):
    hits = [{
        "episode": "Ep", "guest": None, "source_url": None, "source_file": "f",
        "excerpt": "ex", "relevance": 0.9, "chunk_id": "c", "text": "t",
    }]
    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: hits)

    def fake_gen(*a, **k):
        yield ("notice", "fell back to local model")
        yield ("token", "ok ")

    monkeypatch.setattr(agent, "generate_stream", fake_gen)
    events = _run()
    assert ("notice", {"message": "fell back to local model"}) in events


def test_provider_failure_propagates(monkeypatch):
    hit = {"episode": "Ep", "guest": None, "source_url": None, "source_file": "f",
           "excerpt": "ex", "relevance": 0.9, "chunk_id": "c", "text": "t"}
    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [hit])

    def boom(*a, **k):
        raise ProviderUnavailable("down")
        yield  # make it a generator

    monkeypatch.setattr(agent, "generate_stream", boom)
    with pytest.raises(ProviderUnavailable):
        _run()


def test_skill_resolution():
    assert resolve_skill(Skill.auto) == "answer"
    assert resolve_skill(Skill.essay) == "essay"
    assert resolve_skill(Skill.artifact) == "artifact"
