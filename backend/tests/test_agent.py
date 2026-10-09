"""Agent turn-runner tests (pure logic, no DB needed).

Verifies the contract v1 streaming event sequence: notice -> sources ->
token* -> done, with the grounding rule that empty retrieval yields a
normal "not supported" answer (not an error).
"""

from app.schemas import Skill
from app.services.agent import NOTICE_NOT_SUPPORTED, resolve_skill, run_turn


def _events(**kwargs):
    return list(run_turn(None, None, "how do I price?", kwargs.get("skill", Skill.auto)))


def test_event_sequence_for_unsupported_question():
    events = _events()
    names = [e for e, _ in events]
    assert names[0] == "notice"
    assert names[1] == "sources"
    assert "token" in names
    assert names[-1] == "done"
    assert "error" not in names


def test_notice_and_empty_sources_on_empty_retrieval():
    events = dict(_events())
    assert NOTICE_NOT_SUPPORTED in events["notice"]["message"]
    assert events["sources"]["sources"] == []


def test_tokens_reassemble_to_full_answer():
    events = _events()
    text = "".join(d["text"] for e, d in events if e == "token")
    assert "transcript" in text.lower()
    assert len(text) > 50


def test_done_carries_skill_used():
    events = dict(_events())
    assert events["done"]["skill_used"] == "answer"


def test_skill_resolution():
    assert resolve_skill(Skill.auto) == "answer"
    assert resolve_skill(Skill.essay) == "essay"
    assert resolve_skill(Skill.artifact) == "artifact"


def test_explicit_skill_reported_in_done():
    events = dict(_events(skill=Skill.essay))
    assert events["done"]["skill_used"] == "essay"
