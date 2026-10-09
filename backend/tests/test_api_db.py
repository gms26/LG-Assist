"""DB-backed API tests: sessions, chat persistence, auto-titles, retrieval
SQL, user scoping, artifacts — against a real Postgres + pgvector.

Skipped when the test database is unreachable (see conftest.py).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db import get_db
from app.main import app
from app.models import Artifact, ChatSession, Message, TranscriptChunk
from app.routes import chat as chat_route
from app.services import retrieval as retrieval_module


@pytest.fixture()
def client(db):
    def _get_db():
        yield db

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def _uid():
    return str(uuid.uuid4())


def _h(uid):
    return {"X-User-Id": uid}


def test_session_crud_roundtrip(client):
    uid = _uid()
    r = client.post("/api/v1/sessions", json={"title": "My chat"}, headers=_h(uid))
    assert r.status_code == 201
    body = r.json()
    assert body["title"] == "My chat"
    sid = body["id"]

    r = client.get("/api/v1/sessions", headers=_h(uid))
    assert r.json()["total"] == 1
    assert r.json()["sessions"][0]["id"] == sid

    r = client.get(f"/api/v1/sessions/{sid}", headers=_h(uid))
    assert r.json()["session"]["title"] == "My chat"
    assert r.json()["messages"] == []

    assert client.delete(f"/api/v1/sessions/{sid}", headers=_h(uid)).status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}", headers=_h(uid)).status_code == 404


def test_sessions_scoped_per_user(client):
    a, b = _uid(), _uid()
    sid = client.post("/api/v1/sessions", json={}, headers=_h(a)).json()["id"]
    assert client.get(f"/api/v1/sessions/{sid}", headers=_h(b)).status_code == 404
    assert client.get("/api/v1/sessions", headers=_h(b)).json()["total"] == 0
    assert client.delete(f"/api/v1/sessions/{sid}", headers=_h(b)).status_code == 404


def test_list_newest_first_with_pagination(client):
    uid = _uid()
    for t in ["first", "second", "third"]:
        client.post("/api/v1/sessions", json={"title": t}, headers=_h(uid))
    page = client.get("/api/v1/sessions?limit=2&offset=0", headers=_h(uid)).json()
    assert page["total"] == 3
    assert [s["title"] for s in page["sessions"]] == ["third", "second"]
    page2 = client.get("/api/v1/sessions?limit=2&offset=2", headers=_h(uid)).json()
    assert [s["title"] for s in page2["sessions"]] == ["first"]


def _fake_turn(events):
    def _turn(db, chat_session, content, skill, settings=None):
        yield from events

    return _turn


def test_chat_persists_messages_and_sets_title(client, monkeypatch):
    monkeypatch.setattr(
        chat_route, "run_turn",
        _fake_turn([
            ("sources", {"sources": []}),
            ("token", {"text": "hello "}),
            ("done", {"skill_used": "answer"}),
        ]),
    )
    uid = _uid()
    sid = client.post("/api/v1/sessions", json={}, headers=_h(uid)).json()["id"]
    content = "how should we price our b2b saas for mid-market teams"
    r = client.post(f"/api/v1/sessions/{sid}/messages",
                    json={"content": content}, headers=_h(uid))
    assert r.status_code == 200
    assert "event: done" in r.text

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_h(uid)).json()
    assert detail["session"]["title"] == content  # <=60 chars: verbatim
    msgs = detail["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == content
    assert msgs[1]["content"] == "hello "
    assert msgs[1]["skill_used"] == "answer"


def test_chat_long_first_message_title_truncated(client, monkeypatch):
    monkeypatch.setattr(
        chat_route, "run_turn",
        _fake_turn([("done", {"skill_used": "answer"})]),
    )
    uid = _uid()
    sid = client.post("/api/v1/sessions", json={}, headers=_h(uid)).json()["id"]
    content = ("what are the best pricing strategies for a b2b saas startup "
               "selling to mid-market teams in 2026")
    client.post(f"/api/v1/sessions/{sid}/messages",
                json={"content": content}, headers=_h(uid))
    title = client.get(f"/api/v1/sessions/{sid}", headers=_h(uid)).json()["session"]["title"]
    assert len(title) <= 61
    assert title.endswith("…")


def test_chat_second_message_does_not_retitle(client, monkeypatch):
    monkeypatch.setattr(
        chat_route, "run_turn",
        _fake_turn([("done", {"skill_used": "answer"})]),
    )
    uid = _uid()
    sid = client.post("/api/v1/sessions", json={"title": "Kept"}, headers=_h(uid)).json()["id"]
    for text in ["first question here", "second question here"]:
        client.post(f"/api/v1/sessions/{sid}/messages",
                    json={"content": text}, headers=_h(uid))
    detail = client.get(f"/api/v1/sessions/{sid}", headers=_h(uid)).json()
    assert detail["session"]["title"] == "Kept"  # explicit title never overwritten
    assert len(detail["messages"]) == 4


def test_chat_provider_failure_streams_structured_error(client, monkeypatch, db):
    from app.services.llm import ProviderUnavailable

    def _boom(*a, **k):
        raise ProviderUnavailable("down")
        yield

    monkeypatch.setattr(chat_route, "run_turn", _boom)
    uid = _uid()
    sid = client.post("/api/v1/sessions", json={}, headers=_h(uid)).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/messages",
                    json={"content": "hi"}, headers=_h(uid))
    assert "event: error" in r.text
    assert "dependency_unavailable" in r.text
    msgs = client.get(f"/api/v1/sessions/{sid}", headers=_h(uid)).json()["messages"]
    assert [m["role"] for m in msgs] == ["user"]  # user msg kept, no phantom answer


def test_retrieval_sql_ranks_and_thresholds(db, monkeypatch):
    vec = [1.0] * 768
    db.add_all([
        TranscriptChunk(episode="Near", guest="G", source_url="u", source_file="f",
                        speaker="S", seq=0, text="near text", embedding=vec,
                        content_hash="h-near"),
        TranscriptChunk(episode="Far", guest="G", source_url="u", source_file="f",
                        speaker="S", seq=1, text="far text", embedding=[-1.0] * 768,
                        content_hash="h-far"),
    ])
    db.commit()
    monkeypatch.setattr(retrieval_module, "embed_query",
                        lambda settings, text: vec)
    hits = retrieval_module.retrieve(db, get_settings(), "anything", top_k=5)
    assert [h["episode"] for h in hits] == ["Near"]  # far below 0.4 threshold
    assert hits[0]["relevance"] == pytest.approx(1.0)
    assert hits[0]["excerpt"] == "near text"


def test_artifact_roundtrip_and_scoping(client, db):
    uid = _uid()
    sid = uuid.UUID(
        client.post("/api/v1/sessions", json={}, headers=_h(uid)).json()["id"])
    art = Artifact(session_id=sid, type="markdown", title="Doc", content="# Hi")
    db.add(art)
    db.commit()
    db.refresh(art)

    r = client.get(f"/api/v1/artifacts/{art.id}", headers=_h(uid))
    assert r.status_code == 200
    assert r.json()["title"] == "Doc"
    assert r.json()["content"] == "# Hi"

    r = client.get(f"/api/v1/sessions/{sid}/artifacts", headers=_h(uid))
    assert len(r.json()["artifacts"]) == 1

    other = _uid()
    assert client.get(f"/api/v1/artifacts/{art.id}", headers=_h(other)).status_code == 404
    assert client.get("/api/v1/sessions", headers=_h(other)).json()["total"] == 0
