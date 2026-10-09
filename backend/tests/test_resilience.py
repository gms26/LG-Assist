"""Resilience tests (Phase 4): every failure mode gets a structured error.

Runs without Postgres or Ollama (fakes + dependency overrides).
DB-backed coverage of the same paths lives in test_api_db.py.

Covered here:
  - Ollama down mid-chat      -> SSE `error` event, code dependency_unavailable
  - Ollama timeout            -> ProviderUnavailable (same error path)
  - Unexpected generation bug -> SSE `error` event, code internal_error
  - User message persisted even when generation fails (contract rule)
  - Database down             -> HTTP 500 envelope with request_id (never a crash)
  - Empty retrieval / missing key paths are asserted in test_agent.py
"""

import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.db import get_db
from app.main import app
from app.models import ChatSession, Message, User
from app.routes import chat as chat_route
from app.services import llm as llm_module
from app.services.llm import ProviderUnavailable, generate_stream


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def order_by(self, *a):
        return self

    def limit(self, *a):
        return self

    def offset(self, *a):
        return self

    def first(self):
        return self._result[0] if isinstance(self._result, list) else self._result

    def all(self):
        return self._result if isinstance(self._result, list) else []

    def count(self):
        return len(self._result) if isinstance(self._result, list) else 0


class _FakeSession:
    """Stand-in for a SQLAlchemy session: records writes, serves canned rows."""

    def __init__(self, user=None, chat_session=None):
        self._user = user
        self._chat = chat_session
        self.added = []
        self.commits = 0

    def query(self, model):
        if model is User:
            return _FakeQuery(self._user)
        if model is ChatSession:
            return _FakeQuery(self._chat)
        return _FakeQuery([])

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass

    def close(self):
        pass


@pytest.fixture()
def fake_db():
    from types import SimpleNamespace

    session = SimpleNamespace(id=uuid.uuid4(), title=None)
    fake = _FakeSession(chat_session=session)
    app.dependency_overrides[get_db] = lambda: fake
    try:
        yield fake
    finally:
        app.dependency_overrides.clear()


def _client():
    return TestClient(app, raise_server_exceptions=False)


def _post_message(client, monkeypatch, turn_fn):
    monkeypatch.setattr(chat_route, "run_turn", turn_fn)
    sid = uuid.uuid4()
    return client.post(
        f"/api/v1/sessions/{sid}/messages",
        json={"content": "hello"},
        headers={"X-User-Id": str(uuid.uuid4())},
    )


def test_ollama_down_streams_dependency_error(fake_db, monkeypatch):
    def boom(*a, **k):
        raise ProviderUnavailable("ollama unavailable: ConnectError")
        yield

    r = _post_message(_client(), monkeypatch, boom)
    assert r.status_code == 200
    assert "event: error" in r.text
    assert "dependency_unavailable" in r.text
    assert "event: done" not in r.text


def test_unexpected_generation_bug_streams_internal_error(fake_db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bug")
        yield

    r = _post_message(_client(), monkeypatch, boom)
    assert "event: error" in r.text
    assert "internal_error" in r.text


def test_user_message_saved_before_failed_generation(fake_db, monkeypatch):
    def boom(*a, **k):
        raise ProviderUnavailable("down")
        yield

    _post_message(_client(), monkeypatch, boom)
    user_msgs = [o for o in fake_db.added
                 if isinstance(o, Message) and o.role == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "hello"


def test_partial_turn_saved_on_midstream_failure(fake_db, monkeypatch):
    def partial(*a, **k):
        yield ("token", {"text": "half an "})
        raise ProviderUnavailable("down")

    _post_message(_client(), monkeypatch, partial)
    asst = [o for o in fake_db.added
            if isinstance(o, Message) and o.role == "assistant"]
    assert len(asst) == 1
    assert asst[0].content == "half an "


def test_db_failure_returns_structured_500():
    def _bad_db():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))
        yield  # make it a generator like get_db

    app.dependency_overrides[get_db] = _bad_db
    try:
        r = _client().get("/api/v1/sessions",
                          headers={"X-User-Id": str(uuid.uuid4())})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["code"] == "internal_error"
    assert err["message"]
    assert err["request_id"]


def test_ollama_timeout_becomes_provider_unavailable(monkeypatch):
    def _timeout(*a, **k):
        raise httpx.TimeoutException("timed out", request=None)

    monkeypatch.setattr(llm_module, "stream_chat_ollama", _timeout)
    with pytest.raises(ProviderUnavailable):
        list(generate_stream(Settings(), [{"role": "user", "content": "q"}]))


def test_ollama_connection_refused_becomes_provider_unavailable(monkeypatch):
    def _refused(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm_module, "stream_chat_ollama", _refused)
    with pytest.raises(ProviderUnavailable):
        list(generate_stream(Settings(), [{"role": "user", "content": "q"}]))
