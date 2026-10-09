"""SSE formatting tests."""

from app.services.sse import format_sse


def test_sse_line_groups():
    out = format_sse("token", {"text": "hello"})
    assert out == 'event: token\ndata: {"text":"hello"}\n\n'


def test_sse_done_event():
    out = format_sse("done", {"message_id": "abc", "skill_used": "answer"})
    assert out.startswith("event: done\n")
    assert out.endswith("\n\n")
