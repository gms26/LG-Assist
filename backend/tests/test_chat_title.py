"""Session auto-title tests (contract v1.1, pure logic)."""

from app.routes.chat import make_title


def test_short_message_used_verbatim():
    assert make_title("how do I price?") == "how do I price?"


def test_long_message_truncated_at_word_boundary():
    title = make_title(
        "what are the best pricing strategies for a b2b saas startup selling to mid-market teams"
    )
    assert len(title) <= 61  # 60 chars + ellipsis
    assert title.endswith("…")
    assert not title[-2].isspace()  # no dangling space before …


def test_newlines_and_extra_spaces_collapsed():
    assert make_title("hello\n\n   world") == "hello world"


def test_whitespace_only_returns_none():
    assert make_title("   \n  ") is None


def test_exactly_60_chars_untouched():
    text = "x" * 60
    assert make_title(text) == text


def test_no_spaces_falls_back_to_hard_cut():
    title = make_title("y" * 100)
    assert title == "y" * 60 + "…"
