"""Ingestion parsing/chunking tests (no DB, no Ollama needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_transcripts import (  # noqa: E402
    chunk_paragraphs,
    chunk_turns,
    content_hash,
    parse_frontmatter,
    parse_turns,
)

SAMPLE = """---
title: "Test Episode"
date: "2026-01-01"
type: "podcast"
guest: "Jane Doe"
post_url: "https://example.com/p/test"
---

**Jane Doe** (00:00:00):
Hello world, this is the first turn with some content here.

**Lenny Rachitsky** (00:00:23):
And here is a question from Lenny, short.

**Jane Doe** (00:00:30):
"""  # empty turn: back-to-back headers happen in real transcripts


def test_frontmatter_parsed():
    meta, body = parse_frontmatter(SAMPLE)
    assert meta["title"] == "Test Episode"
    assert meta["guest"] == "Jane Doe"
    assert "**Jane Doe**" in body


def test_turns_parsed_and_empty_dropped():
    _, body = parse_frontmatter(SAMPLE)
    turns = parse_turns(body)
    assert len(turns) == 2  # the empty trailing turn is dropped
    assert turns[0]["speaker"] == "Jane Doe"
    assert turns[0]["timestamp"] == "00:00:00"
    assert "Hello world" in turns[0]["text"]


def test_chunking_respects_word_limit():
    turns = [
        {"speaker": "A", "timestamp": "00:00:00", "text": "word " * 300},
        {"speaker": "B", "timestamp": "00:01:00", "text": "word " * 300},
        {"speaker": "A", "timestamp": "00:02:00", "text": "word " * 100},
    ]
    chunks = chunk_turns(turns)
    # [t1], [t1,t2] (overlap), [t2,t3]: overlap intentionally overshoots
    assert len(chunks) == 3
    for c in chunks:
        assert len(c["text"].split()) <= 450 + 300  # limit + one overlapping turn
    # overlap: later chunks start with the previous chunk's last turn
    assert chunks[1]["text"].startswith("A:")
    assert chunks[2]["text"].startswith("B:")


def test_chunk_speaker_is_most_frequent():
    turns = [
        {"speaker": "Guest", "timestamp": "00:00:00", "text": "a b c"},
        {"speaker": "Guest", "timestamp": "00:01:00", "text": "d e f"},
        {"speaker": "Lenny", "timestamp": "00:02:00", "text": "g h"},
    ]
    assert chunk_turns(turns)[0]["speaker"] == "Guest"


def test_paragraph_fallback_for_newsletters():
    body = "\n\n".join(f"paragraph {i} " * 100 for i in range(6))  # 200 words each
    chunks = chunk_paragraphs(body)
    assert len(chunks) == 3  # 2 paras per 400-word chunk
    assert all(c["speaker"] is None for c in chunks)


def test_content_hash_stable_and_unique():
    h1 = content_hash("ep.md", "hello world")
    assert h1 == content_hash("ep.md", "hello world")
    assert h1 != content_hash("ep.md", "hello world!")
    assert h1 != content_hash("other.md", "hello world")
    assert len(h1) == 64
