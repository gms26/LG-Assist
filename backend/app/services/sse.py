"""Server-Sent Events formatting (contract v1 streaming protocol).

One event per line group::

    event: token
    data: {"text": "..."}

Events: token | sources | artifact | notice | done | error
"""

import json
from collections.abc import Iterator


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def chunk_text(text: str, words_per_chunk: int = 12) -> Iterator[str]:
    """Split text into word chunks for token streaming."""
    words = text.split(" ")
    for i in range(0, len(words), words_per_chunk):
        piece = " ".join(words[i : i + words_per_chunk])
        yield piece + (" " if i + words_per_chunk < len(words) else "")
