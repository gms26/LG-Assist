"""Server-Sent Events formatting (contract v1 streaming protocol).

One event per line group::

    event: token
    data: {"text": "..."}

Events: token | sources | artifact | notice | done | error
"""

import json


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
