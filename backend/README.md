# LGA Backend

FastAPI backend for the Lenny Growth Assistant. Free and local by default:
Ollama for chat + embeddings, Postgres with pgvector for storage and retrieval.
No paid API is required.

## Quickstart

```bash
# 1. Postgres with pgvector + API
docker compose up --build

# 2. Ollama on the host (separate terminal)
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # adjust as needed
uvicorn app.main:app --port 8000
```

## Configuration

Everything comes from environment variables prefixed `LGA_` (see `.env.example`).
Safe local defaults are built in; only set `LGA_CLOUD_API_KEY` if you switch
`LGA_PROVIDER` to a cloud provider.

## Layout

```
app/
  main.py        # FastAPI app, router wiring, exception handlers
  config.py      # env-based settings (LGA_*)
  db.py          # engine, session factory, init_db()
  models.py      # users, sessions, messages, transcript_chunks
  schemas.py     # pydantic request/response contracts (provisional — owner defines final contract)
  errors.py      # structured error envelope + handlers
  middleware.py  # request-id middleware
  routes/        # one module per resource (health, sessions, chat, ...)
  services/      # ollama client, retrieval, agent/skills (phase 2/3)
tests/           # automated tests (phase 4)
```

## Key decisions (fixed, changing them later is expensive)

- **Embedding model: `nomic-embed-text` via Ollama, 768 dimensions.** The
  pgvector column width is fixed at table creation, so switching embedding
  models later means wiping `transcript_chunks` and re-ingesting everything
  (the dedupe hash is text-based, so re-embedding would also need a hash
  version bump). This choice is deliberate and documented here.
- **Branch `back`** carries all backend work; the component directory is
  `backend/`. The compose file at `backend/docker-compose.yml` is the
  backend-dev stack only — the repo-root compose (owner's) must not collide
  with it.

## Knowledge base

Transcripts come from Lenny's public starter pack
([lennys-newsletterpodcastdata](https://github.com/LennysNewsletter/lennys-newsletterpodcastdata):
50 podcast transcripts + 10 newsletter posts, with `index.json` metadata).
Ingest with:

```bash
cd backend && python -m scripts.ingest_transcripts [--limit 50] [--include-newsletters] [--force] [--dry-run]
```

The script clones the source repo into `<repo>/data/lenny-transcripts` on
first run (gitignored — raw files are never committed, per the
starter-dataset license: free for personal/non-commercial use, no
redistribution of raw files). Chunks are cut at speaker-turn boundaries
(~450 words, 1-turn overlap), embedded with Ollama `nomic-embed-text`
(768 dims, batched), and upserted into pgvector keyed by content hash, so
re-running never duplicates. Attribution: transcripts © Lenny Rachitsky
([lennysnewsletter.com](https://www.lennysnewsletter.com)).

## API

Contract v1 (`/api/v1`). IDs are UUIDs, timestamps ISO 8601 UTC. The
frontend sends its stored UUID in the `X-User-Id` header; the backend
creates the user row on first sight. Errors always use the envelope
documented above.

- `GET /health` — overall status plus separate `database` and `ollama` checks.
- `POST /sessions` — new chat (`{title?}`) → session.
- `GET /sessions` — list chats newest-first (`limit`, `offset`) → `{sessions, total}`.
- `GET /sessions/{id}` — one chat → `{session, messages}` in order.
- `DELETE /sessions/{id}` — remove chat → 204.
- `POST /sessions/{id}/messages` — send `{content, skill?}` → SSE stream
  (`notice | sources | token* | artifact? | done`, or `error` which ends the
  stream). Unsupported questions return a normal answer with empty sources
  and a `notice` — never an error.
- `GET /config/model` — active provider, model names, available providers,
  whether a fallback is in effect. The frontend never assumes a provider.
- `GET /sessions/{id}/artifacts` — artifacts in a chat.
- `GET /artifacts/{id}` — one artifact (stored exactly as generated; HTML is
  untrusted and rendered only in a sandboxed iframe).

Run the suite: `pytest tests/` (22 tests; DB-backed route tests land in
Phase 4 with a real Postgres).

## Troubleshooting

Backend troubleshooting notes for the README live here from phase 4 onward.
