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

## API

- `GET /api/v1/health` — overall status plus separate `database` and `ollama` checks.
- Session / chat / artifact endpoints: shapes are defined by the repo owner in
  the shared contract; this backend implements them exactly once delivered.

Errors always use the envelope:

```json
{ "error": { "code": "validation_error", "message": "...", "details": [...], "request_id": "..." } }
```

## Troubleshooting

Backend troubleshooting notes for the README live here from phase 4 onward.
