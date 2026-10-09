# First run — from zero to a working demo

Total wall time: roughly **45–75 minutes**, mostly downloads and ingestion
(mostly unattended).

## 0. Prerequisites

- Docker + Docker Compose
- Ollama installed on your laptop: https://ollama.com
- ~8 GB RAM recommended (see "Model choices" below), ~10 GB free disk

## 1. Pull the models (on your laptop, not in Docker)

```bash
ollama pull qwen2.5:7b        # chat model, ~4.7 GB
ollama pull nomic-embed-text  # embedding model, ~274 MB — 768 dimensions (fixed)
```

The embedding model and its **768 vector size are fixed** — the pgvector
column width is created for exactly this size. Do not swap it without
re-ingesting everything (see README "Key decisions").

Check: `ollama list` should show both models. Leave `ollama serve` running
(it starts automatically on most installs).

## 2. Start the stack

```bash
cd backend
docker compose up --build
```

This starts Postgres 16 with pgvector (`:5432`) and the API (`:8000`).
The API boots even if Ollama isn't reachable yet — health will say so.

## 3. Ingest the transcripts

In another terminal:

```bash
cd backend
docker compose exec api python -m scripts.ingest_transcripts
```

What it does: clones the official Lenny starter pack
(LennysNewsletter/lennys-newsletterpodcastdata — 50 podcast transcripts)
into the `lga_data` volume, chunks by speaker turn (~450 words, 1-turn
overlap), embeds with `nomic-embed-text`, and upserts into pgvector keyed by
content hash, so re-running never duplicates.

**Expected run time:** ~50 episodes → ~1,500 chunks → **about 15–30 minutes
on a laptop CPU**, almost entirely embedding-bound (Ollama, batched).
The git clone itself takes under a minute.

Want a quick trial first? Ingest 5 episodes (~2–3 minutes) to validate the
whole pipeline:

```bash
docker compose exec api python -m scripts.ingest_transcripts --limit 5
```

then run the full ingest afterwards — already-ingested chunks are skipped.

## 4. Verify

```bash
curl localhost:8000/api/v1/health | python3 -m json.tool
```

Expect:

```json
{
  "status": "ok",
  "checks": {
    "database": {"status": "ok", ...},
    "ollama": {"status": "ok", "models": ["qwen2.5:7b...", "nomic-embed-text..."]}
  }
}
```

```bash
curl localhost:8000/api/v1/config/model
```

Expect `active_provider: "ollama"`, `fallback_in_effect: false`.

Then point the frontend at `http://localhost:8000` and chat. The first
answer is slow (~30–60 s) while the chat model loads into RAM; after that
it stays warm. A Ship 30 essay runs ~9 sequential local-model calls —
a few minutes on CPU is normal.

## 5. Run the test suite

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

Unit + resilience tests run anywhere. The DB-backed suite
(`tests/test_api_db.py`) needs the stack up and a test database:

```bash
docker compose exec db psql -U lga -c "CREATE DATABASE lga_test;"
LGA_TEST_DATABASE_URL=postgresql+psycopg://lga:lga@localhost:5432/lga_test \
  pytest tests/test_api_db.py -q
```

## Model choices

| Machine | Chat model | Command |
|---|---|---|
| 8 GB RAM (recommended) | **qwen2.5:7b** (default, Q4 ~4.7 GB) | `ollama pull qwen2.5:7b` |
| ~4 GB RAM, tight | **llama3.2:3b** (~2 GB) | `ollama pull llama3.2:3b` + `LGA_OLLAMA_CHAT_MODEL=llama3.2:3b` |
| Very tight | qwen2.5:1.5b (~1 GB) | `ollama pull qwen2.5:1.5b` + same env var |

**Recommendation:** stay on `qwen2.5:7b` if you can — the essay pipeline
leans on instruction-following (outline format, citation discipline), and
3B models degrade visibly there: outlines garble more often (the
deterministic fallback covers it, but essays get blander) and citations get
sloppy. `llama3.2:3b` is the sane fallback: it runs the whole demo,
answers stay grounded, just expect simpler prose. Switch with the
`LGA_OLLAMA_CHAT_MODEL` env var (compose file or `.env`) — no code change.

The embedding model stays `nomic-embed-text` regardless (tiny, fast, fixed
768 dims).

## Troubleshooting

- **`ollama` shows `unreachable` in health:** is `ollama serve` running on
  the host? From inside the container, the host is
  `http://host.docker.internal:11434` (already configured; the compose file
  adds the Linux `host-gateway` mapping).
- **`database` shows `unreachable`:** `docker compose ps` — the `db`
  service must be `healthy`. Port 5432 already in use by a local Postgres?
  Stop it or remap the port.
- **Ingestion fails with an embedding dimension mismatch:** a different
  embedding model got pulled. It must be `nomic-embed-text` (768).
  `ollama list` to check.
- **Ingestion can't clone (network/DNS in container):** clone on the host
  instead (`git clone https://github.com/LennysNewsletter/lennys-newsletterpodcastdata.git
  data/lenny-transcripts`) and run the script on the host with a venv
  (defaults already point at localhost DB + Ollama).
- **Chat answers "not supported by the transcripts":** ingestion hasn't run
  yet, or it ingested 0 chunks — check the script's output counts.
- **Port 8000 in use:** change the mapping in `docker-compose.yml`
  (`"8001:8000"`) or set `LGA_PORT`.
- **Essays are slow:** expected — ~9 sequential model calls on CPU.
  Nothing is stuck; the UI shows a `notice` per stage.
- **`docker compose exec api ...` says no such service:** run it from
  `backend/` where the compose file lives.
