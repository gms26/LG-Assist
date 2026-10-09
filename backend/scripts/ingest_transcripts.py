#!/usr/bin/env python3
"""Ingest Lenny's Podcast transcripts into pgvector.

Source: https://github.com/LennysNewsletter/lennys-newsletterpodcastdata
(50 podcast transcripts + 10 newsletter posts + index.json metadata).

The source repo is cloned on demand into ``<repo>/data/lenny-transcripts``
(gitignored). The raw files are NEVER committed: the starter-dataset license
allows personal/non-commercial use but forbids redistributing the raw files.

Pipeline per episode:
  1. read index.json metadata (title, date, guest, post_url) + the .md file
  2. parse YAML frontmatter and split the body into speaker turns
     (``**Name** (HH:MM:SS):``); files without speaker turns fall back to
     paragraph chunking (newsletter posts)
  3. group turns into ~450-word chunks with 1-turn overlap
  4. sha256 content hash -> skip chunks already ingested (re-runnable,
     refresh without duplicates)
  5. embed new chunks with Ollama (nomic-embed-text, 768 dims, batched)
  6. insert into transcript_chunks

Usage:
  cd backend && python -m scripts.ingest_transcripts [--limit 50]
      [--include-newsletters] [--force] [--dry-run] [--source DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Allow running as `python scripts/ingest_transcripts.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.models import TranscriptChunk  # noqa: E402

SOURCE_REPO = "https://github.com/LennysNewsletter/lennys-newsletterpodcastdata.git"
CHUNK_MAX_WORDS = 450
CHUNK_OVERLAP_TURNS = 1
EMBED_BATCH_SIZE = 16

# **Speaker Name** (HH:MM:SS):  at the start of a line
TURN_RE = re.compile(r"^\*\*(.+?)\*\*\s*\((\d{2}:\d{2}:\d{2})\):\s*$", re.MULTILINE)


def repo_root() -> Path:
    # backend/scripts/ingest_transcripts.py -> repo root
    return Path(__file__).resolve().parents[2]


def ensure_source(data_dir: Path) -> Path:
    """Clone (or pull) the transcript repo into data_dir. Returns its path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "lenny-transcripts"
    if (dest / "index.json").exists():
        print(f"updating existing transcript checkout at {dest}")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"],
                       check=False, capture_output=True)
        return dest
    print(f"cloning transcript repo into {dest} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", SOURCE_REPO, str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "could not clone the transcript repo:\n"
            f"{result.stderr.strip()}\n"
            "check network access, or place the files manually with --source DIR"
        )
    return dest


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from the body. Returns (meta, body)."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            raw = text[3:end].strip()
            meta: dict = {}
            for line in raw.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip('"')
            return meta, text[end + 4 :].lstrip("\n")
    return {}, text


def parse_turns(body: str) -> list[dict]:
    """Split a transcript body into speaker turns.

    Returns [{"speaker", "timestamp", "text"}]. Empty turns (back-to-back
    headers, common in these transcripts) are dropped.
    """
    matches = list(TURN_RE.finditer(body))
    turns: list[dict] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if not text:
            continue
        turns.append(
            {"speaker": match.group(1).strip(),
             "timestamp": match.group(2),
             "text": text}
        )
    return turns


def chunk_paragraphs(body: str) -> list[dict]:
    """Fallback chunker for files without speaker turns (newsletters)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[dict] = []
    current: list[str] = []
    words = 0
    for para in paras:
        w = len(para.split())
        if words + w > CHUNK_MAX_WORDS and current:
            chunks.append({"speaker": None, "text": "\n\n".join(current)})
            current, words = [], 0
        current.append(para)
        words += w
    if current:
        chunks.append({"speaker": None, "text": "\n\n".join(current)})
    return chunks


def chunk_turns(turns: list[dict]) -> list[dict]:
    """Group speaker turns into ~CHUNK_MAX_WORDS chunks with turn overlap.

    Cuts only at turn boundaries; each chunk keeps one overlapping turn
    from the previous chunk so context isn't lost at the seam.
    """
    chunks: list[dict] = []
    current: list[dict] = []
    words = 0
    for turn in turns:
        w = len(turn["text"].split())
        if words + w > CHUNK_MAX_WORDS and current:
            chunks.append(_make_chunk(current))
            current = current[-CHUNK_OVERLAP_TURNS:] if CHUNK_OVERLAP_TURNS else []
            words = sum(len(t["text"].split()) for t in current)
        current.append(turn)
        words += w
    if current:
        chunks.append(_make_chunk(current))
    return chunks


def _make_chunk(turns: list[dict]) -> dict:
    text = "\n\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
    # most frequent speaker in the chunk (usually the guest answering)
    speaker = Counter(t["speaker"] for t in turns).most_common(1)[0][0]
    return {"speaker": speaker, "text": text}


def content_hash(filename: str, text: str) -> str:
    normalized = re.sub(r"\s+", " ", f"{filename}\n{text.strip()}").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def embed_texts(texts: list[str], base_url: str, model: str, vector_dim: int) -> list[list[float]]:
    """Embed a batch of texts with Ollama's /api/embed."""
    import httpx

    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": batch},
            timeout=120.0,
            trust_env=False,  # local service: never via egress proxy
        )
        resp.raise_for_status()
        vectors = resp.json()["embeddings"]
        if vectors and len(vectors[0]) != vector_dim:
            raise SystemExit(
                f"embedding dim mismatch: model returned {len(vectors[0])}, "
                f"expected {vector_dim}. Fix LGA_OLLAMA_EMBED_MODEL / the pgvector column."
            )
        all_vectors.extend(vectors)
    return all_vectors


def load_episodes(source: Path, include_newsletters: bool, limit: int | None) -> list[dict]:
    index = json.loads((source / "index.json").read_text())
    episodes = []
    for entry in index.get("podcasts", []):
        episodes.append({**entry, "kind": "podcast"})
    if include_newsletters:
        for entry in index.get("newsletters", []):
            episodes.append({**entry, "kind": "newsletter"})
    if limit:
        episodes = episodes[:limit]
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Lenny transcripts into pgvector")
    parser.add_argument("--limit", type=int, default=50,
                        help="max episodes to ingest (default: 50 podcasts)")
    parser.add_argument("--include-newsletters", action="store_true",
                        help="also ingest the 10 newsletter posts")
    parser.add_argument("--force", action="store_true",
                        help="re-ingest even if already present (deletes + re-embeds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and chunk only; no DB writes, no Ollama calls")
    parser.add_argument("--source", type=str, default=None,
                        help="use an existing transcript checkout dir instead of cloning")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(os.environ.get("LGA_DATA_DIR", repo_root() / "data"))
    source = Path(args.source) if args.source else ensure_source(data_dir)

    episodes = load_episodes(source, args.include_newsletters, args.limit)
    print(f"{len(episodes)} episodes selected from {source}")

    # Parse + chunk everything first (no network/DB needed).
    pending: list[dict] = []
    for entry in episodes:
        path = source / entry["filename"]
        if not path.exists():
            print(f"  WARN: missing file {entry['filename']}, skipping")
            continue
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        turns = parse_turns(body)
        chunks = chunk_turns(turns) if turns else chunk_paragraphs(body)
        for seq, chunk in enumerate(chunks):
            pending.append({
                "episode": meta.get("title") or entry.get("title", ""),
                "guest": meta.get("guest") or entry.get("guest"),
                "source_url": meta.get("post_url") or entry.get("post_url"),
                "source_file": entry["filename"],
                "speaker": chunk["speaker"],
                "seq": seq,
                "text": chunk["text"],
                "hash": content_hash(entry["filename"], chunk["text"]),
            })
    print(f"parsed {len(pending)} chunks from {len(episodes)} episodes")

    if args.dry_run:
        words = sum(len(c["text"].split()) for c in pending)
        print(f"dry run: {len(pending)} chunks, ~{words} words. no DB writes, no embeddings.")
        return

    init_db()
    db = SessionLocal()
    try:
        if args.force:
            files = {c["source_file"] for c in pending}
            deleted = db.query(TranscriptChunk).filter(
                TranscriptChunk.source_file.in_(files)).delete(synchronize_session=False)
            db.commit()
            print(f"--force: deleted {deleted} existing chunks for re-ingest")
            existing = set()
        else:
            existing = {row[0] for row in db.query(TranscriptChunk.content_hash).all()}
        new = [c for c in pending if c["hash"] not in existing]
        print(f"{len(existing)} chunks already ingested, {len(new)} new")

        for i in range(0, len(new), EMBED_BATCH_SIZE):
            batch = new[i : i + EMBED_BATCH_SIZE]
            vectors = embed_texts(
                [c["text"] for c in batch],
                settings.ollama_base_url, settings.ollama_embed_model, settings.vector_dim,
            )
            for chunk, vector in zip(batch, vectors):
                db.add(TranscriptChunk(
                    episode=chunk["episode"], guest=chunk["guest"],
                    source_url=chunk["source_url"], source_file=chunk["source_file"],
                    speaker=chunk["speaker"], seq=chunk["seq"],
                    text=chunk["text"], embedding=vector,
                    content_hash=chunk["hash"],
                ))
            db.commit()
            print(f"  ingested {min(i + EMBED_BATCH_SIZE, len(new))}/{len(new)}")
        print("done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
