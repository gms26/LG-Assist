"""Ship 30 for 30 essay skill — staged pipeline (item 5).

A STRUCTURED skill definition, not one prompt: principles, stage
definitions, the ~1,250-word target with a length check, and grounding
rules are data + separate stage functions. The official Ship 30 guide's
key rules slot into PRINCIPLES below when they arrive.

Pipeline stages:
  1. core idea   — one-sentence thesis + why-it-matters, from the chunks
  2. outline     — hook angle, 4 sections each tied to source chunk indices,
                   word budgets, takeaway angle (parsed; deterministic
                   fallback if the model garbles the format)
  3. draft       — hook, each section, and takeaway drafted separately,
                   every section grounded ONLY in its assigned excerpts
  4. check+revise— word-count check (target ±10%) and grounding check
                   (every section cites ≥1 assigned excerpt); exactly ONE
                   revision pass, then ship
  5. assemble    — final markdown: hook, headings, bullets, selective bold,
                   one specific takeaway, source list

Thin retrieval (< MIN_CHUNKS usable chunks) -> a notice asking for a
narrower topic instead of an essay. Never an error.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ...config import Settings
from ..llm import ProviderUnavailable, chat_once
from ..retrieval import retrieve
from ..sse import chunk_text

NOTICE_THIN = "not enough source material for an essay — try a narrower topic"

THIN_MESSAGE = (
    "I couldn't find enough in Lenny's transcripts to write a full essay on "
    "this topic. Try narrowing it down — for example, ask about a specific "
    "area like pricing, onboarding, hiring, or growth loops — and I'll build "
    "the essay from the episodes that cover it."
)


@dataclass
class Ship30Spec:
    """The skill's structured definition."""

    name: str = "ship30_essay"
    target_words: int = 1250
    tolerance: float = 0.10  # ±10% -> 1125..1375
    min_chunks: int = 4  # below this, retrieval is "too thin"
    essay_top_k: int = 10  # wider retrieval for essays than Q&A
    n_sections: int = 4

    # Writing principles. The official Ship 30 guide's key rules get
    # appended here verbatim when they arrive — the pipeline reads this
    # list, so no prompt surgery is needed later.
    principles: list[str] = field(default_factory=lambda: [
        "Open with a strong hook — a surprising claim, vivid moment, or "
        "counterintuitive question that earns the next paragraph.",
        "Clear narrative progression: one core idea, built section by section.",
        "Skimmable formatting: headings, bullets, and selective bold for key "
        "terms only — never bold whole sentences.",
        "End with one specific, useful takeaway the reader can act on.",
        "Every substantive claim must be grounded in a cited transcript "
        "excerpt. No claim without a source. Cite inline as (Episode: \"...\").",
        # --- official Ship 30 guide rules (pending; appended on arrival) ---
    ])

    @property
    def min_words(self) -> int:
        return int(self.target_words * (1 - self.tolerance))

    @property
    def max_words(self) -> int:
        return int(self.target_words * (1 + self.tolerance))


SPEC = Ship30Spec()


# ---------------------------------------------------------------------------
# Stage 1: core idea
# ---------------------------------------------------------------------------

CORE_IDEA_SYSTEM = (
    "You distill the single core idea from transcript excerpts. "
    "Reply with exactly two lines and nothing else:\n"
    "IDEA: <one sentence: the core idea the excerpts support>\n"
    "WHY: <one sentence: why it matters to the reader>"
)


def stage_core_idea(question: str, hits: list[dict], settings: Settings) -> str:
    excerpts = "\n\n".join(
        f"(Episode: \"{h['episode']}\") {h['text'][:800]}" for h in hits[:6]
    )
    reply = chat_once(
        settings,
        [
            {"role": "system", "content": CORE_IDEA_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{excerpts}"},
        ],
        timeout_s=60.0,
    )
    for line in reply.splitlines():
        if line.strip().upper().startswith("IDEA:"):
            idea = line.split(":", 1)[1].strip()
            if idea:
                return idea
    first = next((ln.strip() for ln in reply.splitlines() if ln.strip()), "")
    return first or question


# ---------------------------------------------------------------------------
# Stage 2: outline tied to source chunks
# ---------------------------------------------------------------------------

OUTLINE_SYSTEM = (
    "You are an essay outliner. Use the EXACT line format below, nothing else.\n"
    "TITLE: <essay title>\n"
    "HOOK_ANGLE: <one line: the surprising opening angle>\n"
    "SECTION: <section title> | CHUNKS: <idx>,<idx> | WORDS: <number> | ANGLE: <one line: what this section argues>\n"
    "(one SECTION line per section)\n"
    "TAKEAWAY_ANGLE: <one line: the single specific actionable takeaway>\n"
    "Rules: exactly {n} sections; every chunk index 0..{max_idx} appears in at "
    "least one section; WORDS values sum to about {budget}; section titles "
    "are specific, never generic."
)


def _chunk_catalog(hits: list[dict]) -> str:
    return "\n".join(
        f"[{i}] \"{h['episode']}\""
        + (f" — {h['guest']}" if h.get("guest") else "")
        + f": {h['text'][:280]}..."
        for i, h in enumerate(hits)
    )


def parse_outline(reply: str, n_chunks: int, spec: Ship30Spec) -> dict | None:
    """Parse the outline format. Returns None when invalid -> fallback."""
    title = hook_angle = takeaway_angle = None
    sections: list[dict] = []
    for raw in reply.splitlines():
        line = raw.strip()
        if line.startswith("TITLE:"):
            title = line[6:].strip()
        elif line.startswith("HOOK_ANGLE:"):
            hook_angle = line[11:].strip()
        elif line.startswith("TAKEAWAY_ANGLE:"):
            takeaway_angle = line[15:].strip()
        elif line.startswith("SECTION:"):
            parts = [p.strip() for p in line[8:].split("|")]
            if len(parts) != 4:
                return None
            sec_title = parts[0]
            chunks_part = parts[1].upper().replace("CHUNKS:", "").strip()
            words_part = parts[2].upper().replace("WORDS:", "").strip()
            angle = parts[3].upper().replace("ANGLE:", "").strip()
            try:
                idxs = [int(x) for x in chunks_part.replace(" ", "").split(",") if x != ""]
                words = int(words_part)
            except ValueError:
                return None
            if not idxs or any(i < 0 or i >= n_chunks for i in idxs) or words <= 0:
                return None
            sections.append({"title": sec_title, "chunk_idxs": idxs,
                             "words": words, "angle": angle})
    if not (title and hook_angle and takeaway_angle):
        return None
    if len(sections) != spec.n_sections:
        return None
    covered = {i for s in sections for i in s["chunk_idxs"]}
    if covered != set(range(n_chunks)):
        return None
    return {"title": title, "hook_angle": hook_angle,
            "sections": sections, "takeaway_angle": takeaway_angle}


def fallback_outline(question: str, hits: list[dict], spec: Ship30Spec) -> dict:
    """Deterministic outline when the model garbles the format."""
    groups: list[list[int]] = [[] for _ in range(spec.n_sections)]
    for i in range(len(hits)):
        groups[i % spec.n_sections].append(i)
    groups = [g for g in groups if g]
    budget = spec.target_words - 200  # hook + takeaway reserve
    per = budget // len(groups)
    sections = []
    for n, idxs in enumerate(groups):
        ep = hits[idxs[0]]["episode"]
        sections.append({
            "title": f"Lesson {n + 1}: {ep[:60]}",
            "chunk_idxs": idxs,
            "words": per,
            "angle": f"what the excerpts teach about: {question}",
        })
    return {
        "title": question[:80],
        "hook_angle": "the most surprising claim in the excerpts",
        "sections": sections,
        "takeaway_angle": "one specific action for the reader",
    }


def stage_outline(
    question: str, idea: str, hits: list[dict], settings: Settings,
    spec: Ship30Spec = SPEC,
) -> dict:
    budget = spec.target_words - 200
    reply = chat_once(
        settings,
        [
            {"role": "system", "content": OUTLINE_SYSTEM.format(
                n=spec.n_sections, max_idx=len(hits) - 1, budget=budget)},
            {"role": "user",
             "content": f"Question: {question}\nCore idea: {idea}\n\n"
                        f"Source chunks:\n{_chunk_catalog(hits)}"},
        ],
        timeout_s=60.0,
    )
    parsed = parse_outline(reply, len(hits), spec)
    return parsed or fallback_outline(question, hits, spec)


# ---------------------------------------------------------------------------
# Stage 3: section-by-section draft
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = (
    "You write ONE essay section. Ground every claim in the excerpts and cite "
    "inline like (Episode: \"...\"). Use short paragraphs, bullets for lists, "
    "and **bold** sparingly for key terms only. Do NOT write a heading — start "
    "directly with the first sentence."
)

HOOK_SYSTEM = (
    "You write the opening hook of an essay (~80 words): a surprising claim, "
    "vivid moment, or counterintuitive question drawn from the excerpts. Cite "
    "inline like (Episode: \"...\"). Start directly — no heading, no preamble."
)

TAKEAWAY_SYSTEM = (
    "You write the closing takeaway of an essay (~120 words): ONE specific, "
    "actionable takeaway for the reader, grounded in the excerpts. Cite inline "
    "like (Episode: \"...\"). Start directly — no heading."
)


def _strip_heading(text: str) -> str:
    lines = text.strip().splitlines()
    while lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    return "\n".join(lines).strip()


def stage_draft_section(
    section: dict, idea: str, hits: list[dict], settings: Settings
) -> str:
    excerpts = "\n\n".join(
        f"(Episode: \"{hits[i]['episode']}\") {hits[i]['text'][:900]}"
        for i in section["chunk_idxs"]
    )
    reply = chat_once(
        settings,
        [
            {"role": "system", "content": DRAFT_SYSTEM},
            {"role": "user",
             "content": f"Core idea: {idea}\nSection: {section['title']}\n"
                        f"Angle: {section['angle']}\n"
                        f"Target: ~{section['words']} words.\n\n"
                        f"Excerpts (use ONLY these):\n{excerpts}"},
        ],
        timeout_s=120.0,
    )
    return _strip_heading(reply)


def stage_draft_hook(hook_angle: str, hits: list[dict], settings: Settings) -> str:
    excerpts = "\n\n".join(
        f"(Episode: \"{h['episode']}\") {h['text'][:700]}" for h in hits[:2]
    )
    reply = chat_once(
        settings,
        [
            {"role": "system", "content": HOOK_SYSTEM},
            {"role": "user",
             "content": f"Hook angle: {hook_angle}\n\nExcerpts:\n{excerpts}"},
        ],
        timeout_s=60.0,
    )
    return _strip_heading(reply)


def stage_draft_takeaway(
    takeaway_angle: str, idea: str, hits: list[dict], settings: Settings
) -> str:
    excerpts = "\n\n".join(
        f"(Episode: \"{h['episode']}\") {h['text'][:500]}" for h in hits[:3]
    )
    reply = chat_once(
        settings,
        [
            {"role": "system", "content": TAKEAWAY_SYSTEM},
            {"role": "user",
             "content": f"Core idea: {idea}\nTakeaway angle: {takeaway_angle}\n\n"
                        f"Excerpts:\n{excerpts}"},
        ],
        timeout_s=60.0,
    )
    return _strip_heading(reply)


# ---------------------------------------------------------------------------
# Stage 4: word-count + grounding check, one revision pass
# ---------------------------------------------------------------------------

CITATION_MARK = "(Episode:"


def check_essay(parts: dict, spec: Ship30Spec = SPEC) -> dict:
    """Deterministic checks. Returns word counts + sections missing citations."""
    def words(t: str) -> int:
        return len(t.split())

    section_words = [words(s["text"]) for s in parts["sections"]]
    total = words(parts["hook"]) + sum(section_words) + words(parts["takeaway"])
    missing = [i for i, s in enumerate(parts["sections"])
               if CITATION_MARK not in s["text"]]
    return {
        "total_words": total,
        "section_words": section_words,
        "in_range": spec.min_words <= total <= spec.max_words,
        "sections_missing_citations": missing,
    }


def revise_once(
    parts: dict, check: dict, hits: list[dict], settings: Settings,
    spec: Ship30Spec = SPEC,
) -> dict:
    """Exactly one revision pass: fix word count and/or missing citations."""
    # Word-count fix: expand the thinnest or trim the longest section.
    total = check["total_words"]
    target_idx: int | None = None
    mode: str | None = None
    if total < spec.min_words:
        target_idx = min(range(len(parts["sections"])),
                         key=lambda i: check["section_words"][i])
        mode = "expand"
    elif total > spec.max_words:
        target_idx = max(range(len(parts["sections"])),
                         key=lambda i: check["section_words"][i])
        mode = "trim"

    fix: dict[int, str] = {}
    if target_idx is not None:
        fix[target_idx] = mode  # type: ignore[assignment]
    for i in check["sections_missing_citations"]:
        # A section can need both fixes; citation instruction wins the wording.
        fix[i] = "cite" if fix.get(i) is None else fix[i] + "+cite"

    for idx, kind in fix.items():
        section = parts["sections"][idx]
        excerpts = "\n\n".join(
            f"(Episode: \"{hits[i]['episode']}\") {hits[i]['text'][:900]}"
            for i in section["chunk_idxs"]
        )
        if "expand" in kind:
            instruction = (
                f"Expand this section with ~{spec.target_words - total} more words, "
                f"using ONLY the excerpts below. Keep every existing citation.")
        elif "trim" in kind:
            instruction = (
                f"Tighten this section to about {section['words']} words, "
                f"keeping all citations and key points.")
        else:
            instruction = (
                "Rewrite this section so every key claim carries an inline "
                "citation like (Episode: \"...\") from the excerpts below.")
        if "cite" in kind and kind != "cite":
            instruction += " Ensure every key claim carries an inline citation."
        reply = chat_once(
            settings,
            [
                {"role": "system", "content": DRAFT_SYSTEM},
                {"role": "user",
                 "content": f"{instruction}\n\nCurrent section:\n{section['text']}\n\n"
                            f"Excerpts:\n{excerpts}"},
            ],
            timeout_s=120.0,
        )
        parts["sections"][idx]["text"] = _strip_heading(reply)
    return parts


# ---------------------------------------------------------------------------
# Stage 5: assemble
# ---------------------------------------------------------------------------

def stage_assemble(parts: dict, hits: list[dict]) -> str:
    md = f"# {parts['title']}\n\n{parts['hook']}\n\n"
    for s in parts["sections"]:
        md += f"## {s['title']}\n\n{s['text']}\n\n"
    md += f"## The takeaway\n\n{parts['takeaway']}\n\n---\n\n## Sources\n\n"
    seen: set[str] = set()
    for h in hits:
        key = h["episode"]
        if key in seen:
            continue
        seen.add(key)
        line = f"- \"{h['episode']}\""
        if h.get("guest"):
            line += f" — {h['guest']}"
        if h.get("source_url"):
            line += f" ([link]({h['source_url']}))"
        md += line + "\n"
    return md.rstrip() + "\n"


# ---------------------------------------------------------------------------
# Pipeline driver (SSE events)
# ---------------------------------------------------------------------------

def run_essay_skill(
    db,
    chat_session,
    question: str,
    settings: Settings,
    spec: Ship30Spec = SPEC,
) -> Iterator[tuple[str, dict]]:
    """Run the five-stage essay pipeline, yielding SSE (event, data) tuples.

    Raises ProviderUnavailable when no model can serve (the route turns it
    into an error event).
    """
    hits = retrieve(db, settings, question, top_k=spec.essay_top_k)

    if len(hits) < spec.min_chunks:
        yield ("notice", {"message": NOTICE_THIN})
        yield ("sources", {"sources": []})
        for piece in chunk_text(THIN_MESSAGE):
            yield ("token", {"text": piece})
        yield ("done", {"skill_used": "essay"})
        return

    yield ("sources", {"sources": [
        {k: h[k] for k in ("episode", "guest", "source_url", "source_file",
                           "excerpt", "relevance", "chunk_id")}
        for h in hits
    ]})

    yield ("notice", {"message": "Finding the core idea…"})
    idea = stage_core_idea(question, hits, settings)

    yield ("notice", {"message": "Outlining the essay…"})
    outline = stage_outline(question, idea, hits, settings, spec)

    yield ("notice", {"message": "Drafting the hook…"})
    hook = stage_draft_hook(outline["hook_angle"], hits, settings)

    sections = []
    for n, section in enumerate(outline["sections"]):
        yield ("notice", {"message": f"Drafting section {n + 1}/{len(outline['sections'])}…"})
        text = stage_draft_section(section, idea, hits, settings)
        sections.append({**section, "text": text})

    yield ("notice", {"message": "Drafting the takeaway…"})
    takeaway = stage_draft_takeaway(outline["takeaway_angle"], idea, hits, settings)

    parts = {"title": outline["title"], "hook": hook,
             "sections": sections, "takeaway": takeaway}

    yield ("notice", {"message": "Checking length and citations…"})
    check = check_essay(parts, spec)
    if not check["in_range"] or check["sections_missing_citations"]:
        yield ("notice", {"message": "One revision pass…"})
        parts = revise_once(parts, check, hits, settings, spec)

    essay = stage_assemble(parts, hits)
    for piece in chunk_text(essay, words_per_chunk=24):
        yield ("token", {"text": piece})
    yield ("done", {"skill_used": "essay"})
