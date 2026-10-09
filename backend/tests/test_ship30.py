"""Ship 30 skill tests (item 5): outline parsing, checks, assembly, and the
full staged pipeline with a canned LLM — no network or DB needed."""

from types import SimpleNamespace

from app.config import Settings
from app.services.skills import ship30
from app.services.skills.ship30 import (
    NOTICE_THIN,
    SPEC,
    check_essay,
    fallback_outline,
    parse_outline,
    run_essay_skill,
    stage_assemble,
)


def _hit(i):
    return {
        "episode": f"Episode {i}", "guest": f"Guest {i}",
        "source_url": f"https://example.com/{i}", "source_file": f"ep{i}.md",
        "excerpt": f"excerpt {i}", "relevance": 0.9 - i * 0.01,
        "chunk_id": f"chunk-{i}", "text": f"full text {i} " * 50,
    }


HITS = [_hit(i) for i in range(8)]

OUTLINE_REPLY = """TITLE: Pricing Is Positioning
HOOK_ANGLE: the price tag is the pitch
SECTION: Charge more, mean more | CHUNKS: 0,1 | WORDS: 260 | ANGLE: price signals value
SECTION: Onboard to the price | CHUNKS: 2,3 | WORDS: 260 | ANGLE: onboarding justifies cost
SECTION: Hire the storytellers | CHUNKS: 4,5 | WORDS: 260 | ANGLE: team sells value
SECTION: Loop the learnings | CHUNKS: 6,7 | WORDS: 260 | ANGLE: iterate on packaging
TAKEAWAY_ANGLE: raise prices 10% this quarter
"""


def test_parse_outline_valid():
    parsed = parse_outline(OUTLINE_REPLY, 8, SPEC)
    assert parsed is not None
    assert parsed["title"] == "Pricing Is Positioning"
    assert len(parsed["sections"]) == 4
    assert parsed["sections"][0]["chunk_idxs"] == [0, 1]
    assert parsed["sections"][0]["words"] == 260


def test_parse_outline_rejects_bad_format():
    assert parse_outline("TITLE: x\nno sections here", 8, SPEC) is None
    # chunk index out of range
    bad = OUTLINE_REPLY.replace("CHUNKS: 6,7", "CHUNKS: 6,99")
    assert parse_outline(bad, 8, SPEC) is None
    # not all chunks covered
    bad2 = OUTLINE_REPLY.replace("CHUNKS: 6,7 |", "CHUNKS: 6 |")
    assert parse_outline(bad2, 8, SPEC) is None


def test_fallback_outline_covers_all_chunks():
    outline = fallback_outline("q", HITS, SPEC)
    covered = {i for s in outline["sections"] for i in s["chunk_idxs"]}
    assert covered == set(range(8))
    assert len(outline["sections"]) == SPEC.n_sections


def _parts(cite_all=True):
    sections = []
    for i in range(4):
        text = "word " * 250
        if cite_all or i != 1:
            text += ' (Episode: "Episode 0")'
        sections.append({"title": f"S{i}", "text": text, "chunk_idxs": [i]})
    return {"title": "T", "hook": "word " * 100,
            "sections": sections, "takeaway": "word " * 150}


def test_check_essay_word_count_and_citations():
    check = check_essay(_parts(), SPEC)
    # per section: 250 words + 3-word citation; hook 100; takeaway 150
    assert check["total_words"] == 100 + 4 * 253 + 150
    assert check["in_range"]  # 1262 within 1125..1375
    assert check["sections_missing_citations"] == []


def test_check_essay_flags_missing_citations():
    check = check_essay(_parts(cite_all=False), SPEC)
    assert check["sections_missing_citations"] == [1]


def test_check_essay_flags_short_essay():
    parts = _parts()
    parts["sections"] = [{"title": "S", "text": "tiny", "chunk_idxs": [0]}]
    check = check_essay(parts, SPEC)
    assert not check["in_range"]
    assert check["total_words"] < SPEC.min_words


def test_assemble_structure():
    md = stage_assemble(_parts(), HITS)
    assert md.startswith("# T\n")
    for i in range(4):
        assert f"## S{i}\n" in md
    assert "## The takeaway" in md
    assert "## Sources" in md
    # guest + link rendered in the source list
    assert "Guest 0" in md
    assert "https://example.com/0" in md


def test_assemble_dedupes_sources_by_episode():
    hits = [_hit(0), _hit(0), _hit(1)]  # two chunks, same episode
    md = stage_assemble(_parts(), hits)
    sources = md.split("## Sources")[1]
    assert sources.count('"Episode 0"') == 1
    assert sources.count('"Episode 1"') == 1


def test_thin_retrieval_yields_notice_not_essay(monkeypatch):
    monkeypatch.setattr(ship30, "retrieve", lambda *a, **k: [])

    def boom(*a, **k):
        raise AssertionError("LLM must not be called on thin retrieval")

    monkeypatch.setattr(ship30, "chat_once", boom)
    events = list(run_essay_skill(None, SimpleNamespace(id="s"), "q", Settings()))
    names = [e for e, _ in events]
    assert names[0] == "notice"
    assert NOTICE_THIN in events[0][1]["message"]
    assert events[1] == ("sources", {"sources": []})
    assert names[-1] == "done"
    assert events[-1][1]["skill_used"] == "essay"


def test_full_pipeline_with_canned_llm(monkeypatch):
    monkeypatch.setattr(ship30, "retrieve", lambda *a, **k: HITS)

    s_cited = lambda: "word " * 250 + ' (Episode: "Episode 0") '  # noqa: E731
    responses = [
        "IDEA: Pricing is positioning.\nWHY: It shapes perceived value.",  # core idea
        OUTLINE_REPLY,                                                    # outline
        "word " * 100,                                                    # hook
        s_cited(), s_cited(), "word " * 250, s_cited(),                    # sections (s2 no cite)
        "word " * 150,                                                    # takeaway
        "word " * 250 + ' (Episode: "Episode 2") ',                        # revision: fix s2
    ]

    def fake_chat_once(settings, messages, timeout_s=None):
        return responses.pop(0)

    monkeypatch.setattr(ship30, "chat_once", fake_chat_once)
    events = list(run_essay_skill(None, SimpleNamespace(id="s"), "q", Settings()))

    names = [e for e, _ in events]
    assert "error" not in names
    assert names[-1] == "done"
    assert events[-1][1]["skill_used"] == "essay"
    # stage notices were emitted
    notices = [d["message"] for e, d in events if e == "notice"]
    assert any("Outlining" in n for n in notices)
    assert any("revision" in n for n in notices)
    # sources event carries the hits
    sources = next(d["sources"] for e, d in events if e == "sources")
    assert len(sources) == 8
    # assembled essay: structure + revision applied
    essay = "".join(d["text"] for e, d in events if e == "token")
    assert "# Pricing Is Positioning" in essay
    assert essay.count("## ") >= 6  # 4 sections + takeaway + sources
    assert "## The takeaway" in essay
    assert "## Sources" in essay
    assert not responses, "all canned LLM calls consumed exactly once"
