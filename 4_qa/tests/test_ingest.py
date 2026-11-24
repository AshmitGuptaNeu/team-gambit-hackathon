"""Transcript parsing, chunking, decision/action extraction, cards, filler cleaning. No model needed."""
from datetime import datetime, timezone

import pytest

from cards import card_text, topics
from chunking import chunk_meeting
from items import extract_items
from text import clean_speech, tokenize
from transcript import MeetingInput, Utterance, load_session, mmss, parse_transcript

T0 = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)


def meeting(lines, mid="m1", **kw):
    return MeetingInput(mid, T0, [Utterance(float(s), who, text) for s, who, text in lines], **kw)


# ---- transcript
def test_parse_basic_and_sorting():
    us = parse_transcript("[00:10] Bob: second\n[00:02] Alice: first\n")
    assert [(u.start, u.speaker, u.text) for u in us] == [(2.0, "Alice", "first"), (10.0, "Bob", "second")]


def test_parse_long_meetings_and_hours():
    us = parse_transcript("[125:03] Ann: late\n[1:02:03] Ann: with hours")
    assert [u.start for u in us] == [3723.0, 7503.0]


def test_parse_colon_in_text_and_speaker_with_parens():
    us = parse_transcript("[01:00] Multiple speakers (A, B): meet at 10:30: ok")
    assert us[0].speaker == "Multiple speakers (A, B)" and us[0].text == "meet at 10:30: ok"


def test_parse_continuation_blank_lines_and_empty_text():
    us = parse_transcript("\n[00:01] A: hello\nand more\n\n[00:05] B:\n")
    assert len(us) == 1 and us[0].text == "hello and more"


def test_parse_rejects_text_before_first_line():
    with pytest.raises(ValueError):
        parse_transcript("garbage\n[00:01] A: hi")


def test_mmss_matches_ingestion_format():
    assert mmss(0) == "00:00" and mmss(61.9) == "01:01" and mmss(7503) == "125:03"


def test_meeting_requires_timezone():
    with pytest.raises(ValueError):
        MeetingInput("x", datetime(2026, 1, 1), [])


def test_load_session_reads_folder_and_start_time(tmp_path):
    (tmp_path / "transcript.txt").write_text("[00:01] Priya: hello there\n")
    (tmp_path / "pieces.jsonl").write_text('{"start_ms": 1763478000000}\n{"start_ms": 1763477000000}\nnot json\n')
    m = load_session(str(tmp_path), title="Sync")
    assert m.meeting_id == tmp_path.name and m.attendees == ["Priya"]
    assert m.started_at == datetime.fromtimestamp(1763477000, tz=timezone.utc)


# ---- chunking
def test_chunks_cut_at_window_and_never_split_an_utterance():
    lines = [(s, "A", f"word {s} " * 5) for s in range(0, 200, 10)]
    chunks = chunk_meeting(meeting(lines), window_seconds=60, max_words=1000, min_words=0)
    assert [c.start for c in chunks] == [0, 60, 120, 180]
    assert sum(c.text.count("\n") + 1 for c in chunks) == len(lines)
    assert chunks[0].end == 60 and chunks[0].span == "00:00-01:00"
    assert [c.chunk_id for c in chunks][:2] == ["m1#0000", "m1#0001"]


def test_long_monologue_is_cut_by_words():
    lines = [(s, "A", "talk " * 50) for s in range(0, 40, 5)]
    chunks = chunk_meeting(meeting(lines), window_seconds=60, max_words=120, min_words=0)
    assert len(chunks) == 4 and all(len(c.text.split()) <= 2 * 120 for c in chunks)


def test_tiny_last_chunk_is_merged():
    lines = [(0, "A", "a " * 30), (70, "B", "short end")]
    chunks = chunk_meeting(meeting(lines), window_seconds=60, max_words=500, min_words=25)
    assert len(chunks) == 1 and chunks[0].speakers == ["A", "B"]


def test_empty_meeting_has_no_chunks():
    assert chunk_meeting(meeting([])) == []


def test_chunks_keep_source_ids():
    m = MeetingInput("m", T0, [Utterance(0, "A", "hi there", uid="d1"), Utterance(5, "B", "yo", uid="d2")])
    assert chunk_meeting(m)[0].utterance_uids == ["d1", "d2"]


# ---- items
@pytest.mark.parametrize("text,kind,owner", [
    ("Okay, let's go with 59 dollars for the starter plan.", "decision", None),
    ("We decided to postpone dark mode to version two.", "decision", None),
    ("We agreed on a 20 percent discount.", "decision", None),
    ("I'll build the landing page by Friday.", "action", "Priya Shah"),
    ("Dana, can you send the Figma file?", "action", "Dana Kim"),
    ("Please send the final numbers by Thursday.", "action", None),
    ("I think the colours look fine overall.", None, None),
    ("I'll be honest, I do not love it.", None, None),
    ("Okay.", None, None),
])
def test_item_rules(text, kind, owner):
    m = MeetingInput("m", T0, [Utterance(0, "Priya Shah", text), Utterance(1, "Dana Kim", "fine by me then")])
    items = [i for i in extract_items(m, chunk_meeting(m)) if i.text == text]
    if kind is None:
        assert items == []
    else:
        assert len(items) == 1 and items[0].kind == kind and items[0].owner == owner
        assert items[0].chunk_id == "m#0000"


# ---- cards and text
def test_card_has_metadata_and_items():
    m = meeting([(0, "Priya", "We decided to ship the export feature."), (5, "Dana", "export export feature")],
                title="Weekly")
    items = extract_items(m, chunk_meeting(m))
    card = card_text(m, "A summary.", topics(m.utterances, 5), items, 5)
    assert "Meeting: Weekly" in card and "Date: Monday, November 2, 2026" in card
    assert "Attendees: Priya, Dana" in card and "Decisions: We decided" in card and "export" in card


@pytest.mark.parametrize("raw,clean", [
    ("um most people want it,. As um is uh the the.", "Most people want it, As is the."),
    ("Uh, I I think we we should go with it.", "I think we should go with it."),
    ("Mm-hmm.", ""),
])
def test_clean_speech(raw, clean):
    assert clean_speech(raw) == clean


def test_tokenize_stems_and_drops_fillers():
    assert tokenize("Um, the pricing was priced at 59 dollars") == ["price", "price", "59", "dollar"]
