"""Index one finished meeting into the knowledge base (run when a meeting ends).

    meeting -> chunks (~60 s) -> typed items (decisions, actions) -> meeting card -> embed all -> store

Re-indexing the same meeting_id replaces it (one transaction in PgStore).
Each chunk is embedded WITH a short header (meeting title, date, who spoke), so a chunk that only says
"let's do 59 then" still carries what meeting it belongs to ("contextual" chunk embeddings).
"""
import time
from dataclasses import dataclass

import numpy as np

from cards import card_text, topics
from chunking import chunk_meeting
from items import extract_items
from store import MeetingRecord


@dataclass
class IndexReport:
    meeting_id: str
    chunks: int
    items: int
    seconds: float


def chunk_header(meeting, chunk):
    return (f"Meeting: {meeting.title or meeting.meeting_id} | {meeting.started_at.strftime('%B %-d, %Y')} | "
            f"Speakers: {', '.join(chunk.speakers)}")


def index_meeting(store, embedder, meeting, summarizer, chunking, card_cfg):
    t0 = time.perf_counter()
    if not meeting.utterances:
        raise ValueError(f"{meeting.meeting_id}: no utterances to index")
    chunks = chunk_meeting(meeting, **chunking)
    for c in chunks:
        c.embed_text = chunk_header(meeting, c) + "\n" + c.text
    items = extract_items(meeting, chunks)
    summary = summarizer(meeting.utterances) if summarizer else ""
    card = card_text(meeting, summary, topics(meeting.utterances, card_cfg["topics"]), items, card_cfg["max_items"])

    texts = [card] + [c.embed_text for c in chunks] + [i.embed_text() for i in items]
    vecs = embedder.passages(texts)
    card_vec, chunk_vecs, item_vecs = vecs[0], vecs[1:1 + len(chunks)], vecs[1 + len(chunks):]
    assert len(item_vecs) == len(items) and np.all(np.isfinite(vecs))

    record = MeetingRecord(meeting.meeting_id, meeting.title, meeting.started_at, list(meeting.attendees), card,
                           dict(meeting.source))
    store.put_meeting(record, chunks, items, card_vec, chunk_vecs, item_vecs)
    return IndexReport(meeting.meeting_id, len(chunks), len(items), round(time.perf_counter() - t0, 3))
