"""Typed items: decisions and action items, each linked to the chunk it was said in.

This is the "knowledge graph" layer of architecture 2: (meeting) -[HAS]-> (item) -[SAID_IN]-> (chunk),
(item) -[OWNED_BY]-> (person). Stored as rows with foreign keys (see store.py), searchable like chunks.

Baseline extractor: rule-based phrase patterns (the planned Feature 2 baseline, docs/PROJECT.md section 3).
It is deliberately simple and high-precision-ish; Feature 2 can replace extract_items() with an LLM extractor
without changing anything downstream, as long as it returns the same Item objects.
"""
import re
from dataclasses import dataclass

DAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

DECISION_RE = re.compile(
    r"\b(?:we(?:'ve| have)? (?:decided|agreed|settled)"
    r"|(?:it's|it is|that's) (?:decided|settled|final)"
    r"|let's (?:go with|stick with|keep|use)"
    r"|we(?:'ll| will|'re going to| are going to| are| 're) go(?:ing)? with"
    r"|(?:final|the) decision (?:is|was)"
    r"|agreed on|go ahead with|we(?:'re| are) going with)\b", re.I)

ACTION_RE = re.compile(
    r"\b(?:i(?:'ll| will|'m going to| am going to) (?!be\b|say\b|just say\b|think\b|admit\b|tell you\b)[a-z]+"
    r"|(?:can|could|would) you (?:please )?(?:send|share|update|prepare|check|review|write|draft|look|follow|fix|set|book|make|put|take|get|find)"
    r"|please (?:send|share|update|prepare|check|review|write|draft|follow)"
    r"|action items?|to-?do|follow up"
    rf"|by (?:{DAYS}|tomorrow|next week|end of (?:the )?(?:day|week|month)|eod|eow))\b", re.I)


@dataclass
class Item:
    item_id: str
    meeting_id: str
    chunk_id: str
    kind: str            # "decision" | "action"
    text: str
    speaker: str
    owner: str | None    # who is on the hook (actions); None if unclear
    start: float

    def embed_text(self):
        who = f" (owner: {self.owner})" if self.owner else f" (said by {self.speaker})"
        return f"{self.kind.capitalize()}{who}: {self.text}"


def owner_of(text, speaker, attendees):
    """Someone named in the line ("Dana, can you send...") -> them; first person ("I'll") -> the speaker."""
    lowered = text.lower()
    for name in attendees:
        first = name.split()[0].lower()
        if name != speaker and len(first) > 1 and re.search(rf"\b{re.escape(first)}\b", lowered):
            return name
    if re.search(r"\bi(?:'ll| will|'m going to| am going to)\b", lowered):
        return speaker
    return None


def chunk_for(start, chunks):
    """The chunk whose span contains `start` (chunks are sorted by start)."""
    found = chunks[0]
    for c in chunks:
        if c.start <= start:
            found = c
        else:
            break
    return found


def extract_items(meeting, chunks):
    if not chunks:
        return []
    items, counts = [], {"decision": 0, "action": 0}
    for u in meeting.utterances:
        if len(u.text.split()) < 4:
            continue
        kind = "decision" if DECISION_RE.search(u.text) else "action" if ACTION_RE.search(u.text) else None
        if kind is None:
            continue
        counts[kind] += 1
        items.append(Item(
            item_id=f"{meeting.meeting_id}!{kind[0]}{counts[kind]:03d}", meeting_id=meeting.meeting_id,
            chunk_id=chunk_for(u.start, chunks).chunk_id, kind=kind, text=u.text[:400], speaker=u.speaker,
            owner=owner_of(u.text, u.speaker, meeting.attendees) if kind == "action" else None, start=u.start))
    return items
