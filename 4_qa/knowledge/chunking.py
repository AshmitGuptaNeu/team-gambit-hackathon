"""Cut a meeting into chunks of about one minute: the unit that is retrieved, cited and shown.

Each chunk keeps its exact transcript lines ("[mm:ss] Name: text"), so the answer can quote and cite it.
Rules, in order:
  1. a new chunk starts once the current one spans >= window_seconds (an utterance is never split);
  2. a chunk with more than max_words words is closed early (long monologues);
  3. a last chunk under min_words words is merged into the one before.
"""
from dataclasses import dataclass, field

from transcript import mmss, speakers_of


@dataclass
class Chunk:
    chunk_id: str        # "<meeting_id>#0007"
    meeting_id: str
    ord: int             # position in the meeting, from 0
    start: float         # seconds
    end: float           # seconds (start of the next utterance, or the last utterance's end/start)
    speakers: list
    text: str            # transcript lines, one per utterance
    utterance_uids: list = field(default_factory=list)
    embed_text: str = "" # what gets embedded and BM25-indexed: header + text (set by the indexer)

    @property
    def span(self):
        return f"{mmss(self.start)}-{mmss(self.end)}"


def n_words(utterance):
    return len(utterance.text.split())


def group(utterances, window_seconds, max_words, min_words):
    groups, cur, cur_words = [], [], 0
    for u in utterances:
        if cur and (u.start - cur[0].start >= window_seconds or cur_words + n_words(u) > max_words):
            groups.append(cur)
            cur, cur_words = [], 0
        cur.append(u)
        cur_words += n_words(u)
    if cur:
        if groups and cur_words < min_words:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def chunk_meeting(meeting, window_seconds=60, max_words=220, min_words=25):
    """Chunks for one MeetingInput, in meeting order. Empty meeting -> []."""
    groups = group(meeting.utterances, window_seconds, max_words, min_words)
    chunks = []
    for i, g in enumerate(groups):
        nxt = groups[i + 1][0].start if i + 1 < len(groups) else None
        last = g[-1]
        end = nxt if nxt is not None else (last.end if last.end is not None else last.start)
        chunks.append(Chunk(
            chunk_id=f"{meeting.meeting_id}#{i:04d}", meeting_id=meeting.meeting_id, ord=i,
            start=g[0].start, end=max(end, g[0].start), speakers=speakers_of(g),
            text="\n".join(f"[{mmss(u.start)}] {u.speaker}: {u.text}" for u in g),
            utterance_uids=[u.uid for u in g if u.uid is not None]))
    return chunks
