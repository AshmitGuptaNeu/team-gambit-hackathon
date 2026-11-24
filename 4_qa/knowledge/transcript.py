"""The data the knowledge base is built from: one meeting = metadata + time-ordered utterances.

Reads the shared interchange format written by ingestion (1_ingestion/session.py write_transcript):
    [mm:ss] Name: text
Minutes can go past 99 ("[125:03]") because session.mmss never wraps to hours. "[h:mm:ss]" is accepted too.
"""
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

LINE_RE = re.compile(r"^\[(?:(\d+):)?(\d+):([0-5]\d)\]\s*([^:]+?):\s?(.*)$")


@dataclass
class Utterance:
    start: float                 # seconds from the start of the meeting
    speaker: str
    text: str
    end: float | None = None     # seconds; None when the source has no end time
    uid: str | None = None       # source id (e.g. an AMI dialogue act id), used by the evaluation


@dataclass
class MeetingInput:
    meeting_id: str
    started_at: datetime         # timezone-aware
    utterances: list             # list[Utterance], sorted by start
    title: str = ""
    attendees: list = field(default_factory=list)   # names; defaults to everyone who spoke
    source: dict = field(default_factory=dict)      # where it came from (path, corpus, ...)

    def __post_init__(self):
        if self.started_at.tzinfo is None:
            raise ValueError(f"{self.meeting_id}: started_at must be timezone-aware")
        self.utterances = sorted(self.utterances, key=lambda u: u.start)
        if not self.attendees:
            self.attendees = speakers_of(self.utterances)


def speakers_of(utterances):
    """Distinct speakers in order of first appearance."""
    seen = {}
    for u in utterances:
        seen.setdefault(u.speaker, None)
    return list(seen)


def mmss(seconds):
    """Format like ingestion does: minutes never wrap into hours."""
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def parse_transcript(text):
    """Parse transcript.txt text into utterances. A line without a [mm:ss] prefix continues the previous
    utterance (defensive: ingestion writes one line per utterance today). Blank lines are skipped.
    Raises ValueError if text comes before the first timestamped line."""
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            h, mnt, sec, speaker, body = m.groups()
            start = int(h or 0) * 3600 + int(mnt) * 60 + int(sec)
            out.append(Utterance(float(start), speaker.strip(), body.strip()))
        elif out:
            out[-1].text = (out[-1].text + " " + line).strip()
        else:
            raise ValueError(f"line {n}: expected '[mm:ss] Name: text', got {line[:60]!r}")
    return [u for u in sorted(out, key=lambda u: u.start) if u.text]


def session_start(session_dir):
    """Wall-clock start of an ingestion session: the earliest piece start in pieces.jsonl (epoch ms).
    Falls back to transcript.txt's modification time if there is no piece log."""
    path = os.path.join(session_dir, "pieces.jsonl")
    starts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    starts.append(int(json.loads(line)["start_ms"]))
                except (ValueError, KeyError, TypeError):
                    continue
    if starts:
        return datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc)
    return datetime.fromtimestamp(os.path.getmtime(os.path.join(session_dir, "transcript.txt")), tz=timezone.utc)


def load_session(session_dir, title="", started_at=None):
    """Read one ingestion session folder (1_ingestion/sessions/<bot_id>/) as a MeetingInput."""
    session_dir = os.path.abspath(session_dir)
    with open(os.path.join(session_dir, "transcript.txt"), encoding="utf-8") as f:
        utterances = parse_transcript(f.read())
    bot_id = os.path.basename(session_dir.rstrip("/"))
    return MeetingInput(meeting_id=bot_id, started_at=started_at or session_start(session_dir),
                        utterances=utterances, title=title, source={"kind": "ingestion", "path": session_dir})
