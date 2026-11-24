"""AMI retrieval benchmark: queries with known answer locations, built from AMI's own annotations.

AMI's human summaries (abstractive/<m>.abssumm.xml) have four parts: abstract, decisions, actions, problems.
AMI's summary links (extractive/<m>.summlink.xml) connect every summary sentence to the dialogue acts it was
written from. So each linked summary sentence is a query, and its gold answer = the chunk(s) containing
those dialogue acts. Example:
    query: "The team decided to use a kinetic battery."  (decisions)
    gold:  the ~60 s chunk(s) of ES2004c where that was said

The summlink files are read straight from the AMI zip, whose SHA-256 fetch_data.fetch() has just checked
against MANIFEST.json. They are not extracted, so the extracted files and their data_hash stay exactly as
Feature 1 uses them.

Caveat (documented in docs/QA.md): queries are declarative summary sentences, not questions, written by
annotators after the meeting. They are the best freely available gold for "find where this was discussed".
"""
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import ami
import ami_settings
from transcript import MeetingInput, Utterance

NITE = "{http://nite.sourceforge.net/}"
SECTIONS = ("abstract", "decisions", "actions", "problems")


@dataclass
class Query:
    query_id: str        # the summary sentence id
    meeting_id: str
    section: str         # abstract | decisions | actions | problems
    text: str
    gold_uids: list      # dialogue act ids the sentence was written from


def read_dacts(meeting, agent, role):
    """One speaker's dialogue acts as Utterances, keeping AMI's dialogue act id in uid."""
    path = os.path.join(ami_settings.AMI_DIR, "dialogueActs", f"{meeting}.{agent}.dialog-act.xml")
    if not os.path.exists(path):
        return []
    words = ami.read_words(meeting, agent)
    word_ids = list(words)
    position = {w: i for i, w in enumerate(word_ids)}
    out = []
    for dact in ET.parse(path).getroot().iter("dact"):
        child = dact.find(NITE + "child")
        if child is None:
            continue
        first, last = ami.word_range(child.get("href"))
        if first not in position or last not in position:
            continue
        span = word_ids[position[first]: position[last] + 1]
        text = ami.join_words([(words[w][1], words[w][2]) for w in span])
        times = [words[w][0] for w in span if words[w][0] is not None]
        if text and times:
            out.append(Utterance(start=min(times), speaker=ami.ROLE_NAMES.get(role, role), text=text,
                                 uid=dact.get(NITE + "id")))
    return out


def load_meeting(meeting_id, started_at):
    utterances = []
    for agent, role in ami.speaker_roles(meeting_id).items():
        utterances += read_dacts(meeting_id, agent, role)
    return MeetingInput(meeting_id=meeting_id, started_at=started_at, utterances=utterances, title="",
                        source={"kind": "ami", "meeting": meeting_id})


def series_dates(meeting_ids):
    """AMI has no dates. Give each series (ES2004) a week, and meetings a..d consecutive days, so the date
    fields are filled in realistically. No AMI query mentions a date, so these never affect the scores."""
    series = sorted({m[:-1] for m in meeting_ids})
    base = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    return {m: base + timedelta(weeks=series.index(m[:-1]), days="abcd".index(m[-1])) for m in meeting_ids}


def ids_in(href):
    """'X.xml#id(a)' -> ['a'];  'X.xml#id(a)..id(b)' -> ['a', 'b'] (endpoints; ranges are not used by AMI 1.6.2)."""
    return [part.split(")")[0] for part in href.split("id(")[1:]]


def summary_sentences(meeting_id):
    """{sentence id: (section, text)} from the human summary."""
    path = os.path.join(ami_settings.AMI_DIR, "abstractive", f"{meeting_id}.abssumm.xml")
    if not os.path.exists(path):
        return {}
    out = {}
    for section in SECTIONS:
        for part in ET.parse(path).getroot().iter(section):
            for s in part.iter("sentence"):
                if s.text and s.text.strip():
                    out[s.get(NITE + "id")] = (section, s.text.strip())
    return out


def load_queries(meeting_ids, zip_path):
    """Every summary sentence with at least one link to a dialogue act, for the given meetings."""
    queries = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for m in meeting_ids:
            name = f"extractive/{m}.summlink.xml"
            if name not in names:
                continue
            sentences = summary_sentences(m)
            links = {}
            for link in ET.fromstring(z.read(name)).iter("summlink"):
                ptr = {p.get("role"): p.get("href") for p in link.iter(NITE + "pointer")}
                if "extractive" in ptr and "abstractive" in ptr:
                    for sid in ids_in(ptr["abstractive"]):
                        links.setdefault(sid, []).extend(ids_in(ptr["extractive"]))
            for sid, uids in sorted(links.items()):
                if sid in sentences:
                    section, text = sentences[sid]
                    queries.append(Query(sid, m, section, text, sorted(set(uids))))
    return queries


def gold_chunks(query, chunks_by_meeting):
    """Chunk ids (in the query's meeting) that contain at least one of the query's dialogue acts."""
    wanted = set(query.gold_uids)
    return [c.chunk_id for c in chunks_by_meeting.get(query.meeting_id, []) if wanted & set(c.utterance_uids)]
