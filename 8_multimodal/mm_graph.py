"""A small knowledge graph over the captions: who said what, what was decided, who owns which task, what is due.

The same layer as the LiveMinutes knowledge base (4_qa/knowledge/items.py, "architecture 2"):
    (meeting) -[HAS]-> (item) -[SAID_IN]-> (moment)      (item) -[OWNED_BY]-> (person)
    (item) -[SAID_BY]-> (person)                         (item) -[DUE]-> (date phrase)
Items are found by LiveMinutes' phrase rules (decisions, action items) plus one rule added here for lectures
(deadlines: "homework 5 is due Friday"). No model call, so it's instant and can't make anything up: every
item is a real caption line, and an answer from the graph quotes the line and its time.

    build(captions_text) -> Graph
    answer(question, graph) -> dict | None   (None: the graph has nothing on this, so ask the captions instead)
    to_mermaid(graph) -> Mermaid flowchart code
"""
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for folder in ["4_qa/knowledge"]:
    sys.path.insert(0, os.path.join(REPO, folder))
from chunking import chunk_meeting  # noqa: E402
from items import DAYS, extract_items  # noqa: E402
from text import tokenize  # noqa: E402
from transcript import MeetingInput, Utterance, mmss  # noqa: E402

UNKNOWN = "Speaker"   # mic captions have no speaker name
LINE_RE = re.compile(r"^\[(?:(\d+):)?(\d+):([0-5]\d)\]\s*(?:([A-Z][^:\[\]]{0,40}?):\s)?(.*)$")
WHEN = (rf"(?:(?:on |by |this |next )?{DAYS}|tomorrow|tonight|next week|end of (?:the )?(?:day|week|month)"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?"
        r"|\d{1,2}/\d{1,2}|in (?:\d+|one|two|three|a) (?:days?|weeks?))")
WHEN_RE = re.compile(rf"\b{WHEN}\b", re.I)
DEADLINE_RE = re.compile(r"\b(?:(?:is|are|'s|was) due|due (?:on|by|date)|deadline|submit\b.*\bby\b|turn (?:it |them )?in\b)",
                         re.I)
# "Sam, can you ..." / "Sam will ..." / "Sam's going to ..." -> Sam (names in the captions, not only speakers)
NAMED_RE = re.compile(r"\b([A-Z][a-z]{1,15}),? (?:can you|could you|will|'ll|is going to|'s going to|please|you take|takes)\b")
NOT_NAMES = {"I", "We", "It", "That", "This", "The", "You", "They", "He", "She", "Everyone", "Everybody", "Someone",
             "Okay", "Ok", "So", "And", "But", "Then", "Next", "Also", "Who", "What", "Which", "Professor", "Today"}


@dataclass
class Node:
    kind: str           # "decision" | "action" | "deadline"
    text: str           # the caption line, as said
    at: str             # "mm:ss"
    speaker: str
    owner: str | None = None
    when: str | None = None


@dataclass
class Graph:
    items: list = field(default_factory=list)       # list[Node], in caption order
    people: list = field(default_factory=list)      # everyone who spoke or owns something

    def edges(self):
        """(source, relation, target) triples, for display and tests."""
        out = []
        for i, n in enumerate(self.items):
            item = f"{n.kind}:{i}"
            out.append(("meeting", "HAS", item))
            out.append((item, "SAID_AT", n.at))
            if n.speaker != UNKNOWN:
                out.append((item, "SAID_BY", n.speaker))
            if n.owner:
                out.append((item, "OWNED_BY", n.owner))
            if n.when:
                out.append((item, "DUE", n.when))
        return out


def parse(captions):
    out = []
    for line in captions.splitlines():
        m = LINE_RE.match(line.strip())
        if not m or not m.group(5).strip():
            continue
        h, mi, s, who, text = m.groups()
        out.append(Utterance(start=int(h or 0) * 3600 + int(mi) * 60 + int(s), speaker=(who or UNKNOWN).strip(),
                             text=text.strip()))
    return out


def named_owner(text):
    for m in NAMED_RE.finditer(text):
        if m.group(1) not in NOT_NAMES:
            return m.group(1)
    return None


def build(captions):
    utts = parse(captions)
    if not utts:
        return Graph()
    meeting = MeetingInput("captions", datetime.now(timezone.utc), utts)
    found = {(round(i.start), i.text): i for i in extract_items(meeting, chunk_meeting(meeting))}
    nodes = []
    for u in meeting.utterances:
        item = found.get((round(u.start), u.text[:400]))
        deadline = DEADLINE_RE.search(u.text) and len(u.text.split()) >= 4
        if not item and not deadline:
            continue
        kind = item.kind if item else "deadline"
        if kind != "decision" and deadline:
            kind = "deadline"
        owner = named_owner(u.text) or (item.owner if item else None)
        if owner == UNKNOWN:
            owner = None
        when = WHEN_RE.search(u.text)
        nodes.append(Node(kind, u.text, mmss(u.start), u.speaker, owner if kind == "action" else None,
                          when.group(0) if when else None))
    people = list(dict.fromkeys([u.speaker for u in utts if u.speaker != UNKNOWN] + [n.owner for n in nodes if n.owner]))
    return Graph(nodes, people)


# ---------------- questions the graph can answer ----------------

INTENTS = {
    "action": re.compile(r"\b(?:who(?:'s| is)? (?:doing|owns?|handling|responsible|in charge|taking)|who will|"
                         r"tasks?|action items?|to-?dos?|assigned|doing what|what (?:is|does|do|should) \w+ (?:do|doing|need)|"
                         r"(?:my|his|her|their) (?:job|part)|owns?)\b", re.I),
    "decision": re.compile(r"\b(?:decid\w*|decision\w*|agree\w*|settled|chose|choose|picked|go(?:ing)? with)\b", re.I),
    "deadline": re.compile(r"\b(?:due|deadline\w*|when (?:is|are|do|does|should)\b.*\b(?:submit|turn|hand)|submit\w*)\b", re.I),
}
FILLER = set(tokenize("who what when which is are was were the a an do does did doing own owns owner task tasks "
                      "action item items todo to-do assigned decide decided decision decisions agree agreed due "
                      "deadline deadlines submit we they he she it of for on in by to and or our us our their my "
                      "handling responsible charge taking will should need needs job part go going with settled "
                      "chose choose picked lecture meeting class today so far list all any anything everything"))


def answer(question, graph):
    """An answer straight from the graph, or None when it has nothing to say (then the captions are searched)."""
    if not graph.items:
        return None
    kinds = [k for k, rx in INTENTS.items() if rx.search(question)]
    if not kinds:
        return None
    hits = [n for n in graph.items if n.kind in kinds]
    person = next((p for p in graph.people if re.search(rf"\b{re.escape(p.split()[0])}\b", question, re.I)), None)
    if person:
        hits = [n for n in hits if person in (n.owner, n.speaker)]
    topic = set(tokenize(question)) - FILLER - (set(tokenize(person)) if person else set())
    if topic:
        hits = [n for n in hits if topic & set(tokenize(n.text))]
    if not hits:
        return None
    label = {"action": "Task", "decision": "Decision", "deadline": "Due"}
    lines = []
    for n in hits:
        extra = [f"owner: {n.owner}"] if n.owner else []
        extra += [f"when: {n.when}"] if n.when else []
        lines.append(f"**{label[n.kind]}** [{n.at}]: {n.text}" + (f" ({', '.join(extra)})" if extra else ""))
    return {"found": True, "directness": "direct", "answer": "\n".join(lines),
            "citations": [{"quote": f"[{n.at}] {n.speaker + ': ' if n.speaker != UNKNOWN else ''}{n.text}"} for n in hits]}


# ---------------- drawing it ----------------

def _label(text, limit=60):
    t = re.sub(r"\s+", " ", text).strip()
    t = t if len(t) <= limit else t[:limit - 1].rstrip() + "…"
    return t.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


def to_mermaid(graph):
    lines = ["flowchart LR"]
    if not graph.items:
        return "\n".join(lines + ['  empty["No decisions, tasks or deadlines in the captions yet"]'])
    pid = {p: f"p{i}" for i, p in enumerate(graph.people)}
    for p, i in pid.items():
        lines.append(f'  {i}(["{_label(p, 30)}"])')
    titles = {"decision": "Decisions", "action": "Tasks", "deadline": "Deadlines"}
    for kind, title in titles.items():
        group = [(i, n) for i, n in enumerate(graph.items) if n.kind == kind]
        if not group:
            continue
        lines.append(f'  subgraph {kind}s["{title}"]')
        for i, n in group:
            lines.append(f'    {kind[0]}{i}["{n.at} · {_label(n.text)}"]')
        lines.append("  end")
    for i, n in enumerate(graph.items):
        node = f"{n.kind[0]}{i}"
        if n.owner and n.owner in pid:
            lines.append(f"  {pid[n.owner]} -->|owns| {node}")
        elif n.speaker in pid:
            lines.append(f"  {pid[n.speaker]} -.->|said| {node}")
        if n.when:
            lines.append(f'  {node} --> w{i}{{{{"{_label(n.when, 25)}"}}}}')
    lines += ["  classDef decision fill:#e8f0fe,stroke:#4a6fd1", "  classDef action fill:#e6f4ea,stroke:#3a8f5a",
              "  classDef deadline fill:#fdecea,stroke:#c0503f"]
    for kind in titles:
        ids = [f"{kind[0]}{i}" for i, n in enumerate(graph.items) if n.kind == kind]
        if ids:
            lines.append(f"  class {','.join(ids)} {kind}")
    return "\n".join(lines)
