"""Question understanding for live questions: dates, people and "last time", read with plain rules.

    parse_query("what did Priya decide about pricing last week?", now, people)
      -> text "what did Priya decide about pricing", date range = last Monday..Sunday, people = [Priya]

Dates are read in the timezone of `now` (the asker's clock). Supported:
  today, yesterday, this/last week, this/last month, in <Month>, (on|last) <weekday>,
  N days/weeks ago, in the last/past N days/weeks;  recency: last time, last/previous/latest meeting.
No LLM: this runs in well under a millisecond and is fully testable. An LLM rewrite can be added later.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "a": 1, "an": 1}
# Words in role-style names ("Project Manager"): such names are only matched in full, never by first word.
ROLE_WORDS = {"project", "manager", "marketing", "expert", "industrial", "designer", "user", "interface",
              "speaker", "speakers", "unknown", "multiple", "guest", "host"}

RECENCY_RE = re.compile(r"\b(?:the )?(?:last|previous|latest|most recent|prior) (?:time|meeting|call|sync|session|standup)\b"
                        r"|\blast time\b", re.I)


@dataclass
class ParsedQuery:
    text: str                       # the question with date/recency phrases removed (names kept)
    date_from: datetime | None = None
    date_to: datetime | None = None # exclusive
    people: list = field(default_factory=list)
    latest: bool = False            # "last time", "previous meeting": prefer the most recent meeting
    matched: list = field(default_factory=list)   # the phrases that were understood, for debugging


def day_start(d, tz):
    return datetime.combine(d, time.min, tzinfo=tz)


def date_range(q, now):
    """(start, end, phrase) for the first date phrase found, else None. end is exclusive."""
    tz, today = now.tzinfo, now.date()
    monday = today - timedelta(days=today.weekday())
    rules = [
        (r"\btoday\b", lambda m: (today, today + timedelta(1))),
        (r"\byesterday\b", lambda m: (today - timedelta(1), today)),
        (r"\bthis week\b", lambda m: (monday, today + timedelta(1))),
        (r"\blast week\b", lambda m: (monday - timedelta(7), monday)),
        (r"\b(?:in |over )?the (?:last|past) (\d+|one|two|three|four|five|six|seven|a) (day|week)s?\b",
         lambda m: (today - timedelta(n_of(m.group(1)) * (7 if m.group(2) == "week" else 1)), today + timedelta(1))),
        (r"\b(\d+|one|two|three|four|five|six|seven|a|an) (day|week)s? ago\b", lambda m: ago(m, today)),
        (r"\bthis month\b", lambda m: (today.replace(day=1), today + timedelta(1))),
        (r"\blast month\b", lambda m: last_month(today)),
        (rf"\b(?:on |last |this past )?({'|'.join(DAYS)})\b", lambda m: weekday(m.group(1), today)),
        (rf"\b(?:in|during|back in) ({'|'.join(MONTHS)})\b", lambda m: month(m.group(1), today)),
    ]
    for pattern, fn in rules:
        m = re.search(pattern, q, re.I)
        if m:
            start, end = fn(m)
            return day_start(start, tz), day_start(end, tz), m.group(0)
    return None


def n_of(word):
    return int(word) if word.isdigit() else NUM[word.lower()]


def ago(m, today):
    n, unit = n_of(m.group(1)), m.group(2).lower()
    if unit == "day":
        d = today - timedelta(n)
        return d, d + timedelta(1)
    monday = today - timedelta(days=today.weekday()) - timedelta(7 * n)
    return monday, monday + timedelta(7)


def last_month(today):
    first = today.replace(day=1)
    prev = (first - timedelta(1)).replace(day=1)
    return prev, first


def weekday(name, today):
    """The most recent such day strictly before today ("on Monday" asked on a Wednesday = 2 days ago)."""
    back = (today.weekday() - DAYS.index(name.lower())) % 7 or 7
    d = today - timedelta(back)
    return d, d + timedelta(1)


def month(name, today):
    """The most recent such month that is not in the future (asking in November about "June" = this June)."""
    m = MONTHS.index(name.lower()) + 1
    year = today.year if m <= today.month else today.year - 1
    start = today.replace(year=year, month=m, day=1)
    end = start.replace(year=year + (m == 12), month=m % 12 + 1)
    return start, end


def name_patterns(people):
    """{person: regex}. Full name always; the first word too when it is distinctive (not a role word, unique)."""
    firsts = {}
    for p in people:
        firsts.setdefault(p.split()[0].lower(), []).append(p)
    out = {}
    for p in people:
        alts = [re.escape(p)]
        first = p.split()[0]
        words = {w.lower() for w in p.split()}
        if len(p.split()) > 1 and len(first) > 2 and not (words & ROLE_WORDS) and len(firsts[first.lower()]) == 1:
            alts.append(re.escape(first))
        out[p] = re.compile(rf"\b(?:{'|'.join(alts)})(?:'s)?\b", re.I)
    return out


def parse_query(question, now, people=()):
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    parsed = ParsedQuery(text=question)
    text = question
    found = date_range(text, now)
    if found:
        parsed.date_from, parsed.date_to, phrase = found
        parsed.matched.append(phrase)
        text = text.replace(phrase, " ")
    m = RECENCY_RE.search(text)
    if m:
        parsed.latest = True
        parsed.matched.append(m.group(0))
        text = text.replace(m.group(0), " ")
    for person, pattern in name_patterns(people).items():
        if pattern.search(question):
            parsed.people.append(person)
    parsed.text = re.sub(r"\s+([?.!,])", r"\1", " ".join(text.split())) or question
    return parsed
