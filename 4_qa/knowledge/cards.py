"""Meeting card: one short, searchable description of a whole meeting (stage 1 of architecture 2).

    Meeting: <title>
    Date: <weekday, month day, year>
    Attendees: <names>
    Topics: <top keywords>
    Summary: <our own summary>
    Decisions / Action items: <typed items>

The summary comes from a summarizer we pass in (default: the Feature 1 TextRank baseline). It is never a
human-written summary: in the AMI evaluation the questions ARE human summary sentences, so a human summary
in the card would leak the answer.
"""
import os
import sys
from collections import Counter

from text import clean_speech, content_words
from transcript import Utterance

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def topics(utterances, n):
    """Most frequent content words that come up in at least two different utterances."""
    tf, df = Counter(), Counter()
    for u in utterances:
        ws = content_words(u.text)
        tf.update(ws)
        df.update(set(ws))
    return [w for w, _ in sorted(((w, c) for w, c in tf.items() if df[w] >= 2), key=lambda x: (-x[1], x[0]))[:n]]


def first_words(utterances, budget):
    """Fallback summary: the opening lines up to the word budget (for meetings too short for TextRank)."""
    out, used = [], 0
    for u in utterances:
        if used >= budget:
            break
        out.append(u.text)
        used += len(u.text.split())
    return " ".join(out)


def textrank_summarizer(words):
    """The Feature 1 baseline (2_summarization/textrank_baseline.py) as a callable: utterances -> summary."""
    sys.path.insert(0, os.path.join(REPO, "2_summarization"))
    from textrank_baseline import TextRankSummarizer
    model = TextRankSummarizer(words=words)

    def summarize(utterances):
        # Summarize cleaned copies: TextRank only copies lines, so "um"s in the input end up in the card.
        utterances = [Utterance(u.start, u.speaker, clean_speech(u.text)) for u in utterances]
        utterances = [u for u in utterances if len(u.text.split()) >= 4] or utterances
        try:
            return model.summarize(utterances) or first_words(utterances, words)
        except (ValueError, ZeroDivisionError, IndexError):   # summa fails on very small inputs
            return first_words(utterances, words)
    summarize.version = model.version
    return summarize


def card_text(meeting, summary, topic_words, items, max_items):
    lines = [f"Meeting: {meeting.title or meeting.meeting_id}",
             f"Date: {meeting.started_at.strftime('%A, %B %-d, %Y')}",
             f"Attendees: {', '.join(meeting.attendees)}"]
    if topic_words:
        lines.append(f"Topics: {', '.join(topic_words)}")
    if summary:
        lines.append(f"Summary: {summary}")
    for kind, label in [("decision", "Decisions"), ("action", "Action items")]:
        chosen = [i for i in items if i.kind == kind][:max_items]
        if chosen:
            lines.append(f"{label}: " + " | ".join(i.text for i in chosen))
    return "\n".join(lines)
