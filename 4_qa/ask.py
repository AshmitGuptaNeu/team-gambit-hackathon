"""Ask the knowledge base a question; prints the best moments from past meetings, with citations.

    python 4_qa/ask.py "what did we decide about pricing last week?"
    python 4_qa/ask.py "who was going to fix that?" --context live_transcript.txt   # last lines of the live meeting
    python 4_qa/ask.py "..." --now 2026-11-18T15:00:00+00:00 -k 3 --debug

Retrieval only: the LLM that writes an answer from these moments is the next step (4_qa/answer/).
"""
import argparse
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
for folder in ["", "knowledge", "retrieval", "retrieval/bm25", "retrieval/vector", "retrieval/two_stage"]:
    sys.path.insert(0, os.path.join(HERE, folder))
import qa_settings as S  # noqa: E402
from embed import Embedder  # noqa: E402
from store import open_store  # noqa: E402
from two_stage import TwoStageRetriever  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--context", help="text file with the live transcript ([mm:ss] Name: text lines)")
    p.add_argument("--now", help="ISO time to read 'last week' etc. against (default: now)")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    store = open_store(S.DB_URL, S.EMBEDDING["dim"], S.MEMORY_STORE_DIR)
    if not store.meetings():
        sys.exit("the knowledge base is empty: index a meeting first (4_qa/index_meeting.py)")
    context = open(a.context, encoding="utf-8").read() if a.context else None
    now = datetime.fromisoformat(a.now) if a.now else datetime.now(timezone.utc).astimezone()
    r = TwoStageRetriever(store, Embedder(S.EMBEDDING), S.RETRIEVAL)
    titles = {m.meeting_id: (m.title or m.meeting_id, m.started_at) for m in store.meetings()}
    for n, hit in enumerate(r.retrieve(a.question, k=a.k, now=now, context=context), 1):
        title, when = titles[hit["meeting_id"]]
        print(f"\n{n}. {title} ({when:%b %-d, %Y}) at {hit['span']}   [score {hit['score']:.4f}]")
        print("   " + hit["text"].replace("\n", "\n   "))
    if a.debug:
        print("\ndebug:", r.last_debug)


if __name__ == "__main__":
    main()
