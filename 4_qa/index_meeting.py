"""Add a finished meeting to the knowledge base (run when a meeting ends).

    python 4_qa/index_meeting.py 1_ingestion/sessions/<bot_id> --title "Pricing sync"
    python 4_qa/index_meeting.py <session> --date 2026-11-18T15:00:00+00:00   # override the start time

Store: Postgres + pgvector if LIVEMINUTES_DB_URL is set in the environment, else a local folder
(4_qa/kb_local/ or LIVEMINUTES_KB_DIR; kb_local is gitignored because it holds real meeting text).
"""
import argparse
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
for folder in ["", "knowledge"]:
    sys.path.insert(0, os.path.join(HERE, folder))
import qa_settings as S  # noqa: E402
from cards import textrank_summarizer  # noqa: E402
from embed import Embedder  # noqa: E402
from indexer import index_meeting  # noqa: E402
from store import MemoryStore, open_store  # noqa: E402
from transcript import load_session  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session", help="an ingestion session folder (contains transcript.txt)")
    p.add_argument("--title", default="")
    p.add_argument("--date", help="meeting start, ISO 8601 with timezone (default: from pieces.jsonl)")
    a = p.parse_args()
    started = datetime.fromisoformat(a.date) if a.date else None
    if started and started.tzinfo is None:
        sys.exit("--date needs a timezone, e.g. 2026-11-18T15:00:00+00:00")
    meeting = load_session(a.session, title=a.title, started_at=started)
    store = open_store(S.DB_URL, S.EMBEDDING["dim"], S.MEMORY_STORE_DIR)
    report = index_meeting(store, Embedder(S.EMBEDDING), meeting, textrank_summarizer(S.CARD["summary_words"]),
                           S.CHUNKING, S.CARD)
    if isinstance(store, MemoryStore):
        store.save(S.MEMORY_STORE_DIR)
    where = "Postgres" if S.DB_URL else S.MEMORY_STORE_DIR
    print(f"indexed {report.meeting_id} ({meeting.started_at:%Y-%m-%d %H:%M %Z}): {report.chunks} chunks, "
          f"{report.items} decisions/actions, {report.seconds}s -> {where}")


if __name__ == "__main__":
    main()
