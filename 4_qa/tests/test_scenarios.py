"""End-to-end: index the 7 scenario meetings with the real embedding model, ask the 16 labelled questions.

Two-stage must put a gold chunk in its top 3 for every question. The flat hybrid retriever is scored on the
same questions so the test also shows what stage 1 (dates, people, recency, live context) adds.
Run with -s to see the per-question table.
"""
import pytest

import qa_settings as S
from cards import textrank_summarizer
from chunking import Chunk
from indexer import index_meeting
from store import MemoryStore, PgStore
from transcript import mmss
from two_stage import TwoStageRetriever
from vector import HybridRetriever

pytestmark = pytest.mark.model
TOP = 3


def gold_chunks(store, q):
    """Chunks of the gold meeting that contain a gold utterance time."""
    out = []
    for c in store.chunks([q["gold_meeting"]]):
        times = {line[1:line.index("]")] for line in c.text.splitlines()}
        if times & set(q["gold_times"]):
            out.append(c.chunk_id)
    assert out, f"{q['id']}: gold times {q['gold_times']} not found"
    return out


@pytest.fixture(scope="module", params=["memory", "pgvector"])
def indexed(request, scenarios, embedder):
    if request.param == "memory":
        store = MemoryStore()
    else:
        url = request.getfixturevalue("pg_url")
        PgStore.drop(url)
        store = PgStore(url, embedder.dim)
    summarize = textrank_summarizer(S.CARD["summary_words"])
    for m in scenarios["meetings"]:
        index_meeting(store, embedder, m, summarize, S.CHUNKING, S.CARD)
    yield store
    if request.param == "pgvector":
        store.drop_all()


def ask(retriever, q, now):
    return retriever.retrieve(q["question"], k=10, now=now, context=q.get("context"))


def test_two_stage_answers_every_question(indexed, embedder, scenarios, capsys):
    two = TwoStageRetriever(indexed, embedder, S.RETRIEVAL)
    flat = HybridRetriever(indexed, embedder, S.RETRIEVAL["rrf_k"], S.RETRIEVAL["depth"])
    rows, failures = [], []
    for q in scenarios["questions"]:
        gold = gold_chunks(indexed, q)
        rank = lambda res: next((i for i, r in enumerate(res, 1) if r["chunk_id"] in gold), None)  # noqa: E731
        r2, rf = rank(ask(two, q, scenarios["now"])), rank(flat.retrieve(q["question"], k=10))
        rows.append((q["category"], q["id"], rf, r2, two.last_debug))
        if r2 is None or r2 > TOP:
            failures.append(f"{q['id']}: gold rank {r2}, debug {two.last_debug}")
    with capsys.disabled():
        print(f"\n{'category':9} {'question':28} flat  two-stage  stage-1 meetings")
        for cat, qid, rf, r2, dbg in rows:
            print(f"{cat:9} {qid:28} {str(rf):>4}  {str(r2):>9}  {dbg['meetings'][:3]}")
        hits = lambda i: sum(1 for r in rows if r[i] is not None and r[i] <= TOP)  # noqa: E731
        print(f"gold in top {TOP}: flat hybrid {hits(2)}/{len(rows)}, two-stage {hits(3)}/{len(rows)}")
    assert not failures, "\n".join(failures)
    assert sum(1 for r in rows if r[3] and r[3] <= TOP) >= sum(1 for r in rows if r[2] and r[2] <= TOP)


def test_filters_and_debug(indexed, embedder, scenarios):
    two = TwoStageRetriever(indexed, embedder, S.RETRIEVAL)
    res = two.retrieve("What did we decide yesterday about monitoring?", now=scenarios["now"])
    assert two.last_debug["candidates"] == ["ops-weekly-1117"] and all(r["meeting_id"] == "ops-weekly-1117" for r in res)
    two.retrieve("What did Jordan say about monitoring?", now=scenarios["now"])
    assert set(two.last_debug["candidates"]) == {"ops-weekly-1104", "ops-weekly-1117"}
    # nothing in range -> filter dropped (noted), still answers
    res = two.retrieve("What did we decide about pricing in June?", now=scenarios["now"])
    assert res and any("date filter dropped" in n for n in two.last_debug["notes"])
    # exclude the live meeting, restrict to a team's meetings
    res = two.retrieve("monitoring vendor", meeting_ids=["ops-weekly-1104", "ops-weekly-1117"],
                       exclude_meeting_ids=["ops-weekly-1117"], now=scenarios["now"])
    assert res and {r["meeting_id"] for r in res} == {"ops-weekly-1104"}


def test_neighbors_and_result_contract(indexed, embedder, scenarios):
    two = TwoStageRetriever(indexed, embedder, S.RETRIEVAL)
    res = two.retrieve("What price did we agree on in the pricing sync?", k=3, now=scenarios["now"], with_neighbors=True)
    assert len(res) == 3
    for r in res:
        assert {"meeting_id", "chunk_id", "text", "score", "start", "end", "span", "before", "after"} <= set(r)
        assert r["span"] == f"{mmss(r['start'])}-{mmss(r['end'])}"
    first = indexed.chunk(res[0]["chunk_id"])
    assert isinstance(first, Chunk)
    if first.ord == 0:
        assert res[0]["before"] is None


def test_empty_store(embedder):
    assert TwoStageRetriever(MemoryStore(), embedder, S.RETRIEVAL).retrieve("anything") == []
