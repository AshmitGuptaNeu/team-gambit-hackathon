"""Both stores (in-memory and Postgres + pgvector) behave the same. Uses a fake embedder: no model needed.
The pg tests need LIVEMINUTES_TEST_DB_URL (a throwaway database: tables are dropped)."""
import pytest

import qa_settings as S
from bm25 import BM25Retriever
from indexer import index_meeting
from store import MemoryStore, PgStore
from two_stage import TwoStageRetriever

QUERIES = ["starter plan price", "Stripe webhook bug", "dark mode", "on-call rotation", "CSV export timeout"]


def fill(store, emb, meetings):
    for m in meetings:
        index_meeting(store, emb, m, None, S.CHUNKING, S.CARD)
    return store


def same_ranking(a, b):
    """Equal ids in equal order, except where neighbouring scores tie to float32 precision."""
    assert len(a) == len(b)
    for (ia, sa), (ib, sb) in zip(a, b):
        assert sa == pytest.approx(sb, abs=1e-5)
        if ia != ib:
            assert any(x == ib and s == pytest.approx(sa, abs=1e-5) for x, s in a)


@pytest.fixture
def mem(scenarios, fake_embedder):
    return fill(MemoryStore(), fake_embedder, scenarios["meetings"])


@pytest.fixture
def pg(pg_url, scenarios, fake_embedder):
    PgStore.drop(pg_url)
    store = fill(PgStore(pg_url, fake_embedder.dim), fake_embedder, scenarios["meetings"])
    yield store
    store.drop_all()


def check_store_basics(store, scenarios, emb):
    n = len(store.chunks())
    assert len(store.meetings()) == 7 and n > 7
    assert [m.meeting_id for m in store.meetings()][0] == "atlas-kickoff"            # sorted by date
    v0 = store.version
    fill(store, emb, scenarios["meetings"][:1])                                     # re-index: replaced
    assert len(store.chunks()) == n and store.version > v0
    assert store.delete_meeting("atlas-kickoff") and not store.delete_meeting("atlas-kickoff")
    assert not store.chunks(["atlas-kickoff"]) and not store.items(["atlas-kickoff"])
    assert all(i.chunk_id in {c.chunk_id for c in store.chunks()} for i in store.items())   # items -> chunks
    hits = store.search("chunk", emb.query("dark mode"), 5, ["design-review"])
    assert hits and all(h[0].startswith("design-review#") for h in hits)
    assert store.search("chunk", emb.query("x"), 5, []) == [] and store.search("chunk", emb.query("x"), 0) == []
    with pytest.raises(KeyError):
        store.chunk("nope#0000")


def test_memory_store_basics(mem, scenarios, fake_embedder):
    check_store_basics(mem, scenarios, fake_embedder)


def test_memory_store_save_load(mem, tmp_path, fake_embedder):
    mem.save(str(tmp_path))
    loaded = MemoryStore.load(str(tmp_path))
    assert [c.chunk_id for c in loaded.chunks()] == [c.chunk_id for c in mem.chunks()]
    for q in QUERIES:
        same_ranking(loaded.search("chunk", fake_embedder.query(q), 8), mem.search("chunk", fake_embedder.query(q), 8))
    assert MemoryStore.load(str(tmp_path / "empty")).meetings() == []


@pytest.mark.pg
def test_pg_store_basics(pg, scenarios, fake_embedder):
    check_store_basics(pg, scenarios, fake_embedder)


@pytest.mark.pg
def test_pg_matches_memory(pg, mem, fake_embedder):
    assert [c.chunk_id for c in pg.chunks()] == [c.chunk_id for c in mem.chunks()]
    assert [(i.item_id, i.owner) for i in pg.items()] == [(i.item_id, i.owner) for i in mem.items()]
    assert [m.card_text for m in pg.meetings()] == [m.card_text for m in mem.meetings()]
    for q in QUERIES:
        v = fake_embedder.query(q)
        for kind in ("chunk", "card", "item"):
            same_ranking(pg.search(kind, v, 8), mem.search(kind, v, 8))
        same_ranking(pg.search("chunk", v, 5, ["pricing-sync", "atlas-kickoff"]),
                     mem.search("chunk", v, 5, ["pricing-sync", "atlas-kickoff"]))
        assert BM25Retriever(pg).retrieve(q, 5) == BM25Retriever(mem).retrieve(q, 5)


@pytest.mark.pg
def test_pg_two_stage_matches_memory(pg, mem, fake_embedder, scenarios):
    a, b = TwoStageRetriever(pg, fake_embedder, S.RETRIEVAL), TwoStageRetriever(mem, fake_embedder, S.RETRIEVAL)
    for q in scenarios["questions"]:
        ra = a.retrieve(q["question"], 5, now=scenarios["now"], context=q.get("context"))
        rb = b.retrieve(q["question"], 5, now=scenarios["now"], context=q.get("context"))
        assert [r["chunk_id"] for r in ra] == [r["chunk_id"] for r in rb]


@pytest.mark.pg
def test_pg_rejects_wrong_dimension(pg, pg_url):
    with pytest.raises(ValueError):
        PgStore(pg_url, 384)
