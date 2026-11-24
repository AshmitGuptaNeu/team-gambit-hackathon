"""BM25 (Okapi, rank_bm25) over chunks or meeting cards: the lexical baseline, and one half of hybrid search.

The index is built in memory from the store and rebuilt automatically when the store changes (store.version).
At this scale (thousands of chunks) a full rebuild is well under a second; Postgres' own full-text
ranking (ts_rank) is not BM25, which is why BM25 runs here.
"""
import numpy as np
from rank_bm25 import BM25Okapi

from retrieval_common import as_results
from text import tokenize


class BM25Index:
    def __init__(self, ids, owners, texts):
        self.ids, self.owners = list(ids), np.array(owners, dtype=object)
        tokens = [tokenize(t) for t in texts]
        # rank_bm25 divides by the corpus size; an empty corpus (or all-empty docs) gets no model.
        self.model = BM25Okapi(tokens) if tokens and any(tokens) else None

    def search(self, query, k, meeting_ids=None):
        q = tokenize(query)
        if self.model is None or not q or k <= 0:
            return []
        scores = self.model.get_scores(q)
        if meeting_ids is not None:
            scores = np.where(np.isin(self.owners, list(meeting_ids)), scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        # A score of 0 means no query word is in the document: not a match.
        return [(self.ids[j], float(scores[j])) for j in order if np.isfinite(scores[j]) and scores[j] > 0]


class BM25Cache:
    """One BM25Index per kind ("chunk" | "card"), rebuilt when the store's version changes."""

    def __init__(self, store):
        self.store, self._built = store, {}

    def index(self, kind):
        version = self.store.version
        if kind not in self._built or self._built[kind][0] != version:
            if kind == "chunk":
                rows = self.store.chunks()
                idx = BM25Index([c.chunk_id for c in rows], [c.meeting_id for c in rows], [c.embed_text for c in rows])
            elif kind == "card":
                rows = self.store.meetings()
                idx = BM25Index([m.meeting_id for m in rows], [m.meeting_id for m in rows], [m.card_text for m in rows])
            else:
                raise ValueError(kind)
            self._built[kind] = (version, idx)
        return self._built[kind][1]

    def search(self, kind, query, k, meeting_ids=None):
        return self.index(kind).search(query, k, meeting_ids)


class BM25Retriever:
    name = "bm25"

    def __init__(self, store):
        self.store, self.bm25 = store, BM25Cache(store)

    def retrieve(self, question, k=10, meeting_ids=None, **_):
        return as_results(self.store, self.bm25.search("chunk", question, k, meeting_ids), k)
