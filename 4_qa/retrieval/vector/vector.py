"""Dense retrieval (bge-small embeddings, cosine) over chunks, and hybrid = BM25 + dense fused with RRF.

These are the "flat" retrievers (architecture 1): every chunk of every meeting competes in one list.
"""
from bm25 import BM25Cache
from retrieval_common import as_results, rrf


class VectorRetriever:
    name = "vector"

    def __init__(self, store, embedder):
        self.store, self.embedder = store, embedder

    def retrieve(self, question, k=10, meeting_ids=None, **_):
        return as_results(self.store, self.store.search("chunk", self.embedder.query(question), k, meeting_ids), k)


class HybridRetriever:
    name = "hybrid"

    def __init__(self, store, embedder, rrf_k=60, depth=50):
        self.store, self.embedder, self.rrf_k, self.depth = store, embedder, rrf_k, depth
        self.bm25 = BM25Cache(store)

    def ranked(self, question, meeting_ids=None, qvec=None):
        """Fused [(chunk_id, rrf score)] for one question."""
        qvec = self.embedder.query(question) if qvec is None else qvec
        lexical = [cid for cid, _ in self.bm25.search("chunk", question, self.depth, meeting_ids)]
        dense = [cid for cid, _ in self.store.search("chunk", qvec, self.depth, meeting_ids)]
        return rrf([(1.0, lexical), (1.0, dense)], self.rrf_k)

    def retrieve(self, question, k=10, meeting_ids=None, **_):
        return as_results(self.store, self.ranked(question, meeting_ids), k)
