"""Shared by every retriever: the result format (the retriever contract in docs/PROJECT.md section 4) and
reciprocal rank fusion.

    retrieve(question, k) -> [{"meeting_id", "chunk_id", "text", "score", "start", "end", "span"}, ...]
"""
from collections import defaultdict


def rrf(ranked_lists, k=60):
    """Reciprocal rank fusion. ranked_lists: [(weight, [id, ...best first]), ...] -> [(id, score)] best first.
    Ties are broken by id so results never depend on dict order."""
    scores = defaultdict(float)
    for weight, ids in ranked_lists:
        for rank, key in enumerate(ids, 1):
            scores[key] += weight / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def as_results(store, scored, k):
    """[(chunk_id, score)] -> the contract's result dicts, top k."""
    out = []
    for chunk_id, score in scored[:k]:
        c = store.chunk(chunk_id)
        out.append({"meeting_id": c.meeting_id, "chunk_id": c.chunk_id, "text": c.text, "score": round(score, 6),
                    "start": c.start, "end": c.end, "span": c.span})
    return out
