"""Retrieval metrics for Feature 3, per query, then averaged. Binary relevance: a chunk is gold or not.

  hit@k      1 if any gold chunk is in the top k (a.k.a. success@k; what a user feels: "was it on screen?")
  recall@k   fraction of the query's gold chunks found in the top k
  mrr        1 / rank of the first gold chunk (0 if none in the list)
  ndcg@k     DCG of the top k / best possible DCG (gains 1 for gold, log2 discount)
  meeting@1  1 if the top chunk is from the right meeting
Named qa_metrics so it never clashes with 6_evaluation/summarization/metrics.py.
"""
import math


def score_query(ranked_ids, gold, ranked_meetings, gold_meeting, ks=(1, 3, 5, 10)):
    gold = set(gold)
    if not gold:
        raise ValueError("a query needs at least one gold chunk")
    first = next((r for r, cid in enumerate(ranked_ids, 1) if cid in gold), None)
    out = {}
    for k in ks:
        top = ranked_ids[:k]
        out[f"hit@{k}"] = float(any(c in gold for c in top))
        out[f"recall@{k}"] = len(gold & set(top)) / len(gold)
    k = max(ks)
    dcg = sum(1 / math.log2(r + 1) for r, cid in enumerate(ranked_ids[:k], 1) if cid in gold)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    out[f"ndcg@{k}"] = dcg / ideal
    out["mrr"] = 1 / first if first else 0.0
    out["meeting@1"] = float(bool(ranked_meetings) and ranked_meetings[0] == gold_meeting)
    return out


def mean(rows):
    if not rows:
        return {}
    return {key: round(sum(r[key] for r in rows) / len(rows), 4) for key in rows[0]}


def percentile(values, p):
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))]
