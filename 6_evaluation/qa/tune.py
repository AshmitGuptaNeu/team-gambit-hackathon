"""Choose the two-stage weights on AMI DEV only (never eval). Prints the best settings; changes nothing.

    python 6_evaluation/qa/tune.py                 # whole dev corpus (20 meetings)
    python 6_evaluation/qa/tune.py --scope team    # each question searches only its team's 4 meetings

Grid: top_meetings x stage2_weight x card weight x item weight, global_weight fixed at 1.
stage2_weight = 0 and item_weight = 0 is exactly the plain hybrid retriever, so the grid shows whether
any two-stage boost beats it.
"""
import argparse
import copy
import functools
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402  (sets up every import path)
from two_stage import TwoStageRetriever  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["corpus", "team"], default="corpus")
    a = p.parse_args()
    S = R.S
    data_hash = R.fetch_data.fetch()
    dev = R.ami_split.load_split()["dev"]
    embedder = R.Embedder(S.EMBEDDING)
    store, _ = R.build_index(dev, embedder, data_hash)
    embedder.query = functools.lru_cache(maxsize=None)(embedder.query)   # same questions in every config
    by_meeting = {}
    for c in store.chunks(dev):
        by_meeting.setdefault(c.meeting_id, []).append(c)
    zip_path = os.path.join(R.ami_settings.RAW_DIR, R.fetch_data.load_manifest()["file"])
    queries = R.ami_qa.load_queries(dev, zip_path)
    gold = {q.query_id: R.ami_qa.gold_chunks(q, by_meeting) for q in queries}
    queries = [q for q in queries if gold[q.query_id]]

    def score(cfg):
        r, rows = TwoStageRetriever(store, embedder, cfg), []
        for q in queries:
            scope = [m for m in dev if m[:-1] == q.meeting_id[:-1]] if a.scope == "team" else None
            res = r.retrieve(q.text, k=R.K, meeting_ids=scope)
            rows.append(R.qa_metrics.score_query([x["chunk_id"] for x in res], gold[q.query_id],
                                                 [x["meeting_id"] for x in res], q.meeting_id))
        return R.qa_metrics.mean(rows)

    out = []
    for top, s2, card, item in itertools.product([1, 2, 3, 5], [0.0, 0.25, 0.5, 1.0], [0.0, 0.3], [0.0, 0.25, 0.5]):
        cfg = copy.deepcopy(S.RETRIEVAL)
        cfg.update(top_meetings=top, stage2_weight=s2, stage2_weight_latest=s2, global_weight=1.0, item_weight=item)
        cfg["meeting_weights"].update(card_vector=card, card_bm25=card)
        out.append(((top, s2, card, item), score(cfg)))
    out.sort(key=lambda x: -(x[1]["hit@5"] + x[1]["mrr"]))
    print(f"dev, scope {a.scope}, {len(queries)} questions. Best 8 of {len(out)} (ranked by hit@5 + MRR):")
    for (top, s2, card, item), m in out[:8]:
        print(f"  top_meetings={top} stage2={s2} card={card} item={item}:  hit@1 {m['hit@1']:.3f}  "
              f"hit@5 {m['hit@5']:.3f}  hit@10 {m['hit@10']:.3f}  MRR {m['mrr']:.3f}")
    plain = next(m for k, m in out if k[1] == 0 and k[3] == 0)
    print(f"  plain hybrid (stage2=0, item=0):  hit@5 {plain['hit@5']:.3f}  MRR {plain['mrr']:.3f}")


if __name__ == "__main__":
    main()
