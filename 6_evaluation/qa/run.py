"""Feature 3 retrieval benchmark on AMI: index meetings, ask every linked summary sentence, score, log.

    python 6_evaluation/qa/run.py --split dev                          # tune here (20 dev meetings)
    python 6_evaluation/qa/run.py --split eval                         # report here (20 test meetings)
    python 6_evaluation/qa/run.py --split eval --corpus all            # same queries, all 138 meetings indexed
    python 6_evaluation/qa/run.py --split smoke --retrievers bm25      # 3 meetings, quick check

Steps: 1. check data + split   2. index the corpus (cached in 5_data/processed/qa_index/, gitignored)
       3. load queries + gold chunks   4. run each retriever, score every query   5. log

Leakage rules: the queries are human summary sentences, so nothing human-written goes into the index:
meeting cards use OUR TextRank summary. Nothing here is trained; --split dev is for choosing settings,
--split eval only for the final numbers. --corpus all puts train/dev meetings in as distractors only;
queries always come from the chosen split.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for folder in ["4_qa", "4_qa/knowledge", "4_qa/retrieval", "4_qa/retrieval/bm25", "4_qa/retrieval/vector",
               "4_qa/retrieval/two_stage", "5_data/ami", "6_evaluation/qa"]:
    sys.path.insert(0, os.path.join(REPO, folder))
import ami_qa  # noqa: E402
import ami_settings  # noqa: E402
import fetch_data  # noqa: E402
import qa_metrics  # noqa: E402
import qa_settings as S  # noqa: E402
import split as ami_split  # noqa: E402
from bm25 import BM25Retriever  # noqa: E402
from cards import textrank_summarizer  # noqa: E402
from embed import Embedder  # noqa: E402
from indexer import index_meeting  # noqa: E402
from store import MemoryStore  # noqa: E402
from vector import HybridRetriever, VectorRetriever  # noqa: E402

CACHE_DIR = os.path.join(REPO, "5_data", "processed", "qa_index")
K = 10


def short_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_index(meeting_ids, embedder, data_hash):
    """Index the corpus into a MemoryStore, reusing a saved copy when nothing that affects it changed."""
    key = short_hash({"data": data_hash, "meetings": sorted(meeting_ids), "chunking": S.CHUNKING, "card": S.CARD,
                      "embedding": {k: v for k, v in S.EMBEDDING.items() if k not in ("device", "batch_size")},
                      "code": index_code_hash()})
    folder = os.path.join(CACHE_DIR, key)
    if os.path.exists(os.path.join(folder, "kb.json")):
        store = MemoryStore.load(folder)
        print(f"index: loaded {len(store.meetings())} meetings, {len(store.chunks())} chunks from cache {key}")
        return store, key
    store, summarize, dates = MemoryStore(), textrank_summarizer(S.CARD["summary_words"]), ami_qa.series_dates(meeting_ids)
    t0 = time.perf_counter()
    for n, m in enumerate(sorted(meeting_ids), 1):
        rep = index_meeting(store, embedder, ami_qa.load_meeting(m, dates[m]), summarize, S.CHUNKING, S.CARD)
        print(f"  indexed {n:3d}/{len(meeting_ids)} {m}: {rep.chunks} chunks, {rep.items} items, {rep.seconds}s", flush=True)
    print(f"index: built in {time.perf_counter() - t0:.0f}s, saved as {key}")
    store.save(folder)
    return store, key


def index_code_hash():
    """Hash of the code that decides what goes into the index, so a code change rebuilds the cache."""
    h = hashlib.sha256()
    for f in ["transcript.py", "chunking.py", "items.py", "cards.py", "indexer.py", "embed.py", "text.py"]:
        with open(os.path.join(REPO, "4_qa", "knowledge", f), "rb") as fh:
            h.update(fh.read())
    with open(os.path.join(REPO, "6_evaluation", "qa", "ami_qa.py"), "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:12]


def make_retrievers(names, store, embedder):
    makers = {"bm25": lambda: BM25Retriever(store), "vector": lambda: VectorRetriever(store, embedder),
              "hybrid": lambda: HybridRetriever(store, embedder, S.RETRIEVAL["rrf_k"], S.RETRIEVAL["depth"])}
    if "two_stage" in names:
        from two_stage import TwoStageRetriever
        makers["two_stage"] = lambda: TwoStageRetriever(store, embedder, S.RETRIEVAL)
    unknown = set(names) - set(makers)
    if unknown:
        sys.exit(f"unknown retriever(s): {sorted(unknown)}; choose from {sorted(makers)}")
    return [makers[n]() for n in names]


def evaluate(retriever, queries, golds, scope_of=lambda q: None):
    rows, per_section, latencies, examples = [], {}, [], []
    retriever.retrieve(queries[0].text, k=K)          # warm-up (BM25 build, first torch call)
    for q in queries:
        t0 = time.perf_counter()
        res = retriever.retrieve(q.text, k=K, meeting_ids=scope_of(q))
        latencies.append((time.perf_counter() - t0) * 1000)
        row = qa_metrics.score_query([r["chunk_id"] for r in res], golds[q.query_id],
                                     [r["meeting_id"] for r in res], q.meeting_id)
        rows.append(row)
        per_section.setdefault(q.section, []).append(row)
        if len(examples) < 5 and row["hit@5"] == 0:
            examples.append({"query": q.text, "gold": golds[q.query_id], "got": [r["chunk_id"] for r in res[:3]]})
    return {"metrics": qa_metrics.mean(rows),
            "by_section": {s: {"n": len(r), **qa_metrics.mean(r)} for s, r in sorted(per_section.items())},
            "latency_ms": {"p50": round(qa_metrics.percentile(latencies, 50), 1),
                           "p95": round(qa_metrics.percentile(latencies, 95), 1)},
            "misses_sample": examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["smoke", "dev", "eval"], default="dev")
    p.add_argument("--corpus", choices=["split", "all"], default="split",
                   help="index only the split's meetings, or all scenario meetings (more distractors)")
    p.add_argument("--scope", choices=["corpus", "team"], default="corpus",
                   help="search the whole corpus, or only the asking team's own meetings (its AMI series)")
    p.add_argument("--retrievers", default="bm25,vector,hybrid,two_stage")
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()

    # 1. data + split
    data_hash = fetch_data.fetch()
    split = ami_split.load_split()
    split["smoke"] = list(ami_settings.SMOKE_MEETINGS)
    query_meetings = split[a.split]
    corpus = query_meetings if a.corpus == "split" else sorted(split["train"] + split["dev"] + split["eval"])
    zip_path = os.path.join(ami_settings.RAW_DIR, fetch_data.load_manifest()["file"])
    print(f"scope: {a.scope}")
    print(f"split OK; queries from {a.split} ({len(query_meetings)} meetings), corpus {a.corpus} ({len(corpus)} meetings)")

    # 2. index
    embedder = Embedder(S.EMBEDDING)
    store, index_key = build_index(corpus, embedder, data_hash)

    # 3. queries + gold
    by_meeting = {}
    for c in store.chunks(query_meetings):
        by_meeting.setdefault(c.meeting_id, []).append(c)
    all_q = ami_qa.load_queries(query_meetings, zip_path)
    golds = {q.query_id: ami_qa.gold_chunks(q, by_meeting) for q in all_q}
    queries = [q for q in all_q if golds[q.query_id]]
    n_gold = [len(golds[q.query_id]) for q in queries]
    print(f"queries: {len(queries)} with gold ({len(all_q) - len(queries)} dropped: links to no chunk); "
          f"gold chunks per query: mean {sum(n_gold) / len(n_gold):.2f}, max {max(n_gold)}; "
          f"corpus {len(store.chunks())} chunks")

    # 4. retrieve + score
    results = {}
    for r in make_retrievers(a.retrievers.split(","), store, embedder):
        scope_of = ((lambda q: [m for m in corpus if m[:-1] == q.meeting_id[:-1]]) if a.scope == "team"
                    else (lambda q: None))
        results[r.name] = evaluate(r, queries, golds, scope_of)
        m, lat = results[r.name]["metrics"], results[r.name]["latency_ms"]
        print(f"{r.name:10s} hit@1 {m['hit@1']:.3f}  hit@5 {m['hit@5']:.3f}  hit@10 {m['hit@10']:.3f}  "
              f"MRR {m['mrr']:.3f}  nDCG@10 {m['ndcg@10']:.3f}  meeting@1 {m['meeting@1']:.3f}  "
              f"latency p50 {lat['p50']}ms p95 {lat['p95']}ms")
    print("\nhit@5 by summary section (n):")
    sections = sorted({s for v in results.values() for s in v["by_section"]})
    print("  " + "".join(f"{s + ' (' + str(next(iter(results.values()))['by_section'][s]['n']) + ')':>18}" for s in sections))
    for name, v in results.items():
        print(f"  {name:10s}" + "".join(f"{v['by_section'][s]['hit@5']:>18.3f}" for s in sections))

    # 5. log
    config = {"split": a.split, "corpus": a.corpus, "scope": a.scope, "k": K, "chunking": S.CHUNKING, "card": S.CARD,
              "embedding": {k: v for k, v in S.EMBEDDING.items() if k != "batch_size"}, "retrieval": S.RETRIEVAL}
    record = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_qa_{a.split}_{a.corpus}_{a.scope}",
        "commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain", "--", "4_qa", "6_evaluation/qa")),
        "config": config, "config_hash": short_hash(config), "data_hash": data_hash, "index_key": index_key,
        "split_file_sha256": fetch_data.sha256_file(ami_settings.SPLIT_FILE),
        "n_queries": len(queries), "n_corpus_meetings": len(corpus), "n_corpus_chunks": len(store.chunks()),
        "query_meetings": query_meetings, "results": results, "embedder": embedder.version,
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "torch", "transformers", "rank-bm25", "summa"]},
    }
    if not a.no_log:
        os.makedirs(S.RESULTS_DIR, exist_ok=True)
        with open(S.RESULTS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\nlogged {record['run_id']}  config {record['config_hash']}  data {data_hash[:12]}")


if __name__ == "__main__":
    main()
