"""Every setting Feature 3 (Q&A over past meetings) uses, in one place. Eval runs copy these into the results log.

Named qa_settings (not settings) so it never clashes with 1_ingestion/settings.py or 2_summarization/settings.py
when several components are on sys.path at once.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "6_evaluation", "qa", "results")
RESULTS_LOG = os.path.join(RESULTS_DIR, "results_log.jsonl")          # one line per eval run (committed)

# Postgres + pgvector. Empty = no database; the CLIs then use a local file store (MEMORY_STORE_DIR).
DB_URL = os.environ.get("LIVEMINUTES_DB_URL", "")
MEMORY_STORE_DIR = os.environ.get("LIVEMINUTES_KB_DIR", os.path.join(REPO, "4_qa", "kb_local"))  # gitignored

# Chunks: about one minute of meeting each, the same windows BART summarizes (2_summarization BART window_seconds).
CHUNKING = {
    "window_seconds": 60,
    "max_words": 220,      # a window longer than this is cut again (bge-small reads 512 tokens; leave room for the header)
    "min_words": 25,       # a last window shorter than this is merged into the one before
}

# Embedding model: BAAI/bge-small-en-v1.5 (33M params, 384 dims, MIT licence). Small enough for CPU, live-query fast.
EMBEDDING = {
    "model": "BAAI/bge-small-en-v1.5",
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "dim": 384,
    "max_length": 512,
    "batch_size": 32,
    # bge models expect this prefix on queries (not on passages) for retrieval.
    "query_prefix": "Represent this sentence for searching relevant passages: ",
    "device": os.environ.get("LIVEMINUTES_DEVICE", "cpu"),
}

# Meeting card: our own summary of the meeting (never a human-written one, see 6_evaluation/qa/run.py leakage notes).
CARD = {
    "summary_words": 120,  # TextRank word budget for the card summary
    "topics": 12,          # how many topic keywords the card keeps
    "max_items": 12,       # how many decisions / action items the card text lists
}

# Retrieval. RRF = reciprocal rank fusion: score = sum over lists of weight / (RRF_K + rank).
RETRIEVAL = {
    "rrf_k": 60,
    "depth": 50,               # how deep each list (BM25, vector) goes before fusing
    # Two-stage (architecture 2). Tuned on AMI dev (6_evaluation/qa/tune.py): stage2_weight, item_weight,
    # top_meetings and the card weights. Every other value here is hand-set (AMI has no dates, people or
    # live context to tune them on). On dev, boosting the chosen meetings by their own content did not beat
    # plain hybrid search, so the boost is used only when the question says which meeting ("last time").
    # Date and people filters always apply.
    "top_meetings": 3,         # stage 1 keeps this many meetings
    "meeting_weights": {"best_chunk": 1.0, "card_vector": 0.3, "card_bm25": 0.3, "recency": 1.0},
    "stage2_weight": 0.0,      # boost for chunks inside the chosen meetings (no meeting signal in the question)
    "stage2_weight_latest": 1.0,   # ... when the question says "last time" / "previous meeting"
    "global_weight": 1.0,      # chunks from every candidate meeting
    "item_weight": 0.0,        # decisions / action items -> their chunk (0 = off: lowered every AMI dev score)
    "context_weight": 0.5,     # the same search with live-meeting context added to the question
    "context_terms": 8,        # how many key terms of the live context are added
    "context_lines": 12,       # how many recent live-transcript lines are read
}
