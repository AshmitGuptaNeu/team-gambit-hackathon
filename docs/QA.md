# 4. Q&A knowledge base (Feature 3): reference

Ask a question during a meeting; get the moments from past meetings that answer it, with meeting and `[mm:ss]` citations. This folder covers **retrieval**. The answer-writing LLM (`4_qa/answer/`) is not built yet.

## 4.1 Architecture: two-stage hybrid retrieval

```
meeting ends   transcript -> ~60 s chunks (+ header: meeting, date, speakers)
                          -> decisions / action items (rule-based, linked to their chunk)
                          -> meeting card (title, date, attendees, topics, TextRank summary, items)
                          -> embed (bge-small) -> store (Postgres + pgvector, or in-memory)

question       understand: dates ("last week"), people ("with Priya"), "last time"   (query.py, no LLM)
               candidates: meetings passing the date/people filters (relaxed if none pass)
               stage 1: rank meetings (best chunk + card + recency)
               stage 2: hybrid search (BM25 + vector, RRF) in the top meetings
               final: RRF of stage 2 + search over all candidates (+ same with live-meeting context)
```

The boost for chunks inside the stage-1 meetings is on **only when the question names a meeting** ("last time"). On AMI dev, boosting by content alone never beat plain hybrid search (`6_evaluation/qa/tune.py`), so with no such signal two-stage reduces to filtered hybrid search.

| File | What |
|---|---|
| `4_qa/qa_settings.py` | Every setting (chunking, model + pinned revision, retrieval weights) |
| `4_qa/knowledge/` | `transcript.py` (reads `[mm:ss] Name: text`), `chunking.py`, `items.py`, `cards.py`, `embed.py`, `store.py` (PgStore, MemoryStore), `indexer.py` |
| `4_qa/retrieval/bm25/`, `vector/` | BM25, vector, hybrid (flat) retrievers |
| `4_qa/retrieval/two_stage/` | `query.py` (dates, people, recency), `two_stage.py` |
| `4_qa/index_meeting.py`, `4_qa/ask.py` | CLIs: add a session folder; ask a question |
| `6_evaluation/qa/` | AMI benchmark (`ami_qa.py`, `run.py`), metrics, dev tuning (`tune.py`), results log |
| `4_qa/tests/` | 80 tests; `fixtures/scenarios.json` = 7 hand-written meetings + 16 labelled questions |

## 4.2 Commands (repo root)

```bash
.venv/bin/python -m pytest 4_qa/tests -m "not model"          # fast tests, ~2 s
LIVEMINUTES_TEST_DB_URL=postgresql://... .venv/bin/python -m pytest 4_qa/tests   # all, incl. pgvector (DROPS its tables: use a throwaway DB)
.venv/bin/python 6_evaluation/qa/run.py --split eval           # AMI benchmark (add --scope team, --corpus all)
.venv/bin/python 6_evaluation/qa/tune.py                       # choose weights on dev only
.venv/bin/python 4_qa/index_meeting.py 1_ingestion/sessions/<bot_id> --title "Weekly"
.venv/bin/python 4_qa/ask.py "what did we decide about pricing last week?"
```

Postgres: set `LIVEMINUTES_DB_URL` (Postgres 16+ with pgvector; macOS: `brew install postgresql@18 pgvector`). Without it, the CLIs use `4_qa/kb_local/` (gitignored).

## 4.3 AMI benchmark

AMI's human summaries (abstract / decisions / actions / problems) are linked sentence by sentence to the dialogue acts they were written from (`extractive/*.summlink.xml`, read straight from the hash-checked zip). Each linked sentence is a query; gold = the chunk(s) containing those dialogue acts. **Leakage:** nothing human-written goes in the index (cards use our TextRank summary). The stage-2, item, top-meetings and card weights were tuned on dev (`tune.py`); the rest (recency, "last time" boost, live context, BM25:vector 1:1, RRF k, chunk sizes) are hand-set. Eval was run once. **Caveat:** queries are declarative summary sentences, not live questions, and never mention dates or people, so AMI can't test stage 1's filters. The scenario tests cover those.

## 4.4 Results (eval split, 332 queries, k = 10)

| Setting | Retriever | hit@1 | hit@5 | hit@10 | MRR | search p50 |
|---|---|---|---|---|---|---|
| 20 eval meetings | BM25 | 0.325 | **0.581** | 0.690 | **0.434** | 0.4 ms |
| | vector | 0.172 | 0.380 | 0.479 | 0.263 | 17 ms |
| | hybrid | 0.256 | 0.500 | 0.632 | 0.364 | 17 ms |
| | two-stage | 0.250 | 0.494 | 0.626 | 0.358 | 18 ms |
| Team's own 4 meetings (`--scope team`) | BM25 | 0.497 | 0.765 | 0.895 | 0.619 | 0.4 ms |
| | hybrid | 0.464 | **0.789** | 0.879 | 0.597 | 12 ms |
| | two-stage | 0.461 | **0.789** | 0.882 | 0.594 | 12 ms |
| All 138 meetings (`--corpus all`) | BM25 | 0.181 | **0.380** | 0.455 | 0.264 | 2 ms |
| | hybrid | 0.136 | 0.298 | 0.401 | 0.207 | 19 ms |
| | two-stage | 0.130 | 0.292 | 0.395 | 0.201 | 30 ms |

**Scenario tests** (7 meetings, 16 questions about facts, dates, people, "last time" and live context): gold in top 3 for **16/16** with two-stage vs 15/16 with flat hybrid, on both stores. These questions were written by Claude and are not human-reviewed; treat them as tests, not a benchmark.

Findings:
- Most AMI misses are another team's meeting (all teams design the same remote). Scoping to the team's own meetings is the largest gain (+29 points of hit@5).
- The small embedding model is weak on AMI decisions/actions (third-person summaries vs casual speech), so BM25 ties or beats hybrid.
- Picking meetings by content alone does not help; dates, people, recency and live context do (scenario tests).

## 4.5 Known gaps / next

- Answer LLM with citations (`4_qa/answer/`); run `index_meeting` automatically when a session closes.
- Stronger vector side: a larger embedding model or a cross-encoder / LightGBM reranker (`4_qa/retrieval/reranker/`).
- Better card summaries (BART or LLM); LLM extraction of decisions/actions (Feature 2).
- Human-reviewed live-question set (`5_data/synthetic/qa/`).
