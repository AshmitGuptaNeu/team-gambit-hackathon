# 4_qa: ask questions about past meetings (Feature 3)

During a meeting, someone asks *"what price did we pick for the starter plan?"*. This folder finds the
moments in **earlier meetings** that answer it and returns them with the meeting name and a `[mm:ss]`
timestamp.

What exists today is **retrieval** (finding the right moments). The step where an LLM reads those moments
and writes a one-paragraph answer (`answer/`) is not built yet. Once it is, the whole thing is a RAG
(retrieval-augmented generation) system.

Benchmark numbers and full design notes: [`docs/QA.md`](../docs/QA.md).

---

## 1. The idea in one picture

```
WHEN A MEETING ENDS  (index_meeting.py)
  transcript.txt ──► split into ~60 s chunks ──► find decisions / action items ──► write a meeting card
                                │                                                     │
                                └──────────► embed every chunk, item and card (bge-small, 384 numbers each)
                                                              │
                                                              ▼
                                       knowledge base: Postgres + pgvector  (or a local folder)

WHEN SOMEONE ASKS  (ask.py)
  question (+ last lines of the live meeting)
     │
     ├─ 1. understand  : dates ("last week"), people ("with Priya"), "last time"      (rules, no LLM)
     ├─ 2. candidates  : keep only meetings that pass those filters (dropped if nothing passes)
     ├─ 3. stage 1     : rank the candidate meetings (best chunk, meeting card, recency)
     ├─ 4. stage 2     : hybrid search (BM25 + vector) for moments inside the top meetings
     └─ 5. final       : merge the lists with RRF ──► top k moments with meeting + [mm:ss]
```

---

## 2. Every file in one line

### Top level
| File | What it does |
|---|---|
| `qa_settings.py` | Every setting in one place: chunk size, embedding model + pinned version, all retrieval weights, DB location. |
| `index_meeting.py` | Command line: adds one finished meeting (an ingestion session folder) to the knowledge base. |
| `ask.py` | Command line: asks a question and prints the best moments from past meetings with citations. |
| `README.md` | This file. |

### `knowledge/`: turning a meeting into searchable pieces
| File | What it does |
|---|---|
| `transcript.py` | Reads `[mm:ss] Name: text` transcripts and ingestion session folders into `MeetingInput` / `Utterance` objects. |
| `text.py` | Text helpers: tokenizing, stopwords, Porter stemming (for BM25) and removing "um"/"uh" and stutters. |
| `chunking.py` | Splits a meeting into ~60-second chunks (220-word cap; a tiny last chunk is merged into the one before). |
| `items.py` | Finds decisions ("we agreed…") and action items ("I'll…") with phrase rules and guesses each owner. |
| `cards.py` | Builds a meeting card (title, date, attendees, topic words, TextRank summary, decisions/actions). |
| `embed.py` | Loads the `bge-small-en-v1.5` model and turns text into 384-number unit vectors. |
| `store.py` | The knowledge base itself: `PgStore` (Postgres + pgvector) and `MemoryStore` (numpy, saved to a folder), same methods. |
| `indexer.py` | Runs the whole "meeting ends" pipeline: chunk → items → card → embed → save. |

### `retrieval/`: finding the right moments
| File | What it does |
|---|---|
| `retrieval_common.py` | Reciprocal Rank Fusion (merges ranked lists) and the shared result format. |
| `bm25/bm25.py` | Keyword search (BM25) over chunks or meeting cards; rebuilds itself when the store changes. |
| `vector/vector.py` | Meaning search (cosine similarity) and plain hybrid search (BM25 + vector merged with RRF). |
| `two_stage/query.py` | Reads the question for date ranges, people's names and "last time" (pure rules). |
| `two_stage/two_stage.py` | The main retriever: filter meetings → pick the best meetings → find the moments → merge. |
| `graph/`, `reranker/` | Empty placeholders (a knowledge-graph or reranker retriever could go here later). |

### `answer/`
| File | What it does |
|---|---|
| `.gitkeep` | Placeholder for the LLM that will write the final answer from the retrieved moments (not built). |

### `tests/` (80 tests)
| File | What it does |
|---|---|
| `conftest.py` | Test setup: import paths, the fake embedder, the real model, the throwaway-Postgres fixture. |
| `test_ingest.py` | Transcript parsing, chunking, items, cards and indexing. |
| `test_query_fusion_metrics.py` | Date/people/"last time" parsing, RRF, live-context words and the benchmark metrics. |
| `test_store.py` | `MemoryStore` and `PgStore` give identical results (save, load, replace, delete, search, filters). |
| `test_scenarios.py` | End to end: 16 realistic questions over 7 meetings, the right moment must be in the top 3. |
| `test_ami_benchmark.py` | The AMI benchmark builder: questions, gold answers, no leakage. |
| `fixtures/scenarios.json` | The 7 test meetings and 16 labelled questions (written by Claude, not human-reviewed yet). |

### Related files outside this folder
| File | What it does |
|---|---|
| `../6_evaluation/qa/ami_qa.py` | Turns AMI's human summaries into questions with known correct moments. |
| `../6_evaluation/qa/run.py` | Runs the AMI benchmark for BM25 / vector / hybrid / two-stage and logs the results. |
| `../6_evaluation/qa/tune.py` | Tries weight combinations on the AMI **dev** split only. |
| `../6_evaluation/qa/qa_metrics.py` | hit@k, recall@k, MRR, nDCG@10, latency. |
| `../docs/QA.md` | Full reference: architecture, commands, benchmark results, known gaps. |
| `../2_summarization/textrank_baseline.py` | The TextRank summarizer reused for meeting cards. |

---

## 3. The approaches we use

We combine **two kinds of search**, then add **meeting-level reasoning** on top.

| Approach | Finds | Misses |
|---|---|---|
| **Keyword search (BM25)** | Exact words: names, numbers, "Stripe", "starter plan" | Same meaning, different words ("cost" vs "price") |
| **Meaning search (vector + cosine)** | Same idea, different words | Exact rare terms, numbers |
| **Hybrid (both, merged with RRF)** | Most of both | |
| **Two-stage (ours)** | Hybrid + filters for dates, people, "last time" and the live discussion | |

Why not just one? On real meetings (AMI), keyword search alone was the strongest single method. Questions
like *"what did Priya say last week?"* need dates and names, which neither search understands, so the
two-stage layer handles those with rules.

### 3.1 What is BM25?

BM25 is a **keyword-scoring formula**, the classic search-engine ranking. Nothing is trained. For each
chunk it adds up, for every question word that appears in the chunk:

```
score += IDF(word) × (tf × (k1 + 1)) / (tf + k1 × (1 − b + b × chunk_length / average_length))
```

- **tf** = how many times the word appears in this chunk (more is better, with diminishing returns, controlled by `k1 = 1.5`).
- **IDF** = how rare the word is across all chunks. A word in every chunk is worth ~0; a word in one chunk is worth a lot.
- **length normalisation** (`b = 0.75`): long chunks don't win just by having more words.

Before scoring, text is lowercased, common words ("the", "did", "we", "um") are removed and words are
reduced to their stem with the Porter stemmer ("pricing" → "price", "picking" → "pick"). Code:
`knowledge/text.py` (`tokenize`) and `retrieval/bm25/bm25.py` (uses `rank_bm25.BM25Okapi`).

BM25 runs **in Python memory**, not in Postgres. Postgres' built-in text ranking (`ts_rank`) is not BM25.
The index is rebuilt automatically whenever a meeting is added, which takes well under a second at our scale.

### 3.2 What is an embedding (encoding)?

An embedding model reads a piece of text and outputs a list of numbers (a **vector**) that captures its
meaning. Texts that mean similar things get vectors that point in similar directions.

- Model: **`BAAI/bge-small-en-v1.5`**, 33M parameters, MIT licence, pinned to revision `5c38ec7c…` so results are reproducible.
- Output: **384 numbers** per text.
- How (`knowledge/embed.py`): tokenize (max 512 tokens) → run the model → take the first token's output (the `[CLS]` token) → scale it to length 1.
- **Questions** get the prefix `"Represent this sentence for searching relevant passages: "` (bge is trained that way); **chunks do not**.
- Each **chunk** is embedded with a short header in front, so a line like "let's do 59 then" still says which meeting it came from:
  ```
  Meeting: Atlas pricing sync | October 27, 2026 | Speakers: Priya Shah, Marcus Lee
  [01:10] Priya Shah: What if we move the starter plan to 59 dollars a month?
  ...
  ```
- Runs on CPU by default (set `LIVEMINUTES_DEVICE=mps` or `cuda` to change it). It's downloaded from
  Hugging Face on first use (~130 MB, cached in `~/.cache/huggingface`).

### 3.3 What is cosine similarity?

Cosine similarity measures **how closely two vectors point in the same direction**:

```
cosine(a, b) = (a · b) / (|a| × |b|)        1 = same meaning, 0 = unrelated
```

All our vectors are scaled to length 1, so cosine is simply the **dot product** (multiply matching
numbers and add them up). Meaning search = embed the question, compute its cosine with every chunk vector,
then sort from highest to lowest.

- `MemoryStore`: an exact computation, one matrix multiplication (`matrix @ question_vector`) over all chunks.
- `PgStore`: pgvector's `<=>` operator (cosine distance = 1 − cosine) with an **HNSW index**, a graph
  structure that finds the nearest vectors without checking every row (approximate but very fast).

### 3.4 What is Reciprocal Rank Fusion (RRF)?

BM25 scores (like 2.15) and cosine scores (like 0.75) are on different scales, so we can't add them.
RRF uses only **positions** in each list:

```
score(chunk) = Σ over lists   weight / (60 + rank of the chunk in that list)
```

A chunk that is near the top of **both** lists beats one that tops only one list. Code: `retrieval_common.rrf`.

### 3.5 What the two-stage retriever adds (`retrieval/two_stage/`)

1. **Understand the question** (`query.py`, regex rules, no LLM): "last week", "yesterday", "3 days ago",
   "in October", "on Monday" become a date range; known attendee names become a people filter;
   "last time" / "previous meeting" turn on a recency signal.
2. **Candidates**: keep only meetings in that date range or with those people. If nothing passes, the filter is dropped and a note goes into the debug output.
3. **Stage 1, rank meetings**: by where each meeting's best chunk lands, plus the meeting card (vector + BM25), plus recency for "last time" questions.
4. **Stage 2, search inside the top 3 meetings**: hybrid search. This list is boosted **only** for "last time" questions. On AMI, boosting by content alone never helped.
5. **Live context**: up to 8 key words from the last 12 lines of the live meeting are added to the question and searched again (weight 0.5). This is how "who's fixing *that*?" finds the Stripe bug.
6. **Final RRF** of all the lists → the top k chunks.

Settings that matter, all in `qa_settings.py → RETRIEVAL`:
- **tuned on AMI dev** (`6_evaluation/qa/tune.py`): `stage2_weight` = 0, `item_weight` = 0, `top_meetings` = 3, and the card weights (0.3).
- **set by hand**: everything else (recency, "last time" boost, live-context weight, BM25:vector = 1:1, RRF k = 60, depth = 50, chunk sizes).

---

## 4. Where is the vector database?

`knowledge/store.py → open_store()` decides, using one environment variable:

| `LIVEMINUTES_DB_URL` | Store used | Where the vectors live |
|---|---|---|
| set, e.g. `postgresql://user@host:5432/liveminutes` | **`PgStore`** | Postgres tables with `vector(384)` columns and HNSW indexes (pgvector extension) |
| not set | **`MemoryStore`** | Loaded into RAM from `4_qa/kb_local/` (or `LIVEMINUTES_KB_DIR`): `kb.json` (text + metadata) and `vectors.npz` (numpy vectors) |

`kb_local/` is gitignored because it holds real meeting text.

**Postgres tables** (created automatically on first connect):
```
meetings (meeting_id, title, started_at, attendees[], source, card_text, card_vec vector(384))
chunks   (chunk_id, meeting_id, ord, start_s, end_s, speakers[], text, embed_text, vec vector(384))
items    (item_id, meeting_id, chunk_id, kind, text, speaker, owner, start_s, vec vector(384))
kb_meta  (key, value)            -- model name + dimension check, version counter
```
The search query it runs:
```sql
SELECT chunk_id, 1 - (vec <=> :question_vec) FROM chunks
WHERE meeting_id = ANY(:meetings) ORDER BY vec <=> :question_vec LIMIT 50;
```
Re-indexing a meeting replaces all its rows in one transaction, so it's safe to run twice. Postgres 16+
with pgvector works locally (`brew install postgresql@18 pgvector`) and on AWS RDS.

---

## 5. Libraries

| Library | Version | Used for |
|---|---|---|
| `transformers` | 5.20.2 | Loading and running the bge-small embedding model |
| `torch` | 2.14.1 | Runs the model (CPU / MPS / CUDA) |
| `numpy` | 2.5.4 | Vectors, cosine via matrix multiply, `MemoryStore` |
| `rank-bm25` | 0.2.2 | BM25 Okapi scoring |
| `nltk` | 3.10.3 | Porter stemmer only |
| `summa` | 1.2.0 | TextRank summary for meeting cards (via `2_summarization/textrank_baseline.py`) |
| `psycopg` + `psycopg-binary` | 3.3.9 | Talking to Postgres |
| `pgvector` (Python) | 0.5.1 | Sending numpy vectors to Postgres `vector` columns |
| pgvector (Postgres extension) | 0.8.x | Vector columns, `<=>` cosine distance, HNSW index |
| `pytest` | 9.2.0 | Tests |

There's no LLM, no LangChain and no hosted vector service. Everything runs locally.

---

## 6. One retrieval, step by step (real output)

Two demo meetings are indexed: **"Atlas pricing sync"** (Oct 27) and **"Atlas launch readiness"** (Nov 10),
each split into 4 chunks (8 chunks in total).

**Question:** `what price did we pick for the starter plan?`

**Step 1: understand.** No date, no person, no "last time" → no filters, both meetings are candidates.

**Step 2: BM25 tokens.** `['price', 'pick', 'starter', 'plan']`

| Word | IDF | Appears in | Why |
|---|---|---|---|
| price | 0.0 | 4 of 8 chunks | Every pricing-sync chunk header says "pricing" → too common to help |
| plan | 0.0 | 4 of 8 chunks | Too common |
| starter | 1.61 | `bot_demo1#0001` only | Rare → strong signal |
| pick | 1.61 | `bot_demo2#0002` only | Matches "**picking** a template": a false match |

BM25 result: `bot_demo1#0001` (2.15), `bot_demo2#0002` (1.66). Only these two contain a useful word.

**Step 3: embed the question.**
```
"Represent this sentence for searching relevant passages: what price did we pick for the starter plan?"
  → [-0.0533, -0.0248, -0.0633, -0.0039, 0.0046, 0.0452, … 384 numbers], length 1.0
```
For comparison, the sentence "59 dollars keeps us profitable" (no shared words) has cosine **0.57** with
the question: the model knows "price" and "dollars" are related.

**Step 4: vector search (cosine with every chunk).**
| Rank | Chunk | Cosine | Content |
|---|---|---|---|
| 1 | `bot_demo1#0001` | 0.751 | starter plan → 59 dollars ✔ |
| 2 | `bot_demo1#0002` | 0.657 | annual plan 20% discount |
| 3 | `bot_demo2#0001` | 0.590 | launch moved to Dec 8 |
| … | … | … | |
| 8 | `bot_demo2#0002` | 0.523 | Stripe bug, template picking |

**Step 5: RRF merge** (k = 60):
```
bot_demo1#0001 : 1/(60+1) BM25 + 1/(60+1) vector = 0.0328   ← top in both lists
bot_demo2#0002 : 1/(60+2) BM25 + 1/(60+8) vector = 0.0308
bot_demo1#0002 :            0  + 1/(60+2) vector = 0.0161
```

**Step 6: final answer** (`ask.py`):
```
1. Atlas pricing sync (Oct 27, 2026) at 01:10-02:40
   [01:10] Priya Shah: What if we move the starter plan to 59 dollars a month?
   [01:38] Marcus Lee: 59 dollars keeps us profitable and is still below the main competitor.
   [02:02] Priya Shah: Okay, we agreed on 59 dollars a month for the starter plan, replacing the 49 dollar price.
2. Atlas launch readiness (Nov 10, 2026) at 02:20-03:35     ← keyword false match ("picking")
3. Atlas pricing sync (Oct 27, 2026) at 02:40-03:45
```
The right moment is #1. The #2 result shows why both searches are used and why a reranker or the answer
LLM is the next step: BM25 alone was fooled by "pick", and vector alone ranked that chunk last.

Total search time: about 15–30 ms (the model is already loaded).

---

## 7. How to run

From the repo root:
```bash
# add a finished meeting (Postgres if LIVEMINUTES_DB_URL is set, else 4_qa/kb_local/)
.venv/bin/python 4_qa/index_meeting.py 1_ingestion/sessions/<bot_id> --title "Atlas pricing sync"

# ask
.venv/bin/python 4_qa/ask.py "what price did we pick for the starter plan?"
.venv/bin/python 4_qa/ask.py "who is fixing that?" --context live_transcript.txt   # use the live meeting
.venv/bin/python 4_qa/ask.py "what did we decide last week?" --now 2026-11-18T15:00:00+00:00 --debug

# tests
.venv/bin/python -m pytest 4_qa/tests -m "not model"            # fast, ~2 s
LIVEMINUTES_TEST_DB_URL=postgresql://... .venv/bin/python -m pytest 4_qa/tests   # all; DROPS tables, use a throwaway DB

# AMI benchmark
.venv/bin/python 6_evaluation/qa/run.py --split eval --scope team
```

---

## 8. Results in one table (AMI, 332 questions, right moment in the top 5)

| Where we search | BM25 | Vector | Hybrid | Two-stage |
|---|---|---|---|---|
| Own meetings (real use) | 77% | 62% | **79%** | **79%** |
| 20 meetings from 5 teams | **58%** | 38% | 50% | 49% |
| All 138 meetings | **38%** | 19% | 30% | 29% |

Our 16 scenario questions (dates, people, "last time", live context): two-stage finds the right moment in
the top 3 for **16/16**; plain hybrid finds it for 15/16.

## 9. Not built yet

- The answer LLM with citations (`answer/`).
- Indexing automatically when the meeting bot finishes.
- A stronger meaning search: a bigger embedding model or a reranker (`retrieval/reranker/`).
- A human-reviewed question set to tune the hand-set weights.
