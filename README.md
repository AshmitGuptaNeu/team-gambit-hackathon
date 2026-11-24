# LiveMinutes Local: an AI assistant a deaf student owns

**Team Gambit** · built by Ashmit Gupta ([@AshmitGuptaNeu](https://github.com/AshmitGuptaNeu))

Hackathon build ("Build AI we own"). One web page on the student's own laptop that **listens, reads, sees and plans** with them. Every model is open-weight and runs on this machine: no cloud AI, no API keys, and a lecture never leaves the laptop.

**The user:** a deaf or hard-of-hearing CS student. In lectures, the student needs captions they can trust. In office hours, they need answers from the syllabus and the whiteboard. In group projects, they need the plan written down.

**Showing it to someone?** Follow [DEMO.md](DEMO.md): a step-by-step demo with what to click, what you'll see and what to say.

## What it does

| # | Feature | Input | How |
|---|---|---|---|
| 1 | Live captions from the mic, adjustable size, a level meter (the user can't check the mic by ear) | audio | Whisper large-v3-turbo |
| 2 | Captions from a recording (.wav/.mp3/.m4a), split into whole sentences | audio | Whisper |
| 3 | Questions about the captions, with quotes; says so when it wasn't said | text | gemma3:12b |
| 4 | Describe a photo (whiteboard, slide) and answer questions about it; "the image doesn't show that" | image | gemma3:12b (vision) |
| 5 | Questions about a document (PDF/TXT/MD) with the quote and page; "that isn't in this document" | text | bge-small search + gemma3:12b, three checks |
| 6 | Diagrams (flowchart, sequence, UML class, state, ER, Gantt, mind map, timeline) from a request, the captions or a whiteboard photo. The diagram code is checked, repaired, drawn, editable and exportable as SVG | code | qwen2.5-coder:7b + the official Mermaid parser |
| 6b | **Who's doing what**: a knowledge graph over the captions. People, decisions, tasks with owners, and deadlines with dates, linked to the moment each was said. Drawn as a graph, and it answers "what is Sam doing?" / "what did we decide?" instantly, quoting the line | text | phrase rules from the LiveMinutes knowledge base (`4_qa/knowledge/items.py`), no model |
| 7 | General questions, labelled *General answer* so they're never mistaken for course facts | text | gemma3:12b |
| 8 | Meeting bot (optional): joins a Google Meet call, asks everyone for permission in chat, and only then sends captions with speaker names | audio | Attendee (open source, Docker) + our bridge, using the same local Whisper |

Not built: video. Nothing was trained or fine-tuned. These are off-the-shelf open-weight models, plus the system around them: retrieval, answer checks, diagram validation and repair, and consent.

## Models (all local)

| Job | Model | Runs in |
|---|---|---|
| Speech to text | Whisper `large-v3-turbo` (faster-whisper) | inside `app.py`, on the Mac's GPU (MPS) |
| Vision, answers | `gemma3:12b` | Ollama |
| Diagram code | `qwen2.5-coder:7b` | Ollama |
| Document search | `BAAI/bge-small-en-v1.5` + BM25 | inside `app.py` |
| Diagram check | `mermaid` 11.4.1 + jsdom | Node, offline |

Built and measured on a MacBook Pro (Apple M3 Max: 14-core CPU, 30-core GPU, 36 GB memory, macOS 14.8.3).

## Quick start

```bash
# once
.venv/bin/pip install -r requirements.txt
ollama pull gemma3:12b && ollama pull qwen2.5-coder:7b
(cd 8_multimodal/mermaid_check && npm ci)

# run
ollama serve                         # if the Ollama app isn't already running
.venv/bin/python 8_multimodal/app.py # -> http://127.0.0.1:8780 (only this computer can reach it)
```

Optional meeting bot (needs Docker Desktop; one-time setup in [docs/INGESTION.md](docs/INGESTION.md)):

```bash
bash 1_ingestion/start_bot.sh        # Attendee + bridge in Docker; then paste a Meet link under "Join a meeting…"
```

Tests (offline, models faked): `.venv/bin/python -m pytest 8_multimodal/tests local_models/tests`

## Measured (Nov 2026, Apple M3 Max, 36 GB)

| | Result |
|---|---|
| Captions | 0.83 s to transcribe a 5.7 s clip; a 12.6 s spoken clip came out word for word |
| Photo | Describing it takes 16–19 s; follow-up questions about 3 s |
| Document Q&A (demo syllabus, 20 questions) | 10/10 answerable questions answered with the right page; of 10 unanswerable questions, 0 got a made-up answer (8 refused, 2 flagged as partial); median 4.7 s |
| Diagrams (5 requests) | 5/5 valid on the first try, 3–6 s each (21 s from a whiteboard photo) |

**Caveats:** the prompts were tuned on these same questions and requests, so these are demo results, not benchmarks. The meeting bot can't join Teams on this Mac, because Attendee's Chrome is x86-only, runs under Rosetta, and the Teams web app never finishes loading. Google Meet is the supported path. Some diagram content can be wrong even when the code is valid. That's why the code is shown and editable.

## Where things are

| Folder | What |
|---|---|
| [`8_multimodal/`](8_multimodal/) | **The hackathon app**: `app.py` (server), `static/index.html` (the page), `mm_image.py`, `mm_document.py`, `mm_diagram.py`, demo files, evals, tests. Details: [8_multimodal/README.md](8_multimodal/README.md) |
| [`local_models/`](local_models/) | The local Whisper server and model notes: [local_models/README.md](local_models/README.md) |
| [`1_ingestion/`](1_ingestion/) | The consent-first meeting bot (Attendee + `bridge.py`) |
| [`4_qa/`](4_qa/) | The LiveMinutes knowledge base: hybrid search (BM25 + vectors) reused by document Q&A, and the decisions/tasks layer reused by the graph |
| [`2_summarization/`](2_summarization/), `5_data/`, `6_evaluation/` | The LiveMinutes summarization baseline this build started from: [docs/BASELINE.md](docs/BASELINE.md) |
