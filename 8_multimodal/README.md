# 8. Multimodal: one local assistant for a deaf or hard-of-hearing student

All models are open-weight and run on this machine (`local_models/README.md`). Settings: `mm_settings.py`.

```bash
ollama serve                                         # if it isn't running already
(cd 8_multimodal/mermaid_check && npm ci)            # once
.venv/bin/python 8_multimodal/app.py                 # -> http://127.0.0.1:8780  (only this computer can reach it)
```

## Join a meeting (the LiveMinutes bot, with everyone's permission)

**Join a meeting…** (captions panel) sends the existing LiveMinutes bot (`1_ingestion/`: open-source Attendee in Docker, plus our bridge) to a Google Meet link. (Teams: the bot's x86 Chrome runs under Rosetta on this Mac, and the Teams web app never finishes loading there.)
1. It asks in the meeting chat. Nothing is transcribed until **everyone present** says yes. A late joiner pauses it, and anyone saying no stops it and deletes the session.
2. The page shows who has said yes and who it's still waiting for.
3. Then the call's lines appear in the captions with each speaker's name. Questions and diagrams use them like any other captions.

The bridge sends its audio back to this app's Whisper (port 8777), so one local Whisper process serves both the mic and the meeting. Live notes run on local gemma3.
```bash
bash 1_ingestion/start_bot.sh        # Docker Desktop: Attendee + bridge; one-time setup: docs/INGESTION.md
```

## The interface (`app.py`, `static/index.html`)

One page with two panels. It needs no Docker and no meeting bot.
- **Captions** (left): *Start captions* listens to the mic. Each piece ends at a pause; with no pause, at the quietest moment by 8 s, never mid-word. The pieces go to Whisper in order. The previous caption is sent along as context, and Whisper's "Thank you." on silence is dropped (the same rule as the meeting bridge). A level meter shows the mic is hearing sound, since the user can't check that by ear. Text size goes from S to XL. A dropped recording (.wav/.mp3/.m4a, turned into WAV in the browser) adds its transcript here, split into whole sentences.
- **Ask & draw** (right): *Ask* answers from the attached photo, the attached document, or the captions, in that order. Document and caption answers come with quotes, and say so plainly when the answer isn't there. With none of the three, the answer is general and labelled *General answer · not from your captions, photos or documents*. *Draw a diagram* writes Mermaid code from the request, the captions and/or the photo, and draws it with mermaid.js served from this machine. The code can be edited and redrawn, copied, or downloaded as SVG.

Tested Nov 20 2026 in Chrome (Playwright), on a 12.6 s spoken clip (macOS `say`) and the demo files: recording → captions → a question → "not found" → a diagram from the captions → a photo and a question about it. Then the same clip through Chrome's fake mic: 2 caption lines, word for word, with the cut between sentences. No console errors, no horizontal scroll at 390 px. `tests/test_app.py` covers the routing with the models faked.

| Input | Module | Status |
|---|---|---|
| Audio: transcribe | `local_models/stt_server.py` (Whisper) | done |
| Image: describe + questions | `mm_image.py` (gemma3:12b) | done |
| Text: questions about a document, says when it isn't there | `mm_document.py` | done |
| Code: diagram code (Mermaid) from a prompt, a transcript or a whiteboard photo | `mm_diagram.py` (qwen2.5-coder:7b) | done |
| Knowledge graph over the captions: who owns what, decisions, deadlines | `mm_graph.py` (rules from `4_qa/knowledge/items.py`) | done |
| Video: describe + questions | | not built |
| One interface | `app.py` + `static/index.html` | done |

## Who's doing what: a knowledge graph over the captions (`mm_graph.py`)

The LiveMinutes knowledge base's "graph layer" (`4_qa/knowledge/items.py`, architecture 2 in `LiveMinutes_KnowledgeBase_Overview.pdf`), built over the captions:
`(meeting) -HAS-> (decision | task | deadline) -SAID_AT-> (time)`, `-SAID_BY-> (person)`, `-OWNED_BY-> (person)`, `-DUE-> (date)`.
- **Found by:** LiveMinutes' phrase rules for decisions and action items, plus two rules added here: deadlines ("homework five is due Friday") and owners named in the line ("Sam, can you…"). There's no model call, so it's instant, and every box is a real caption line.
- **Who's doing what** (captions toolbar): draws the graph as Mermaid (people → owns → task → date), editable like any diagram.
- **Questions** about tasks, owners, decisions or deadlines are answered from the graph first, quoting the line and its time. If the graph has nothing that matches (say, "who owns the frontend?" when nobody said it), the question goes on to the normal caption search, which can still say "that wasn't said".
- **Limits:** the rules miss things said in unusual ways, and mic captions have no speaker names ("I'll do it" has no owner). LiveMinutes compared a full LLM-built graph (Neo4j) and rejected it as slow, costly and unreliable on speech. That's still not built.

## Document Q&A: three checks before an answer is shown

1. **Retrieval** (BM25 + bge-small vectors, RRF, from `4_qa/`): a question far off the document's topic is refused before the model is asked at all.
2. **Quote check**: the model must copy the sentences that support its answer. Each one must really be in the excerpt it cites: a fuzzy match, with the same negation words ("may be used" ≠ "cannot be used").
3. **Directness**: a second model call grades the quotes as *direct* (answer shown), *partial* (shown as "The document doesn't say exactly that. The closest it gets: ...") or *no* (refused).

```bash
.venv/bin/python 8_multimodal/eval_doc.py          # demo syllabus, 20 questions (needs Ollama)
.venv/bin/python -m pytest 8_multimodal/tests      # offline, fake models
```

Demo syllabus, 20 questions (Nov 19 2026, gemma3:12b, M3 Max): answerable 10/10 answered directly with the right page cited. Unanswerable: 8/10 said "not in the document", 2/10 flagged as partial, 0 made up. Median 4.7 s per question.
**Caveat:** the prompts were tuned on these same 20 questions, and Claude wrote them. A held-out document and question set are still needed before this is a real number.

## Image (demo whiteboard, `demo/whiteboard_observer.jpg`)

The description read every line of the handwriting, the UML and the code (16 s). Follow-up questions take about 3 s. "What color is the professor's shirt?" gets "The image doesn't show that."

## Code: diagrams from a prompt (`mm_diagram.py`)

In a meeting or lecture, people rarely need code that runs, but they do need to plan. So the code input writes
**Mermaid diagram code**: a flowchart, sequence, UML class, state, ER, Gantt, mind map or timeline. The student can see it, edit it and paste it into GitHub or Notion.

1. **Choose** (qwen2.5-coder:7b, a small JSON reply): the diagram type, a title and a one-line note.
2. **Write** (the same model, plain text): shown only that type's example, on an unrelated topic (a coffee shop), plus that type's rules.
3. **Check, locally**: `lint()` catches mistakes the parser lets through, then the official Mermaid parser runs in Node (`mermaid_check/`, with jsdom, no browser and no network).
4. **Repair**: on an error, the model gets its code and the error back and tries again, up to 3 tries.

A whiteboard photo is first read by gemma3 (`mm_image.describe`), then drawn by the code model.

```bash
(cd 8_multimodal/mermaid_check && npm ci)            # once: mermaid 11.4.1 + jsdom
.venv/bin/python 8_multimodal/eval_diagram.py        # 5 requests (needs Ollama)
```

5 requests (Nov 20 2026, M3 Max): a typed flowchart, a typed sequence diagram, meeting → Gantt plan, meeting → mind map, whiteboard → UML.
**5/5 valid on the first try, 3–6 s each (21 s with the whiteboard).** The sequence, mind map and UML diagrams are right. The Gantt plan has the right owners and order, but starts Sam's task on its due date. The flowchart's logic is muddled (a question in a box, a "Not applicable" branch).
What changed along the way: with gemma3:12b and all eight examples in one prompt, it was 2/5 valid, up to 316 s, and it copied example content into the diagrams.
**Caveat:** the same 5 requests were used for tuning. Mistakes in the content are why the interface shows the code next to the drawing and lets the student edit it.
