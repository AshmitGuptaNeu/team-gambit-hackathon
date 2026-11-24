# Demo walkthrough: LiveMinutes Local

A step-by-step script to show the whole project: what to click, what you'll see, and what to say. Every step below was run on this laptop on Nov 20 2026, and the answers and times shown are from that run. What the project is: [README.md](README.md). How it works inside: [8_multimodal/README.md](8_multimodal/README.md).

**The story:** a deaf CS student goes through one day: a lecture, a syllabus question, a whiteboard, a group meeting. One laptop helps with all of it, and nothing leaves the laptop.

**Time:** about 6 minutes for the core demo (steps 1–9). Add 2 minutes for the optional meeting bot (step 11).

---

## Before you start (30 minutes before)

```bash
cd ~/Desktop/liveminutes-work/hackathon
ollama list                                   # must show gemma3:12b and qwen2.5-coder:7b
MM_WARM=1 .venv/bin/python 8_multimodal/app.py  # leave this terminal open
```
Open **http://127.0.0.1:8780** in Chrome. The top right should read `local · whisper large-v3-turbo · gemma3:12b · qwen2.5-coder:7b`. If it's red, Ollama isn't running: open the Ollama app, or run `ollama serve`.

**Warm up once.** The first call to each model loads it into memory and is slow. That way none of that happens on stage.
1. Attach `8_multimodal/demo/whiteboard_observer.jpg` and wait for the description.
2. Switch to **Draw a diagram** and type `a flowchart of making tea`.
3. Press **Clear** in the captions panel, then reload the page.

**Checklist:**
- [ ] Chrome has microphone permission for 127.0.0.1. Press **Start captions** once and allow it.
- [ ] The demo files are easy to reach, all in `8_multimodal/demo/`: `group_meeting.wav`, `syllabus_cs3500.pdf`, `whiteboard_observer.jpg`.
- [ ] Do Not Disturb is on, and other apps are closed (the models use about 13 GB of the 36 GB).
- [ ] If you'll use the meeting bot: Docker Desktop is running and `bash 1_ingestion/start_bot.sh` printed `Ready`. See step 11.

---

## The demo

### 1. Live captions (about 30 s)
**Do:** click **Start captions** (left panel), say a sentence out loud, and set the size to **L**.
**You'll see:** the level bar moves while you talk, and your sentence appears a few seconds later.
**Say:** *"Whisper large-v3-turbo runs on this laptop's GPU. A 6-second sentence is transcribed in under a second. The level bar is there because the student can't hear whether the mic is working."*
Click **Stop captions** when done.

### 2. A recording becomes captions (about 15 s)
**Do:** **Attach…** → `demo/group_meeting.wav` (a 17-second group meeting).
**You'll see** (about 1.3 s), word for word, as five caption lines:
> Okay everyone, we decided to use Postgres for the database. · Sam, can you write the API endpoints by Friday? · Dana will draft the design doc for the observer pattern by Wednesday. · I'll set up the CI pipeline next week. · And remember, homework 5 is due on Friday at midnight.

**Say:** *"It works on a recording too, if the student missed the meeting."*

### 3. Who's doing what: the knowledge graph (instant)
**Do:** click **Who's doing what** (captions toolbar).
**You'll see:** a graph: Sam → owns → "write the API endpoints" → *by Friday*; Dana → owns → the design doc; plus a *Decisions* box (Postgres) and a *Deadlines* box (homework 5, *on Friday*). The note reads "1 decision, 3 tasks, 1 deadline, 2 people".
**Say:** *"This is a small knowledge graph: people, decisions, tasks and deadlines, each linked to the moment it was said. It's built by rules, with no AI model, so every box is a real sentence and nothing can be made up."*

### 4. Questions about the captions (instant to 8 s)
Type these into the right panel, one at a time (**Ask a question** mode):

| Type | You'll see |
|---|---|
| `What is Sam doing?` | **From the decisions & tasks graph**: "Task [00:04]: Sam, can you write the API endpoints by Friday? (owner: Sam, when: by Friday)", with the quote. Instant. |
| `When is homework 5 due?` | "Due [00:13]: … due on Friday at midnight." Instant. |
| `Who is setting up the CI pipeline?` | Not in the graph (mic captions have no speaker names), so it searches the captions: "I will set up the CI pipeline next week." About 8 s. |
| `What did the professor say about the midterm?` | ✕ **"That wasn't said in the captions so far."** |

**Say (on the last one):** *"This is the most important answer in the demo. It doesn't guess."*

### 5. A diagram from the meeting (about 15–30 s)
**Do:** switch to **Draw a diagram** and type `turn this into a timeline with owners`.
**You'll see:** a Gantt chart with a section per person and the homework as a milestone. Open **Mermaid code** to show it can be edited; press **Download SVG**.
**Say:** *"In a meeting you don't need code that runs, you need a plan you can see. The model writes Mermaid code. The official Mermaid parser checks it on this laptop, and if it's broken, the model gets the error and fixes it."*
⚠️ **Dates can be off** (it guessed which Friday). Say so, then fix one in the code box and press **Redraw**: it shows why the code is editable.

### 6. The syllabus (about 6 s per question)
**Do:** switch back to **Ask a question**. **Attach…** → `demo/syllabus_cs3500.pdf` (3 pages, indexed in 0.4 s).

| Type | You'll see |
|---|---|
| `Can I use late days on the final project?` | "No, late days cannot be used for the final project.", with the quote and **page 2** |
| `How do I get live captioning for lectures?` | "Register with the Disability Resource Center and send the instructor your accommodation letter…", **page 1** |
| `What room is the midterm in?` | **Partly answered**: "The document doesn't say exactly that. The closest it gets: 'Midterm exam on Wednesday October 21, in class.' (page 3)" |

**Say:** *"Three checks before any answer is shown. Is the question about this document at all? Is every quote really in the document, word for word, including 'may' versus 'cannot'? Does it answer this exact question? On our test syllabus: 10 out of 10 answered with the right page, and nothing made up on 10 it can't answer."*

### 7. The whiteboard photo (about 19 s, then about 3 s per question)
**Do:** remove the PDF (✕ on the chip). **Attach…** → `demo/whiteboard_observer.jpg`.
**You'll see:** a description that reads all the handwriting: the Observer UML (Subject, Observer, CaptionView), the `notifyAll()` code, and "HW5 due Fri Nov 6".

| Type | You'll see |
|---|---|
| `When is HW5 due?` | HW5 is due "Fri Nov 6". |
| `What colour is the professor's shirt?` | **"The image doesn't show that."** |

### 8. Whiteboard → clean UML (about 25 s)
**Do:** with the photo still attached, switch to **Draw a diagram** and type `turn the whiteboard into a UML class diagram`.
**You'll see:** a clean class diagram (Observer interface, Subject, CaptionView) that can be edited and pasted into GitHub.
**Say:** *"Two models together: Gemma reads the photo, Qwen Coder writes the diagram."*

### 9. Turn the Wi-Fi off (the strongest moment)
**Do:** click the Wi-Fi icon → **off**. Remove the attachment, switch to **Ask a question**, type `What is the observer pattern, in two sentences?`
**You'll see:** an answer in about 3 s, labelled **General answer · not from your captions, photos or documents**.
**Say:** *"No internet. Every model you just saw runs on this MacBook Pro: an Apple M3 Max with 36 gigabytes of memory, no cloud, no API keys. The student's lectures and private conversations never leave their own laptop."*
**Turn the Wi-Fi back on** before step 11.

### 10. Close (say this)
*"We didn't train anything. We built the system around open models that knows the difference between what it heard, what it saw, and what it doesn't know."*

---

## 11. Optional: the meeting bot joins a Google Meet call

Use **Google Meet only**: the bot can't join Teams on this Mac (its Chrome is x86-only and runs under Rosetta, and the Teams web app never finishes loading). **Rehearse this at least once before the demo.** It hasn't been tested end to end on Meet yet.

**Setup (before the demo):**
```bash
open -a Docker                       # wait until Docker Desktop says it's running
bash 1_ingestion/start_bot.sh        # waits for the bridge and prints "Ready"
```
Create a Meet with a **personal Gmail** account and set **Host controls → Meeting access → Open**, so the bot can join without being let in.

**Do:**
1. Click **Join a meeting…** → paste the Meet link → **Send the bot**.
2. The page shows the bot's status: joining, then asking for permission.
3. In the Meet chat, the bot posts: *"LiveMinutes is here to take notes. Reply yes to allow recording, or no to stop."*
4. Everyone types **yes**. The page shows each person's name turn into a ✓.
5. Talk. Captions appear on the left with each **speaker's name**. Then press **Who's doing what**: with real names, "I'll do it" now gets an owner.
6. **Remove the bot** when done.

**Say:** *"Nothing is recorded until every person in the call says yes. A new person joining pauses it, and one 'no' stops it and deletes everything."*

---

## Questions judges may ask (short, honest answers)

| Question | Answer |
|---|---|
| Did you train or fine-tune anything? | No. Off-the-shelf open-weight models. What we built is the system around them: retrieval, the answer checks, diagram validation and repair, the knowledge graph, consent. |
| Is it really local? | Yes. Whisper, Gemma 3 12B, Qwen 2.5 Coder 7B, bge-small and the Mermaid parser all run on this laptop. The page is only reachable from this computer (127.0.0.1). |
| How do you stop it making things up? | Documents: three checks (on-topic retrieval, quotes verified word for word, a second call that grades whether the quote answers this exact question). Images: told to say "the image doesn't show that". Graph: rules only, every item is a real line. |
| How did you measure it? | Demo sets: syllabus 20 questions (10/10 right page, 0 made up), diagrams 5/5 valid. The prompts were tuned on those same sets, so they're demo results, not benchmarks. The LiveMinutes AMI results: BART beat TextRank on every ROUGE score, and search finds the right meeting moment in the top 5 for 79% of questions (team's own meetings). |
| Is the knowledge graph Neo4j? | No. A rule-based graph of people, decisions, tasks and deadlines, built from the captions. An LLM-built Neo4j graph was compared earlier in LiveMinutes and rejected as slow and unreliable on speech. |
| Is there a vector store? | The document and captions are embedded (bge-small) and searched with BM25 plus vectors merged with RRF, in memory. Saving past meetings to a store on disk isn't built yet. |
| Video? | Not built. We focused on audio, image, documents and diagrams. |
| What hardware? | MacBook Pro, Apple M3 Max (14-core CPU, 30-core GPU), 36 GB of memory shared by the CPU and GPU, macOS 14. |

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Top right is red / "Ollama is not reachable" | Open the Ollama app or run `ollama serve`, then reload the page |
| The first answer or diagram takes 30+ s | A model is loading. Do the warm-up above, and don't switch models right before going on stage |
| No captions from the mic | Chrome's address bar → mic icon → allow. Check that the level bar moves |
| Captions show "Thank you." on silence | It's dropped automatically. If it appears anyway, it's Whisper guessing on noise: press **Clear** |
| Diagram says "needs a fix" | Open the code box: it shows the parser's error. Edit and **Redraw**, or ask again in fewer words |
| "reply cut off at max_tokens" | Fixed on Nov 20: the diagram writer now retries shorter. If you still see it, ask for a smaller diagram |
| The page says the bot isn't running | `bash 1_ingestion/start_bot.sh` (Docker Desktop must be open) |
| The bot never joins | Use Google Meet, not Teams. Set the Meet's access to **Open**. Check `cd 1_ingestion/attendee && docker compose -f dev.docker-compose.yaml -f ../compose.local.yaml logs --tail 50 attendee-worker-local` |
| Anything else | The server's terminal prints the full error. Restart with `Ctrl+C`, then `MM_WARM=1 .venv/bin/python 8_multimodal/app.py` |

**Reset between run-throughs:** press **Clear** (captions), remove the attachment (✕), and reload the page.
**Shut down:** `Ctrl+C` in the app's terminal; `bash 1_ingestion/start_bot.sh stop` for the bot.
