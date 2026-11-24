"""The one interface: http://127.0.0.1:8780, one page for a deaf or hard-of-hearing student. Everything runs on
this machine (Whisper, gemma3, qwen2.5-coder, bge-small, the Mermaid parser and mermaid.js).

    .venv/bin/python 8_multimodal/app.py          (Ollama must be running: `ollama serve`)

  GET  /                    the page (static/index.html)
  GET  /api/status          Ollama running, models pulled?
  GET  /static/...          the page's JS/CSS; mermaid.min.js comes from mermaid_check/node_modules
  POST /api/transcribe      body: a 16 kHz WAV piece from the mic -> {"text", "segments"}  (live captions)
  POST /api/attach          multipart "file": image -> described; PDF/txt/md -> indexed; WAV -> transcribed
  POST /api/ask             JSON {"text", "mode": "ask"|"diagram", "attachment", "captions", "history"}
  POST /api/graph           JSON {"captions"}: the decisions / tasks / deadlines graph (mm_graph.py) as Mermaid
  POST /api/meeting/join    JSON {"link"}: the meeting bot (1_ingestion, in Docker) joins a Meet/Teams call and
                            asks everyone in chat for permission; nothing is transcribed until all say yes
  GET  /api/meeting/<bot>   that bot's state, who has said yes, and the transcript (?v=: long poll)
  POST /api/meeting/<bot>/leave   the bot leaves the call (Attendee API, key from the repo's .env)
  POST /v1/audio/transcriptions   the OpenAI-style speech-to-text the bridge calls (local_models/stt_server.py's
                            handler), also on port 8777: one Whisper process serves the mic AND the meeting

What /api/ask answers from, in order: diagram mode -> diagram code (with the captions and/or the attached image
as context); an attached image -> the image; an attached document -> the document (says when it isn't there);
captions so far -> first the knowledge graph over them (who owns what, what was decided, what is due: instant,
quotes the line), else a search of the captions, with the same "isn't there" checks; nothing -> a general answer, labelled so.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import uuid

from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "local_models"))
import mm_diagram  # noqa: E402
import mm_document  # noqa: E402
import mm_graph  # noqa: E402
import mm_image  # noqa: E402
import mm_llm  # noqa: E402

HOST = os.environ.get("MM_HOST", "127.0.0.1")      # this computer only
PORT = int(os.environ.get("MM_PORT", 8780))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")   # as in llm_providers.py
BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8776").rstrip("/")      # the meeting bot's bridge
ATTENDEE_URL = os.environ.get("ATTENDEE_URL", "http://localhost:8000").rstrip("/")  # "localhost": its ALLOWED_HOSTS
STT_PORT = int(os.environ.get("STT_PORT", 8777))   # where compose.local.yaml points the bridge's speech-to-text
BOT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STATIC = os.path.join(HERE, "static")
MERMAID_JS = os.path.join(HERE, "mermaid_check", "node_modules", "mermaid", "dist", "mermaid.min.js")
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".bmp")
DOC_TYPES = (".pdf", ".txt", ".md")
# Whisper writes "Thank you." over silence; the meeting bridge drops the same things (1_ingestion/settings.py).
HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "."}

log = logging.getLogger("mm_app")
attachments = {}      # id -> {"kind": "image"|"document", "name", "bytes" | "doc"}; one local user, in memory
caption_docs = {}     # hash of the captions text -> Document over it (re-indexed only when the captions change)
_llm_lock = asyncio.Lock()   # one model call at a time: Ollama would queue them anyway, this keeps the order


def keep_segment(seg):
    text = seg.get("text", "").strip()
    no_speech = seg.get("no_speech_prob", 0)
    return text and not (no_speech > 0.6 or (text.lower() in HALLUCINATIONS and no_speech > 0.3))


async def transcribe(wav, prompt=None):
    """prompt: the previous caption, so a piece that starts mid-sentence is read in context."""
    import stt_server
    async with stt_server._lock:   # the same lock as the bridge's requests: one transcription at a time
        out = await asyncio.to_thread(stt_server.transcribe, wav, "en", prompt or None)
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in out["segments"] if keep_segment(s)]
    return {"text": " ".join(s["text"] for s in segments), "segments": sentences(segments),
            "duration": out["duration"]}


def sentences(segments):
    """Whisper's segments end mid-sentence ("...for the caption" / "feed. The subject..."): regroup the text into
    whole sentences, each stamped with the time of the segment it starts in. Easier to read as captions."""
    out, text, start = [], "", None
    for seg in segments:
        for piece in re.split(r"(?<=[.!?])\s+", seg["text"]):
            if not piece:
                continue
            if start is None:
                start = seg["start"]
            text = f"{text} {piece}".strip()
            if re.search(r"[.!?]$", piece):
                out.append({"start": start, "end": seg["end"], "text": text})
                text, start = "", None
    if text:
        out.append({"start": start, "end": segments[-1]["end"], "text": text})
    return out


async def run_llm(fn, *args, **kw):
    async with _llm_lock:
        return await asyncio.to_thread(fn, *args, **kw)


def error(message, status=400):
    return web.json_response({"error": message}, status=status)


# ---------------- routes ----------------

async def index(request):
    return web.FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})


async def mermaid_js(request):
    if not os.path.exists(MERMAID_JS):
        return web.Response(status=404, text="run: (cd 8_multimodal/mermaid_check && npm ci)")
    return web.FileResponse(MERMAID_JS)


async def api_status(request):
    """Is Ollama up, with both models pulled? (Whisper loads on the first caption.)"""
    import urllib.request
    import mm_settings as S
    wanted = [S.VISION_MODEL, S.CODE_MODEL]
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as r:
            have = {m["name"] for m in json.load(r)["models"]}
    except OSError:
        return web.json_response({"ok": False, "problem": "Ollama isn't running: start it with `ollama serve`"})
    missing = [m for m in wanted if m not in have]
    if missing:
        return web.json_response({"ok": False, "problem": "not pulled: " + ", ".join(missing)})
    return web.json_response({"ok": True, "models": ["whisper large-v3-turbo", *wanted]})


async def api_transcribe(request):
    wav = await request.read()
    if wav[:4] != b"RIFF":
        return error("send a WAV piece")
    return web.json_response(await transcribe(wav, request.query.get("prompt", "")[-500:]))


async def api_attach(request):
    form = await request.post()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "file"):
        return error("no file")
    name, data = upload.filename or "file", upload.file.read()
    ext = os.path.splitext(name.lower())[1]
    aid = uuid.uuid4().hex[:12]
    if ext in IMAGE_TYPES or (upload.content_type or "").startswith("image/"):
        description = await run_llm(mm_image.describe, data)
        attachments[aid] = {"kind": "image", "name": name, "bytes": data}
        return web.json_response({"id": aid, "kind": "image", "name": name, "description": description})
    if ext in DOC_TYPES:
        try:
            doc = await asyncio.to_thread(mm_document.Document.from_bytes, data, name)
        except ValueError as e:
            return error(str(e))
        attachments[aid] = {"kind": "document", "name": name, "doc": doc}
        return web.json_response({"id": aid, "kind": "document", "name": name, "pages": len(doc.pages),
                                  "chunks": len(doc.chunks)})
    if ext == ".wav" or data[:4] == b"RIFF":   # the page turns mp3/m4a/... into WAV before sending
        out = await transcribe(data)
        return web.json_response({"kind": "audio", "name": name, **out})
    return error(f"can't read {ext or 'this'} files: send a photo, a PDF/.txt/.md document or a recording")


def captions_document(captions):
    key = hashlib.sha1(captions.encode()).hexdigest()
    if key not in caption_docs:
        caption_docs.clear()
        caption_docs[key] = mm_document.Document("lecture captions", [(1, captions)])
    return caption_docs[key]


GENERAL_SYSTEM = ("You help a deaf or hard-of-hearing university student. Answer clearly and briefly, in plain "
                  "written English. You have no lecture, document or image to go on: answer from general knowledge.")


async def api_ask(request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return error("send JSON")
    text = (body.get("text") or "").strip()
    if not text:
        return error("type a question or a diagram request")
    captions = (body.get("captions") or "").strip()
    att = attachments.get(body.get("attachment") or "")
    history = [(h["q"], h["a"]) for h in body.get("history") or [] if "q" in h and "a" in h][-6:]

    if body.get("mode") == "diagram":
        image = att["bytes"] if att and att["kind"] == "image" else None
        r = await run_llm(mm_diagram.generate, text, transcript=captions or None, image=image)
        r.pop("whiteboard", None)
        return web.json_response({"kind": "diagram", **r,
                                  "context": [c for c, on in (("captions", captions), ("image", image)) if on]})
    if att and att["kind"] == "image":
        answer = await run_llm(mm_image.ask, att["bytes"], text, history)
        return web.json_response({"kind": "image", "answer": answer, "source": att["name"]})
    if att and att["kind"] == "document":
        a = await run_llm(att["doc"].ask, text)
        return web.json_response({"kind": "document", "source": att["name"], **a.to_dict()})
    if captions:
        g = mm_graph.answer(text, mm_graph.build(captions))
        if g:
            return web.json_response({"kind": "graph", "source": "the decisions & tasks graph", **g})
        doc = await asyncio.to_thread(captions_document, captions)
        a = await run_llm(doc.ask, text)
        d = a.to_dict()
        if not d["found"]:
            d["answer"] = "That wasn't said in the captions so far."
        return web.json_response({"kind": "captions", "source": "the captions", **d})
    prompt = "".join(f"Earlier question: {q}\nYour answer: {a}\n\n" for q, a in history) + f"Question: {text}"
    answer = await run_llm(mm_llm.ask, prompt, system=GENERAL_SYSTEM, max_tokens=800)
    return web.json_response({"kind": "general", "answer": answer})


async def api_graph(request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return error("send JSON")
    g = mm_graph.build((body.get("captions") or "").strip())
    code = mm_graph.to_mermaid(g)
    counts = {k: sum(n.kind == k for n in g.items) for k in ("decision", "action", "deadline")}
    return web.json_response({
        "kind": "diagram", "diagram_type": "knowledge graph", "title": "Who's doing what", "code": code,
        "valid": True, "tries": 1, "context": ["captions"], "people": g.people,
        "edges": [list(e) for e in g.edges()],
        "notes": (", ".join(f"{n} {w if n == 1 else w + 's'}" for n, w in [(counts["decision"], "decision"),
                  (counts["action"], "task"), (counts["deadline"], "deadline")])
                  + f", {len(g.people)} {'person' if len(g.people) == 1 else 'people'}. Found by phrase rules on the captions (no AI model), so each box is a real line.")})


# ---------------- the meeting bot (1_ingestion/bridge.py, in Docker) ----------------

async def bridge_call(method, path, body=None, timeout=30):
    import aiohttp
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as http:
            async with http.request(method, BRIDGE_URL + path, json=body) as r:
                return r.status, await r.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return 0, None


async def api_meeting_join(request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return error("send JSON")
    status, out = await bridge_call("POST", "/live/api/send", {"link": body.get("link", ""), "name": "LiveMinutes"})
    if status == 0:
        return error("The meeting bot isn't running. Start it with: bash 1_ingestion/start_bot.sh", 503)
    if status != 200:
        return error((out or {}).get("error", f"bridge answered HTTP {status}"), status)
    return web.json_response(out)


async def api_meeting(request):
    bot_id = request.match_info["bot_id"]
    if not BOT_ID.match(bot_id):
        return error("bad bot id")
    v = request.query.get("v")
    status, out = await bridge_call("GET", f"/live/{bot_id}/data" + (f"?v={int(v)}" if v and v.isdigit() else ""),
                                    timeout=40)
    if status != 200 or not isinstance(out, dict):
        return error("The meeting bot isn't reachable" if status == 0 else f"bridge answered HTTP {status}", 502)
    keep = ("bot_id", "state", "live", "recording", "declined", "consent", "people", "transcript", "version")
    return web.json_response({k: out.get(k) for k in keep})


def attendee_key():
    """ATTENDEE_API_KEY from the environment or the repo's .env (make_account.sh writes it there)."""
    if os.environ.get("ATTENDEE_API_KEY"):
        return os.environ["ATTENDEE_API_KEY"]
    try:
        for line in open(os.path.join(REPO, ".env")):
            if line.startswith("ATTENDEE_API_KEY="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


async def api_meeting_leave(request):
    import aiohttp
    bot_id = request.match_info["bot_id"]
    if not BOT_ID.match(bot_id):
        return error("bad bot id")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as http:
            async with http.post(f"{ATTENDEE_URL}/api/v1/bots/{bot_id}/leave",
                                 headers={"Authorization": f"Token {attendee_key()}"}) as r:
                if r.status >= 400:
                    return error(f"Attendee answered HTTP {r.status}", 502)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return error("Attendee isn't reachable", 503)
    return web.json_response({"ok": True})


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except (Exception, SystemExit) as e:   # SystemExit: Ollama down or a model not pulled (llm_providers.check)
        log.exception("request failed")
        return error(str(e) or e.__class__.__name__, status=500)


def make_app():
    app = web.Application(middlewares=[errors], client_max_size=200 * 1024 * 1024)
    app.router.add_get("/", index)
    app.router.add_get("/static/mermaid.min.js", mermaid_js)
    app.router.add_static("/static", STATIC)
    app.router.add_get("/api/status", api_status)
    app.router.add_post("/api/transcribe", api_transcribe)
    app.router.add_post("/api/attach", api_attach)
    app.router.add_post("/api/ask", api_ask)
    app.router.add_post("/api/graph", api_graph)
    app.router.add_post("/api/meeting/join", api_meeting_join)
    app.router.add_get("/api/meeting/{bot_id}", api_meeting)
    app.router.add_post("/api/meeting/{bot_id}/leave", api_meeting_leave)
    import stt_server
    app.router.add_post("/v1/audio/transcriptions", stt_server.transcriptions)
    app.router.add_post("/openai/v1/audio/transcriptions", stt_server.transcriptions)
    app.router.add_get("/health", stt_server.health)
    return app


async def serve():
    """The page on PORT; the speech-to-text routes on STT_PORT too, where the bridge looks for them."""
    runner = web.AppRunner(make_app())
    await runner.setup()
    await web.TCPSite(runner, HOST, PORT).start()
    try:
        await web.TCPSite(runner, HOST, STT_PORT).start()
        log.info("speech-to-text for the meeting bot: http://%s:%d/v1/audio/transcriptions", HOST, STT_PORT)
    except OSError:
        log.warning("port %d is taken (local_models/stt_server.py running?): the bridge will use that", STT_PORT)
    print(f"open http://{HOST}:{PORT}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Load Whisper and the embedder now, on the main thread: the first caption/question isn't slowed by it, and a
    # first-time `transformers` import inside a worker thread failed once (ImportError: AutoModel) in testing.
    if os.environ.get("MM_WARM", "1") == "1":
        import stt_server
        stt_server.get_model()
        mm_document.embedder()
    asyncio.run(serve())
