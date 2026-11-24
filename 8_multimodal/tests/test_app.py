"""The interface's server: routing of /api/ask and /api/attach, with every model faked (no Ollama, no Whisper)."""
import io
import sys
import types

import pytest
from aiohttp import FormData

import app as A
from mm_document import Document


class FakeEmbed:
    """Bag-of-letters vectors: enough for retrieval to run."""
    def _v(self, t):
        import numpy as np
        v = np.array([t.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz"], dtype=float) + 1e-6
        return v / np.linalg.norm(v)

    def passages(self, texts):
        import numpy as np
        return np.stack([self._v(t) for t in texts])

    def query(self, q):
        return self._v(q)


@pytest.fixture
async def client(aiohttp_client, monkeypatch):
    A.attachments.clear(); A.caption_docs.clear()
    monkeypatch.setattr(A.mm_document, "embedder", lambda: FakeEmbed())
    return await aiohttp_client(A.make_app())


def test_sentences_regroups_whisper_segments():
    segs = [{"start": 0.0, "end": 5.0, "text": "Homework 5 is due Friday. Use the observer"},
            {"start": 5.0, "end": 9.0, "text": "pattern. Office hours move to"},
            {"start": 9.0, "end": 10.0, "text": "Thursday"}]
    out = A.sentences(segs)
    assert [s["text"] for s in out] == ["Homework 5 is due Friday.", "Use the observer pattern.",
                                        "Office hours move to Thursday"]
    assert [s["start"] for s in out] == [0.0, 0.0, 5.0]


def test_keep_segment_drops_whisper_silence_text():
    assert not A.keep_segment({"text": "Thank you.", "no_speech_prob": 0.4})
    assert not A.keep_segment({"text": "Real words", "no_speech_prob": 0.9})
    assert A.keep_segment({"text": "Thank you.", "no_speech_prob": 0.1})


async def test_diagram_mode_passes_captions_and_image(client, monkeypatch):
    seen = {}

    def fake_generate(request, transcript=None, image=None):
        seen.update(request=request, transcript=transcript, image=image)
        return {"diagram_type": "flowchart", "title": "T", "notes": "", "code": "flowchart TD\n A-->B",
                "valid": True, "tries": 1, "errors": [], "whiteboard": {"x": 1}}
    monkeypatch.setattr(A.mm_diagram, "generate", fake_generate)
    A.attachments["img1"] = {"kind": "image", "name": "wb.jpg", "bytes": b"JPEG"}
    r = await client.post("/api/ask", json={"text": "draw it", "mode": "diagram", "captions": "[00:00] hi",
                                            "attachment": "img1"})
    body = await r.json()
    assert body["kind"] == "diagram" and body["context"] == ["captions", "image"] and "whiteboard" not in body
    assert seen == {"request": "draw it", "transcript": "[00:00] hi", "image": b"JPEG"}


async def test_ask_routes_to_image_then_document_then_captions_then_general(client, monkeypatch):
    monkeypatch.setattr(A.mm_image, "ask", lambda img, q, h: f"image says {q}")
    A.attachments["img1"] = {"kind": "image", "name": "wb.jpg", "bytes": b"JPEG"}
    r = await (await client.post("/api/ask", json={"text": "what?", "attachment": "img1", "captions": "x"})).json()
    assert r == {"kind": "image", "answer": "image says what?", "source": "wb.jpg"}

    calls = []
    monkeypatch.setattr(Document, "ask", lambda self, q: calls.append((self.name, q)) or
                        types.SimpleNamespace(to_dict=lambda: {"found": False, "answer": "That isn't in this document."}))
    doc = Document("syllabus.txt", [(1, "Homework is due on Fridays.")], embed=FakeEmbed())
    A.attachments["doc1"] = {"kind": "document", "name": "syllabus.txt", "doc": doc}
    r = await (await client.post("/api/ask", json={"text": "room?", "attachment": "doc1"})).json()
    assert r["kind"] == "document" and calls[-1] == ("syllabus.txt", "room?")

    r = await (await client.post("/api/ask", json={"text": "room?", "captions": "[00:00] Office hours move."})).json()
    assert r["kind"] == "captions" and calls[-1][0] == "lecture captions"
    assert r["answer"] == "That wasn't said in the captions so far."

    monkeypatch.setattr(A.mm_llm, "ask", lambda prompt, **kw: "general answer")
    r = await (await client.post("/api/ask", json={"text": "what is OOP?"})).json()
    assert r == {"kind": "general", "answer": "general answer"}


async def test_attach_document_and_unknown_type(client):
    form = FormData(); form.add_field("file", io.BytesIO(b"Late days: three.\fNo final exam."), filename="s.txt")
    r = await (await client.post("/api/attach", data=form)).json()
    assert r["kind"] == "document" and r["pages"] == 2 and r["id"] in A.attachments
    form = FormData(); form.add_field("file", io.BytesIO(b"PK\x03\x04"), filename="slides.zip")
    resp = await client.post("/api/attach", data=form)
    assert resp.status == 400 and "can't read .zip" in (await resp.json())["error"]


async def test_errors_become_json(client, monkeypatch):
    def down(*a, **kw):
        raise SystemExit("Ollama isn't running at http://127.0.0.1:11434")
    monkeypatch.setattr(A.mm_llm, "ask", down)
    resp = await client.post("/api/ask", json={"text": "hi"})
    assert resp.status == 500 and "Ollama" in (await resp.json())["error"]
    assert (await client.post("/api/ask", json={"text": " "})).status == 400


async def test_meeting_join_and_poll_go_through_the_bridge(client, monkeypatch):
    calls = []

    async def fake_bridge(method, path, body=None, timeout=30):
        calls.append((method, path, body))
        if path == "/live/api/send":
            return 200, {"bot_id": "bot_abc", "state": "joining"}
        return 200, {"bot_id": "bot_abc", "state": "joined_recording", "live": True, "recording": False,
                     "people": [{"name": "Priya", "present": True, "decision": None}], "transcript": [],
                     "version": 3, "activity": ["internal"], "notes": {"x": 1}}
    monkeypatch.setattr(A, "bridge_call", fake_bridge)
    r = await (await client.post("/api/meeting/join", json={"link": "meet.google.com/abc-defg-hij"})).json()
    assert r["bot_id"] == "bot_abc" and calls[0][2]["link"] == "meet.google.com/abc-defg-hij"
    r = await (await client.get("/api/meeting/bot_abc?v=2")).json()
    assert calls[1][1] == "/live/bot_abc/data?v=2"
    assert r["people"][0]["name"] == "Priya" and "activity" not in r and "notes" not in r
    assert (await client.get("/api/meeting/..%2Fetc")).status in (400, 404)


async def test_meeting_join_says_how_to_start_the_bot(client, monkeypatch):
    async def down(*a, **kw):
        return 0, None
    monkeypatch.setattr(A, "bridge_call", down)
    resp = await client.post("/api/meeting/join", json={"link": "meet.google.com/abc-defg-hij"})
    assert resp.status == 503 and "start_bot.sh" in (await resp.json())["error"]


def test_attendee_key_comes_from_env_or_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("ATTENDEE_API_KEY", "k1")
    assert A.attendee_key() == "k1"
    monkeypatch.delenv("ATTENDEE_API_KEY")
    (tmp_path / ".env").write_text("STT_BACKEND=local\nATTENDEE_API_KEY=k2\n")
    monkeypatch.setattr(A, "REPO", str(tmp_path))
    assert A.attendee_key() == "k2"
