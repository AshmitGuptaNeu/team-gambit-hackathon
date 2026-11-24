"""Offline tests for image input (no Ollama): the downscaling, and what is sent to the model."""
import base64
import io
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mm_image  # noqa: E402
import mm_llm  # noqa: E402


def png(w, h):
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), (255, 0, 0, 128)).save(buf, format="PNG")
    return buf.getvalue()


def test_big_photos_are_downscaled_to_jpeg():
    out = Image.open(io.BytesIO(base64.b64decode(mm_llm.prepare_image(png(4000, 3000), max_side=1280))))
    assert out.format == "JPEG" and max(out.size) == 1280 and out.size == (1280, 960)


def test_describe_sends_the_image_and_schema(monkeypatch):
    sent = {}
    monkeypatch.setattr(mm_llm, "ask", lambda prompt, **kw: sent.update(kw) or {"kind": "whiteboard"})
    assert mm_image.describe(b"img")["kind"] == "whiteboard"
    assert sent["images"] == [b"img"] and sent["schema"] is mm_image.DESCRIBE_SCHEMA


def test_follow_up_questions_carry_the_earlier_ones(monkeypatch):
    seen = []
    monkeypatch.setattr(mm_llm, "ask", lambda prompt, **kw: seen.append(prompt) or "ok")
    mm_image.ask(b"img", "and the second box?", history=[("what is the title?", "Observer pattern")])
    assert "what is the title?" in seen[0] and seen[0].endswith("Question: and the second box?")
