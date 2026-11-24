"""Offline tests for the local speech-to-text server: a fake Whisper model stands in, so nothing downloads.

    .venv/bin/python -m pytest local_models/tests
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "1_ingestion"))
import stt_server  # noqa: E402


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        segs = [SimpleNamespace(start=0.0, end=1.5, text=" Hello there.", avg_logprob=-0.2, compression_ratio=1.1,
                                no_speech_prob=0.01, temperature=0.0, words=None),
                SimpleNamespace(start=1.5, end=2.0, text=" Thank you.", avg_logprob=-1.5, compression_ratio=1.0,
                                no_speech_prob=0.9, temperature=0.0, words=None)]
        return iter(segs), SimpleNamespace(language="en", duration=2.0)


def wav(seconds=0.5, rate=8000, channels=2):
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(seconds * rate) * channels)
    return buf.getvalue()


def test_decode_wav_to_16k_mono():
    pcm = stt_server.decode(wav(seconds=0.5, rate=8000, channels=2))
    assert pcm.dtype.name == "float32" and pcm.ndim == 1 and len(pcm) == 8000


@pytest.fixture
def fake(monkeypatch):
    model = FakeWhisper()
    monkeypatch.setattr(stt_server, "_model", model)
    return model


def test_verbose_json_is_what_the_bridge_reads(fake):
    """The bridge's spoken_text() must keep real speech and drop the low-confidence hallucination."""
    out = stt_server.transcribe(wav(), language="en", prompt="earlier words")
    assert out["text"] == "Hello there. Thank you." and out["language"] == "en"
    assert [s["no_speech_prob"] for s in out["segments"]] == [0.01, 0.9]
    assert fake.calls[0]["initial_prompt"] == "earlier words" and fake.calls[0]["language"] == "en"
    import bridge
    text, dropped = bridge.spoken_text(json.dumps(out))
    assert text == "Hello there." and dropped == ["Thank you."]


async def test_endpoint_speaks_the_openai_form(fake, aiohttp_client):
    import aiohttp
    client = await aiohttp_client(stt_server.make_app())
    form = aiohttp.FormData()
    form.add_field("file", wav(), filename="piece.wav", content_type="audio/wav")
    for k, v in {"model": "whisper-large-v3-turbo", "language": "en", "response_format": "verbose_json",
                 "temperature": "0", "prompt": "hi"}.items():
        form.add_field(k, v)
    r = await client.post("/v1/audio/transcriptions", data=form)
    assert r.status == 200
    body = await r.json()
    assert body["segments"][0]["text"] == " Hello there." and "x-local-stt-ms" in r.headers
    r = await client.post("/v1/audio/transcriptions", data=aiohttp.FormData({"x": "1"}))
    assert r.status == 400
