"""Local speech-to-text server: the same API as Groq's Whisper endpoint, running on this machine.

The bridge (1_ingestion/bridge.py) posts each audio piece to an OpenAI-style /v1/audio/transcriptions endpoint
and reads `verbose_json` (segments with no_speech_prob). This server answers that exact request with
faster-whisper (open weights, CTranslate2), so the bridge only needs a different URL: nothing leaves the machine.

    .venv/bin/python local_models/stt_server.py                 # http://127.0.0.1:8777, model large-v3-turbo
    STT_MODEL=small.en .venv/bin/python local_models/stt_server.py   # smaller / faster model

Then set STT_BACKEND=local in the repo's .env (the default). The first request downloads the model (~1.6 GB for
large-v3-turbo) from Hugging Face; after that it works offline.
"""

import asyncio
import io
import json
import os
import platform
import sys
import time
import wave

from aiohttp import web

HOST = os.environ.get("STT_HOST", "127.0.0.1")   # 0.0.0.0 if the bridge runs in Docker (see local_models/README.md)
PORT = int(os.environ.get("STT_PORT", 8777))
MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
# Apple Silicon: the GPU (mps) is ~3x faster than CPU (5.7 s clip: 0.83 s vs 2.6 s on an M3 Max, Nov 2026).
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
DEVICE = os.environ.get("STT_DEVICE", "mps" if APPLE_SILICON else "auto")   # mps / cpu / cuda / auto
COMPUTE = os.environ.get("STT_COMPUTE", "default")   # the model's own type; int8 was slower on the M3 Max CPU

_model = None
_lock = asyncio.Lock()   # one transcription at a time: the model isn't shared across threads safely


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel   # imported here so --help works without it
        t = time.monotonic()
        _model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
        print(f"loaded whisper {MODEL} on {DEVICE} in {time.monotonic() - t:.1f}s", flush=True)
    return _model


def decode(audio_bytes, rate=16000):
    """Audio bytes -> 16 kHz mono float32 numpy array, which faster-whisper takes as is.
    WAV (what the bridge sends) with the standard library; anything else through PyAV, if it works."""
    import numpy as np
    if audio_bytes[:4] != b"RIFF":
        return io.BytesIO(audio_bytes)            # mp3, m4a, ...: faster-whisper decodes it with PyAV
    with wave.open(io.BytesIO(audio_bytes)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"only 16-bit WAV is supported, got {8 * w.getsampwidth()}-bit")
        channels, source_rate = w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if source_rate != rate and len(pcm):          # linear resampling: fine for speech recognition
        n = int(round(len(pcm) * rate / source_rate))
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


def transcribe(audio_bytes, language=None, prompt=None, temperature=0.0, word_timestamps=False):
    """faster-whisper -> the OpenAI/Groq verbose_json shape (text, language, duration, segments)."""
    segments, info = get_model().transcribe(
        decode(audio_bytes), language=language or None, initial_prompt=prompt or None,
        temperature=temperature, beam_size=5, vad_filter=False, word_timestamps=word_timestamps)
    out = []
    for i, s in enumerate(segments):
        seg = {"id": i, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text,
               "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio,
               "no_speech_prob": s.no_speech_prob, "temperature": s.temperature}
        if word_timestamps and s.words:
            seg["words"] = [{"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)} for w in s.words]
        out.append(seg)
    return {"task": "transcribe", "language": info.language, "duration": round(info.duration, 2),
            "text": " ".join(" ".join(s["text"] for s in out).split()), "segments": out}


async def transcriptions(request):
    try:
        form = await request.post()
    except Exception as e:
        return web.json_response({"error": {"message": f"bad form: {e}"}}, status=400)
    upload = form.get("file")
    if upload is None or not hasattr(upload, "file"):
        return web.json_response({"error": {"message": "missing 'file'"}}, status=400)
    audio = upload.file.read()
    fmt = form.get("response_format", "json")
    started = time.monotonic()
    try:
        async with _lock:
            result = await asyncio.to_thread(
                transcribe, audio, form.get("language"), form.get("prompt"),
                float(form.get("temperature") or 0), form.get("timestamp_granularities[]") == "word")
    except Exception as e:                        # unreadable audio etc.: a clear 400, not a bare 500
        return web.json_response({"error": {"message": f"{type(e).__name__}: {e}"}}, status=400)
    headers = {"x-local-stt-ms": str(int((time.monotonic() - started) * 1000))}
    if fmt == "text":
        return web.Response(text=result["text"], headers=headers)
    if fmt == "json":
        return web.json_response({"text": result["text"]}, headers=headers)
    return web.json_response(result, headers=headers, dumps=lambda o: json.dumps(o, ensure_ascii=False))


async def health(request):
    return web.Response(text="ok")


def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/audio/transcriptions", transcriptions)
    app.router.add_post("/openai/v1/audio/transcriptions", transcriptions)   # Groq's path, too
    app.router.add_get("/health", health)
    return app


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    get_model()   # load at start, so the first meeting line isn't slow
    print(f"local STT: http://{HOST}:{PORT}/v1/audio/transcriptions  (model {MODEL})", flush=True)
    web.run_app(make_app(), host=HOST, port=PORT, print=None)
