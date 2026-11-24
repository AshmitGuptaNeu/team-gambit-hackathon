# Local models

Everything the hackathon build needs runs on this machine with open-weight models. Nothing is sent to a hosted API.

| Job | Model | Runs in | Used by |
|---|---|---|---|
| Speech to text | Whisper `large-v3-turbo` (faster-whisper / CTranslate2) | `stt_server.py`, port 8777 | `1_ingestion/bridge.py` (`STT_BACKEND=local`, the default) |
| Notes, answers (and later: images, video frames) | `gemma3:12b` (Ollama) | Ollama, port 11434 | `2_summarization/llm_providers.py` `OllamaProvider` (the default provider) |
| Retrieval embeddings | `BAAI/bge-small-en-v1.5` | in-process (transformers) | `4_qa/` |

## Setup (once)

```bash
# Python packages (repo root): requirements.txt section 7 is faster-whisper
.venv/bin/pip install -r requirements.txt

# Ollama: https://ollama.com/download, or on macOS
brew install ollama
ollama serve &                 # or open the Ollama app
ollama pull gemma3:12b         # ~8 GB, once
```

## Run

```bash
.venv/bin/python local_models/stt_server.py       # downloads Whisper (~1.6 GB) the first time
curl http://127.0.0.1:8777/health                  # -> ok
```

Check it with a spoken test clip (macOS):

```bash
say -o /tmp/t.wav --data-format=LEI16@16000 "Priya will send the deck by Friday."
curl -s -F file=@/tmp/t.wav -F response_format=verbose_json http://127.0.0.1:8777/v1/audio/transcriptions
```

Settings (environment variables): `STT_MODEL` (e.g. `small.en` for a smaller machine), `STT_DEVICE` (`mps` on Apple Silicon by default, `cpu`, `cuda`), `STT_PORT`, `STT_HOST`. Model choice for notes: `LIVEMINUTES_LLM_MODEL` in `.env`.

## Bridge in Docker

The bridge container reaches the host as `host.docker.internal` (set in `1_ingestion/compose.local.yaml`). Docker Desktop forwards that to the host's localhost. With Colima or on Linux, start the servers so they listen beyond localhost: `STT_HOST=0.0.0.0 .venv/bin/python local_models/stt_server.py` and `OLLAMA_HOST=0.0.0.0 ollama serve`.

## Measured (Apple M3 Max, 36 GB, Nov 2026)

| | Time for a 5.7 s clip |
|---|---|
| Whisper large-v3-turbo, `mps` | 0.83 s |
| same, CPU float32 | 2.6 s |
| same, CPU int8 | 4.9 s |

Tests (no model download): `.venv/bin/python -m pytest local_models/tests`
