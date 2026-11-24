#!/usr/bin/env bash
# Start the meeting bot: the Docker VM (Colima, if Docker isn't up), Attendee and the bridge, then check them.
# One-time setup first: docs/INGESTION.md section 1.4 (clone Attendee, .env, build, certs, account).
#   bash 1_ingestion/start_bot.sh          start
#   bash 1_ingestion/start_bot.sh stop     stop the containers (and the Colima VM)
# Speech-to-text and the live notes stay on this Mac (not in Docker): run 8_multimodal/app.py (or
# local_models/stt_server.py) and `ollama serve`. The bridge reaches them as host.docker.internal.
set -euo pipefail
cd "$(dirname "$0")"
# Docker Desktop's tools (its credential helper is needed to pull images), then ~/.local/bin (Colima, if used)
export PATH="/Applications/Docker.app/Contents/Resources/bin:$HOME/.local/bin:$PATH"
dc() { (cd attendee && docker compose -f dev.docker-compose.yaml -f ../compose.local.yaml "$@"); }

if [ "${1:-}" = "stop" ]; then
  dc stop
  exit 0
fi

[ -d attendee ] || { echo "attendee/ is missing: do the one-time setup in docs/INGESTION.md first"; exit 1; }
if [ -d /Applications/Docker.app ]; then
  docker context use desktop-linux >/dev/null 2>&1 || true
  if ! docker info >/dev/null 2>&1; then
    echo "starting Docker Desktop..."; open -a Docker
    for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 3; done
  fi
elif ! docker info >/dev/null 2>&1 && command -v colima >/dev/null; then
  echo "starting the Docker VM (Colima)..."; colima start
fi
docker info >/dev/null 2>&1 || { echo "Docker isn't running: open Docker Desktop"; exit 1; }

dc up -d postgres redis attendee-app-local attendee-worker-local attendee-scheduler-local liveminutes-bridge
printf "waiting for the bridge"
for _ in $(seq 1 60); do
  if curl -fsS -m 2 http://127.0.0.1:8776/health >/dev/null 2>&1; then echo " ok"; break; fi
  printf "."; sleep 2
done
curl -fsS -m 2 http://127.0.0.1:8776/health >/dev/null || { echo " the bridge didn't come up: dc logs liveminutes-bridge"; exit 1; }
curl -fsS -m 2 -o /dev/null http://localhost:8000/accounts/login/ && echo "Attendee: http://localhost:8000"
curl -fsS -m 2 -o /dev/null http://127.0.0.1:8777/health 2>/dev/null \
  && echo "speech-to-text: up on :8777" || echo "speech-to-text isn't up yet: start .venv/bin/python 8_multimodal/app.py"
echo "Ready: paste a meeting link under 'Join a meeting' at http://127.0.0.1:8780"
