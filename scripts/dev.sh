#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/uvicorn || ! -d apps/web/node_modules ]]; then
  echo 'Run ./scripts/setup.sh first.' >&2
  exit 1
fi
export PYTHONPATH="$PWD/apps/api"
.venv/bin/uvicorn surv.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
cleanup(){ kill "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for attempt in {1..50}; do
  if curl --silent --fail http://127.0.0.1:8000/api/v1/health >/dev/null; then break; fi
  sleep 0.1
done
npm --prefix apps/web run dev
