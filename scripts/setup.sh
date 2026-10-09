#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
fi
if ! command -v psql >/dev/null 2>&1; then
  echo 'PostgreSQL is required. Install it and start the server first.' >&2
  exit 1
fi
if ! psql -d postgres -Atqc 'SELECT 1' >/dev/null 2>&1; then
  echo 'PostgreSQL is not accepting local connections.' >&2
  exit 1
fi
if [[ "$(psql -d postgres -Atqc "SELECT 1 FROM pg_database WHERE datname='surveillance_qa'")" != '1' ]]; then
  createdb surveillance_qa
fi
psql -d surveillance_qa -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION IF NOT EXISTS vector' >/dev/null
"$PYTHON_BIN" -m venv .venv
.venv/bin/pip install -r requirements.txt
npm --prefix apps/web install
printf 'Setup complete. Add GEMINI_API_KEY to .env, then run ./scripts/dev.sh\n'
