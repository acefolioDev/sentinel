#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
fi
"$PYTHON_BIN" -m venv .venv
.venv/bin/pip install -r requirements.txt
npm --prefix apps/web install
printf 'Setup complete. Run ./scripts/dev.sh\n'
