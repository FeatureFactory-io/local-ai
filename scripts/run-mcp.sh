#!/bin/bash
# Start the Local AI dispatcher on stdio for Cursor or Claude Desktop.
# The working directory is the repo root so logs/app.log is created here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m local_ai
