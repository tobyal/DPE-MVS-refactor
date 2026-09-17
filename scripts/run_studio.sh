#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "Usage: $0 <experiment_root> [dense_root] [port]"; exit 1; fi
ROOT="$1"; DENSE="${2:-}"; PORT="${3:-8765}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ARGS=(--root "$ROOT" --port "$PORT")
[[ -n "$DENSE" ]] && ARGS+=(--dense "$DENSE")
python3 "$HERE/studio/backend/server.py" "${ARGS[@]}"
