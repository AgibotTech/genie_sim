#!/usr/bin/env bash

set -euo pipefail

PY_PATH="/geniesim/main/source/geniesim/generator/app.py"
PYTHON="/geniesim/generator_env/bin/python"

# Fail early if the script is missing
[[ -f "$PY_PATH" ]] || { echo "ERROR: $PY_PATH not found" >&2; exit 1; }

# Use sudo to switch user and preserve PYTHONPATH
# cd /geniesim/main/source
# "$PYTHON" "$PY_PATH" "$@"
# PYTHONPATH="${PYTHONPATH:-}:${PYTHONPATH:+/}geniesim/main/source" "$PYTHON" "$PY_PATH" "$@"

sudo -u "#1000" -g "#1000" env PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}/geniesim/main/source" "$PYTHON" "$PY_PATH" "$@"
