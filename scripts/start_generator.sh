#!/usr/bin/env bash

# current folder as WORD_DIR
CURRENT_DIR=$(pwd)

set -euo pipefail

# Fail fast: we must be in the repo root
if [[ ! -d .git ]]; then
    cat <<EOF
ERROR: $(basename "$0") must be run from the repository root.

Current directory: $(pwd)

Please run:
    ./scripts/$(basename "$0")   [options]
EOF
    exit 1
fi

# OPENAI_API_BASE_URL OPENAI_API_KEY
# for var in SIM_ASSETS ; do
#     if [[ -z "${!var:-}" ]]; then
#         echo "ERROR: You must set the environment variable '$var' before running this script."
#     fi
#     echo "using $var='${!var}'"
# done

echo $UID
# echo $GID
# export UID=$(id -u)
export GID=$(id -g)

# If we get here we are in the repo root – continue safely
docker compose -f source/geniesim/generator/compose.yaml up --build
