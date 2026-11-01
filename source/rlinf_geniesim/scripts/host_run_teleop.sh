#!/usr/bin/env bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
# Connect to a task initialized with init_task.py. Backend selection belongs
# to that task; this wrapper forwards teleoperation arguments unchanged.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SIM_REPO_ROOT:-${TELEOP_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"

if [[ -z "${TELEOP_CONTAINER:-}" ]]; then
  export PYTHONPATH="$REPO/source:$REPO/source/geniesim_teleop/src:$REPO/source/geniesim_benchmark/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "${TELEOP_PYTHON:-python3}" "$REPO/source/rlinf_geniesim/scripts/teleop_pico_debug.py" "$@"
fi

FORWARD=()
while IFS= read -r -d '' item; do
  case "${item%%=*}" in
    RLINF_*|ROS_*|RMW_*|CUDA_*|NVIDIA_*|GENIESIM_*|DISPLAY)
      FORWARD+=(-e "$item") ;;
  esac
done < <(env -0)
ROOT="${TELEOP_CONTAINER_REPO:-/workspace}"
exec docker exec -i --user "$(id -u):$(id -g)" -e HOME=/home/isaac-sim \
  "${FORWARD[@]}" "$TELEOP_CONTAINER" bash -c '
  set -e
  export SIM_REPO_ROOT="$1"
  source "$SIM_REPO_ROOT/source/rlinf_geniesim/scripts/rlinf_env.sh"
  export PYTHONPATH="$SIM_REPO_ROOT/source/geniesim_teleop/src:$PYTHONPATH"
  shift
  exec python3 "$SIM_REPO_ROOT/source/rlinf_geniesim/scripts/teleop_pico_debug.py" "$@"
' teleop "$ROOT" "$@"
