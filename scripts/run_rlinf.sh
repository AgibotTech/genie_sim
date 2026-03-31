#!/usr/bin/env bash
# run_rlinf.sh — in-container wrapper that sets up the environment and runs
# an RLinf command.  Baked into the Docker image at /geniesim/scripts/run_rlinf.sh.
#
# Usage (inside the container):
#   /geniesim/scripts/run_rlinf.sh <command> [args...]

set -eo pipefail

GENIESIM_ROOT="${GENIESIM_ROOT:-/geniesim/main}"
SIM_REPO_ROOT="${SIM_REPO_ROOT:-${GENIESIM_ROOT}}"
RLINF_ROOT="${RLINF_ROOT:-/geniesim/RLinf}"
ROS_WS_INSTALL="${ROS_WS_INSTALL:-/geniesim/ros_ws_build/install}"
RLINF_VENV="${RLINF_VENV:-/opt/rlinf_venv/rlinf}"

export GENIESIM_ROOT SIM_REPO_ROOT
export GENIESIM_CONTAINER="${GENIESIM_CONTAINER:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=1

if [ -f "${RLINF_VENV}/bin/activate" ]; then
    source "${RLINF_VENV}/bin/activate"
else
    echo "[run_rlinf.sh] WARNING: RLinf venv not found at ${RLINF_VENV}" >&2
fi

export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment}"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

if [ -f "${ROS_WS_INSTALL}/setup.bash" ]; then
    source "${ROS_WS_INSTALL}/setup.bash"
fi

cd "${RLINF_ROOT}"

if [ $# -eq 0 ]; then
    echo "[run_rlinf.sh] No command given — dropping into bash."
    exec bash
else
    exec "$@"
fi
