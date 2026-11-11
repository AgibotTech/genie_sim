#!/bin/bash
# Automatic data processing: run process_data.py inside the container to
# post-process recorded teleop data for the specified task.
# Usage: ./scripts/autoteleop_post_process.sh <task_name> [user_name]
#   task_name: required, corresponds to <repo>/output/recording_data/<task_name>
#   user_name: optional, default is GenieSim

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/teleop_runtime.sh"
teleop_select_runtime "$@"
set -- "${TELEOP_RUNTIME_ARGS[@]}"
CONTAINER_NAME="${GENIESIM_CONTAINER:-$TELEOP_DEFAULT_CONTAINER}"
REPO_IN_CONTAINER="${GENIESIM_WORKSPACE:-/workspace}"
TELEOP_APP="${REPO_IN_CONTAINER}/source/geniesim_teleop/src/geniesim_teleop/app"
# Use apt NumPy for cv2 and rosbags image decoding.
NUMPY1_PATH="/usr/lib/python3/dist-packages"
USER_NAME="${2:-GenieSim}"

usage() {
    echo "Usage: $0 [--isaac-version 6.2] <task_name> [user_name]"
    echo "  task_name  required, name of the recording data dir (<repo>/output/recording_data/<task_name>)"
    echo "  user_name  optional, default is GenieSim"
    exit 1
}

if [ -z "${1:-}" ]; then
    echo "Error: task_name must be provided"
    usage
fi

TASK_NAME="$1"

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Error: container '$CONTAINER_NAME' is not running. Start it first (e.g. geniesim docker${TELEOP_ISAAC_VERSION} up),"
    echo "       or set GENIESIM_CONTAINER=<name>."
    exit 1
fi

echo "Processing data: task_name=$TASK_NAME, user_name=$USER_NAME"

# These dependencies are baked into the geniesim4 image.
if ! docker exec -e "PYTHONPATH=$NUMPY1_PATH" "$CONTAINER_NAME" python3 -c 'import rosbags, rosbags.image, h5py' ; then
    echo "Error: recording dependencies are missing or incompatible. Rebuild with geniesim docker${TELEOP_ISAAC_VERSION} build and recreate the container."
    exit 1
fi

PROCESS_HELPER="${REPO_IN_CONTAINER}/source/geniesim_teleop/scripts/teleop_process.sh"
PROCESS_PID_FILE=""
EXEC_PID=""

cleanup() {
    local status=$?
    trap - EXIT
    trap '' INT TERM HUP
    if [[ -n "$PROCESS_PID_FILE" ]]; then
        if ! docker exec "$CONTAINER_NAME" bash "$PROCESS_HELPER" stop "$PROCESS_PID_FILE"; then
            status=1
            echo "Error: post-processing cleanup failed; check $PROCESS_PID_FILE in $CONTAINER_NAME." >&2
            if [[ -n "$EXEC_PID" ]] && jobs -pr | grep -qx "$EXEC_PID"; then
                kill -TERM "$EXEC_PID" 2>/dev/null || true
            fi
        fi
    fi
    if [[ -n "$EXEC_PID" ]]; then
        wait "$EXEC_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

PROCESS_PID_FILE=$(docker exec "$CONTAINER_NAME" mktemp /tmp/geniesim-teleop-post-process.XXXXXX)
docker exec -e "SIM_REPO_ROOT=$REPO_IN_CONTAINER" "$CONTAINER_NAME" \
    bash "$PROCESS_HELPER" start "$PROCESS_PID_FILE" bash -c '
        set -e
        source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
        source "$1/bin/env.sh"
        export PYTHONPATH="$2:${PYTHONPATH:-}"
        exec python3 -u -m geniesim_teleop.data_recording.process_data --data_dir "$3" --user "$4"
    ' _ "$TELEOP_APP" "$NUMPY1_PATH" "$TASK_NAME" "$USER_NAME" &
EXEC_PID=$!
wait "$EXEC_PID"
EXEC_PID=""
echo "Data processing completed."
