#!/bin/bash
# Automatic data processing: execute process_data.py inside the container to post-process the recorded data for the specified task
# Usage: ./scripts/autoteleop_post_process.sh <task_name> [user_name]
#   task_name: required, corresponds to main/output/recording_data/<task_name>
#   user_name: optional, default is GenieSim

set -e

CONTAINER_NAME="genie_sim_benchmark"
USER_NAME="${2:-GenieSim}"

usage() {
    echo "Usage: $0 <task_name> [user_name]"
    echo "  task_name  required, name of the recording data directory (e.g. main/output/recording_data/<task_name>)"
    echo "  user_name  optional, default is GenieSim"
    exit 1
}

if [ -z "$1" ]; then
    echo "Error: task_name must be provided"
    usage
fi

TASK_NAME="$1"

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Error: container $CONTAINER_NAME is not running, please start it first (e.g. ./scripts/start_gui.sh)"
    exit 1
fi

echo "Processing data: task_name=$TASK_NAME, user_name=$USER_NAME"
docker exec "$CONTAINER_NAME" bash -c "
    source /geniesim/teleop_env/bin/activate
    source /opt/ros/jazzy/setup.bash
    source /geniesim/main/source/teleop/app/bin/env.sh
    cd /geniesim/main/source/teleop
    PYTHONUNBUFFERED=1 python3 -u ./data_recording/process_data.py --data_dir '$TASK_NAME' --user '$USER_NAME'
"
echo "Data processing completed."
