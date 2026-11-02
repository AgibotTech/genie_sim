#!/bin/bash

# Usage: ./scripts/start_gui.sh [exec|start|restart|run] [container_name]
# Default action: run
# Default container name: geniesim3-data-collection

ACTION=${1:-run}
CONTAINER_NAME=${2:-geniesim3-data-collection}

# current folder as WORD_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

set -eo pipefail

# exec action: enter container
if [ "$ACTION" == "exec" ]; then
    echo "Executing: docker exec -it $CONTAINER_NAME /bin/bash"
    docker exec -it $CONTAINER_NAME /bin/bash
    exit 0
fi

# start action: start container
if [ "$ACTION" == "start" ]; then
    echo "Executing: docker start $CONTAINER_NAME"
    docker start $CONTAINER_NAME
    exit 0
fi

# restart action: restart container
if [ "$ACTION" == "restart" ]; then
    echo "Executing: docker restart $CONTAINER_NAME"
    docker restart $CONTAINER_NAME
    exit 0
fi

# run action: create and run new container
if [ "$ACTION" == "run" ]; then
    set -eo pipefail

    echo "using SIM_REPO_ROOT='$CURRENT_DIR'"
    SIM_ASSETS="$(python3 "$SCRIPT_DIR/resolve_assets.py")"
    export SIM_ASSETS

    # Keep the task library outside the image; remap its host path for the client.
    LLM_TASK_ROOT="$(python3 "$SCRIPT_DIR/resolve_llm_task_root.py")" || exit $?
    LLM_TASK_DOCKER_ARGS=()
    if [ -n "$LLM_TASK_ROOT" ]; then
        sudo setfacl -R -m u:1234:rwX "$LLM_TASK_ROOT"
        LLM_TASK_DOCKER_ARGS=(-v "$LLM_TASK_ROOT:/llm_tasks:rw" -e LLM_TASK_ROOT=/llm_tasks -e GENIESIM_LLM_TASK_ROOT=/llm_tasks)
    fi
    echo "using SIM_ASSETS='$SIM_ASSETS'"

    mkdir -p ~/docker/isaac-sim/cache/main/ov
    mkdir -p ~/docker/isaac-sim/cache/main/warp
    mkdir -p ~/docker/isaac-sim/cache/computecache
    mkdir -p ~/docker/isaac-sim/config
    mkdir -p ~/docker/isaac-sim/data/documents
    mkdir -p ~/docker/isaac-sim/data/Kit
    mkdir -p ~/docker/isaac-sim/logs
    mkdir -p ~/docker/isaac-sim/pkg
    sudo chown -R 1234:1234 ~/docker/isaac-sim
    sudo setfacl -m u:1234:rwX $CURRENT_DIR/scripts
    sudo mkdir -p $CURRENT_DIR/saved_task && sudo setfacl -m u:1234:rwX -R $CURRENT_DIR/saved_task
    sudo mkdir -p $CURRENT_DIR/recording_data && sudo setfacl -m u:1234:rwX -R $CURRENT_DIR/recording_data
    [ -d $CURRENT_DIR/config ] && sudo setfacl -m u:1234:rwX -R $CURRENT_DIR/config
    echo "LLM task root: ${LLM_TASK_ROOT:-not configured}"
    echo "Container name: $CONTAINER_NAME"
    xhost +
    docker run -it --name $CONTAINER_NAME \
        --user 1234:1234 \
        --entrypoint ./scripts/entry_point.sh \
        --rm \
        --gpus all \
        --network=host \
        --privileged \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=Y" \
        -e "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" \
        -e "OMNI_USER=geniesim" \
        -e "OMNI_PASS=geniesim" \
        -e "SIM_ASSETS=/geniesim/main/source/geniesim/assets" \
        -e DISPLAY \
        -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
        -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
        -v /dev/input:/dev/input:rw \
        "${LLM_TASK_DOCKER_ARGS[@]}" \
        -v "$SIM_ASSETS:/geniesim/main/source/geniesim/assets:rw" \
        -v $CURRENT_DIR:/geniesim/main/data_collection:rw \
        -w /geniesim/main/data_collection \
        "${DATA_COLLECTION_IMAGE:-registry.agibot.com/genie-sim/geniesim3-data-collection:latest}" \
        bash
fi

# If action doesn't match, show help information
echo "Error: Unknown action '$ACTION'"
echo ""
echo "Usage: ./scripts/start_gui.sh [exec|start|restart|run] [container_name]"
echo "  Actions:"
echo "    exec    - Enter container (docker exec -it container_name /bin/bash)"
echo "    start   - Start container (docker start container_name)"
echo "    restart - Restart container (docker restart container_name)"
echo "    run     - Create and run new container (default action)"
echo "  Container name: Default is 'geniesim3-data-collection'"
echo ""
echo "Examples:"
echo "  ./scripts/start_gui.sh run my_container"
echo "  ./scripts/start_gui.sh exec my_container"
echo "  ./scripts/start_gui.sh start my_container"
echo "  ./scripts/start_gui.sh restart my_container"
exit 1
