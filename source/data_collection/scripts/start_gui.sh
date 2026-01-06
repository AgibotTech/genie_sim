#!/bin/bash

# Usage: ./scripts/start_gui.sh [exec|start|restart|run] [container_name]
# Default action: run
# Default container name: data_collection_gui

ACTION=${1:-run}
CONTAINER_NAME=${2:-data_collection_open_source}

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
    if [ -z "$SIM_ASSETS" ]; then
        echo "You need to set \$SIM_ASSETS eg. SIM_ASSETS=~/assets"
        exit 1
    else
        echo "using SIM_ASSETS='$SIM_ASSETS'"
    fi

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
        -v $SIM_ASSETS:/geniesim/main/source/geniesim/assets:rw \
        -v $CURRENT_DIR:/geniesim/main/data_collection:rw \
        -w /geniesim/main/data_collection \
        registry.agibot.com/genie-sim/open_source-data-collection:latest \
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
echo "  Container name: Default is 'data_collection_gui'"
echo ""
echo "Examples:"
echo "  ./scripts/start_gui.sh run my_container"
echo "  ./scripts/start_gui.sh exec my_container"
echo "  ./scripts/start_gui.sh start my_container"
echo "  ./scripts/start_gui.sh restart my_container"
exit 1
