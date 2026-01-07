#!/bin/bash

# Enhanced Data Collection Startup Script
# Usage: ./scripts/run_data_collection.sh [--headless] [--no-record] [--task TASK_PATH] [--watch] [--container-name NAME]

set -eo pipefail

# Default values
HEADLESS=false
RECORD=true
TASK="tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json"
STANDALONE=false  # false = print to terminal, true = only save to file
CONTAINER_NAME="data_collection_open_source"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS=true
            shift
            ;;
        --no-record)
            RECORD=false
            shift
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --standalone)
            STANDALONE=true
            shift
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./scripts/run_data_collection.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --headless              Run in headless mode (default: false)"
            echo "  --no-record             Disable recording (default: record enabled)"
            echo "  --task TASK_PATH        Task template path (default: tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json)"
            echo "  --standalone             Run in standalone mode (only save logs, no terminal output) (default: false, prints to terminal)"
            echo "  --container-name NAME   Container name (default: data_collection_open_source)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./scripts/run_data_collection.sh --headless --task tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json"
            echo "  ./scripts/run_data_collection.sh --standalone --headless"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if SIM_ASSETS is set
if [ -z "$SIM_ASSETS" ]; then
    echo "Error: SIM_ASSETS environment variable is not set"
    echo "Please set it, e.g., export SIM_ASSETS=~/assets"
    exit 1
fi

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

sudo setfacl -m u:1234:rwX $CURRENT_DIR/scripts
sudo mkdir -p $CURRENT_DIR/saved_task && sudo setfacl -m u:1234:rwX -R $CURRENT_DIR/saved_task
sudo mkdir -p $CURRENT_DIR/recording_data && sudo setfacl -m u:1234:rwX -R $CURRENT_DIR/recording_data
[ -d $CURRENT_DIR/config ] && sudo setfacl -m u:1234:rwX -R $CURRENT_DIR/config
# Extract task name from task path or JSON file
TASK_NAME=""
if [ -f "$TASK" ]; then
    # Try to extract task name from JSON file
    TASK_NAME=$(python3 -c "import json, sys;
try:
    with open('$TASK', 'r') as f:
        data = json.load(f)
        task_name = data.get('task', '')
        if task_name:
            print(task_name)
except:
    pass" 2>/dev/null || echo "")
fi

# If task name not found in JSON, use filename without extension
if [ -z "$TASK_NAME" ] || [ "$TASK_NAME" = "" ]; then
    TASK_NAME=$(basename "$TASK" .json)
fi

# Sanitize task name for directory name (remove special characters)
TASK_NAME=$(echo "$TASK_NAME" | sed 's/[^a-zA-Z0-9_-]/_/g')

# Create log directory based on task name
LOG_DIR="${CURRENT_DIR}/logs/${TASK_NAME}"
mkdir -p "$LOG_DIR"

# Set permissions for log directory (container runs as user 1234:1234)
# Make sure the directory is writable by the container user
chmod 777 "$LOG_DIR" 2>/dev/null || true
# Also try to set ownership if possible (may require sudo)
if command -v sudo >/dev/null 2>&1; then
    sudo chown -R 1234:1234 "$LOG_DIR" 2>/dev/null || true
    sudo chmod -R 777 "$LOG_DIR" 2>/dev/null || true
fi

# Log file for script output
SCRIPT_LOG="${LOG_DIR}/run_data_collection_sh.log"

# Function to log and optionally print
log_and_print() {
    if [ "$STANDALONE" = true ]; then
        # Standalone mode: only save to file
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" >> "$SCRIPT_LOG"
    else
        # Default mode: print to terminal and save to file
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$SCRIPT_LOG"
    fi
}

# Function to cleanup container
cleanup_container() {
    log_and_print "Cleaning up container..."
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_and_print "Stopping container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        log_and_print "Removing container: $CONTAINER_NAME"
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        log_and_print "Container cleaned up"
    fi
}

# Set trap for cleanup
trap cleanup_container EXIT INT TERM

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_and_print "Container '$CONTAINER_NAME' already exists. Removing it..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

log_and_print "=========================================="
log_and_print "Data Collection Startup"
log_and_print "=========================================="
log_and_print "Configuration:"
log_and_print "  Headless mode: $HEADLESS"
log_and_print "  Recording: $RECORD"
log_and_print "  Task template: $TASK"
log_and_print "  Task name: $TASK_NAME"
log_and_print "  Container name: $CONTAINER_NAME"
log_and_print "  Log directory: $LOG_DIR"
log_and_print "  Standalone mode: $STANDALONE (false = print to terminal, true = only save to file)"
log_and_print "=========================================="
log_and_print ""

# Build entrypoint arguments
DISPLAY_ARGS=""
ENTRYPOINT_ARGS=""
if [ "$HEADLESS" = true ]; then
    ENTRYPOINT_ARGS="$ENTRYPOINT_ARGS --headless"
else
    xhost +
    DISPLAY_ARGS="-e DISPLAY"
fi
if [ "$RECORD" = false ]; then
    ENTRYPOINT_ARGS="$ENTRYPOINT_ARGS --no-record"
fi
if [ -n "$TASK" ]; then
    ENTRYPOINT_ARGS="$ENTRYPOINT_ARGS --task $TASK"
fi

# Start container with log directory mounted
log_and_print "Starting Docker container..."
log_and_print "Command: docker run -d --name $CONTAINER_NAME ..."
log_and_print "Log directory mounted: $LOG_DIR -> /geniesim/main/data_collection/logs/${TASK_NAME}"

CONTAINER_ID=$(docker run -d --name $CONTAINER_NAME \
    --user 1234:1234 \
    --entrypoint ./scripts/data_collection_entrypoint.sh \
    --gpus all \
    --network=host \
    --privileged \
    $DISPLAY_ARGS \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" \
    -e "OMNI_USER=geniesim" \
    -e "OMNI_PASS=geniesim" \
    -e "LOG_DIR=/geniesim/main/data_collection/logs/${TASK_NAME}" \
    -e "SIM_ASSETS=/geniesim/main/source/geniesim/assets" \
    -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
    -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
    -v /dev/input:/dev/input:rw \
    -v $SIM_ASSETS:/geniesim/main/source/geniesim/assets:rw \
    -v $CURRENT_DIR:/geniesim/main/data_collection:rw \
    -v $LOG_DIR:/geniesim/main/data_collection/logs/${TASK_NAME}:rw \
    -w /geniesim/main/data_collection \
    registry.agibot.com/genie-sim/open_source-data-collection:latest \
    $ENTRYPOINT_ARGS)

if [ -z "$CONTAINER_ID" ]; then
    log_and_print "Error: Failed to start container"
    exit 1
fi

log_and_print "Container started: $CONTAINER_ID"
log_and_print ""

# Wait a bit for container to initialize
sleep 3

# Function to monitor container and optionally print logs
monitor_container() {
    if [ "$STANDALONE" = true ]; then
        # Standalone mode: only monitor, logs are already being written directly
        log_and_print "Monitoring container in standalone mode (logs saved to: $LOG_DIR)..."
        log_and_print "Press Ctrl+C to stop"
        log_and_print ""

        while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; do
            sleep 10
        done

        log_and_print "Container stopped"
    else
        # Default mode: print logs to terminal while saving to file
        log_and_print "Monitoring container (logs printed to terminal and saved to: $LOG_DIR)..."
        log_and_print "Press Ctrl+C to stop"
        log_and_print ""

        # Tail container logs to terminal and file
        # Container logs (from entrypoint script) go to docker logs
        docker logs -f "$CONTAINER_NAME" 2>&1 | tee -a "${LOG_DIR}/container.log" &
        CONTAINER_LOG_PID=$!

        # Wait a bit for application logs to be created
        sleep 5

        # Tail application logs if they exist
        SERVER_LOG_PID=""
        MAIN_LOG_PID=""

        if [ -f "${LOG_DIR}/data_collector_server.log" ]; then
            tail -f "${LOG_DIR}/data_collector_server.log" &
            SERVER_LOG_PID=$!
        fi

        if [ -f "${LOG_DIR}/run_data_collection.log" ]; then
            tail -f "${LOG_DIR}/run_data_collection.log" &
            MAIN_LOG_PID=$!
        fi

        # Wait for container to stop
        wait $CONTAINER_LOG_PID 2>/dev/null || true

        # Kill log tail processes
        [ -n "$SERVER_LOG_PID" ] && kill $SERVER_LOG_PID 2>/dev/null || true
        [ -n "$MAIN_LOG_PID" ] && kill $MAIN_LOG_PID 2>/dev/null || true

        log_and_print "Container stopped"
    fi
}

# Start monitoring
monitor_container

# Logs are already being written directly, no need to copy

log_and_print ""
log_and_print "=========================================="
log_and_print "Data collection completed"
log_and_print "=========================================="
log_and_print "Logs saved to: $LOG_DIR"
log_and_print "  - run_data_collection_sh.log (script output)"
log_and_print "  - container.log (container logs)"
log_and_print "  - data_collector_server.log (if available)"
log_and_print "  - run_data_collection.log (if available)"
log_and_print ""

# Cleanup will be done by trap
log_and_print "Exiting (container will be cleaned up automatically)..."
