#!/bin/bash

# Data Collection Entrypoint Script
# This script starts both data_collector_server.py and run_data_collection.py
# Usage: ./data_collection_entrypoint.sh [--headless] [--no-record] [--task TASK_PATH]

set -eo pipefail

# Default values
HEADLESS=false
RECORD=true
TASK="tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json"

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
        --help|-h)
            echo "Usage: ./data_collection_entrypoint.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --headless              Run in headless mode (default: false)"
            echo "  --no-record             Disable recording (default: record enabled)"
            echo "  --task TASK_PATH        Task template path"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# First, execute entry_point.sh setup (but skip the exec at the end)
echo "=========================================="
echo "Executing entry_point.sh setup..."
echo "=========================================="

ENTRY_POINT_SCRIPT="/geniesim/main/data_collection/scripts/entry_point.sh"
if [ -f "$ENTRY_POINT_SCRIPT" ]; then
    # Source entry_point.sh but skip the exec at the end
    # We'll execute it in a subshell and capture the environment setup
    (
        # Source the entry_point.sh script
        # We need to prevent the exec from running, so we'll source it and stop before exec
        set +e  # Don't exit on error in this subshell
        source "$ENTRY_POINT_SCRIPT" /bin/true 2>/dev/null || {
            # If entry_point.sh tries to exec, it will fail with /bin/true
            # But the environment setup should have been done
            true
        }
    )

    # Execute entry_point.sh logic manually (since exec would replace the process)
    export ROS_DISTRO=jazzy
    export ISAACSIM_HOME=/isaac-sim
    export CUROBO_PATH=/tmp/curobo
    export SIM_REPO_ROOT=/geniesim/main/data_collection

    # user 1234 access
    sudo setfacl -m u:1234:rwX /isaac-sim/.cache 2>/dev/null || true
    sudo setfacl -m u:1234:rwX /isaac-sim/.nv/ComputeCache 2>/dev/null || true
    sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/logs 2>/dev/null || true
    sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/config 2>/dev/null || true
    sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/data 2>/dev/null || true
    sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/pkg 2>/dev/null || true

    # bashrc configuration (entry_point.sh should have done this, but ensure it's done)
    if ! grep -q "export SIM_REPO_ROOT=/geniesim/main/data_collection" ~/.bashrc 2>/dev/null; then
        echo "export SIM_REPO_ROOT=/geniesim/main/data_collection" >>~/.bashrc
    fi
    if ! grep -q "export SIM_ASSETS=" ~/.bashrc 2>/dev/null; then
        echo "export SIM_ASSETS=/geniesim/main/source/geniesim/assets" >>~/.bashrc
    fi
    if ! grep -q "export ENABLE_SIM=" ~/.bashrc 2>/dev/null; then
        echo "export ENABLE_SIM=1" >>~/.bashrc
    fi
    if ! grep -q "export ROS_DISTRO=" ~/.bashrc 2>/dev/null; then
        echo "export ROS_DISTRO=${ROS_DISTRO}" >>~/.bashrc
    fi
    if ! grep -q "export ROS_VERSION=" ~/.bashrc 2>/dev/null; then
        echo "export ROS_VERSION=2" >>~/.bashrc
    fi
    if ! grep -q "export ROS_PYTHON_VERSION=" ~/.bashrc 2>/dev/null; then
        echo "export ROS_PYTHON_VERSION=3" >>~/.bashrc
    fi
    if ! grep -q "export ROS_LOCALHOST_ONLY=" ~/.bashrc 2>/dev/null; then
        echo "export ROS_LOCALHOST_ONLY=1" >>~/.bashrc
    fi
    if ! grep -q "export RMW_IMPLEMENTATION=" ~/.bashrc 2>/dev/null; then
        echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >>~/.bashrc
    fi
    if ! grep -q "LD_LIBRARY_PATH.*isaacsim.ros2.bridge" ~/.bashrc 2>/dev/null; then
        echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:+\$LD_LIBRARY_PATH:}${ISAACSIM_HOME}/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib\"" >>~/.bashrc
    fi
    if ! grep -q "export ROS_CMD_DISTRO=" ~/.bashrc 2>/dev/null; then
        echo "export ROS_CMD_DISTRO=jazzy" >>~/.bashrc
    fi
    if ! grep -q "source /isaac-sim/setup_ros_env.sh" ~/.bashrc 2>/dev/null; then
        echo "source ${ISAACSIM_HOME}/setup_ros_env.sh" >>~/.bashrc
    fi
    if ! grep -q "alias omni_python=" ~/.bashrc 2>/dev/null; then
        echo "alias omni_python='${ISAACSIM_HOME}/python.sh'" >>~/.bashrc
    fi
    if ! grep -q "alias isaacsim=" ~/.bashrc 2>/dev/null; then
        echo "alias isaacsim='${ISAACSIM_HOME}/runapp.sh'" >>~/.bashrc
    fi
    if ! grep -q "alias geniesim=" ~/.bashrc 2>/dev/null; then
        echo "alias geniesim='omni_python /geniesim/main/source/geniesim/app/app.py'" >>~/.bashrc
    fi
    if ! grep -q "alias source_ros_py311=" ~/.bashrc 2>/dev/null; then
        echo "alias source_ros_py311='unset LD_LIBRARY_PATH && source /opt/ros/jazzy/setup.bash'" >>~/.bashrc
    fi

    # Copy curobo assets
    sudo cp -r ${SIM_ASSETS}/robot/curobo_robot/assets/robot $ISAACSIM_HOME/kit/python/lib/python3.11/site-packages/curobo/content/assets/ 2>/dev/null || true
    sudo cp -r ${SIM_REPO_ROOT}/config/curobo/configs $ISAACSIM_HOME/kit/python/lib/python3.11/site-packages/curobo/content/ 2>/dev/null || true

    echo "✓ entry_point.sh setup completed"
else
    echo "Warning: entry_point.sh not found at $ENTRY_POINT_SCRIPT"
    echo "Setting up basic environment..."
    export ROS_DISTRO=jazzy
    export ISAACSIM_HOME=/isaac-sim
    export CUROBO_PATH=/tmp/curobo
    export SIM_REPO_ROOT=/geniesim/main/data_collection
fi

echo ""
echo "Setting up environment variables directly..."
# Set all required environment variables directly (no need to source bashrc)
export ROS_DISTRO=jazzy
export ISAACSIM_HOME=/isaac-sim
export CUROBO_PATH=/tmp/curobo
export SIM_REPO_ROOT=/geniesim/main/data_collection
export SIM_ASSETS=/geniesim/main/source/geniesim/assets
export ENABLE_SIM=1
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_CMD_DISTRO=jazzy

# Set up LD_LIBRARY_PATH
ROS_BRIDGE_LIB="${ISAACSIM_HOME}/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib"
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="${ROS_BRIDGE_LIB}"
else
    # Only add if not already present
    if [[ ":$LD_LIBRARY_PATH:" != *":${ROS_BRIDGE_LIB}:"* ]]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ROS_BRIDGE_LIB}"
    fi
fi

# Source ROS environment if available
if [ -f "${ISAACSIM_HOME}/setup_ros_env.sh" ]; then
    source "${ISAACSIM_HOME}/setup_ros_env.sh" 2>/dev/null || true
fi

# Verify critical environment variables
echo ""
echo "Environment variables:"
echo "  SIM_ASSETS=$SIM_ASSETS"
echo "  SIM_REPO_ROOT=$SIM_REPO_ROOT"
echo "  ISAACSIM_HOME=$ISAACSIM_HOME"
echo "  ROS_DISTRO=$ROS_DISTRO"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "✓ Environment variables set directly"
echo ""

# Set up paths
ISAACSIM_PYTHON="${ISAACSIM_HOME}/python.sh"
WORK_DIR="/geniesim/main/data_collection"

# Use LOG_DIR from environment if set, otherwise use default
if [ -n "$LOG_DIR" ]; then
    # LOG_DIR is set from host (via docker run -e)
    # It should be something like /geniesim/main/data_collection/logs/task_name
    mkdir -p "$LOG_DIR"
    # Ensure directory is writable by user 1234 (container user)
    # Try sudo first, then fallback to regular chmod
    sudo chown -R 1234:1234 "$LOG_DIR" 2>/dev/null || true
    sudo chmod -R 777 "$LOG_DIR" 2>/dev/null || chmod -R 777 "$LOG_DIR" 2>/dev/null || true
    # Verify we can write to the directory
    touch "${LOG_DIR}/.test_write" 2>/dev/null && rm -f "${LOG_DIR}/.test_write" || {
        echo "Warning: Cannot write to LOG_DIR: $LOG_DIR"
        echo "Attempting to fix permissions..."
        sudo chmod -R 777 "$LOG_DIR" 2>/dev/null || true
    }
else
    # Default log directory
    LOG_DIR="${WORK_DIR}/logs"
    mkdir -p "$LOG_DIR"
    sudo chown -R 1234:1234 "$LOG_DIR" 2>/dev/null || true
    sudo chmod -R 777 "$LOG_DIR" 2>/dev/null || chmod -R 777 "$LOG_DIR" 2>/dev/null || true
fi

# Change to work directory
cd "${WORK_DIR}"

echo "=========================================="
echo "Data Collection Entrypoint"
echo "=========================================="
echo "Configuration:"
echo "  Headless mode: $HEADLESS"
echo "  Recording: $RECORD"
echo "  Task template: $TASK"
echo "  Working directory: $WORK_DIR"
echo "  Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Build command arguments
DATA_COLLECTOR_ARGS="--enable_physics --enable_curobo"
MAIN_ARGS=""

if [ "$HEADLESS" = true ]; then
    DATA_COLLECTOR_ARGS="$DATA_COLLECTOR_ARGS --headless"
fi

if [ "$RECORD" = true ]; then
    DATA_COLLECTOR_ARGS="$DATA_COLLECTOR_ARGS --publish_ros"
    MAIN_ARGS="$MAIN_ARGS --use_recording"
fi

if [ -n "$TASK" ]; then
    MAIN_ARGS="$MAIN_ARGS --task_template $TASK"
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    echo "=========================================="
    # Kill processes by name
    echo "Stopping data_collector_server..."
    pkill -f "data_collector_server.py" 2>/dev/null || true
    echo "Stopping run_data_collection.py..."
    pkill -f "run_data_collection.py" 2>/dev/null || true
    sleep 2
    echo "Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Start data_collector_server in background
echo "Starting data_collector_server..."
echo "Command: ${ISAACSIM_PYTHON} scripts/data_collector_server.py $DATA_COLLECTOR_ARGS"
echo "Logs: ${LOG_DIR}/data_collector_server.log"
echo "Environment: SIM_ASSETS=$SIM_ASSETS, SIM_REPO_ROOT=$SIM_REPO_ROOT"
# All environment variables are already exported above, Python will inherit them
${ISAACSIM_PYTHON} scripts/data_collector_server.py $DATA_COLLECTOR_ARGS > "${LOG_DIR}/data_collector_server.log" 2>&1 &
DATA_COLLECTOR_PID=$!

echo "✓ data_collector_server started (PID: $DATA_COLLECTOR_PID)"

# Wait for the server to initialize
if [ "$HEADLESS" = true ]; then
    echo "Waiting for data_collector_server to initialize (headless mode, may take longer)..."
    sleep 15
else
    echo "Waiting for data_collector_server to initialize..."
    sleep 10
fi

# Check if data_collector_server is still running
if ! kill -0 $DATA_COLLECTOR_PID 2>/dev/null; then
    echo "Error: data_collector_server process died"
    echo "Last 30 lines of log:"
    tail -30 "${LOG_DIR}/data_collector_server.log" || true
    exit 1
fi
echo "✓ data_collector_server is running"

# Start run_data_collection.py in background
echo ""
echo "Starting run_data_collection.py..."
echo "Command: ${ISAACSIM_PYTHON} scripts/run_data_collection.py $MAIN_ARGS"
echo "Logs: ${LOG_DIR}/run_data_collection.log"
echo "Environment: SIM_ASSETS=$SIM_ASSETS, SIM_REPO_ROOT=$SIM_REPO_ROOT"
# All environment variables are already exported above, Python will inherit them
${ISAACSIM_PYTHON} scripts/run_data_collection.py $MAIN_ARGS > "${LOG_DIR}/run_data_collection.log" 2>&1 &
MAIN_PID=$!

echo "✓ run_data_collection.py started (PID: $MAIN_PID)"

# Wait a bit and verify run_data_collection.py is running
sleep 5
if ! kill -0 $MAIN_PID 2>/dev/null; then
    echo "Warning: run_data_collection.py process died"
    echo "Last 30 lines of log:"
    tail -30 "${LOG_DIR}/run_data_collection.log" || true
    # Don't exit, let it try to continue
else
    echo "✓ run_data_collection.py is running"
fi

echo ""
echo "=========================================="
echo "Both services are running!"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  data_collector_server: $DATA_COLLECTOR_PID"
echo "  run_data_collection.py: $MAIN_PID"
echo ""
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/data_collector_server.log"
echo "  tail -f ${LOG_DIR}/run_data_collection.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Monitor processes
MONITOR_COUNT=0
while true; do
    sleep 10
    MONITOR_COUNT=$((MONITOR_COUNT + 1))

    # Check if processes are still running
    DATA_COLLECTOR_RUNNING=true
    MAIN_RUNNING=true

    if ! kill -0 $DATA_COLLECTOR_PID 2>/dev/null; then
        DATA_COLLECTOR_RUNNING=false
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Error: data_collector_server.py process died"
        echo "Last 30 lines of log:"
        tail -30 "${LOG_DIR}/data_collector_server.log" || true
        exit 1
    fi

    if ! kill -0 $MAIN_PID 2>/dev/null; then
        MAIN_RUNNING=false
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Warning: run_data_collection.py process died"
        echo "Last 30 lines of log:"
        tail -30 "${LOG_DIR}/run_data_collection.log" || true
        # Check if it's a normal exit or error
        if grep -q "job done" "${LOG_DIR}/run_data_collection.log" 2>/dev/null; then
            echo "Main process completed successfully"
            exit 0
        else
            exit 1
        fi
    fi

    # Print status every 6 checks (1 minute)
    if [ $((MONITOR_COUNT % 6)) -eq 0 ]; then
        if [ "$DATA_COLLECTOR_RUNNING" = true ] && [ "$MAIN_RUNNING" = true ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Status: Both services running normally"
        fi
    fi
done
