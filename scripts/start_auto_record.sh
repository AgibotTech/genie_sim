#!/bin/bash
# Auto recording startup script

set -eo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Auto Image Recording and Extraction Tool ===${NC}"
echo ""

PROJECT_ROOT=$(pwd)

source $PROJECT_ROOT/../record_env/bin/activate

# Default parameters
OUTPUT_DIR="$PROJECT_ROOT/output/recording_data"
FINAL_OUTPUT_DIR=""
TIMEOUT=60
IMAGE_TIMEOUT=30.0
RESIZE_WIDTH=""
RESIZE_HEIGHT=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --final_output_dir)
            FINAL_OUTPUT_DIR="$2"
            shift 2
            ;;
        --resize_width)
            RESIZE_WIDTH="$2"
            shift 2
            ;;
        --resize_height)
            RESIZE_HEIGHT="$2"
            shift 2
            ;;
        --image_timeout)
            IMAGE_TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--final_output_dir DIR] [--resize_width WIDTH] [--resize_height HEIGHT] [--image_timeout SEC]"
            exit 1
            ;;
    esac
done

echo "Configuration parameters:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Final output directory: ${FINAL_OUTPUT_DIR:-<not set>}"
echo "  Timeout: ${TIMEOUT}s (genie_sim heartbeat, node exit)"
echo "  Image timeout: ${IMAGE_TIMEOUT}s (auto stop recording & convert)"
if [ -n "$RESIZE_WIDTH" ] || [ -n "$RESIZE_HEIGHT" ]; then
    echo "  Image resize: ${RESIZE_WIDTH:-orig}x${RESIZE_HEIGHT:-orig}"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Starting recording node...${NC}"
echo "Waiting for CompressedImage topics starting with /record/..."
echo "Recording will automatically stop after ${IMAGE_TIMEOUT}s with no new image messages"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop gracefully${NC}"
echo ""

unset LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash

PYTHON_SCRIPT="$(dirname "$0")/auto_record_and_extract.py"
PY_PID=""

cleanup() {
    local exit_status=$?
    echo ""

    if [[ -n "$PY_PID" ]] && kill -0 "$PY_PID" 2>/dev/null; then
        echo -e "${YELLOW}Received interrupt, shutting down node gracefully...${NC}"
        kill -INT "$PY_PID" 2>/dev/null
        # Wait up to 15s for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 "$PY_PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        # If still alive, SIGKILL
        if kill -0 "$PY_PID" 2>/dev/null; then
            echo -e "${RED}Node did not exit gracefully, forcing...${NC}"
            kill -9 "$PY_PID" 2>/dev/null
            wait "$PY_PID" 2>/dev/null
        else
            wait "$PY_PID" 2>/dev/null
            exit_status=$?
        fi
    fi

    echo -e "${YELLOW}Cleanup done.${NC}"
    exit $exit_status
}

trap cleanup SIGINT SIGTERM EXIT

echo -e "${GREEN}Launching Python node (PID $$)...${NC}"

# Build command
CMD=(python3 "$PYTHON_SCRIPT" --output_dir "$OUTPUT_DIR" --timeout "$TIMEOUT")
[[ -n "$FINAL_OUTPUT_DIR" ]] && CMD+=(--final_output_dir "$FINAL_OUTPUT_DIR")
[[ -n "$IMAGE_TIMEOUT" ]] && CMD+=(--image_timeout "$IMAGE_TIMEOUT")
[[ -n "$RESIZE_WIDTH" ]] && CMD+=(--resize_width "$RESIZE_WIDTH")
[[ -n "$RESIZE_HEIGHT" ]] && CMD+=(--resize_height "$RESIZE_HEIGHT")

# Disable SIGINT inheritance so only this shell (not python) gets Ctrl+C directly.
# We handle it via the trap above.
"${CMD[@]}" &
PY_PID=$!

# Wait for the python process; this respects signals received by the shell.
wait "$PY_PID"
EXIT_CODE=$?

# Unset trap before normal exit to avoid double-trigger
trap - SIGINT SIGTERM EXIT

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}=== Recording Complete ===${NC}"
    echo "Output location: $OUTPUT_DIR"
    echo ""
else
    echo ""
    echo -e "${RED}Program exited with code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
