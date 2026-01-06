#!/bin/bash

# List of process names to kill
PROCESS_NAMES=(
    "run_tasks.sh"
    "omni_python"
    "python.sh"
    "isaac-sim"
    "app.py"
)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Process Cleanup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Kill each process
for PROCESS_NAME in "${PROCESS_NAMES[@]}"; do
    echo -e "${YELLOW}Checking for process: ${PROCESS_NAME}${NC}"

    # Check if process exists
    if pgrep -f "${PROCESS_NAME}" > /dev/null; then
        echo -e "${RED}  Killing processes matching '${PROCESS_NAME}'...${NC}"
        pkill -9 -f "${PROCESS_NAME}"

        # Wait a moment and verify
        sleep 0.5
        if pgrep -f "${PROCESS_NAME}" > /dev/null; then
            echo -e "${RED}  Warning: Some processes may still be running${NC}"
        else
            echo -e "${GREEN}  ✓ Successfully killed${NC}"
        fi
    else
        echo -e "  No processes found"
    fi
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup complete${NC}"
echo -e "${GREEN}========================================${NC}"
