#!/bin/bash

# Configuration variables
INFER_HOST="localhost"
INFER_PORT=8999
NUM_EPISODE=1
ROBOT_TYPE="G2"  # G1 or G2

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLM_TASK_DIR="${PROJECT_ROOT}/source/geniesim/benchmark/config/llm_task"
TASK_MAPPING_FILE="${PROJECT_ROOT}/source/geniesim/benchmark/config/task_config_mapping.py"
TEMPLATE_YAML="${PROJECT_ROOT}/source/geniesim/config/template.yaml"
TEMP_YAML="/tmp/temp_run_${USER}_$$.yaml"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Task Batch Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Configuration:"
echo -e "  INFER_HOST: ${YELLOW}${INFER_HOST}${NC}"
echo -e "  INFER_PORT: ${YELLOW}${INFER_PORT}${NC}"
echo -e "  NUM_EPISODE: ${YELLOW}${NUM_EPISODE}${NC}"
echo -e "  ROBOT_TYPE: ${YELLOW}${ROBOT_TYPE}${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Function to get task_name from task_config_mapping.py
get_task_name() {
    local sub_task_name=$1
    local robot_type=$2

    # Extract the background task name using Python
    python3 << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}/source/geniesim/benchmark/config')
from task_config_mapping import TASK_MAPPING

sub_task = "${sub_task_name}"
robot = "${robot_type}"

if sub_task in TASK_MAPPING:
    background = TASK_MAPPING[sub_task].get("background", {})
    if robot in background:
        print(background[robot])
    else:
        print("NOT_FOUND")
else:
    print("NOT_FOUND")
EOF
}

# Get all subdirectories in llm_task folder
if [ ! -d "${LLM_TASK_DIR}" ]; then
    echo -e "${RED}Error: Task directory not found: ${LLM_TASK_DIR}${NC}"
    exit 1
fi

# Read all folder names
SUB_TASKS=($(ls -d ${LLM_TASK_DIR}/*/ 2>/dev/null | xargs -n 1 basename))

if [ ${#SUB_TASKS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No sub-tasks found in ${LLM_TASK_DIR}${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${#SUB_TASKS[@]} sub-tasks to process${NC}\n"

# Counter for successful and failed runs
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
declare -a FAILED_TASKS
declare -a SKIPPED_TASKS

# Process each sub-task
for SUB_TASK_NAME in "${SUB_TASKS[@]}"; do
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Processing: ${YELLOW}${SUB_TASK_NAME}${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Get the corresponding task_name (background)
    TASK_NAME=$(get_task_name "${SUB_TASK_NAME}" "${ROBOT_TYPE}")

    if [ "${TASK_NAME}" == "NOT_FOUND" ] || [ -z "${TASK_NAME}" ]; then
        echo -e "${YELLOW}Warning: No background task found for '${SUB_TASK_NAME}' with robot type '${ROBOT_TYPE}'${NC}"
        echo -e "${YELLOW}Skipping this task...${NC}\n"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_TASKS+=("${SUB_TASK_NAME}")
        continue
    fi

    echo -e "  Sub-task: ${YELLOW}${SUB_TASK_NAME}${NC}"
    echo -e "  Task (background): ${YELLOW}${TASK_NAME}${NC}"

    # Create a temporary yaml file with replaced values
    cp "${TEMPLATE_YAML}" "${TEMP_YAML}"

    # Replace values in the temporary yaml file using sed
    sed -i "s|infer_host:.*|infer_host: \"${INFER_HOST}\"|g" "${TEMP_YAML}"
    sed -i "s|infer_port:.*|infer_port: ${INFER_PORT}|g" "${TEMP_YAML}"
    sed -i "s|num_episode:.*|num_episode: ${NUM_EPISODE}|g" "${TEMP_YAML}"
    sed -i "s|task_name:.*|task_name: \"${TASK_NAME}\"|g" "${TEMP_YAML}"
    sed -i "s|sub_task_name:.*|sub_task_name: \"${SUB_TASK_NAME}\"|g" "${TEMP_YAML}"

    echo -e "\n${GREEN}Running benchmark...${NC}"

    # Run the benchmark (change to project root directory first)
    cd "${PROJECT_ROOT}"
    /isaac-sim/python.sh source/geniesim/app/app.py --config "${TEMP_YAML}"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully completed: ${SUB_TASK_NAME}${NC}\n"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ Failed: ${SUB_TASK_NAME}${NC}\n"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_TASKS+=("${SUB_TASK_NAME}")
    fi

    # Clean up temporary file
    rm -f "${TEMP_YAML}"

    echo ""
done

# Print summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Batch Run Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total tasks: ${#SUB_TASKS[@]}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAILED_COUNT}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED_COUNT}${NC}"

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo -e "\n${RED}Failed tasks:${NC}"
    for task in "${FAILED_TASKS[@]}"; do
        echo -e "  - ${task}"
    done
fi

if [ ${SKIPPED_COUNT} -gt 0 ]; then
    echo -e "\n${YELLOW}Skipped tasks (no mapping for ${ROBOT_TYPE}):${NC}"
    for task in "${SKIPPED_TASKS[@]}"; do
        echo -e "  - ${task}"
    done
fi

echo -e "${GREEN}========================================${NC}"

# Clean up any remaining temporary files
rm -f /tmp/temp_run_${USER}_*.yaml 2>/dev/null

# Exit with error if any tasks failed
if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi

exit 0
