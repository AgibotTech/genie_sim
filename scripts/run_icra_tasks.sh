#!/bin/bash

# Run all icra_*.yaml configs under source/geniesim/config/ in order.
# Usage: ./scripts/run_icra_tasks.sh [--infer-host HOST:PORT]

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/source/geniesim/config"
OUTPUT_DIR="${PROJECT_ROOT}/output"
BENCHMARK_DIR="${OUTPUT_DIR}/benchmark"

# Parse arguments
INFER_HOST=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --infer-host)
            INFER_HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--infer-host HOST:PORT]"
            exit 1
            ;;
    esac
done

# Trap Ctrl+C to run clean.sh before exiting
trap_cleanup() {
    echo -e "\n${YELLOW}Interrupted! Running cleanup...${NC}"
    bash "${SCRIPT_DIR}/clean.sh"
    exit 130
}
trap trap_cleanup SIGINT

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ICRA Task Batch Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config directory: ${YELLOW}${CONFIG_DIR}${NC}"
if [ -n "${INFER_HOST}" ]; then
    echo -e "Override infer_host: ${YELLOW}${INFER_HOST}${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

if [ ! -d "${CONFIG_DIR}" ]; then
    echo -e "${RED}Error: Config directory not found: ${CONFIG_DIR}${NC}"
    exit 1
fi

# Sorted list of icra_*.yaml (deterministic order)
ICRA_YAMLS=($(find "${CONFIG_DIR}" -maxdepth 1 -name "icra_*.yaml" -type f | sort))

# If --infer-host is provided, create copies in /tmp with overridden infer_host
TEMP_CONFIG_DIR=""
if [ -n "${INFER_HOST}" ]; then
    TEMP_CONFIG_DIR="/tmp/icra_configs_$$"
    mkdir -p "${TEMP_CONFIG_DIR}"
    NEW_YAMLS=()
    for yaml in "${ICRA_YAMLS[@]}"; do
        TMP_YAML="${TEMP_CONFIG_DIR}/$(basename ${yaml})"
        sed "s|infer_host:.*|infer_host: \"${INFER_HOST}\"|" "${yaml}" > "${TMP_YAML}"
        echo -e "  ${YELLOW}$(basename ${yaml})${NC}: $(grep infer_host "${TMP_YAML}")"
        NEW_YAMLS+=("${TMP_YAML}")
    done
    ICRA_YAMLS=("${NEW_YAMLS[@]}")
    echo -e "${GREEN}Created ${#ICRA_YAMLS[@]} temp configs with infer_host=${INFER_HOST}${NC}\n"
fi

if [ ${#ICRA_YAMLS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No icra_*.yaml files found in ${CONFIG_DIR}${NC}"
    exit 1
fi

# Backup existing output/benchmark to output/benchmark_{YYYY-MM-DD-HH}
if [ -d "${BENCHMARK_DIR}" ]; then
    TIMESTAMP=$(date +%Y-%m-%d-%H)
    BACKUP_DIR="${OUTPUT_DIR}/benchmark_${TIMESTAMP}"
    echo -e "${YELLOW}Backing up ${BENCHMARK_DIR} -> ${BACKUP_DIR}${NC}"
    mv "${BENCHMARK_DIR}" "${BACKUP_DIR}"
    echo -e "${GREEN}Backup done.${NC}\n"
fi

echo -e "${GREEN}Found ${#ICRA_YAMLS[@]} icra config(s) to run${NC}\n"

SUCCESS_COUNT=0
FAILED_COUNT=0
declare -a FAILED_TASKS

for YAML_PATH in "${ICRA_YAMLS[@]}"; do
    CONFIG_NAME="$(basename "${YAML_PATH}" .yaml)"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Processing: ${YELLOW}${CONFIG_NAME}${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "  Config: ${YAML_PATH}\n"

    cd "${PROJECT_ROOT}"
    /isaac-sim/python.sh source/geniesim/app/app.py --config "${YAML_PATH}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully completed: ${CONFIG_NAME}${NC}\n"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ Failed: ${CONFIG_NAME}${NC}\n"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_TASKS+=("${CONFIG_NAME}")
    fi

    echo ""
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Batch Run Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total configs: ${#ICRA_YAMLS[@]}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAILED_COUNT}${NC}"

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo -e "\n${RED}Failed configs:${NC}"
    for task in "${FAILED_TASKS[@]}"; do
        echo -e "  - ${task}"
    done
fi

echo -e "${GREEN}========================================${NC}"

# Clean up temp configs
if [ -n "${TEMP_CONFIG_DIR}" ] && [ -d "${TEMP_CONFIG_DIR}" ]; then
    rm -rf "${TEMP_CONFIG_DIR}"
fi

# Auto-run score statistics
echo -e "\n${GREEN}Running score statistics...${NC}"
python3 "${SCRIPT_DIR}/stat_average.py" "${BENCHMARK_DIR}"

if [ ${FAILED_COUNT} -gt 0 ]; then
    exit 1
fi
exit 0
