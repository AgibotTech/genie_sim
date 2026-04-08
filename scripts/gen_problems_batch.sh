#!/bin/bash
set -euo pipefail

# Batch-generate problems.json for all scenes under a llm_task/<task>/ directory.
#
# Usage:
#   ./scripts/gen_problems_batch.sh place_block_into_box
#   ./scripts/gen_problems_batch.sh place_block_into_box --force
#
# Notes:
# - Scene directories are expected at: source/geniesim/benchmark/config/llm_task/<task>/<scene_id>/
# - Each scene directory should contain instructions.json.
# - problems.json will be written into each scene directory by eval_gen.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_GEN="${PROJECT_ROOT}/source/geniesim/evaluator/generators/eval_gen.py"
LLM_TASK_ROOT="${PROJECT_ROOT}/source/geniesim/benchmark/config/llm_task"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_name> [--force]"
  exit 2
fi

TASK_NAME="$1"
shift || true

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 <task_name> [--force]"
      exit 2
      ;;
  esac
done

TASK_DIR="${LLM_TASK_ROOT}/${TASK_NAME}"
if [[ ! -d "${TASK_DIR}" ]]; then
  echo -e "${RED}Error: task directory not found:${NC} ${TASK_DIR}"
  exit 1
fi

if [[ ! -f "${EVAL_GEN}" ]]; then
  echo -e "${RED}Error: eval_gen.py not found:${NC} ${EVAL_GEN}"
  exit 1
fi

# Ensure geniesim package is importable when running eval_gen.py directly.
export PYTHONPATH="${PROJECT_ROOT}/source:${PYTHONPATH:-}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Problems Batch Generator${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Task: ${YELLOW}${TASK_NAME}${NC}"
echo -e "Dir : ${YELLOW}${TASK_DIR}${NC}"
if [[ "${FORCE}" -eq 1 ]]; then
  echo -e "Mode: ${YELLOW}force overwrite problems.json${NC}"
else
  echo -e "Mode: ${YELLOW}skip when problems.json exists${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
declare -a FAILED_SCENES

shopt -s nullglob
SCENE_DIRS=( "${TASK_DIR}"/*/ )
shopt -u nullglob

if [[ ${#SCENE_DIRS[@]} -eq 0 ]]; then
  echo -e "${RED}Error: no scene subdirectories found under:${NC} ${TASK_DIR}"
  exit 1
fi

for SCENE_DIR in "${SCENE_DIRS[@]}"; do
  SCENE_DIR="${SCENE_DIR%/}"
  SCENE_ID="$(basename "${SCENE_DIR}")"

  if [[ ! -f "${SCENE_DIR}/instructions.json" ]]; then
    echo -e "${YELLOW}[skip]${NC} ${TASK_NAME}/${SCENE_ID} (missing instructions.json)"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  if [[ "${FORCE}" -ne 1 && -f "${SCENE_DIR}/problems.json" ]]; then
    echo -e "${YELLOW}[skip]${NC} ${TASK_NAME}/${SCENE_ID} (problems.json exists)"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  echo -e "${GREEN}[run]${NC} ${TASK_NAME}/${SCENE_ID}"

  set +e
  python3 "${EVAL_GEN}" --scene_dir "${SCENE_DIR}"
  RC=$?
  set -e

  if [[ ${RC} -eq 0 ]]; then
    if [[ -f "${SCENE_DIR}/problems.json" ]]; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      echo -e "${RED}[fail]${NC} ${TASK_NAME}/${SCENE_ID} (exit 0 but problems.json not created)"
      FAILED_COUNT=$((FAILED_COUNT + 1))
      FAILED_SCENES+=("${TASK_NAME}/${SCENE_ID}")
    fi
  else
    echo -e "${RED}[fail]${NC} ${TASK_NAME}/${SCENE_ID} (exit ${RC})"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    FAILED_SCENES+=("${TASK_NAME}/${SCENE_ID}")
  fi
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total scenes : ${#SCENE_DIRS[@]}"
echo -e "${GREEN}Success     : ${SUCCESS_COUNT}${NC}"
echo -e "${YELLOW}Skipped     : ${SKIPPED_COUNT}${NC}"
echo -e "${RED}Failed      : ${FAILED_COUNT}${NC}"

if [[ ${FAILED_COUNT} -gt 0 ]]; then
  echo -e "\n${RED}Failed scenes:${NC}"
  for s in "${FAILED_SCENES[@]}"; do
    echo " - ${s}"
  done
  exit 1
fi

exit 0
