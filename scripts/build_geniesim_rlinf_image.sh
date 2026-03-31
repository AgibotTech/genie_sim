#!/bin/bash
# Build the geniesim-rlinf Docker image.
#
# Usage:
#   bash main/scripts/build_geniesim_rlinf_image.sh [tag]
#
# Build context is main/scripts/ (not main/) to avoid sending large asset
# directories to the Docker daemon.

set -eo pipefail

IMAGE_TAG="${1:-geniesim-rlinf:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[build] Building ${IMAGE_TAG}"
echo "[build] Dockerfile : ${SCRIPT_DIR}/dockerfile_geniesim_rlinf"
echo "[build] Context    : ${SCRIPT_DIR}"

docker build \
  -f "${SCRIPT_DIR}/dockerfile_geniesim_rlinf" \
  -t "${IMAGE_TAG}" \
  "${SCRIPT_DIR}"

echo "[build] Done: ${IMAGE_TAG}"
