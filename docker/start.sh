#!/usr/bin/env bash
# Launch the GenieSim container.
#
# Invoked by `geniesim docker[<variant>] up`. The CLI wrapper injects all
# variant-specific values as env vars before calling this script, so a
# single shared script handles all Isaac Sim versions.
#
# Env vars set by the CLI (all have sensible defaults for standalone use):
#
#   GENIESIM_IMAGE            docker image to run
#   GENIESIM_CONTAINER        container name
#   GENIESIM_WORKSPACE        host path to bind-mount at /workspace
#                               (default: $(pwd))
#   GENIESIM_CACHE_ROOT       host path for Isaac Sim cache
#                               (default: $HOME/docker/isaac-sim)
#   GENIESIM_VARIANT_LABEL    label used in messages (e.g. "docker5.1")
#   GENIESIM_EXTRA_CACHE_DIRS space-separated extra subdirs under cache/main
#                               (e.g. "kit pip numba" for Isaac Sim 6.0)
#
#   Passed through into the container for entrypoint.sh:
#   ROS_DISTRO                jazzy or humble
#   GENIESIM_PY_CMD           python3 or omni_python
#   GENIESIM_BREAK_SYSTEM_PKGS  1 = add --break-system-packages to pip
#   GENIESIM_CHOWN_KIT_PATH   path to chown for isaacsim/kit (empty = skip)
#
# Flags:
#   --headless    Skip X11 mounts / DISPLAY plumbing / /dev/input mount.
#                 Use for CI / batch / no-GUI workflows.

set -eo pipefail

# ANSI colors
RST='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
CYAN='\033[36m'
MAGENTA='\033[35m'

HEADLESS=0
for arg in "$@"; do
    case "$arg" in
        --headless) HEADLESS=1 ;;
        *) echo -e "${RED}❌ [geniesim docker] unknown arg: $arg${RST}" >&2; exit 2 ;;
    esac
done

IMAGE="${GENIESIM_IMAGE:-registry.agibot.com/genie-sim/geniesim4:latest}"
NAME="${GENIESIM_CONTAINER:-geniesim}"
HOST_WS="${GENIESIM_WORKSPACE:-$(pwd)}"
CACHE_ROOT="${GENIESIM_CACHE_ROOT:-$HOME/docker/isaac-sim}"
VARIANT_LABEL="${GENIESIM_VARIANT_LABEL:-docker}"

mkdir -p "${CACHE_ROOT}/cache/main/ov" \
         "${CACHE_ROOT}/cache/main/warp" \
         "${CACHE_ROOT}/cache/computecache" \
         "${CACHE_ROOT}/config" \
         "${CACHE_ROOT}/data/documents" \
         "${CACHE_ROOT}/data/Kit" \
         "${CACHE_ROOT}/logs" \
         "${CACHE_ROOT}/pkg"

for extra_dir in ${GENIESIM_EXTRA_CACHE_DIRS:-}; do
    mkdir -p "${CACHE_ROOT}/cache/main/${extra_dir}"
done

# geniesim_assets source root (passed by CLI via GENIESIM_ASSETS_SRC).
ASSETS_SRC="${GENIESIM_ASSETS_SRC:-}"

if [ -z "${ASSETS_SRC}" ] || [ ! -f "${ASSETS_SRC}/pyproject.toml" ]; then
    echo -e "${RED}❌ geniesim_assets is not pip-installed (editable) on the host.${RST}"
    echo -e "   ${DIM}Install it first:${RST}  ${CYAN}pip install -e /path/to/geniesim_assets${RST}"
    exit 1
fi
echo -e "   ${DIM}Assets:${RST}    ${CYAN}${ASSETS_SRC}${RST}"

# Common docker run args — image-independent, present in both modes.
RUN_ARGS=(
    -itd
    --name "${NAME}"
    --init
    --gpus all
    --network=host
    --privileged
    -e "ACCEPT_EULA=Y"
    -e "OMNI_KIT_ACCEPT_EULA=Y"
    -e "PRIVACY_CONSENT=Y"
    -e "HOST_UID=$(id -u)"
    -e "HOST_GID=$(id -g)"
    -e "ROS_DISTRO=${ROS_DISTRO:-jazzy}"
    -e "GENIESIM_PY_CMD=${GENIESIM_PY_CMD:-python3}"
    -e "GENIESIM_BREAK_SYSTEM_PKGS=${GENIESIM_BREAK_SYSTEM_PKGS:-1}"
    -e "GENIESIM_CHOWN_KIT_PATH=${GENIESIM_CHOWN_KIT_PATH:-}"
    -e "GENIESIM_ISAACSIM_KIT_CACHE_PATH=${GENIESIM_ISAACSIM_KIT_CACHE_PATH:-}"
    -e "GENIESIM_OVRTX_CACHE_PATH=${GENIESIM_OVRTX_CACHE_PATH:-}"
    -v "${CACHE_ROOT}/cache/main:/home/isaac-sim/.cache:rw"
    -v "${CACHE_ROOT}/cache/computecache:/home/isaac-sim/.nv/ComputeCache:rw"
    -v "${CACHE_ROOT}/logs:/home/isaac-sim/.nvidia-omniverse/logs:rw"
    -v "${CACHE_ROOT}/config:/home/isaac-sim/.nvidia-omniverse/config:rw"
    -v "${CACHE_ROOT}/data:/home/isaac-sim/.local/share/ov/data:rw"
    -v "${CACHE_ROOT}/pkg:/home/isaac-sim/.local/share/ov/pkg:rw"
    -v "${HOST_WS}:/workspace:rw"
    -v "${ASSETS_SRC}:/geniesim_assets:rw"
    -v "${HOST_WS}/docker/entrypoint.sh:/usr/local/bin/geniesim-entrypoint:ro"
    -w /workspace
)

# Named volumes for in-package shader caches — seeded from image on first create,
# persisted across container restarts. Only mounted when the path is set for the variant.
# OptiX AI denoiser weights, shipped with the host NVIDIA driver and absent from
# the Isaac Sim image. Path-traced rendering loads them every frame and logs
# "Unable to load denoiser weights" on each attempt when the file is missing;
# raytraced lighting never touches it. Only affects image noise, not physics.
if [ -f /usr/share/nvidia/nvoptix.bin ]; then
    RUN_ARGS+=(-v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro)
fi

if [ -n "${GENIESIM_ISAACSIM_KIT_CACHE_PATH:-}" ]; then
    RUN_ARGS+=(--mount "source=${NAME}-isaacsim-kit-cache,target=${GENIESIM_ISAACSIM_KIT_CACHE_PATH}")
fi
if [ -n "${GENIESIM_OVRTX_CACHE_PATH:-}" ]; then
    RUN_ARGS+=(--mount "source=${NAME}-ovrtx-cache,target=${GENIESIM_OVRTX_CACHE_PATH}")
fi

# GUI-only additions: X11 socket, Xauthority, joystick / device input.
if [ "${HEADLESS}" -eq 0 ]; then
    xhost +local: >/dev/null 2>&1 || true
    RUN_ARGS+=(
        -e DISPLAY
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
        -v "${HOME}/.Xauthority:/home/isaac-sim/.Xauthority:ro"
        -v /dev/input:/dev/input:rw
    )
fi

docker run "${RUN_ARGS[@]}" "${IMAGE}" tail -f /dev/null >/dev/null

# Stream entrypoint logs until "ready" appears.
docker logs -f "${NAME}" 2>&1 &
LOGS_PID=$!
for _ in $(seq 1 60); do
    if docker logs "${NAME}" 2>&1 | grep -q "GENIESIM_READY"; then
        break
    fi
    sleep 0.5
done
kill "${LOGS_PID}" 2>/dev/null || true
wait "${LOGS_PID}" 2>/dev/null || true

echo ""
MODE_LABEL="gui"; [ "${HEADLESS}" -eq 1 ] && MODE_LABEL="headless"
echo -e "${GREEN}${BOLD}✅ container '${NAME}' is up (${MODE_LABEL})${RST} — enter with: ${BOLD}geniesim ${VARIANT_LABEL} into${RST}"
