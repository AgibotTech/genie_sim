#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/teleop_runtime.sh"
teleop_select_runtime "$@"
set -- "${TELEOP_RUNTIME_ARGS[@]}"
CONTAINER="${GENIESIM_WUJI_CONTAINER:-$TELEOP_DEFAULT_CONTAINER}"
WORKSPACE="${GENIESIM_WUJI_WORKSPACE:-/workspace}"

# Help and dry-run do not start Docker, ROS nodes or gloves; dry-run needs xacro.
for arg in "$@"; do
    case "$arg" in
        --help|-h|--dry-run)
            echo "Isaac Sim 6.2: starts simulation and manual collection; --external-simulator starts controls only."
            export SIM_REPO_ROOT="$HOST_REPO"
            export PYTHONPATH="$HOST_REPO/source/geniesim_teleop/src:$HOST_REPO/source/geniesim_cli/src:$HOST_REPO/source/geniesim_benchmark/src:${PYTHONPATH:-}"
            exec python3 -m geniesim_teleop.wuji.launch "$@"
            ;;
    esac
done

TTY_ARGS=(-i)
if [[ -t 0 && -t 1 ]]; then
    TTY_ARGS+=(-t)
elif [[ " $* " != *" --check "* ]]; then
    echo "Run Wuji from an interactive terminal so Ctrl-C can stop this session; use --check for preflight." >&2
    exit 2
fi

HOST_TELEOP_APP="${HOST_REPO}/source/geniesim_teleop/src/geniesim_teleop/app"
PINOCCHIO_LIB="${HOST_TELEOP_APP}/vendors/lib/libpinocchio_casadi.so.3.7.0"
PINOCCHIO_TAR="${HOST_TELEOP_APP}/vendors/lib/libpinocchio.tar.gz"
if [ ! -f "$PINOCCHIO_LIB" ] && [ -f "$PINOCCHIO_TAR" ]; then
    echo "Extracting libpinocchio.tar.gz to vendors/lib ..."
    tar -xzvf "$PINOCCHIO_TAR" -C "${HOST_TELEOP_APP}/vendors/lib"
fi

exec docker exec "${TTY_ARGS[@]}" \
    -e "SIM_REPO_ROOT=$WORKSPACE" \
    -e "GENIESIM_WUJI_SIM_PY=${GENIESIM_WUJI_SIM_PY:-$TELEOP_DEFAULT_SIM_PY}" \
    "$CONTAINER" bash -lc '
        set -e
        source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
        # Avoid the optional shared /tmp Fast DDS profile from env.sh; this session
        # selects its own domain in the supervisor and uses local ROS peers.
        unset USE_CONTAINER
        source "$SIM_REPO_ROOT/source/geniesim_teleop/src/geniesim_teleop/app/bin/env.sh"
        export PYTHONPATH="$SIM_REPO_ROOT/source/geniesim_teleop/src:$SIM_REPO_ROOT/source/geniesim_cli/src:$SIM_REPO_ROOT/source/geniesim_benchmark/src:/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
        exec python3 -m geniesim_teleop.wuji.launch "$@"
    ' wuji "$@"
