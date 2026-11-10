#!/usr/bin/env bash
# GenieSim container entrypoint — shared across all Isaac Sim variants.
#
# Two responsibilities:
#   1. Remap the in-container "isaac-sim" user (built at uid:gid 1234:1234
#      in the NVIDIA base image) to match the host user, so files written
#      to bind-mounts have the right ownership on the host.
#   2. Wire up the per-container ROS 2 + Isaac Sim environment and drop
#      privileges before exec'ing the user-supplied command.
#
# Variant-specific behaviour is driven by env vars injected by start.sh:
#
#   ROS_DISTRO                jazzy (default) or humble
#   GENIESIM_PY_CMD           python3 (default) or omni_python
#   GENIESIM_BREAK_SYSTEM_PKGS  1 = add --break-system-packages to pip
#   GENIESIM_CHOWN_KIT_PATH   path to chown for isaacsim/kit (empty = skip)

set -e

# ANSI colors
RST='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
CYAN='\033[36m'
MAGENTA='\033[35m'

echo -e "${MAGENTA}${BOLD}🧞 [entrypoint]${RST} starting GenieSim container setup..."

# Variant defaults (overridden by env vars from start.sh)
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
GENIESIM_PY_CMD="${GENIESIM_PY_CMD:-python3}"
GENIESIM_BREAK_SYSTEM_PKGS="${GENIESIM_BREAK_SYSTEM_PKGS:-1}"
GENIESIM_CHOWN_KIT_PATH="${GENIESIM_CHOWN_KIT_PATH:-}"
GENIESIM_ISAACSIM_KIT_CACHE_PATH="${GENIESIM_ISAACSIM_KIT_CACHE_PATH:-}"
GENIESIM_OVRTX_CACHE_PATH="${GENIESIM_OVRTX_CACHE_PATH:-}"

# ---------------------------------------------------------------------------
# Phase 1 — runtime UID/GID remap (runs as root)
# ---------------------------------------------------------------------------
HOST_UID="${HOST_UID:-1234}"
HOST_GID="${HOST_GID:-1234}"
TARGET_USER="isaac-sim"

if [ "${HOST_UID}" = "1234" ] && [ "${HOST_GID}" = "1234" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  [entrypoint]${RST} HOST_UID/HOST_GID not set — running as uid 1234."
    echo -e "   ${DIM}Files written to bind-mounts will be owned by uid 1234 on the host.${RST}"
    echo -e "   ${DIM}Pass -e HOST_UID=\$(id -u) -e HOST_GID=\$(id -g), or use 'geniesim docker up'.${RST}"
    echo ""
fi

if [ "$(id -u)" = "0" ] && { [ "${HOST_UID}" != "1234" ] || [ "${HOST_GID}" != "1234" ]; }; then
    echo -e "${CYAN}🔄 [entrypoint]${RST} remapping ${BOLD}${TARGET_USER}${RST} 1234:1234 → ${BOLD}${HOST_UID}:${HOST_GID}${RST}"

    EXISTING_USER=$(getent passwd "${HOST_UID}" | cut -d: -f1)
    if [ -n "${EXISTING_USER}" ] && [ "${EXISTING_USER}" != "${TARGET_USER}" ]; then
        userdel "${EXISTING_USER}" 2>/dev/null || true
    fi
    EXISTING_GROUP=$(getent group "${HOST_GID}" | cut -d: -f1)
    if [ -n "${EXISTING_GROUP}" ] && [ "${EXISTING_GROUP}" != "${TARGET_USER}" ]; then
        groupdel "${EXISTING_GROUP}" 2>/dev/null || true
    fi

    if [ "${HOST_GID}" != "1234" ]; then
        groupmod -g "${HOST_GID}" "${TARGET_USER}"
    fi

    if [ "${HOST_UID}" != "1234" ]; then
        usermod -u "${HOST_UID}" -d /home/isaac-sim "${TARGET_USER}"
    fi

    mkdir -p /home/isaac-sim
    chown "${HOST_UID}:${HOST_GID}" /home/isaac-sim
    chown -R "${HOST_UID}:${HOST_GID}" /home/isaac-sim 2>/dev/null || true

    # The selected Kit needs writable EULA/runtime paths for the remapped user.
    if [ -n "${GENIESIM_CHOWN_KIT_PATH}" ]; then
        chown "${HOST_UID}:${HOST_GID}" "${GENIESIM_CHOWN_KIT_PATH}" 2>/dev/null || true
        if [ -f "${GENIESIM_CHOWN_KIT_PATH%/kit}/python_packages/isaacsim/__init__.py" ]; then
            for directory in config data logs; do
                mkdir -p "${GENIESIM_CHOWN_KIT_PATH}/${directory}"
                chown -R "${HOST_UID}:${HOST_GID}" "${GENIESIM_CHOWN_KIT_PATH}/${directory}"
            done
        fi
    fi
    # Recursively chown the named volume cache dirs so the remapped user can write shaders.
    if [ -n "${GENIESIM_ISAACSIM_KIT_CACHE_PATH}" ] && [ -d "${GENIESIM_ISAACSIM_KIT_CACHE_PATH}" ]; then
        chown -R "${HOST_UID}:${HOST_GID}" "${GENIESIM_ISAACSIM_KIT_CACHE_PATH}" 2>/dev/null || true
    fi
    if [ -n "${GENIESIM_OVRTX_CACHE_PATH}" ] && [ -d "${GENIESIM_OVRTX_CACHE_PATH}" ]; then
        chmod -R a+w "${GENIESIM_OVRTX_CACHE_PATH}" 2>/dev/null || true
    fi
    # Ensure /isaac-sim is traversable by the remapped user (NVIDIA base image sets drwxr-x---).
    [ -d /isaac-sim ] && chmod o+rx /isaac-sim 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Write omni_python wrapper (5.1 env isolated to this script only).
# Overwritten on every start so it stays current regardless of image age.
# ---------------------------------------------------------------------------
if [ -f /isaac-sim/python.sh ]; then
    cat >/usr/local/bin/omni_python <<'OMNI_EOF'
#!/bin/bash
export ISAACSIM_HOME=/isaac-sim
_ros_distro="${ROS_DISTRO:-jazzy}"

# Isaac Sim ships its own ROS 2 built for the bundled Python (3.11). The
# system ROS under /opt/ros is built for a different Python (3.12) ABI and
# gets sourced into PYTHONPATH/LD_LIBRARY_PATH by /etc/profile.d/geniesim.sh.
# If both trees stay on the path, rclpy loads message classes from one build
# and typesupport .so from the other and aborts with rcl_interfaces
# ...convert_from_py assertion failures. Strip /opt/ros from both paths so
# omni_python uses only Isaac's internal bridge.
_strip_opt_ros() {
    local out="" entry IFS=':'
    for entry in $1; do
        case "$entry" in
            /opt/ros/*|"") ;;
            *) out="${out:+$out:}$entry" ;;
        esac
    done
    printf '%s' "$out"
}
PYTHONPATH="$(_strip_opt_ros "$PYTHONPATH")"
LD_LIBRARY_PATH="$(_strip_opt_ros "$LD_LIBRARY_PATH")"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/isaac-sim/exts/isaacsim.ros2.bridge/${_ros_distro}/lib"
export PYTHONPATH="/isaac-sim/exts/isaacsim.ros2.bridge/${_ros_distro}/rclpy:${PYTHONPATH}"
[ -f /isaac-sim/setup_ros_env.sh ] && source /isaac-sim/setup_ros_env.sh
exec /isaac-sim/python.sh "$@"
OMNI_EOF
    chmod +x /usr/local/bin/omni_python
    echo -e "${CYAN}🔧 [entrypoint]${RST} omni_python wrapper written to /usr/local/bin/omni_python"
fi

# ---------------------------------------------------------------------------
# Phase 2 — environment setup (runs once at container start)
# ---------------------------------------------------------------------------
echo -e "${CYAN}🔧 [entrypoint]${RST} configuring environment..."
TARGET_HOME=$(getent passwd "${TARGET_USER}" | cut -d: -f6)
if [ -n "${TARGET_HOME}" ] && [ -d /root/.ros/rosdep ] && [ ! -e "${TARGET_HOME}/.ros/rosdep" ]; then
    mkdir -p "${TARGET_HOME}/.ros"
    chown -R "${HOST_UID}:${HOST_GID}" /root/.ros 2>/dev/null || true
    chmod 755 /root 2>/dev/null || true
    ln -s /root/.ros/rosdep "${TARGET_HOME}/.ros/rosdep"
    chown -R "${HOST_UID}:${HOST_GID}" "${TARGET_HOME}/.ros" 2>/dev/null || true
fi
export ROS_DISTRO
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export GENIESIM_IN_CONTAINER=1
export WARP_CACHE_PATH="/tmp/geniesim-newton-warp-${HOST_UID}"
mkdir -p "${WARP_CACHE_PATH}"
chown "${HOST_UID}:${HOST_GID}" "${WARP_CACHE_PATH}" 2>/dev/null || true

# Write /etc/profile.d/geniesim.sh — sourced by all login shells system-wide.
# Overwritten on every container start so ROS_DISTRO is always current.
if [ "${ROS_DISTRO}" = "humble" ]; then
    _ROS_DISCOVERY_LINE="export ROS_LOCALHOST_ONLY=1"
else
    _ROS_DISCOVERY_LINE="export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST"
fi
cat >/etc/profile.d/geniesim.sh <<EOF
export PATH="/usr/local/bin:\${HOME}/.local/bin:\${PATH}"
export XDG_RUNTIME_DIR="/tmp/runtime-\$(id -u)"
mkdir -p "\${XDG_RUNTIME_DIR}" && chmod 700 "\${XDG_RUNTIME_DIR}" 2>/dev/null || true
export GENIESIM_IN_CONTAINER=1
export WARP_CACHE_PATH="/tmp/geniesim-newton-warp-\$(id -u)"
export ROS_DISTRO=${ROS_DISTRO}
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
${_ROS_DISCOVERY_LINE}
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces></General><Discovery><Peers><Peer address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
export GENIESIM_ASSETS_PATH=/geniesim_assets
export SIM_REPO_ROOT=/workspace
export ENABLE_SIM=1
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
[ -f /opt/ros/${ROS_DISTRO}/setup.bash ] && source /opt/ros/${ROS_DISTRO}/setup.bash
if [[ \$- == *i* ]]; then
    alias isaacsim='/isaac-sim/runapp.sh'
    command -v geniesim >/dev/null 2>&1 && eval "\$(geniesim completion bash)" 2>/dev/null || true
fi
EOF
chmod 644 /etc/profile.d/geniesim.sh
# Hook into /etc/bash.bashrc so non-login interactive shells (bash -i) also get it.
if ! grep -q "profile.d/geniesim.sh" /etc/bash.bashrc 2>/dev/null; then
    echo '[ -f /etc/profile.d/geniesim.sh ] && source /etc/profile.d/geniesim.sh' >> /etc/bash.bashrc
fi

# ---------------------------------------------------------------------------
# Phase 3 — install tier-1 geniesim peers (editable, from bind-mounted workspace)
# ---------------------------------------------------------------------------
echo -e "${CYAN}📦 [entrypoint]${RST} installing tier-1 geniesim peers (editable)..."

_pip_install() {
    local py_cmd="$1"; shift
    local pip_break="$1"; shift
    local break_flag=""
    [ "${pip_break}" = "1" ] && break_flag="--break-system-packages"
    # Run as the remapped user when root and using system python3, so
    # egg-info dirs written to the bind-mounted workspace are owned by the
    # host user, not root. omni_python (/isaac-sim/python.sh) is not
    # world-executable after uid remap, so it must run as root.
    local prefix=()
    [ "$(id -u)" = "0" ] && [ "${py_cmd}" = "python3" ] && prefix=(gosu "${TARGET_USER}")
    stdbuf -oL "${prefix[@]}" ${py_cmd} -m pip install \
        ${break_flag} --no-deps --no-build-isolation --root-user-action=ignore \
        "$@" 2>&1 \
        | stdbuf -oL sed -u 's/\x1b\[[0-9;]*m//g' \
        | stdbuf -oL sed -u -E "s/^(Successfully.*)/${GREEN}\1${RST}/; s/^(ERROR.*)/${RED}\1${RST}/" \
        | while IFS= read -r line; do [ -n "$line" ] && echo -e "   ${DIM}${line}${RST}"; done || true
}

# Single source of truth for the tier-1 peer list: the umbrella's
# ``source/geniesim/pyproject.toml``. ``docker/collect_deps.py --peers``
# reads that pyproject (same parser as ``geniesim_cli._tiers``) and
# prints the peer names one per line. Tier-2 peers (geniesim_teleop /
# scene agent / world) are not installed by default. Users opt in via
# ``pip install -e "source/geniesim/[teleop|scene_agent|world|all]"``
# inside the container.
#
# Two non-pyproject inputs the script can't derive:
#   * ``/geniesim_assets`` — distributed out-of-band, baked into the image;
#     installed first so siblings can ``import geniesim_assets`` at install.
#   * ``geniesim_cli`` — always installed under system python3 so the
#     console script lands in /usr/local/bin (handled by the regular loop).
EDITABLE_ARGS=""
if [ -f /geniesim_assets/pyproject.toml ]; then
    EDITABLE_ARGS="${EDITABLE_ARGS} -e /geniesim_assets"
    echo -e "   ${DIM}+ geniesim_assets${RST}"
fi

# Derive the rest from the umbrella's pyproject.
_PEERS_SCRIPT="/workspace/docker/collect_deps.py"
if [ -x "${_PEERS_SCRIPT}" ] || [ -f "${_PEERS_SCRIPT}" ]; then
    while IFS= read -r peer; do
        # Skip peers we already added explicitly (geniesim_assets) or
        # that aren't checked out in the mounted workspace.
        [ -z "${peer}" ] && continue
        [ "${peer}" = "geniesim_assets" ] && continue
        # GENIESIM_SKIP_PEERS: optional space/comma separated skip list. Some
        # peers' setup.py depend on an external checkout outside the repo (e.g.
        # geniesim_world needs external/ml-sharp); installing them in a runtime
        # that does not need them (rlinf sim / teleop) only makes the whole pip
        # command fail by association. Unset means behaviour is unchanged.
        case ",${GENIESIM_SKIP_PEERS//[ ]/,}," in
            *",${peer},"*) echo -e "   ${DIM}- ${peer} (GENIESIM_SKIP_PEERS)${RST}"; continue ;;
        esac
        pkg_dir="/workspace/source/${peer}"
        if [ -f "${pkg_dir}/pyproject.toml" ]; then
            EDITABLE_ARGS="${EDITABLE_ARGS} -e ${pkg_dir}"
            echo -e "   ${DIM}+ ${peer}${RST}"
        else
            echo -e "   ${YELLOW}⚠ ${peer} listed in umbrella pyproject but not mounted at ${pkg_dir} — skipping${RST}"
        fi
    done < <(python3 "${_PEERS_SCRIPT}" /workspace/source --peers 2>/dev/null)
else
    echo -e "   ${RED}❌ ${_PEERS_SCRIPT} missing — cannot derive tier-1 peer list. Aborting install.${RST}"
    EDITABLE_ARGS=""
fi

if [ -n "${EDITABLE_ARGS}" ]; then
    # Always install into system python3 (console scripts land in /usr/local/bin).
    _pip_install python3 "1" ${EDITABLE_ARGS}

    # For omni_python variants, also install into the bundled interpreter so
    # simulation code can import geniesim packages directly.
    if [ "${GENIESIM_PY_CMD}" != "python3" ]; then
        echo -e "   ${DIM}(also installing into ${GENIESIM_PY_CMD})${RST}"
        _pip_install "${GENIESIM_PY_CMD}" "${GENIESIM_BREAK_SYSTEM_PKGS}" ${EDITABLE_ARGS}
    fi
fi

# ---------------------------------------------------------------------------
# Phase 3b — optional native libuipc through the shared bootstrap entry point
# ---------------------------------------------------------------------------
# Only images that prepare native libuipc at build time enable this phase.
# The dedicated 6.2 image uses system CPython 3.12 for the native runtime.
# The mounted CLI source also works before the remapped user's editable install
# is visible to root. Native files belong to the shared evaluation interpreter.
if [ -f /opt/geniesim/docker/install_libuipc.py ]; then
    if ! GENIESIM_REPO_ROOT=/workspace \
        PYTHONPATH="/workspace/source/geniesim_cli/src${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 -m geniesim_cli bootstrap --libuipc-only; then
        echo -e "${YELLOW}⚠️ [entrypoint]${RST} libuipc bootstrap unavailable. Continuing container startup."
    fi
fi

# ---------------------------------------------------------------------------
# Phase 4 — drop privileges and exec the command
# ---------------------------------------------------------------------------
echo -e "${GREEN}${BOLD}✅ [entrypoint]${RST} ${GREEN}GENIESIM_READY${RST}"
if [ "$(id -u)" = "0" ]; then
    exec gosu "${TARGET_USER}" "$@"
fi
exec "$@"
