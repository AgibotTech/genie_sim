#!/bin/bash
# GenieSim × RLinf container entrypoint.
#
# Responsibilities:
#   1. Source ROS 2 (Jazzy) for the current session.
#   2. Build the geniesim_rl_interfaces colcon workspace if it hasn't been
#      built yet for the container's Python version (avoids stale cross-distro
#      artifacts from a host humble build).
#   3. Source the workspace so LD_LIBRARY_PATH / AMENT_PREFIX_PATH are set.
#   4. Install the geniesim Python source into IsaacSim python (editable, best-effort).
#   5. Write ~/.bashrc additions so interactive `docker exec` shells are ready.
#   6. exec "$@"  — run whatever CMD was passed (default: tail -f /dev/null).
#
# All setup is idempotent: re-running the script (container restart) is safe.

set -eo pipefail
# Note: -u (treat unset vars as error) is intentionally omitted.
# ROS setup.bash references variables like AMENT_TRACE_SETUP_FILES without
# guarding them, which would cause spurious failures under set -u.

export ISAACSIM_HOME="${ISAACSIM_HOME:-/isaac-sim}"
export GENIESIM_ROOT="${GENIESIM_ROOT:-/geniesim/main}"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
# SIM_REPO_ROOT is expected by geniesim's system_utils.py for path resolution.
# It points to the main code directory (which contains source/geniesim/...).
export SIM_REPO_ROOT="${SIM_REPO_ROOT:-${GENIESIM_ROOT}}"

# ---------------------------------------------------------------------------
# ACL: ensure user 1234 can write into the mounted repo dirs.
# We only set the ACL recursively on directories that need write access
# at container startup (workspace build, log dirs, etc.).
# The setfacl calls modify the host filesystem via the bind mount.
# ---------------------------------------------------------------------------
if command -v sudo >/dev/null 2>&1 && command -v setfacl >/dev/null 2>&1; then
  # Top-level dirs: need traversal only
  sudo setfacl -m u:1234:rwX "${GENIESIM_ROOT}"           2>/dev/null || true
  sudo setfacl -m u:1234:rwX "${SIM_REPO_ROOT}"           2>/dev/null || true
  # ros_interfaces: colcon source directory
  if [ -d "${SIM_REPO_ROOT}/source/geniesim/rl/ros_interfaces" ]; then
    sudo setfacl -Rm u:1234:rwX "${SIM_REPO_ROOT}/source/geniesim/rl/ros_interfaces" 2>/dev/null || true
  fi
fi

# ---------------------------------------------------------------------------
# ~/.bashrc additions for interactive shells (idempotent guard)
# ---------------------------------------------------------------------------
if ! grep -q "# >>> GenieSim RLinf container setup" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc << BASHRC
# >>> GenieSim RLinf container setup
export GENIESIM_ROOT=${GENIESIM_ROOT}
export ISAACSIM_HOME=${ISAACSIM_HOME}
export SIM_REPO_ROOT=${SIM_REPO_ROOT}
export PYTHONPATH=\${GENIESIM_ROOT}/RLinf:\${PYTHONPATH:-}
alias omni_python='${ISAACSIM_HOME}/python.sh'
export RMW_IMPLEMENTATION=\${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export ROS_LOCALHOST_ONLY=\${ROS_LOCALHOST_ONLY:-1}
export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}
source /opt/ros/${ROS_DISTRO}/setup.bash
BASHRC
fi

# ---------------------------------------------------------------------------
# Source ROS for the current session
# ---------------------------------------------------------------------------
source /opt/ros/${ROS_DISTRO}/setup.bash

# ---------------------------------------------------------------------------
# Build geniesim_rl_interfaces if missing or built for wrong Python version.
#
# The source lives in the MOUNTED repo (read-only from container perspective
# without extra setup).  We build into /geniesim/ros_ws_build INSIDE the
# container so that:
#   - We avoid permission issues with the bind-mounted source tree.
#   - The build artifacts are owned by uid 1234 (no chown needed).
#   - A named volume can optionally be attached at /geniesim/ros_ws_build to
#     persist the build across container restarts.
# ---------------------------------------------------------------------------
WS_SRC="${SIM_REPO_ROOT}/source/geniesim/rl/ros_interfaces"
WS_BUILD="/geniesim/ros_ws_build"   # writable container-local path

if [ -d "${WS_SRC}" ]; then
  mkdir -p "${WS_BUILD}/src"

  # Symlink the source package into the build workspace (read-only source is fine).
  if [ ! -e "${WS_BUILD}/src/geniesim_rl_interfaces" ]; then
    ln -sfn "${WS_SRC}" "${WS_BUILD}/src/geniesim_rl_interfaces"
  fi

  # Detect the Python version the system python3 uses (must match colcon build).
  PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.x")
  PKG_SITE="${WS_BUILD}/install/geniesim_rl_interfaces/lib/python${PYVER}/site-packages"

  if [ ! -d "${PKG_SITE}" ]; then
    echo "[entrypoint] Building geniesim_rl_interfaces for python${PYVER} / ROS ${ROS_DISTRO}..."
    (cd "${WS_BUILD}" && colcon build --packages-select geniesim_rl_interfaces --symlink-install)
    echo "[entrypoint] Build complete."
  else
    echo "[entrypoint] geniesim_rl_interfaces already built for python${PYVER}."
  fi

  # Source workspace (sets LD_LIBRARY_PATH, AMENT_PREFIX_PATH, PYTHONPATH).
  source "${WS_BUILD}/install/setup.bash"

  # Persist workspace source in ~/.bashrc for interactive shells.
  if ! grep -q "ros_ws_build/install/setup.bash" ~/.bashrc 2>/dev/null; then
    echo "source ${WS_BUILD}/install/setup.bash" >> ~/.bashrc
  fi
else
  echo "[entrypoint] WARNING: ${WS_SRC} not found." >&2
  echo "[entrypoint] ROS services will be unavailable until the workspace is built." >&2
fi

# ---------------------------------------------------------------------------
# Install geniesim Python source into IsaacSim python (editable, best-effort)
# This allows rl_renderer.py (which runs under /isaac-sim/python.sh) to
# import geniesim packages.
# ---------------------------------------------------------------------------
echo "[entrypoint] Installing geniesim into IsaacSim python (editable)..."
"${ISAACSIM_HOME}/python.sh" -m pip install -q -U pip 2>/dev/null || true
"${ISAACSIM_HOME}/python.sh" -m pip install -q -e "${SIM_REPO_ROOT}/source" 2>/dev/null || true

echo "[entrypoint] Container initialised. ROS_DISTRO=${ROS_DISTRO}, GENIESIM_ROOT=${GENIESIM_ROOT}"

# ---------------------------------------------------------------------------
# Hand off to the container's main command (default: tail -f /dev/null).
# ---------------------------------------------------------------------------
exec "$@"
