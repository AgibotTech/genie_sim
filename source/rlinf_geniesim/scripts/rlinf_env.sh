#!/usr/bin/env bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
# Source this file to select the mounted repository and installed ROS workspace.

export SIM_REPO_ROOT="${SIM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export GENIESIM_ASSETS_PATH="${GENIESIM_ASSETS_PATH:-${SIM_ASSETS_ROOT:-/geniesim_assets}}"
export RLINF_WS_BUILD="${RLINF_WS_BUILD:-${GENIESIM_ENGINE_WS:-/tmp/ros_ws_build}}"
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" || return 1
if [ -f "$RLINF_WS_BUILD/install/setup.bash" ]; then
  source "$RLINF_WS_BUILD/install/setup.bash" || return 1
fi
export PYTHONPATH="$SIM_REPO_ROOT/source:$SIM_REPO_ROOT/source/geniesim_benchmark/src${PYTHONPATH:+:$PYTHONPATH}"
export GENIESIM_PERF_LOG="${GENIESIM_PERF_LOG:-0}"
export GENIESIM_RENDER_PERF_LOG="${GENIESIM_RENDER_PERF_LOG:-0}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}" OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}" PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
