#!/usr/bin/env bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
# Prepare the mounted runtime; the batch launcher owns the ROS workspace build.
set -e

export SIM_REPO_ROOT="${SIM_REPO_ROOT:-/workspace}"
export GENIESIM_ASSETS_PATH="${GENIESIM_ASSETS_PATH:-${SIM_ASSETS_ROOT:-/geniesim_assets}}"
export RLINF_WS_BUILD="${RLINF_WS_BUILD:-/tmp/ros_ws_build}"
export GENIESIM_SKIP_PEERS="${GENIESIM_SKIP_PEERS:-geniesim_world}"

# Portable Kit stores runtime files below its installation directory.
if [ "$(id -u)" = 0 ] && [ -d /isaac-sim/kit ]; then
  for directory in cache config data logs; do
    mkdir -p "/isaac-sim/kit/$directory"
    chown -R "${HOST_UID:-1234}:${HOST_GID:-1234}" "/isaac-sim/kit/$directory"
  done
fi

if [ -f "$GENIESIM_ASSETS_PATH/pyproject.toml" ]; then
  pip3 install --break-system-packages --no-deps -q -e "$GENIESIM_ASSETS_PATH"
fi
exec /usr/local/bin/geniesim-entrypoint "$@"
