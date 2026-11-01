#!/usr/bin/env bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

set -euo pipefail

mkdir -p "${HOME}" "${SCENE_AGENT_CACHE}" "${SCENE_AGENT_OUTPUT}" "${SCENE_AGENT_MODEL_DIR}"

if [ -d "${GENIESIM_ASSETS}/objects" ]; then
    python /opt/scene-agent/examples/bind_assets.py --assets-root "${GENIESIM_ASSETS}"
fi

exec "$@"
