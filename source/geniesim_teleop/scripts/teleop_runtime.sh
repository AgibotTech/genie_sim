#!/usr/bin/env bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

# Shared host-side selection for collection, Wuji and post-processing.
teleop_select_runtime() {
    TELEOP_ISAAC_VERSION="${GENIESIM_TELEOP_ISAAC_VERSION:-6.2}"
    TELEOP_RUNTIME_ARGS=()
    while (($#)); do
        case "$1" in
            --isaac-version)
                if (($# < 2)); then
                    echo "--isaac-version requires 5.1 or 6.2" >&2
                    return 2
                fi
                TELEOP_ISAAC_VERSION="$2"
                shift 2 ;;
            --isaac-version=*) TELEOP_ISAAC_VERSION="${1#*=}"; shift ;;
            *) TELEOP_RUNTIME_ARGS+=("$1"); shift ;;
        esac
    done
    case "$TELEOP_ISAAC_VERSION" in
        5.1)
            TELEOP_DEFAULT_CONTAINER=geniesim3
            TELEOP_DEFAULT_SIM_PY=omni_python
            TELEOP_SKIP_SIMULATOR=0 ;;
        6.2)
            TELEOP_DEFAULT_CONTAINER=geniesim4
            TELEOP_DEFAULT_SIM_PY=python3
            TELEOP_SKIP_SIMULATOR=1 ;;
        *) echo "Unsupported Isaac version: $TELEOP_ISAAC_VERSION (use 5.1 or 6.2)" >&2; return 2 ;;
    esac
}
