#!/usr/bin/env bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
# Start an idle ZMQ server. Clients select tasks, physics, and rendering at init.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SIM_REPO_ROOT:-$(cd "$HERE/../../.." && pwd)}"
ASSETS_DIR="${SIM_ASSETS_ROOT:-${GENIESIM_ASSETS_PATH:-/geniesim_assets}}"
PORT=5601; GPU=""; ROS_DOMAIN=""; ACTION=start; FOREGROUND=0; FOLLOW=0
FABRIC="${RLINF_TILED_FABRIC:-1}"; HEADLESS="${HEADLESS:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"; REBUILD_WS=0
BARRIER_CAP_S="${RLINF_TILED_BARRIER_CAP_S:-}"
export RLINF_WS_BUILD="${RLINF_WS_BUILD:-${GENIESIM_ENGINE_WS:-/tmp/ros_ws_build}}"

usage() {
  cat <<'HELP'
Usage: start_sim_server.sh [options]
  --port PORT             ZMQ port (default: 5601)
  --code DIR              repository root (default: SIM_REPO_ROOT or this checkout)
  --assets DIR            asset package root (default: SIM_ASSETS_ROOT or /geniesim_assets)
  --device GPU            CUDA_VISIBLE_DEVICES for this server (--gpu is an alias)
  --domain ID             ROS domain (default: derived from the port)
  --no-fabric             disable Fabric rendering
  --barrier-cap-s SECONDS tiled render barrier timeout cap
  --rebuild-ws            rebuild the installed ROS workspace
  --foreground            keep the launcher attached to the server
  --status | --stop       inspect or stop only this port's recorded process
  --logs [--follow]       read the last 100 log lines; optionally follow replacements
HELP
}
need_value() { [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "Missing value for $1" >&2; exit 2; }; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) need_value "$@"; PORT="$2"; shift 2;;
    --code) need_value "$@"; CODE_DIR="$2"; shift 2;;
    --assets) need_value "$@"; ASSETS_DIR="$2"; shift 2;;
    --device|--gpu) need_value "$@"; GPU="$2"; shift 2;;
    --domain|--ros-domain) need_value "$@"; ROS_DOMAIN="$2"; shift 2;;
    --barrier-cap-s) need_value "$@"; BARRIER_CAP_S="$2"; shift 2;;
    --no-fabric) FABRIC=0; shift;;
    --rebuild-ws) REBUILD_WS=1; shift;;
    --foreground) FOREGROUND=1; shift;;
    --stop) ACTION=stop; shift;;
    --status) ACTION=status; shift;;
    --logs) ACTION=logs; shift;;
    --follow) FOLLOW=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
. "$HERE/_server_slot.sh"
. "$HERE/_ws_build_stamp.sh"
slot_derive "$PORT" "$ROS_DOMAIN" || exit 2
if [ "$FOLLOW" = 1 ] && [ "$ACTION" != logs ]; then
  echo '--follow requires --logs' >&2; exit 2
fi
if [ "$ACTION" = logs ]; then
  [ -f "$SLOT_LOG" ] || { echo "Server log not found: $SLOT_LOG" >&2; exit 1; }
  tail_args=(-n 100)
  if [ "$FOLLOW" = 1 ]; then tail_args+=(-F); fi
  exec tail "${tail_args[@]}" -- "$SLOT_LOG"
fi
if [ "$ACTION" = status ]; then
  if [ -f "$SLOT_PIDFILE" ] && slot_alive "$(cat "$SLOT_PIDFILE")"; then
    echo "$SLOT_TAG running (pid=$(cat "$SLOT_PIDFILE")); log=$SLOT_LOG"
  else
    echo "$SLOT_TAG stopped"
  fi
  exit 0
fi
mkdir -p "$SLOT_RUN_DIR"
exec 8>"$SLOT_RUN_DIR/launcher.lock"
flock 8
if [ "$ACTION" = stop ]; then slot_stop; exit $?; fi
if [ -f "$SLOT_PIDFILE" ] && slot_alive "$(cat "$SLOT_PIDFILE")"; then
  echo "$SLOT_TAG already has a running server; use --stop first" >&2
  exit 3
fi
[[ "$GPU" =~ ^([0-9]+(,[0-9]+)*)?$ ]] || { echo "Invalid GPU list: $GPU" >&2; exit 2; }
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid ready timeout" >&2; exit 2; }
[[ "$FABRIC" =~ ^[01]$ && "$HEADLESS" =~ ^[01]$ ]] || { echo "Fabric/headless must be 0 or 1" >&2; exit 2; }
if [ -n "$BARRIER_CAP_S" ]; then
  [[ "$BARRIER_CAP_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid barrier cap" >&2; exit 2; }
  export RLINF_TILED_BARRIER_CAP_S="$BARRIER_CAP_S"
fi
[ -d "$CODE_DIR/source/rlinf_geniesim" ] || { echo "Repository missing at $CODE_DIR" >&2; exit 2; }
[ -d "$ASSETS_DIR" ] || { echo "Assets missing at $ASSETS_DIR" >&2; exit 2; }
CODE_DIR="$(cd "$CODE_DIR" && pwd -P)"; ASSETS_DIR="$(cd "$ASSETS_DIR" && pwd -P)"
export SIM_REPO_ROOT="$CODE_DIR" GENIESIM_ASSETS_PATH="$ASSETS_DIR"
export ROS_DOMAIN_ID="$SLOT_DOMAIN" RLINF_SERVER_TAG="$SLOT_TAG"
export RLINF_TILED_HEADLESS="$HEADLESS" RLINF_TILED_READY_TIMEOUT="$READY_TIMEOUT" RLINF_TILED_FABRIC="$FABRIC"
export RLINF_FUSED_READY_TIMEOUT="${RLINF_FUSED_READY_TIMEOUT:-$READY_TIMEOUT}"
export RLINF_ISAAC_PYTHON="${RLINF_ISAAC_PYTHON:-/isaac-sim/python.sh}"
[ -z "$GPU" ] || export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$(dirname "$RLINF_WS_BUILD")"
exec 9>"$RLINF_WS_BUILD.lock"
flock 9
ws_prepare "$CODE_DIR/source/geniesim_ros/src/ros_ws/src" "$RLINF_WS_BUILD" "$REBUILD_WS"
source "$HERE/rlinf_env.sh"
printf 'comm:\n  backend: zmq\n  bind: tcp://0.0.0.0:%s\n  jpeg_quality: 85\n' "$PORT" > "$SLOT_COMM_CFG"
rm -f "$SLOT_PIDFILE"
# Passing positional arguments avoids shell interpretation of paths and flags.
setsid bash -c '
  echo "$$" > "$1"
  exec "$2" "$3/source/rlinf_geniesim/scripts/sim_server.py" --comm-config "$4"
' _ "$SLOT_PIDFILE" "${RLINF_SERVER_PYTHON:-python3}" "$CODE_DIR" "$SLOT_COMM_CFG" \
  8>&- 9>&- >"$SLOT_LOG" 2>&1 </dev/null &
server_pid=$!
for ((attempt=0; attempt<100; attempt++)); do
  [ -s "$SLOT_PIDFILE" ] && break
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 0.05
done
exec 9>&-
cleanup_failed_start() { slot_stop 2; }
trap 'cleanup_failed_start; exit 130' INT TERM
ready=0
for ((attempt=0; attempt<READY_TIMEOUT; attempt++)); do
  if grep -qE 'serving \(idle' "$SLOT_LOG"; then ready=1; break; fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "Server failed to become ready; log: $SLOT_LOG" >&2
  tail -n 12 "$SLOT_LOG" >&2
  cleanup_failed_start
  exit 1
fi
# The batch launcher receives the identity before this slot can be replaced.
if [ -n "${_RLINF_LAUNCH_RECEIPT:-}" ]; then
  receipt_pid="$(cat "$SLOT_PIDFILE" 2>/dev/null)" || receipt_pid=""
  receipt_start="$(slot_process_start "$receipt_pid")" || receipt_start=""
  if [ -z "$receipt_start" ] || ! printf '%s:%s\n' "$receipt_pid" "$receipt_start" > "$_RLINF_LAUNCH_RECEIPT"; then
    cleanup_failed_start
    exit 1
  fi
fi
echo "$SLOT_TAG serving at tcp://127.0.0.1:$PORT; ROS domain=$SLOT_DOMAIN; log=$SLOT_LOG"
exec 8>&-
if [ "$FOREGROUND" = 1 ]; then
  wait "$server_pid"
else
  trap - INT TERM
fi
