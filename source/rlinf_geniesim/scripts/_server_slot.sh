# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
# Per-port process, ROS domain, and shared-memory ownership.
command -v log >/dev/null 2>&1 || log() { echo "[slot] $*"; }

slot_derive() {
  _port="$1"; _domain="${2:-}"; _root="${3:-${RLINF_RUN_ROOT:-/tmp}}"
  [[ "$_port" =~ ^[1-9][0-9]{0,4}$ ]] || { log "ERROR: invalid port '$_port'"; return 1; }
  if [ "$_port" -le 0 ] || [ "$_port" -ge 65536 ]; then
    log "ERROR: port outside 1..65535: $_port"; return 1
  fi

  SLOT_TAG="p$_port"
  SLOT_RUN_DIR="$_root/rlinf_srv_$SLOT_TAG"
  SLOT_LOG="$SLOT_RUN_DIR/sim.log"
  SLOT_COMM_CFG="$SLOT_RUN_DIR/comm.yaml"
  SLOT_PIDFILE="$SLOT_RUN_DIR/sim_server.pid"

  if [ -n "$_domain" ]; then
    [[ "$_domain" =~ ^(0|[1-9][0-9]{0,2})$ ]] || { log "ERROR: invalid domain '$_domain'"; return 1; }
    if [ "$_domain" -lt 0 ] || [ "$_domain" -gt 101 ]; then
      log "ERROR: domain outside 0..101: $_domain"; return 1
    fi
    SLOT_DOMAIN="$_domain"
  else
    SLOT_DOMAIN=$(( _port - 5600 ))
    if [ "$SLOT_DOMAIN" -lt 1 ] || [ "$SLOT_DOMAIN" -gt 101 ]; then
      SLOT_DOMAIN=$(( (_port % 101) + 1 ))
    fi
  fi
  return 0
}


SLOT_PROC_PATTERN="${SLOT_PROC_PATTERN:-sim_server\.py}"
slot_alive() {
  [ -n "$1" ] && [ -d "/proc/$1" ] || return 1
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -qE "$SLOT_PROC_PATTERN"
}

slot_process_start() {
  local stat
  read -r stat 2>/dev/null < "/proc/$1/stat" || return 1
  # The command name may contain spaces or closing parentheses.
  set -- ${stat##*) }
  [ "$#" -ge 20 ] || return 1
  printf '%s\n' "${20}"
}

slot_descendants() {
  _queue="$1"; _out=""
  while [ -n "$_queue" ]; do
    set -- $_queue
    _cur="$1"; shift; _queue="$*"
    for _k in $(ps -eo pid=,ppid= 2>/dev/null | awk -v p="$_cur" '$2==p {print $1}'); do
      _out="$_out $_k"; _queue="$_queue $_k"
    done
  done
  echo "$_out"
}

slot_cleanup_shm() {
  local _pid="$1" _f _generation
  case "$_pid" in ''|*[!0-9]*) return 0;; esac
  for _f in /dev/shm/*_rlinf_zmq_"$_pid" /dev/shm/*_rlinf_zmq_"$_pid"_g* \
            /dev/shm/rlinf_tiled_"$_pid" /dev/shm/rlinf_tiled_"$_pid"_* \
            /dev/shm/sem.*rlinf*_"$_pid"; do
    [ -e "$_f" ] && rm -f "$_f" 2>/dev/null
  done
  for _f in /dev/shm/rlinf_fused_state_"$_pid"_* /dev/shm/rlinf_fused_tiled_"$_pid"_* \
            /dev/shm/sem.rlinf_fused_req_"$_pid"_* /dev/shm/sem.rlinf_fused_resp_"$_pid"_*; do
    _generation="${_f##*_}"
    [[ "$_generation" =~ ^[0-9]+$ ]] || continue
    [ -e "$_f" ] && rm -f "$_f" 2>/dev/null
  done
  return 0
}


slot_stop() {
  local _grace _pid _start _tree _p _birth _identity _i _n
  local -a _identities=()
  _grace="${1:-15}"
  if [ ! -f "$SLOT_PIDFILE" ]; then
    log "No server record at $SLOT_PIDFILE"
    return 0
  fi
  _pid="$(cat "$SLOT_PIDFILE" 2>/dev/null)"
  case "$_pid" in ''|*[!0-9]*) log "Removing invalid server record"; rm -f "$SLOT_PIDFILE"; return 0;; esac
  _start="$(slot_process_start "$_pid")" || _start=""
  if ! slot_alive "$_pid"; then
    log "Removing stale server record for pid=${_pid:-empty}"
    slot_cleanup_shm "${_pid:-}"; rm -f "$SLOT_PIDFILE"; return 0
  fi
  # A reused PID must also belong to this slot, not just another sim server.
  if [ -n "${SLOT_COMM_CFG:-}" ] && ! tr '\0' '\n' < "/proc/$_pid/cmdline" | grep -Fxq -- "$SLOT_COMM_CFG"; then
    log "Refusing to stop pid=$_pid: command does not own $SLOT_COMM_CFG"
    return 1
  fi
  _tree="$(slot_descendants "$_pid")"
  for _p in $_tree; do
    _birth="$(slot_process_start "$_p")" || continue
    _identities+=("$_p:$_birth")
  done
  _identities+=("$_pid:$_start")
  [ -n "$_start" ] && [ "$(slot_process_start "$_pid")" = "$_start" ] || {
    log "Server pid=$_pid exited before shutdown; leaving replacement processes untouched"
    return 1
  }
  log "Stopping $SLOT_TAG: pid=$_pid; requesting graceful shutdown"
  kill -TERM "$_pid" 2>/dev/null || true
  _i=0
  while [ "$_i" -lt $(( _grace * 2 )) ]; do
    [ "$(slot_process_start "$_pid")" = "$_start" ] || break
    slot_alive "$_pid" || break
    sleep 0.5; _i=$(( _i + 1 ))
  done
  _n=0
  for _identity in "${_identities[@]}"; do
    _p="${_identity%%:*}"; _birth="${_identity#*:}"
    # Children can exit during the grace period; never signal a reused PID.
    [ "$(slot_process_start "$_p")" = "$_birth" ] && {
      kill -9 "$_p" 2>/dev/null && _n=$(( _n + 1 ))
    }
  done
  sleep 1
  slot_cleanup_shm "$_pid"; rm -f "$SLOT_PIDFILE"
  log "Stopped $SLOT_TAG; forcibly terminated $_n remaining processes"
  return 0
}

slot_require_idle_build() {
  local record pid
  for record in "${1:-/tmp}"/rlinf_srv_p*/sim_server.pid; do
    [ -f "$record" ] || continue
    pid="$(tr -dc '0-9' < "$record")"
    if slot_alive "$pid"; then
      log "ERROR: clean rebuild refused: active server $pid ($record); stop it first"
      return 1
    fi
  done
}
