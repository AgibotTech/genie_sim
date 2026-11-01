#!/usr/bin/env bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
# Validate all slots, then start them serially with rollback on failure.
set -o pipefail
_BATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$_BATCH_DIR/_server_slot.sh"
log() { echo "[servers] $*"; }
err() { echo "[servers] ERROR: $*" >&2; }
_SIM_ROOT_FALLBACK="$(cd "$_BATCH_DIR/../../.." && pwd)"

# Expand only environment-variable references; configuration is never shell code.
_expand_conf_value() {
  _where="$1"; _val="$2"; _out=""
  while [ -n "$_val" ]; do
    case "$_val" in
      *'$'*) ;;
      *) _out="$_out$_val"; break ;;
    esac
    _out="$_out${_val%%'$'*}"                     # the literal part before $
    _rest="${_val#*'$'}"
    case "$_rest" in
      '{'*)
        _name="${_rest#'{'}"
        case "$_name" in
          *'}'*) ;;
          *) err "$_where: '\${$_name' is missing a closing }"; return 1 ;;
        esac
        _rest="${_name#*'}'}"; _name="${_name%%'}'*}" ;;
      *)
        _name="$(printf '%s' "$_rest" | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p')"
        _rest="${_rest#"$_name"}" ;;
    esac
    case "$_name" in
      ''|[0-9]*|*[!A-Za-z0-9_]*)
        err "$_where: '\$$_name' is not a valid variable name"; return 1 ;;
    esac
    _sub="${!_name-}"
    if [ -z "$_sub" ]; then
      if [ "$_name" = SIM_ROOT ] || [ "$_name" = SIM_REPO_ROOT ]; then
        _sub="$_SIM_ROOT_FALLBACK"
        log "$_where: Repository variable unset; using script location $_sub" >&2
      else
        err "$_where: \$$_name is empty or undefined -- did the environment variable not get through? "
        return 1
      fi
    fi
    _out="$_out$_sub"; _val="$_rest"
  done
  printf '%s' "$_out"
}

_GLOBAL_KEYS=" code assets stagger rebuild "
_SERVER_KEYS=" port gpu domain fabric headless ready_timeout "

batch_parse() {
  _conf="$1"
  [ -f "$_conf" ] || { err "Configuration file missing: $_conf"; return 1; }

  BATCH_CODE=""; BATCH_ASSETS=""; BATCH_STAGGER="0"; BATCH_REBUILD="auto"
  BATCH_ROWS=""
  _rows=""; _ports=""; _domains=""; _gpus=""; _lineno=0; _bad=0

  while IFS= read -r _line || [ -n "$_line" ]; do
    _lineno=$(( _lineno + 1 ))
    _line="${_line%%#*}"                       # Remove inline comments
    case "$_line" in *[![:space:]]*) ;; *) continue;; esac   # Skip blank rows

    read -r -a _tokens <<< "$_line"
    set -- "${_tokens[@]}"
    _kind="$1"; shift
    case "$_kind" in
      global) _allowed="$_GLOBAL_KEYS" ;;
      server) _allowed="$_SERVER_KEYS" ;;
      *) err "$_conf:$_lineno: Expected global or server, got '$_kind'"; _bad=1; continue ;;
    esac

    _port=""; _gpu=""; _domain=""; _fabric="1"; _headless="1"; _ready="600"
    for _kv in "$@"; do
      case "$_kv" in *=*) ;; *) err "$_conf:$_lineno: '$_kv' must be key=value"; _bad=1; continue 2;; esac
      _k="${_kv%%=*}"; _v="${_kv#*=}"
      case "$_allowed" in
        *" $_k "*) ;;
        *) err "$_conf:$_lineno: $_kind row has unknown key '$_k'(allowed:$_allowed)"; _bad=1; continue 2 ;;
      esac
      [ -n "$_v" ] || { err "$_conf:$_lineno: '$_k=' has an empty value"; _bad=1; continue 2; }
      case "$_k" in
        code) BATCH_CODE="$(_expand_conf_value "$_conf:$_lineno" "$_v")" || { _bad=1; continue 2; } ;;
        assets) BATCH_ASSETS="$(_expand_conf_value "$_conf:$_lineno" "$_v")" || { _bad=1; continue 2; } ;;
        stagger) BATCH_STAGGER="$_v" ;;
        rebuild) BATCH_REBUILD="$_v" ;;
        port) _port="$_v" ;;
        gpu) _gpu="$_v" ;;
        domain) _domain="$_v" ;;
        fabric) _fabric="$_v" ;;
        headless) _headless="$_v" ;;
        ready_timeout) _ready="$_v" ;;
      esac
    done

    [ "$_kind" = "global" ] && continue

    if [ -z "$_port" ]; then
      err "$_conf:$_lineno: server requires port"
      _bad=1; continue
    fi
    if ! slot_derive "$_port" "$_domain" >/dev/null 2>&1; then
      err "$_conf:$_lineno: port=$_port domain=${_domain:-<derived>} invalid (port 1..65535, domain 0..101)"
      _bad=1; continue
    fi
    _domain="$SLOT_DOMAIN"

    case "$_fabric" in 0|1) ;; *) err "$_conf:$_lineno: fabric must be 0 or 1, got '$_fabric'"; _bad=1; continue;; esac
    case "$_headless" in 0|1) ;; *) err "$_conf:$_lineno: headless must be 0 or 1, got '$_headless'"; _bad=1; continue;; esac
    [[ "$_ready" =~ ^[1-9][0-9]*$ ]] || { err "$_conf:$_lineno: ready_timeout must be positive"; _bad=1; continue; }
    [[ "$_gpu" =~ ^([0-9]+(,[0-9]+)*)?$ ]] || { err "$_conf:$_lineno: invalid gpu list"; _bad=1; continue; }

    case " $_ports " in *" $_port "*) err "$_conf:$_lineno: Duplicate port $_port"; _bad=1; continue;; esac
    case " $_domains " in
      *" $_domain "*)
        err "$_conf:$_lineno: Duplicate ROS domain $_domain: servers would share topics; choose distinct port/domain values"
        _bad=1; continue ;;
    esac
    _ports="$_ports $_port"; _domains="$_domains $_domain"

    case " $_gpus " in
      *" $_gpu "*) [ -n "$_gpu" ] && log "GPU $_gpu is shared by multiple servers" ;;
    esac
    [ -n "$_gpu" ] && _gpus="$_gpus $_gpu"

    _rows="$_rows$_port|$_gpu|$_domain|$_fabric|$_headless|$_ready
"
  done < "$_conf"

  [ "$_bad" = 0 ] || return 1
  [ -n "$_rows" ] || { err "$_conf contains no server rows"; return 1; }
  case "$BATCH_STAGGER" in ''|*[!0-9]*) err "stagger must be numeric, got '$BATCH_STAGGER'"; return 1;; esac
  case "$BATCH_REBUILD" in auto|force) ;; *) err "rebuild must be auto or force, got '$BATCH_REBUILD'"; return 1;; esac
  BATCH_ROWS="$_rows"
  return 0
}

_selected() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0;; esac
  return 1
}

usage() {
  cat <<'HELP'
Usage: start_sim_servers.sh PLAN.conf [--only P1,P2] [--dry-run|--status|--stop]
       start_sim_servers.sh --count N [--base-port P] [--devices 0,1] [options]
Options for generated plans: --code DIR --assets DIR --no-fabric --rebuild-ws
Shared option: --barrier-cap-s SECONDS
Plans contain global key=value settings and server port=P [gpu=N ...] rows.
All selected slots must be stopped before starting. Failure rolls back this run.
HELP
}

main() {
  local conf="" count="" base_port=5601 devices="" code="" assets=""
  local fabric=1 rebuild=auto cap="" port gpu domain headless ready row
  local -a dev_list=() started=() args=()
  local i per_device=0 first=1 rc selected=0 receipt="" identity=""
  local -A identities=()
  ACTION=start; ONLY=""; DRY_RUN=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --count|--base-port|--devices|--code|--assets|--only|--barrier-cap-s)
        [ "$#" -ge 2 ] && [ -n "$2" ] || { err "Missing value for $1"; return 2; }
        case "$1" in
          --count) count="$2";; --base-port) base_port="$2";; --devices) devices="$2";;
          --code) code="$2";; --assets) assets="$2";; --only) ONLY="$2";; --barrier-cap-s) cap="$2";;
        esac
        shift 2;;
      --no-fabric) fabric=0; shift;; --rebuild-ws) rebuild=force; shift;;
      --stop) ACTION=stop; shift;; --status) ACTION=status; shift;;
      --dry-run) DRY_RUN=1; shift;;
      -h|--help) usage; return 0;;
      -*) err "Unknown option: $1"; return 2;;
      *) [ -z "$conf" ] || { err "Only one plan file is allowed"; return 2; }; conf="$1"; shift;;
    esac
  done
  [ -z "$cap" ] || [[ "$cap" =~ ^[0-9]+([.][0-9]+)?$ ]] || { err "Invalid barrier cap"; return 2; }
  [ -z "$ONLY" ] || [[ "$ONLY" =~ ^[0-9]+(,[0-9]+)*$ ]] || { err "Invalid --only port list"; return 2; }
  if [ -n "$conf" ]; then
    [ -z "$count$devices$code$assets" ] && [ "$fabric" = 1 ] && [ "$rebuild" = auto ] && [ "$base_port" = 5601 ] || {
      err "Use either a plan file or generated-plan options"; return 2;
    }
    batch_parse "$conf" || return 2
  else
    [[ "$count" =~ ^[1-9][0-9]{0,2}$ && "$base_port" =~ ^[1-9][0-9]{0,4}$ ]] || { usage >&2; return 2; }
    (( count <= 101 && base_port + count <= 65536 )) || { err "Invalid port/count range"; return 2; }
    if [ -n "$devices" ]; then
      [[ "$devices" =~ ^[0-9]+(,[0-9]+)*$ ]] || { err "Invalid devices"; return 2; }
      IFS=',' read -r -a dev_list <<< "$devices"
      (( count % ${#dev_list[@]} == 0 )) || { err "Count must be divisible by device count"; return 2; }
      per_device=$((count / ${#dev_list[@]}))
    fi
    BATCH_CODE="$code"; BATCH_ASSETS="$assets"; BATCH_STAGGER=0; BATCH_REBUILD="$rebuild"; BATCH_ROWS=""
    for ((i=0; i<count; i++)); do
      port=$((base_port+i)); gpu=""
      slot_derive "$port" || return 2
      (( per_device == 0 )) || gpu="${dev_list[$((i/per_device))]}"
      BATCH_ROWS+="$port|$gpu|$SLOT_DOMAIN|$fabric|1|600"$'\n'
    done
  fi
  local -A domains=() available=()
  while IFS='|' read -r port gpu domain fabric headless ready; do
    [ -n "$port" ] || continue
    available[$port]=1
    _selected "$port" || continue
    selected=$((selected+1))
    [ -z "${domains[$domain]:-}" ] || { err "Selected ports collide in ROS domain $domain"; return 2; }
    domains[$domain]=1
    slot_derive "$port" "$domain" || return 2
    if [ "$ACTION" = start ] && [ "$DRY_RUN" = 0 ] && [ -f "$SLOT_PIDFILE" ] && slot_alive "$(cat "$SLOT_PIDFILE")"; then
      err "$SLOT_TAG is already running; stop it explicitly first"; return 1
    fi
  done <<< "$BATCH_ROWS"
  if [ -n "$ONLY" ]; then
    local -a requested=()
    IFS=',' read -r -a requested <<< "$ONLY"
    for port in "${requested[@]}"; do
      [ -n "${available[$port]:-}" ] || { err "Port $port is not in the plan"; return 2; }
    done
  fi
  [ "$selected" -gt 0 ] || { err "No slots selected"; return 2; }
  rollback() {
    local cleanup_port
    for cleanup_port in "${started[@]}"; do
      (
        slot_derive "$cleanup_port" || exit 0
        [ -d "$SLOT_RUN_DIR" ] || exit 0
        exec 8>"$SLOT_RUN_DIR/launcher.lock"
        flock 8 || exit 0
        expected="${identities[$cleanup_port]}"
        current="$(cat "$SLOT_PIDFILE" 2>/dev/null)"
        if [ "$current" = "${expected%%:*}" ] &&
            [ "$(slot_process_start "$current")" = "${expected#*:}" ]; then
          slot_stop
        else
          log "Skipping rollback for $SLOT_TAG: the recorded server has changed"
        fi
      ) || true
    done
    [ -z "$receipt" ] || rm -f "$receipt"
  }
  trap 'rollback; exit 130' INT TERM
  while IFS='|' read -r port gpu domain fabric headless ready; do
    [ -n "$port" ] || continue
    _selected "$port" || continue
    if [ "$ACTION" != start ]; then
      args=(--port "$port" "--$ACTION")
    else
      args=(--port "$port" --domain "$domain")
      [ -z "$gpu" ] || args+=(--device "$gpu")
      [ -z "$BATCH_CODE" ] || args+=(--code "$BATCH_CODE")
      [ -z "$BATCH_ASSETS" ] || args+=(--assets "$BATCH_ASSETS")
      [ "$fabric" = 1 ] || args+=(--no-fabric)
      [ -z "$cap" ] || args+=(--barrier-cap-s "$cap")
      if [ "$BATCH_REBUILD" = force ] && [ "$first" = 1 ]; then args+=(--rebuild-ws); fi
      first=0
    fi
    if [ "$DRY_RUN" = 1 ]; then
      printf 'HEADLESS=%q READY_TIMEOUT=%q bash %q ' "$headless" "$ready" "$_BATCH_DIR/start_sim_server.sh"
      printf '%q ' "${args[@]}"; printf '\n'
      continue
    fi
    if [ "$ACTION" = start ]; then
      receipt="$(mktemp "${TMPDIR:-/tmp}/rlinf-launch.XXXXXX")" || { rollback; trap - INT TERM; return 1; }
    fi
    _RLINF_LAUNCH_RECEIPT="$receipt" HEADLESS="$headless" READY_TIMEOUT="$ready" \
      bash "$_BATCH_DIR/start_sim_server.sh" "${args[@]}"
    rc=$?
    if [ "$ACTION" = start ]; then
      identity="$(cat "$receipt" 2>/dev/null)"
      rm -f "$receipt"; receipt=""
      # Failed launches clean up their own slot before releasing its lock.
      if [ "$rc" = 0 ]; then
        if [[ "$identity" =~ ^[1-9][0-9]*:[0-9]+$ ]]; then
          started+=("$port"); identities[$port]="$identity"
        else
          err "$port returned no valid launch identity"; rc=1
        fi
      fi
      if [ "$rc" != 0 ]; then rollback; trap - INT TERM; return "$rc"; fi
      [ "$BATCH_STAGGER" = 0 ] || sleep "$BATCH_STAGGER"
    elif [ "$rc" != 0 ]; then
      trap - INT TERM; return "$rc"
    fi
  done <<< "$BATCH_ROWS"
  trap - INT TERM
  if [ "$ACTION" = start ] && [ "$DRY_RUN" = 0 ]; then
    for port in "${started[@]}"; do printf 'tcp://127.0.0.1:%s\n' "$port"; done
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
