#!/bin/bash
# Own one container-side process group for a collection session.
set -e

[[ $# -ge 2 ]] || { echo "Usage: $0 start|stop|stop-wait PID_FILE [COMMAND ...]" >&2; exit 2; }
mode=$1
pid_file=$2
shift 2

case "$mode" in
    start)
        [[ $# -gt 0 ]] || exit 2
        [[ -f "$pid_file" ]] || exit 1
        setsid --wait bash -c '
            pid_file=$1
            shift
            [[ -f "$pid_file" ]] || exit 1
            echo "$$" > "$pid_file"
            exec "$@"
        ' _ "$pid_file" "$@" &
        waiter=$!
        signal_child() {
            if [[ -s "$pid_file" ]]; then
                read -r child < "$pid_file"
                kill -TERM -- "-$child" 2>/dev/null || true
            fi
        }
        trap signal_child EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM HUP
        wait "$waiter"
        ;;
    stop|stop-wait)
        if [[ ! -s "$pid_file" ]]; then
            rm -f -- "$pid_file"
            exit 0
        fi
        read -r child < "$pid_file"
        [[ "$child" =~ ^[0-9]+$ ]] && ((child > 1)) || exit 1
        group_alive() {
            ps -eo pgid=,stat= | awk -v group="$child" '
                $1 == group && $2 !~ /^Z/ { alive=1 }
                END { exit !alive }
            '
        }
        if [[ "$mode" == stop-wait ]]; then
            trap '' INT TERM HUP
            kill -INT -- "-$child" 2>/dev/null || true
            while group_alive; do
                sleep 0.2
            done
            rm -f -- "$pid_file"
            exit 0
        fi
        forced=0
        for stage in TERM:150 INT:25 KILL:10; do
            group_alive || break
            sig=${stage%:*}
            if [[ "$sig" == KILL ]]; then
                echo "Error: forcing process group $child to stop; recording finalization is unverified." >&2
                forced=1
            fi
            kill -"$sig" -- "-$child" 2>/dev/null || true
            for ((i=0; i<${stage#*:}; i++)); do
                group_alive || break
                sleep 0.2
            done
        done
        if group_alive; then
            echo "Error: process group $child is still alive; retained $pid_file." >&2
            exit 1
        fi
        rm -f -- "$pid_file"
        ((forced == 0))
        ;;
    *)
        echo "Usage: $0 start|stop|stop-wait PID_FILE [COMMAND ...]" >&2
        exit 2
        ;;
esac
