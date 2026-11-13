#!/bin/bash
# Launch GenieSim teleoperation from the host.
# Requires the container to be running. Opens the bridge, MC,
# chassis servo (except arm-only), and Pico terminals using the same task.
# Starts Isaac Sim 6.2 in manual collection mode.
#
# Finally it waits for y/N to decide whether the recorded episode is kept.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/teleop_runtime.sh"
teleop_select_runtime "$@" || exit $?
set -- "${TELEOP_RUNTIME_ARGS[@]}"
CONTAINER_NAME="${GENIESIM_CONTAINER:-$TELEOP_DEFAULT_CONTAINER}"
TERMINAL_ENV="autorun"
EXTERNAL_SIMULATOR=0
for arg in "$@"; do
    case "$arg" in
        --external-simulator) EXTERNAL_SIMULATOR=1 ;;
        --help|-h)
            echo "Usage: $0 [--isaac-version 6.2] [--external-simulator]"
            echo "Starts Isaac Sim 6.2 and manual collection; --external-simulator starts controls only."
            echo "GENIESIM_TELEOP_CONFIG selects the shared container-side task YAML."
            echo "GENIESIM_TELEOP_SIM_PY selects a prepared simulator interpreter (default: python3.12)."
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done
if ((EXTERNAL_SIMULATOR)); then
    echo "Info: Isaac $TELEOP_ISAAC_VERSION control-only session; the simulator must provide joint states and TF separately."
fi

# GENIESIM_TELEOP_ARM_ONLY=1 disables gripper and chassis control.
# For combined Pico/glove collection, use autoteleop_wuji.sh.
TELEOP_ARM_ONLY_ARGS=""
if [[ "${GENIESIM_TELEOP_ARM_ONLY:-0}" == "1" ]]; then
    TELEOP_ARM_ONLY_ARGS=" --arm_only"
    echo "Info: Pico teleop arm-only mode enabled"
fi

REPO_IN_CONTAINER="${GENIESIM_WORKSPACE:-/workspace}"
TELEOP_SRC="${REPO_IN_CONTAINER}/source/geniesim_teleop/src"
TELEOP_PKG="${TELEOP_SRC}/geniesim_teleop"
TELEOP_APP="${TELEOP_PKG}/app"
BENCHMARK_PKG="${REPO_IN_CONTAINER}/source/geniesim_benchmark/src/geniesim_benchmark"
TELEOP_YAML="${GENIESIM_TELEOP_CONFIG:-${BENCHMARK_PKG}/config/teleop.yaml}"

HOST_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
teleop_ensure_app "$HOST_REPO" || exit $?

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    echo "Start it first, then re-run this script. For the new CLI image:"
    echo "    geniesim docker${TELEOP_ISAAC_VERSION} up"
    echo "Or set GENIESIM_CONTAINER=<name> if your container has a different name:"
    echo "    docker ps --format '{{.Names}}'"
    exit 1
else
    echo "Info: Container $CONTAINER_NAME already running"
fi

# Resolve the Pico network before installing dependencies or launching processes.
# Multiple interfaces require GENIESIM_TELEOP_INTERFACE or GENIESIM_TELEOP_HOST_IP.
if ! TELEOP_HOST_IP=$(docker exec \
    -e GENIESIM_TELEOP_INTERFACE -e GENIESIM_TELEOP_HOST_IP \
    -e PYTHONPATH="$TELEOP_SRC" "$CONTAINER_NAME" \
    python3 -m geniesim_teleop.utils.network); then
    exit 1
fi
echo "Info: Pico network address: $TELEOP_HOST_IP"

if ! docker exec -e PYTHONPATH="$TELEOP_SRC:${BENCHMARK_PKG%/*}" -e SIM_REPO_ROOT="$REPO_IN_CONTAINER" \
    -e GENIESIM_TELEOP_CONFIG="$TELEOP_YAML" "$CONTAINER_NAME" \
    python3 -m geniesim_teleop.config.task_config; then
    exit 1
fi

# Install the package into system Python when needed. --no-deps preserves
# the image's NumPy 2.x; teleop/bridge use the apt NumPy 1.x overlay below.
if ! docker exec "$CONTAINER_NAME" python3 -c 'import geniesim_teleop' 2>/dev/null; then
    echo "Info: installing geniesim_teleop (editable, --no-deps) into $CONTAINER_NAME ..."
    docker exec "$CONTAINER_NAME" python3 -m pip install -e "${REPO_IN_CONTAINER}/source/geniesim_teleop" --no-deps || exit 1
fi
# The bridge and Pico use apt NumPy 1.x for cv_bridge compatibility.
# Source the canonical dev overlay (or the installed wheel's ROS overlay).
ROS_SETUP="${REPO_IN_CONTAINER}/devel/setup.bash"
if [[ "${GENIESIM_TELEOP_ARM_ONLY:-0}" != "1" ]]; then
    if ! docker exec "$CONTAINER_NAME" test -f "$ROS_SETUP"; then
        ROS_SETUP=$(docker exec "$CONTAINER_NAME" python3 -c 'from geniesim_ros import setup_bash_path; print(setup_bash_path() or "")')
    fi
    if [[ -z "$ROS_SETUP" ]] || ! docker exec "$CONTAINER_NAME" bash -c \
        'source /opt/ros/jazzy/setup.bash && source "$1" && ros2 pkg prefix genie_sim_controllers >/dev/null' _ "$ROS_SETUP"; then
        echo "Error: chassis servo is not built. Run geniesim ros build dev in the container at $REPO_IN_CONTAINER."
        exit 1
    fi
fi

PROCESS_HELPER="${REPO_IN_CONTAINER}/source/geniesim_teleop/scripts/teleop_process.sh"
SIM_PID_FILE=""
declare -a CONTROL_PID_FILES=()
declare -a COMMANDS=()
stop_simulator() {
    [[ -n "$SIM_PID_FILE" ]] || return 0
    trap '' INT TERM HUP
    echo "Waiting for recording files to finish writing and Isaac Sim to exit..."
    docker exec "$CONTAINER_NAME" bash "$PROCESS_HELPER" stop-wait "$SIM_PID_FILE" || return 1
    SIM_PID_FILE=""
}
stop_controls() {
    local failed=0
    for pid_file in "${CONTROL_PID_FILES[@]}"; do
        docker exec "$CONTAINER_NAME" bash "$PROCESS_HELPER" stop "$pid_file" || failed=1
    done
    ((failed == 0))
}
cleanup_session() {
    local failed=0
    stop_simulator || failed=1
    stop_controls || failed=1
    return "$failed"
}
trap 'status=$?; cleanup_session || status=1; exit "$status"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

add_control_command() {
    local pid_file command
    pid_file=$(docker exec "$CONTAINER_NAME" mktemp /tmp/geniesim-teleop-control.XXXXXX) || exit 1
    CONTROL_PID_FILES+=("$pid_file")
    printf -v command '%q ' docker exec -it \
        -e "GENIESIM_TELEOP_CONFIG=$TELEOP_YAML" \
        -e "SIM_REPO_ROOT=$REPO_IN_CONTAINER" \
        -e GENIESIM_TELEOP_INTERFACE= -e "GENIESIM_TELEOP_HOST_IP=$TELEOP_HOST_IP" \
        "$CONTAINER_NAME" bash "$PROCESS_HELPER" start "$pid_file" bash -ic "$1"
    COMMANDS+=("$command")
}
printf -v CONTROL_SETUP 'source /opt/ros/jazzy/setup.bash && source %q && ' "${TELEOP_APP}/bin/env.sh"
NUMPY_SETUP='export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH && '
add_control_command "${CONTROL_SETUP}${NUMPY_SETUP}exec python3 -m geniesim_teleop.bridge"
add_control_command "${CONTROL_SETUP}exec python3 -m geniesim_teleop.start_mc"
if [[ "${GENIESIM_TELEOP_ARM_ONLY:-0}" != "1" ]]; then
    printf -v CHASSIS_SETUP 'source %q && ' "$ROS_SETUP"
    add_control_command "${CONTROL_SETUP}${CHASSIS_SETUP}exec python3 -m geniesim_teleop.start_chassis"
fi
add_control_command "${CONTROL_SETUP}${NUMPY_SETUP}exec python3 -m geniesim_teleop.teleop${TELEOP_ARM_ONLY_ARGS}"
declare -a DELAYS=(1 15 3 5 5)

TERMINAL_CMD=""
for term in gnome-terminal konsole xterm terminator; do
    if command -v "$term" &>/dev/null; then
        case "$term" in
        gnome-terminal) TERMINAL_CMD="gnome-terminal -- bash -c" ;;
        konsole) TERMINAL_CMD="konsole -e bash -c" ;;
        xterm) TERMINAL_CMD="xterm -e" ;;
        terminator) TERMINAL_CMD="terminator -e" ;;
        esac
        break
    fi
done

if [ -z "$TERMINAL_CMD" ]; then
    echo "No terminal emulator found. Please install one and try again."
    exit 1
fi

if ((EXTERNAL_SIMULATOR == 0)); then
    EXISTING_RECORDINGS=$(docker exec "$CONTAINER_NAME" python3 "${TELEOP_PKG}/data_recording/patch_recording_info.py" \
        --config "$TELEOP_YAML" --base "${REPO_IN_CONTAINER}/output/recording_data" --snapshot) || exit 1
    SIM_PID_FILE=$(docker exec "$CONTAINER_NAME" mktemp /tmp/geniesim-teleop-sim.XXXXXX) || exit 1
    # Keep Isaac on its SDK Python/NumPy; app/env.sh is for control processes.
    SIM_SCRIPT='
        set -e
        cd "$SIM_REPO_ROOT"
        export GENIESIM_SKIP_AUTOBOOT=1
        source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
        export PYTHONPATH="$SIM_REPO_ROOT/source/geniesim_cli/src:$SIM_REPO_ROOT/source/geniesim_benchmark/src:$SIM_REPO_ROOT/source/geniesim_teleop/src"
        # Match the ordinary control processes domain when env.sh uses container mode.
        if [[ -n "${USE_CONTAINER:-}" ]]; then export ROS_DOMAIN_ID=1; fi
        export ROS_LOCALHOST_ONLY="${USE_CONTAINER:+0}"
        export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
        exec bash "$GENIESIM_TELEOP_PROCESS_HELPER" start "$GENIESIM_TELEOP_SIM_PID_FILE" \
            python3 -m geniesim_cli benchmark run "$GENIESIM_TELEOP_CONFIG" \
                --benchmark.model_arc=pi --benchmark.policy_class=DemoPolicy \
                --benchmark.env_class=DummyEnv --benchmark.interactive=true --benchmark.record=false
    '
    printf -v SIM_COMMAND '%q ' docker exec -it \
        -e "SIM_REPO_ROOT=$REPO_IN_CONTAINER" \
        -e "GENIESIM_TELEOP_CONFIG=$TELEOP_YAML" \
        -e "GENIESIM_TELEOP_SIM_PID_FILE=$SIM_PID_FILE" \
        -e "GENIESIM_TELEOP_PROCESS_HELPER=$PROCESS_HELPER" \
        -e "GENIESIM_PY_CMD=${GENIESIM_TELEOP_SIM_PY:-$TELEOP_DEFAULT_SIM_PY}" \
        "$CONTAINER_NAME" bash -lc "$SIM_SCRIPT"
    COMMANDS=("$SIM_COMMAND" "${COMMANDS[@]}")
    DELAYS=(1 1 15 3 5)
    echo "Info: starting Isaac Sim and collection with $TELEOP_YAML"
    echo "Info: Pico waits for simulator TF initialization before tracking."
fi

for i in "${!COMMANDS[@]}"; do
    sleep "${DELAYS[$i]}"
    if [[ "$TERMINAL_CMD" == "gnome-terminal"* ]]; then
        gnome-terminal -- bash -c "export TERMINAL_ENV=$TERMINAL_ENV; ${COMMANDS[$i]}; exec bash" &
    else
        $TERMINAL_CMD "${COMMANDS[$i]}" &
    fi
    if ((EXTERNAL_SIMULATOR == 0 && i == 0)); then
        simulator_started=0
        for ((attempt=0; attempt<30; attempt++)); do
            if docker exec "$CONTAINER_NAME" test -s "$SIM_PID_FILE"; then
                simulator_started=1
                break
            fi
            sleep 1
        done
        if ((simulator_started == 0)); then
            echo "Error: simulator terminal did not start; inspect its output." >&2
            exit 1
        fi
    fi
done

echo -e "\nAll terminals started. Press 'y' or 'Y' = teleoperation succeeded, keep data; 'n' or 'N' = failed, do not keep data ..."
while read -n 1 -s input; do
    if [[ "$input" == "Y" || "$input" == "y" ]]; then
        echo "Keeping the recording after session shutdown."
        stop_simulator || exit 1
        stop_controls || exit 1
        echo "Patching recording_info.json: add teleop_result"
        docker exec "$CONTAINER_NAME" python3 ${TELEOP_PKG}/data_recording/patch_recording_info.py \
            --config "$TELEOP_YAML" \
            --base ${REPO_IN_CONTAINER}/output/recording_data \
            --existing "${EXISTING_RECORDINGS:-[]}" \
            || exit 1
        echo "Recording saved."

        break
    elif [[ "$input" == "N" || "$input" == "n" ]]; then
        stop_simulator || exit 1
        stop_controls || exit 1
        echo "Patching recording_info.json: add teleop_result=false"
        docker exec "$CONTAINER_NAME" python3 ${TELEOP_PKG}/data_recording/patch_recording_info.py \
            --config "$TELEOP_YAML" \
            --base ${REPO_IN_CONTAINER}/output/recording_data \
            --existing "${EXISTING_RECORDINGS:-[]}" \
            --teleop-result false \
            || exit 1
        break
    fi
done

reset
