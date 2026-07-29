#!/bin/bash
# Launch GenieSim teleoperation from OUTSIDE the container.
#
# Same behaviour as the legacy scripts/autoteleop.sh, but with all paths
# updated for the new package layout:
#   - teleop code now lives in source/geniesim_teleop/src/geniesim_teleop/
#   - the simulator app + teleop.yaml now live in the geniesim_benchmark package
#   - teleop.py is launched as a module (python3 -m geniesim_teleop.teleop)
#     because its imports are now absolute (geniesim_teleop.*)
#
# It ensures the GUI container is running (starting it via start_gui.sh if
# needed), then opens four terminals that bring up, in order:
#   1. the Isaac Sim simulator (teleop.yaml scene)
#   2. the image pub/sub bridge
#   3. the motion-control binary (genie_motion_control)
#   4. the teleop loop (opens the VR server, waits for the Pico headset)
#
# Finally it enters a multi-episode loop: press the controller's record button
# to start an episode, then y/n here to keep/discard it and re-arm for the next
# episode, and q to stop everything.

# Container name + repo mount point. The new geniesim CLI container is
# named "geniesim3" and mounts the repo at /workspace (the legacy
# autoteleop.sh used "genie_sim_benchmark" + /geniesim/main). Override via
# env if your container differs.
CONTAINER_NAME="${GENIESIM_CONTAINER:-geniesim3}"
TERMINAL_ENV="autorun"
PROCESS_CLIENT="teleop|ros|geniesim_teleop"

# --- Paths (container-side, repo mounted at /workspace) --------------------
REPO_IN_CONTAINER="${GENIESIM_WORKSPACE:-/workspace}"
# teleop package: source tree root that holds the importable package dir
TELEOP_SRC="${REPO_IN_CONTAINER}/source/geniesim_teleop/src"
TELEOP_PKG="${TELEOP_SRC}/geniesim_teleop"
TELEOP_APP="${TELEOP_PKG}/app"
# simulator app + teleop config (now inside the geniesim_benchmark package)
BENCHMARK_PKG="${REPO_IN_CONTAINER}/source/geniesim_benchmark/src/geniesim_benchmark"
SIM_APP="${BENCHMARK_PKG}/app/app.py"
TELEOP_YAML="${BENCHMARK_PKG}/config/teleop.yaml"

# --- Host-side paths (for the pinocchio unpack step) -----------------------
HOST_TELEOP_APP="$PWD/source/geniesim_teleop/src/geniesim_teleop/app"
PINOCCHIO_LIB="${HOST_TELEOP_APP}/vendors/lib/libpinocchio_casadi.so.3.7.0"
PINOCCHIO_TAR="${HOST_TELEOP_APP}/vendors/lib/libpinocchio.tar.gz"

# If the pinocchio library does not exist in vendors/lib, extract it
if [ ! -f "$PINOCCHIO_LIB" ] && [ -f "$PINOCCHIO_TAR" ]; then
    echo "Extracting libpinocchio.tar.gz to vendors/lib ..."
    tar -xzvf "$PINOCCHIO_TAR" -C "${HOST_TELEOP_APP}/vendors/lib"
fi

# --- Ensure the container is running ---------------------------------------
if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    echo "Start it first, then re-run this script. For the new CLI image:"
    echo "    geniesim docker5.1 up      # or: geniesim docker up"
    echo "Or set GENIESIM_CONTAINER=<name> if your container has a different name:"
    echo "    docker ps --format '{{.Names}}'"
    exit 1
else
    echo "Info: Container $CONTAINER_NAME already running"
fi

# --- Ensure geniesim_teleop is importable in the container -----------------
# teleop.py is launched as `python3 -m geniesim_teleop.teleop`, so the package
# must be pip-installed (editable) for the system python3 inside the container.
# A fresh checkout / rebuilt container won't have it yet, which fails with
# `No module named 'geniesim_teleop'`. Install it once, idempotently, here.
#
# --no-deps is REQUIRED: geniesim_teleop's pyproject pins `numpy<2.0`, but the
# image deliberately keeps NumPy 2.x global (Isaac/torch need it) and shims
# NumPy 1.x onto PYTHONPATH for just the teleop/bridge processes (see below).
# A deps-resolving install would downgrade the global NumPy and break Isaac.
if ! docker exec "$CONTAINER_NAME" python3 -c 'import geniesim_teleop' 2>/dev/null; then
    echo "Info: installing geniesim_teleop (editable, --no-deps) into $CONTAINER_NAME ..."
    docker exec "$CONTAINER_NAME" python3 -m pip install -e "${REPO_IN_CONTAINER}/source/geniesim_teleop" --no-deps
fi
# pynput is the one live-loop dependency not baked into the image; install it
# on its own (it pulls only evdev / python-xlib, never NumPy) so the teleop
# loop's `from pynput import keyboard` resolves. The heavier data_recording
# deps (open3d/h5py/rosbags) are only needed by post-processing, not here.
if ! docker exec "$CONTAINER_NAME" python3 -c 'import pynput' 2>/dev/null; then
    echo "Info: installing pynput into $CONTAINER_NAME ..."
    docker exec "$CONTAINER_NAME" python3 -m pip install pynput
fi

# --- The four teleop processes ---------------------------------------------
# Note: geniesim_teleop is pip-installed (editable) in the container, so
# teleop.py runs as a module under the system python3 (which has rclpy from
# ROS jazzy). No teleop_env venv is needed on the new image.
#
# Process 1 (simulator) prepends the Isaac Sim ROS 2 bridge lib dir to
# LD_LIBRARY_PATH. omni_python loads Isaac's own rclpy, but the login-shell
# profile sources /opt/ros/jazzy first, leaving the system rosidl typesupport
# libs ahead of Isaac's on LD_LIBRARY_PATH — that mismatch crashes the bridge
# with a `convert_from_py` assertion. Prepending Isaac's lib dir fixes the
# resolution order.
ISAAC_BRIDGE_LIB="/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib"
# Processes 2 & 4 (bridge.py / teleop.py) import cv_bridge, whose C extension
# was built against NumPy 1.x and crashes under the pip-installed NumPy 2.x in
# /usr/local. Prepend the apt-installed NumPy 1.26 (/usr/lib/python3/dist-
# packages) on PYTHONPATH for just these two processes — the global NumPy 2.x
# stays untouched (Isaac Sim / mujoco / torch depend on it).
NUMPY1_PATH="/usr/lib/python3/dist-packages"
declare -a COMMANDS=(
    "docker exec -it $CONTAINER_NAME bash -ic 'export LD_LIBRARY_PATH=${ISAAC_BRIDGE_LIB}:\$LD_LIBRARY_PATH && omni_python ${SIM_APP} --config ${TELEOP_YAML}'"
    "docker exec -it $CONTAINER_NAME bash -ic 'source /opt/ros/jazzy/setup.bash && source ${TELEOP_APP}/bin/env.sh && export PYTHONPATH=${NUMPY1_PATH}:\$PYTHONPATH && python3 ${TELEOP_PKG}/bridge.py'"
    "docker exec -it $CONTAINER_NAME bash -ic 'source /opt/ros/jazzy/setup.bash && source ${TELEOP_APP}/bin/env.sh && ${TELEOP_APP}/bin/start_mc.sh --no-tool'"
    "docker exec -it $CONTAINER_NAME bash -ic 'source /opt/ros/jazzy/setup.bash && source ${TELEOP_APP}/bin/env.sh && export SIM_REPO_ROOT=${REPO_IN_CONTAINER} && export PYTHONPATH=${NUMPY1_PATH}:\$PYTHONPATH && python3 -m geniesim_teleop.teleop'"
)
declare -a DELAYS=(1 15 3 5 5)

# --- Pick a terminal emulator ----------------------------------------------
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

# --- Launch each process in its own terminal -------------------------------
for i in "${!COMMANDS[@]}"; do
    sleep "${DELAYS[$i]}"
    if [[ "$TERMINAL_CMD" == "gnome-terminal"* ]]; then
        gnome-terminal -- bash -c "export TERMINAL_ENV=$TERMINAL_ENV; ${COMMANDS[$i]}; exec bash" &
    else
        $TERMINAL_CMD "${COMMANDS[$i]}" &
    fi
done

# --- Multi-episode keep/discard loop ---------------------------------------
# Legacy behaviour: a single y/N press killed every teleop process (simulator
# included) and the script exited, so collecting N episodes meant restarting
# all four terminals N times. Now y/n only ends the *current* episode: the
# simulator finalizes the ros2 bag gracefully (metadata.yaml intact) and
# re-arms, so the next record-button press starts the next episode. Press q
# to stop everything and close the terminals.

RECORDING_BASE="${REPO_IN_CONTAINER}/output/recording_data"

# End the current episode: broadcast on /sim/stop_episode (both the simulator
# and the teleop loop subscribe to it).
stop_current_episode() {
    echo "  -> asking the simulator to finalize the current episode..."
    docker exec "$CONTAINER_NAME" bash -c \
        "source /opt/ros/jazzy/setup.bash && timeout 15 ros2 topic pub --times 3 /sim/stop_episode std_msgs/msg/Bool '{data: true}' >/dev/null 2>&1" \
        || echo "  warning: failed to send stop signal (container or ROS not ready?)"
    sleep 2 # give ros2 bag a moment to write metadata.yaml
}

# Show whether the latest episode looks complete (.mcap = data recorded,
# metadata.yaml = bag finalized, recording_info.json = post-processing input).
show_latest_episode_status() {
    docker exec "$CONTAINER_NAME" bash -c "
        latest=\$(find $RECORDING_BASE -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -z \"\$latest\" ]; then
            echo '  status: no recording data yet (did you forget the record button?)'
            exit 0
        fi
        echo \"  latest episode dir: \$latest\"
        if ls \"\$latest\"/*.mcap >/dev/null 2>&1; then mcap=ok; else mcap=MISSING; fi
        if [ -f \"\$latest/metadata.yaml\" ]; then meta=ok; else meta=MISSING; fi
        if [ -f \"\$latest/recording_info.json\" ]; then info=ok; else info=MISSING; fi
        echo \"  .mcap: \$mcap | metadata.yaml: \$meta | recording_info.json: \$info\"
    " 2>/dev/null || echo "  status check failed (container not running?)"
}

echo ""
echo "=================================================================="
echo "All terminals started. Controls:"
echo "  1. press the controller's record button to start an episode"
echo "  2. when the episode is done, come back here and press:"
echo "     y = success: keep the data, then record the next episode"
echo "     n = failure: mark the data discarded, then record the next episode"
echo "     q = quit: stop everything and close the terminals"
echo "=================================================================="
while read -n 1 -s input; do
    if [[ "$input" == "Y" || "$input" == "y" ]]; then
        echo ""
        echo "[y] keeping this episode..."
        stop_current_episode
        docker exec "$CONTAINER_NAME" python3 ${TELEOP_PKG}/data_recording/patch_recording_info.py \
            --config ${TELEOP_YAML} \
            --base ${RECORDING_BASE} \
            || true
        show_latest_episode_status
        echo "  done. Press the record button for the next episode, or y/n/q here."
    elif [[ "$input" == "N" || "$input" == "n" ]]; then
        echo ""
        echo "[n] discarding this episode (teleop_result=false; post-processing drops it)..."
        stop_current_episode
        docker exec "$CONTAINER_NAME" python3 ${TELEOP_PKG}/data_recording/patch_recording_info.py \
            --config ${TELEOP_YAML} \
            --base ${RECORDING_BASE} \
            --teleop-result false \
            || true
        show_latest_episode_status
        echo "  done. Press the record button for the next episode, or y/n/q here."
    elif [[ "$input" == "Q" || "$input" == "q" ]]; then
        echo ""
        echo "[q] stopping all teleop processes..."
        # Finalize any episode still recording first (avoids a corrupt bag),
        # then kill the processes.
        stop_current_episode
        docker exec "$CONTAINER_NAME" bash -c "pkill -SIGTERM -f '$PROCESS_CLIENT' 2>/dev/null || true"
        sleep 1
        break
    fi
done

reset
