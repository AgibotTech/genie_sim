#!/bin/bash
CONTAINER_NAME="genie_sim_benchmark"
START_SCRIPT="$PWD/scripts/start_gui.sh"
TERMINAL_ENV="autorun"
PROCESS_CLIENT="teleop|ros"

# If the pinocchio library does not exist in vendors/lib, extract it
PINOCCHIO_LIB="$PWD/source/teleop/app/vendors/lib/libpinocchio_casadi.so.3.7.0"
PINOCCHIO_TAR="$PWD/source/teleop/app/vendors/lib/libpinocchio.tar.gz"
if [ ! -f "$PINOCCHIO_LIB" ]; then
    echo "Extracting libpinocchio.tar.gz to vendors/lib ..."
    tar -xzvf "$PINOCCHIO_TAR" -C "$PWD/source/teleop/app/vendors/lib"
fi

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Warning: Container $CONTAINER_NAME not running, try to start..."

    if [ -x "$START_SCRIPT" ]; then
        echo "Executing script: $START_SCRIPT (in background)"
        "$START_SCRIPT" &
        START_PID=$!
        MAX_WAIT=60
        ELAPSED=0
        while [ $ELAPSED -lt $MAX_WAIT ]; do
            sleep 3
            ELAPSED=$((ELAPSED + 3))
            if docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
                echo "Info: Container $CONTAINER_NAME started (after ${ELAPSED}s)"
                break
            fi
        done
        if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
            echo "Error: Container failed to start within ${MAX_WAIT}s"
            echo "Tip: run './scripts/start_gui.sh' in foreground to see errors, or inspect with: docker ps -a | grep $CONTAINER_NAME"
            kill $START_PID 2>/dev/null || true
            exit 1
        fi
    else
        echo "Error: Start script $START_SCRIPT not exist or not executable"
        exit 1
    fi
else
    echo "Info: Container $CONTAINER_NAME already running"
fi

declare -a COMMANDS=(
    "docker exec -it $CONTAINER_NAME bash -ic 'omni_python ./source/geniesim/app/app.py --config ./source/geniesim/config/teleop.yaml'"
    "docker exec -it $CONTAINER_NAME bash -ic 'source /opt/ros/jazzy/setup.bash && source /geniesim/main/source/teleop/app/bin/env.sh && python3 ./source/teleop/bridge.py'"
    "docker exec -it $CONTAINER_NAME bash -ic 'source /opt/ros/jazzy/setup.bash && source /geniesim/main/source/teleop/app/bin/env.sh && /geniesim/main/source/teleop/app/bin/start_mc.sh --no-tool'"
    "docker exec -it $CONTAINER_NAME bash -ic 'source /geniesim/teleop_env/bin/activate && source /opt/ros/jazzy/setup.bash && source /geniesim/main/source/teleop/app/bin/env.sh && python3 ./source/teleop/teleop.py'"
)
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

for i in "${!COMMANDS[@]}"; do
    sleep "${DELAYS[$i]}"
    if [[ "$TERMINAL_CMD" == "gnome-terminal"* ]]; then
        gnome-terminal -- bash -c "export TERMINAL_ENV=$TERMINAL_ENV; ${COMMANDS[$i]}; exec bash" &
    else
        $TERMINAL_CMD "${COMMANDS[$i]}" &
    fi
done

echo -e "\nAll terminals started. Press 'y' or 'Y' = teleoperation succeeded, keep data; 'n' or 'N' = failed, do not keep data ..."
while read -n 1 -s input; do
    if [[ "$input" == "Y" || "$input" == "y" ]]; then
        echo "Save the remote operation data.....Congratulations!"
        echo -e "Sending SIGTERM to teleop processes..."
        docker exec "$CONTAINER_NAME" bash -c "pkill -SIGTERM -f '$PROCESS_CLIENT' 2>/dev/null || true"
        sleep 1
        echo "Patching recording_info.json: add teleop_result"
        docker exec "$CONTAINER_NAME" python3 /geniesim/main/source/teleop/data_recording/patch_recording_info.py \
            --config /geniesim/main/source/geniesim/config/teleop.yaml \
            --base /geniesim/main/output/recording_data \
            || true

        break
    elif [[ "$input" == "N" || "$input" == "n" ]]; then
        echo -e "Sending SIGTERM to teleop processes..."
        docker exec "$CONTAINER_NAME" bash -c "pkill -SIGTERM -f '$PROCESS_CLIENT' 2>/dev/null || true"
        sleep 1
        echo "Patching recording_info.json: add teleop_result=false"
        docker exec "$CONTAINER_NAME" python3 /geniesim/main/source/teleop/data_recording/patch_recording_info.py \
            --config /geniesim/main/source/geniesim/config/teleop.yaml \
            --base /geniesim/main/output/recording_data \
            --teleop-result false \
            || true
        break
    fi
done


reset
