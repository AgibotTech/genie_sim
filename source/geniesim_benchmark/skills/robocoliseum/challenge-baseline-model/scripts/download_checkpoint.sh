#!/bin/bash

set -e

DATASET_ID="agibot_world/GenieSim3.0-Dataset"

VALID_CHECKPOINTS=(
    "instruction_and_robust_pi05"
    "manipulation_pi05"
    "spatial_pi05"
)

show_help() {
    echo "Usage: $0 [CHECKPOINT_NAME] [LOCAL_DIR]"
    echo ""
    echo "Download a baseline checkpoint from ModelScope dataset: $DATASET_ID"
    echo "Remote checkpoints live under: checkpoints/<CHECKPOINT_NAME>"
    echo ""
    echo "Arguments:"
    echo "  CHECKPOINT_NAME   Name of the checkpoint to download (default: instruction_and_robust_pi05)"
    echo "  LOCAL_DIR         Base directory for checkpoints (default: ./openpi/checkpoints/)"
    echo "  -h, --help        Show this help"
    echo ""
    echo "Available checkpoints:"
    for ckpt in "${VALID_CHECKPOINTS[@]}"; do
        echo "  - $ckpt"
    done
    echo ""
    echo "Examples:"
    echo "  $0                                        # instruction_and_robust_pi05 -> ./openpi/checkpoints/instruction_and_robust_pi05/"
    echo "  $0 manipulation_pi05                      # manipulation_pi05 -> ./openpi/checkpoints/manipulation_pi05/"
    echo "  $0 spatial_pi05 /path/to/save             # spatial_pi05 -> /path/to/save/spatial_pi05/"
}

CHECKPOINT_NAME=""
LOCAL_DIR=""

for arg in "$@"; do
    case $arg in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$CHECKPOINT_NAME" ]; then
                CHECKPOINT_NAME="$arg"
            elif [ -z "$LOCAL_DIR" ]; then
                LOCAL_DIR="$arg"
            else
                echo "Error: too many arguments"
                echo ""
                show_help
                exit 1
            fi
            ;;
    esac
done

CHECKPOINT_NAME="${CHECKPOINT_NAME:-instruction_and_robust_pi05}"
LOCAL_DIR="${LOCAL_DIR:-./openpi/checkpoints/}"

is_valid=false
for ckpt in "${VALID_CHECKPOINTS[@]}"; do
    if [ "$CHECKPOINT_NAME" = "$ckpt" ]; then
        is_valid=true
        break
    fi
done

if [ "$is_valid" = false ]; then
    echo "Error: invalid checkpoint name '$CHECKPOINT_NAME'"
    echo ""
    echo "Available checkpoints:"
    for ckpt in "${VALID_CHECKPOINTS[@]}"; do
        echo "  - $ckpt"
    done
    exit 1
fi

if ! command -v modelscope &> /dev/null; then
    echo "Error: modelscope not found; install with: pip install modelscope"
    exit 1
fi

REMOTE_PATH="checkpoints/$CHECKPOINT_NAME"
FINAL_DIR="$LOCAL_DIR/$CHECKPOINT_NAME"
mkdir -p "$FINAL_DIR"

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "========================================="
echo "Downloading checkpoint: $CHECKPOINT_NAME"
echo "Dataset: $DATASET_ID"
echo "Remote path: $REMOTE_PATH"
echo "Local target: $FINAL_DIR"
echo "========================================="

modelscope download --dataset "$DATASET_ID" --include "$REMOTE_PATH/**" --local_dir "$TEMP_DIR"

if [ $? -eq 0 ]; then
    cp -a "$TEMP_DIR/$REMOTE_PATH"/. "$FINAL_DIR"/
    echo ""
    echo "========================================="
    echo "✓ Checkpoint '$CHECKPOINT_NAME' downloaded to $FINAL_DIR"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ Download failed"
    echo "========================================="
    exit 1
fi
