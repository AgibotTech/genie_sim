#!/bin/bash

set -e

DATASET_ID="agibot_world/GenieSim3.0-Dataset"

VALID_CHECKPOINTS=(
    "grasp_targets"
    "handover_chips"
    "organize_items"
    "pack_in_supermarket"
    "place_block_in_drawer"
    "recognize_size"
    "select_color"
    "sort_fruit"
)

show_help() {
    echo "Usage: $0 [CHECKPOINT_NAME] [LOCAL_DIR]"
    echo ""
    echo "Download checkpoint from ModelScope dataset: $DATASET_ID"
    echo ""
    echo "Arguments:"
    echo "  CHECKPOINT_NAME   Name of the checkpoint to download (default: select_color)"
    echo "  LOCAL_DIR         Base directory for checkpoints (default: ./openpi/checkpoints/)"
    echo "  -h, --help        Show this help"
    echo ""
    echo "Available checkpoints:"
    for ckpt in "${VALID_CHECKPOINTS[@]}"; do
        echo "  - $ckpt"
    done
    echo ""
    echo "Examples:"
    echo "  $0                                  # Download select_color to ./openpi/checkpoints/select_color/"
    echo "  $0 grasp_targets                    # Download grasp_targets to ./openpi/checkpoints/grasp_targets/"
    echo "  $0 sort_fruit /path/to/save         # Download sort_fruit to /path/to/save/sort_fruit/"
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

CHECKPOINT_NAME="${CHECKPOINT_NAME:-select_color}"
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

mkdir -p "$LOCAL_DIR"

echo "========================================="
echo "Downloading checkpoint: $CHECKPOINT_NAME"
echo "Dataset: $DATASET_ID"
echo "Target directory: $LOCAL_DIR"
echo "========================================="

if ! command -v modelscope &> /dev/null; then
    echo "Error: modelscope not found; install with: pip install modelscope"
    exit 1
fi

CHECKPOINTS_WITH_SUFFIX=("select_color" "grasp_targets" "organize_items" "recognize_size")
REMOTE_PATH="checkpoints/$CHECKPOINT_NAME"
HAS_SUFFIX=false
for ckpt in "${CHECKPOINTS_WITH_SUFFIX[@]}"; do
    if [ "$CHECKPOINT_NAME" = "$ckpt" ]; then
        REMOTE_PATH="checkpoints/$CHECKPOINT_NAME/29999"
        HAS_SUFFIX=true
        break
    fi
done

FINAL_DIR="$LOCAL_DIR/$CHECKPOINT_NAME"
mkdir -p "$FINAL_DIR"

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Remote path: $REMOTE_PATH"
echo "Local target: $FINAL_DIR"
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
