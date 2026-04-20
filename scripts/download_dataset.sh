#!/bin/bash

set -e

DATASET_ID="agibot_world/GenieSim3.0-Dataset"

VALID_SUITES=(
    "instruction"
    "manipulation"
    "sim2real"
)

show_help() {
    echo "Usage: $0 [SUITE_NAME] [LOCAL_DIR]"
    echo ""
    echo "Download task suite data from ModelScope dataset: $DATASET_ID"
    echo ""
    echo "Arguments:"
    echo "  SUITE_NAME   Name of the task suite to download (default: all)"
    echo "  LOCAL_DIR    Base directory for datasets (default: ./data/)"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Available task suites:"
    for suite in "${VALID_SUITES[@]}"; do
        echo "  - $suite"
    done
    echo ""
    echo "Examples:"
    echo "  $0                                  # Download all task suites to ./data/"
    echo "  $0 instruction                      # Download instruction suite to ./data/"
    echo "  $0 sim2real /path/to/save           # Download sim2real suite to /path/to/save/"
}

SUITE_NAME=""
LOCAL_DIR=""

for arg in "$@"; do
    case $arg in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$SUITE_NAME" ]; then
                SUITE_NAME="$arg"
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

LOCAL_DIR="${LOCAL_DIR:-./data/}"

if ! command -v modelscope &> /dev/null; then
    echo "Error: modelscope not found; install with: pip install modelscope"
    exit 1
fi

download_suite() {
    local suite=$1
    local target_dir=$2

    mkdir -p "$target_dir"

    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    echo "========================================="
    echo "Downloading task suite: $suite"
    echo "Dataset: $DATASET_ID"
    echo "Remote path: task_suite/$suite"
    echo "Target directory: $target_dir"
    echo "========================================="

    modelscope download --dataset "$DATASET_ID" --include "task_suite/$suite/**" --local_dir "$TEMP_DIR"

    if [ $? -eq 0 ]; then
        cp -a "$TEMP_DIR/task_suite/$suite"/. "$target_dir"/
        echo ""
        echo "========================================="
        echo "✓ Task suite '$suite' downloaded to $target_dir"
        echo "========================================="
    else
        echo ""
        echo "========================================="
        echo "✗ Download failed for '$suite'"
        echo "========================================="
        exit 1
    fi

    rm -rf "$TEMP_DIR"
    trap - EXIT
}

if [ -z "$SUITE_NAME" ]; then
    echo "Downloading all task suites..."
    echo ""
    for suite in "${VALID_SUITES[@]}"; do
        download_suite "$suite" "$LOCAL_DIR/$suite"
    done
    echo ""
    echo "========================================="
    echo "✓ All task suites downloaded to $LOCAL_DIR"
    echo "========================================="
else
    is_valid=false
    for suite in "${VALID_SUITES[@]}"; do
        if [ "$SUITE_NAME" = "$suite" ]; then
            is_valid=true
            break
        fi
    done

    if [ "$is_valid" = false ]; then
        echo "Error: invalid task suite name '$SUITE_NAME'"
        echo ""
        echo "Available task suites:"
        for suite in "${VALID_SUITES[@]}"; do
            echo "  - $suite"
        done
        exit 1
    fi

    download_suite "$SUITE_NAME" "$LOCAL_DIR/$SUITE_NAME"
fi
