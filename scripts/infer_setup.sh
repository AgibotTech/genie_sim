#!/bin/bash

set -e

REPO_URL="https://github.com/AgibotTech/ACoT-VLA.git"
BRANCH="genie_sim"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/openpi"

echo "========================================="
echo "Cloning ACoT-VLA repository..."
echo "Repository: $REPO_URL"
echo "Target directory: $TARGET_DIR"
echo "========================================="

if [ -d "$TARGET_DIR" ]; then
    echo "Warning: Directory $TARGET_DIR already exists."
    read -p "Do you want to remove it and clone again? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing directory..."
        rm -rf "$TARGET_DIR"
    else
        echo "Aborting. Please remove or rename the existing directory first."
        exit 1
    fi
fi

MAX_RETRIES=3
RETRY_DELAY=5

for i in $(seq 1 $MAX_RETRIES); do
    echo "Cloning repository (attempt $i/$MAX_RETRIES)..."
    if git clone -b "$BRANCH" -c http.postBuffer=524288000 "$REPO_URL" "$TARGET_DIR" 2>&1; then
        echo "✓ Successfully cloned $REPO_URL (branch: $BRANCH) to $TARGET_DIR"
        break
    else
        if [ $i -lt $MAX_RETRIES ]; then
            echo "Clone failed, retrying in ${RETRY_DELAY}s..."
            rm -rf "$TARGET_DIR"
            sleep $RETRY_DELAY
        else
            echo "✗ Failed to clone after $MAX_RETRIES attempts."
            echo ""
            echo "Possible fixes:"
            echo "  1. Check your network connection / proxy settings"
            echo "  2. Try: git config --global http.sslVerify false"
            echo "  3. Set a proxy: git config --global http.proxy <your_proxy>"
            echo "  4. Use SSH: git clone -b $BRANCH git@github.com:AgibotTech/ACoT-VLA.git $TARGET_DIR"
            exit 1
        fi
    fi
done
