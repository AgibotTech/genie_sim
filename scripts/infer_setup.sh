#!/bin/bash

# Script to clone openpi repository with specific branch
# Clones https://github.com/Physical-Intelligence/openpi.git
# Branch: kevin/pi05-support
# Target directory: openpi (in project root)

set -e  # Exit on error

REPO_URL="https://github.com/Physical-Intelligence/openpi.git"
BRANCH="kevin/pi05-support"

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/openpi"

echo "Cloning openpi repository..."
echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo "Target directory: $TARGET_DIR"

# Check if directory already exists
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

# Clone the repository with the specific branch
echo "Cloning repository..."
git clone -b "$BRANCH" "$REPO_URL" "$TARGET_DIR"

echo "Successfully cloned $REPO_URL (branch: $BRANCH) to $TARGET_DIR"

# Deploy local infer files
cp $SCRIPT_DIR/infer_deps/serve_policy.Dockerfile $PROJECT_ROOT/openpi/scripts/docker/serve_policy.Dockerfile
cp $SCRIPT_DIR/infer_deps/serve_policy.py $PROJECT_ROOT/openpi/scripts/serve_policy.py
cp $SCRIPT_DIR/infer_deps/go1_policy.py $PROJECT_ROOT/openpi/src/openpi/policies/go1_policy.py
cp $SCRIPT_DIR/infer_deps/config.py $PROJECT_ROOT/openpi/src/openpi/training/config.py
cp $SCRIPT_DIR/infer_deps/data_loader.py $PROJECT_ROOT/openpi/src/openpi/training/data_loader.py

echo "Successfully deployed local infer files"
