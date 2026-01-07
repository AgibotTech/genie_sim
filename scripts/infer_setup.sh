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
cd $TARGET_DIR
git apply $SCRIPT_DIR/infer_changes.patch

echo "Successfully deployed local infer files"

mkdir -p $TARGET_DIR/checkpoints/select_color
mkdir -p $TARGET_DIR/checkpoints/size_recogize
mkdir -p $TARGET_DIR/checkpoints/grasp_targets
mkdir -p $TARGET_DIR/checkpoints/organize_items

echo "Successfully created checkpoints directories"
