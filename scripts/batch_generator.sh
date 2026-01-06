#!/bin/bash

# Script to run geniesim/generator/app.py 10 times in isaacsim5 conda environment
# Author: Auto-generated
# Date: 2025-12-21

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim5

scene_id=$1
run_times=$2


# Check if conda environment activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate isaacsim5 conda environment"
    exit 1
fi

echo "Starting generation"
echo "=============================================="

cd ../source
# Run the script run_times times
for ((i=1; i<=run_times; i++))
do
    echo ""
    echo "Run #$i started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------------"

    python3 geniesim/generator/app.py --scene_id $scene_id --task_gen

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Run #$i completed successfully"
    else
        echo "Run #$i failed with exit code $exit_code"
        echo "Stopping execution due to error"
        exit $exit_code
    fi

    echo "----------------------------------------------"
done

echo ""
echo "=============================================="
echo "All 10 runs completed at $(date '+%Y-%m-%d %H:%M:%S')"
