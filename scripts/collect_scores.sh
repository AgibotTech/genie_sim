#!/bin/bash
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Script to collect scores from evaluate_ret_*.json files and generate CSV output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="$PROJECT_ROOT/output/benchmark"
TASK_CONFIG_FILE="$PROJECT_ROOT/source/geniesim/benchmark/config/task_config_mapping.py"
STAT_SCORE_SCRIPT="$SCRIPT_DIR/stat_score.py"
OUTPUT_CSV="$PROJECT_ROOT/output/task_scores.csv"

# Temporary file for Python processing
TEMP_JSON=$(mktemp)
trap "rm -f $TEMP_JSON" EXIT

echo "Collecting scores from benchmark results..."
echo "Benchmark directory: $BENCHMARK_DIR"
echo "Output file: $OUTPUT_CSV"

# Create CSV header
echo "task_name,avg_score,operation_label,cognitive_label" > "$OUTPUT_CSV"

# Python script to extract eval_dims from task_config_mapping.py
python3 << PYTHON_EOF > "$TEMP_JSON"
import sys
import json
import os

# Add the source directory to path to import task_config_mapping
source_dir = "$PROJECT_ROOT/source"
sys.path.insert(0, source_dir)

try:
    from geniesim.benchmark.config.task_config_mapping import TASK_MAPPING

    # Create a mapping of sub_task_name -> eval_dims
    result = {}
    for task_name, config in TASK_MAPPING.items():
        eval_dims = config.get("eval_dims", {})
        if isinstance(eval_dims, dict):
            result[task_name] = {
                "manip": eval_dims.get("manip", ""),
                "cognition": eval_dims.get("cognition", "")
            }
        elif isinstance(eval_dims, str):
            # Handle special case like "long-horizon"
            result[task_name] = {
                "manip": eval_dims,
                "cognition": eval_dims
            }
        else:
            result[task_name] = {
                "manip": "",
                "cognition": ""
            }

    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

# Check if Python script succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract task mapping" >&2
    exit 1
fi

# Read the task mapping JSON
TASK_MAPPING_JSON=$(cat "$TEMP_JSON")

# Function to get eval_dims for a sub_task_name
get_eval_dims() {
    local sub_task_name=$1
    # Use a temporary file to avoid shell escaping issues
    local temp_file=$(mktemp)
    echo "$TASK_MAPPING_JSON" > "$temp_file"
    python3 << PYTHON_EOF
import json
import sys

with open("$temp_file", "r") as f:
    task_mapping = json.load(f)

sub_task_name = "$sub_task_name"
if sub_task_name in task_mapping:
    eval_dims = task_mapping[sub_task_name]
    print(f"{eval_dims['manip']}|{eval_dims['cognition']}")
else:
    print("|")
PYTHON_EOF
    rm -f "$temp_file"
}

# Function to format labels
format_operation_label() {
    local manip=$1
    case "$manip" in
        "pick") echo "Planer pick" ;;
        "planer_pick_place_") echo "Planer pickplace" ;;
        "spatial_pick") echo "Spatial pick" ;;
        "spatial_pick_place") echo "Spatial pickplace" ;;
        "insert") echo "Insert" ;;
        "straighten") echo "Straighten" ;;
        "push") echo "Push" ;;
        "pull") echo "Pull" ;;
        "open") echo "Open" ;;
        "press") echo "Press" ;;
        "turn") echo "Turn" ;;
        "bimanual_hold") echo "Bimanual hold" ;;
        "dump") echo "Dump" ;;
        "pick_place") echo "Pick place" ;;
        "hang") echo "Hang" ;;
        "long-horizon") echo "Long-horizon" ;;
        *) echo "$manip" ;;
    esac
}

format_cognitive_label() {
    local cognition=$1
    case "$cognition" in
        "color") echo "Color" ;;
        "shape") echo "Shape" ;;
        "size") echo "Size" ;;
        "number") echo "Number" ;;
        "category") echo "Category" ;;
        "semantic") echo "Semantic" ;;
        "logic") echo "Logic" ;;
        "position") echo "Position" ;;
        "common_sense") echo "Common Sense" ;;
        "long-horizon") echo "Long-horizon" ;;
        *) echo "$cognition" ;;
    esac
}

# Collect all unique sub_task_name directories
declare -A processed_tasks

# Find all directories containing evaluate_ret_*.json files
while IFS= read -r json_file; do
    # Get the directory containing the JSON file
    task_dir=$(dirname "$json_file")

    # Extract sub_task_name (the directory name containing the JSON file)
    sub_task_name=$(basename "$task_dir")

    # Skip if we've already processed this sub_task_name
    if [ -n "${processed_tasks[$sub_task_name]}" ]; then
        continue
    fi

    processed_tasks[$sub_task_name]=$task_dir
done < <(find "$BENCHMARK_DIR" -type f -name "evaluate_ret_*.json")

# Process each unique sub_task_name
for sub_task_name in "${!processed_tasks[@]}"; do
    task_dir="${processed_tasks[$sub_task_name]}"

    echo "Processing: $sub_task_name"

    # Get eval_dims from task_config_mapping
    eval_dims=$(get_eval_dims "$sub_task_name")
    manip=$(echo "$eval_dims" | cut -d'|' -f1)
    cognition=$(echo "$eval_dims" | cut -d'|' -f2)

    if [ -z "$manip" ] && [ -z "$cognition" ]; then
        echo "Warning: No eval_dims found for $sub_task_name, skipping..." >&2
        continue
    fi

    # Get average score using stat_score.py
    temp_result_file=$(mktemp)
    # Run stat_score.py and redirect stdout to /dev/null to avoid mixing with CSV output
    # We only need the JSON file, not the stdout message
    python3 "$STAT_SCORE_SCRIPT" --dir "$task_dir" -o "$temp_result_file" >/dev/null 2>&1 || true

    # Read the score from JSON file
    avg_score=$(python3 -c "
import json
import sys
try:
    with open('$temp_result_file', 'r') as f:
        data = json.load(f)
    if 'e2e' in data and 'average' in data['e2e'] and data['e2e']['average'] is not None:
        print(f\"{data['e2e']['average']:.4f}\")
    else:
        print('0.0000')
except Exception as e:
    print('0.0000')
" 2>/dev/null || echo "0.0000")
    rm -f "$temp_result_file"

    # Format labels
    operation_label=$(format_operation_label "$manip")
    cognitive_label=$(format_cognitive_label "$cognition")

    # Clean variables: remove any newlines, carriage returns, and trim whitespace
    sub_task_name=$(echo -n "$sub_task_name" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    avg_score=$(echo -n "$avg_score" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    operation_label=$(echo -n "$operation_label" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    cognitive_label=$(echo -n "$cognitive_label" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # Append to CSV (ensure proper format: task_name,avg_score,operation_label,cognitive_label)
    # Use printf to ensure clean output without extra newlines
    printf "%s,%s,%s,%s\n" "$sub_task_name" "$avg_score" "$operation_label" "$cognitive_label" >> "$OUTPUT_CSV"

    echo "  Score: $avg_score, Manip: $operation_label, Cognition: $cognitive_label"
done

echo ""
echo "Done! Results saved to: $OUTPUT_CSV"
echo "Total tasks processed: $(tail -n +2 "$OUTPUT_CSV" | wc -l)"
