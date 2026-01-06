#!/bin/bash
# Auto recording startup script

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Auto Image Recording and Extraction Tool ===${NC}"
echo ""

PROJECT_ROOT=$(pwd)

source $PROJECT_ROOT/../record_env/bin/activate

# Set default parameters
OUTPUT_DIR="$PROJECT_ROOT/output/recording_data"
FINAL_OUTPUT_DIR="$PROJECT_ROOT/output/recording_data"
TIMEOUT=10

echo "Configuration parameters:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Final output directory: $FINAL_OUTPUT_DIR"
echo "  Timeout: ${TIMEOUT} seconds"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Starting recording node...${NC}"
echo "Waiting for CompressedImage topics starting with /record/..."
echo "Recording will automatically stop after ${TIMEOUT} seconds with no new messages"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop manually${NC}"
echo ""

unset $LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash

# Run script (parameters parsed via argparse)
python3 "$(dirname "$0")/auto_record_and_extract.py" \
    --output_dir "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" \
    --final_output_dir "$FINAL_OUTPUT_DIR" \
    --delete_db3_after

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Recording Complete ===${NC}"
    echo "Output location: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - rosbag: $OUTPUT_DIR/recording_*/"
    echo "  - images: $OUTPUT_DIR/images/"
    echo "  - videos: $OUTPUT_DIR/videos/"
    echo ""

    # Display video list
    if [ -d "$OUTPUT_DIR/videos" ]; then
        VIDEO_COUNT=$(ls -1 "$OUTPUT_DIR/videos"/*.webm 2>/dev/null | wc -l)
        if [ $VIDEO_COUNT -gt 0 ]; then
            echo "Generated video files:"
            ls -lh "$OUTPUT_DIR/videos"/*.webm
        fi
    fi
else
    echo ""
    echo -e "${RED}Program exited abnormally (exit code: $EXIT_CODE)${NC}"
    exit $EXIT_CODE
fi
