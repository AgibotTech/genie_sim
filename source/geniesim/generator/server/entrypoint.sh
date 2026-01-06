#!/bin/bash

# Debug info: Check mounted directories
echo "=== Debug Info ==="
echo "Current working directory: $(pwd)"
echo "Checking /geniesim directory:"
ls -la /geniesim 2>/dev/null  || echo "/geniesim does not exist"
echo "Checking /geniesim/assets directory:"
ls -la /geniesim/assets 2>/dev/null || echo "/geniesim/assets does not exist"
echo "Checking mcp_config.json:"
ls -la /geniesim/generator/server/mcp_config.json 2>/dev/null || echo "mcp_config.json does not exist"
echo "================"

exec "$@"
