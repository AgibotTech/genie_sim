#!/bin/bash
set -e

# 1. Check Contrainer mount
echo "--- Container Directory Check ---"
ls -d /geniesim/main/3rdparty/ || echo "Cannot find /geniesim/main/3rdparty/"

# 2. Docker runs on root
mkdir -p /geniesim/main/source/teleop/app/share
mkdir -p /geniesim/main/source/teleop/app/bin/.cache
chmod -R 777 /geniesim/main/source/teleop/app/bin

# 3. Install package (uv is faster and safe than just pip)
rm -rf /geniesim/main/source/GenieSim.egg-info

# 3rdparty
WHL_PATH=$(find /geniesim/main/3rdparty -name "ik_solver-*.whl" | head -n 1)

if [ -n "$WHL_PATH" ]; then
    echo "Found: $WHL_PATH. Installing..."
    uv pip install "$WHL_PATH"
else
    echo "ERROR: File not found in /geniesim/main/3rdparty/"
    exit 1
fi

# 4. Install source code
uv pip install -e /geniesim/main/source

echo "Setup Complete."
exec "$@"
