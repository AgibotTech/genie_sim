#!/bin/bash

# current folder as WORD_DIR
CURRENT_DIR=$(pwd)

echo $CURRENT_DIR
xhost +local:
docker run -it --name pi05_infer\
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    --privileged \
    -e DISPLAY \
    -v /dev/input:/dev/input:rw \
    -v $CURRENT_DIR/openpi:/root/openpi:rw \
    -w /root/openpi \
    openpi_server:latest \
    bash ./scripts/entrypoint.sh
