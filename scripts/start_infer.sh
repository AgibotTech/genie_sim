#!/bin/bash

HOST_ROOT_DIR=$(readlink -f .)
xhost +local:

# 기존 컨테이너 삭제
docker rm -f pi05_infer 2>/dev/null

docker run -it --name pi05_infer \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    --privileged \
    -e DISPLAY \
    -v /dev/input:/dev/input:rw \
    -v "$HOST_ROOT_DIR":/geniesim/main:rw \
    -w /geniesim/main \
    openpi_server:latest \
    /bin/bash ./scripts/entrypoint.sh /bin/bash
