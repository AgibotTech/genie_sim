# GenieSim Docker image — Isaac Sim 6.X + ROS 2 Jazzy + system Python 3.12.
#
# Build:
#   geniesim docker6.2 build
#   geniesim docker6.2 build --china      # select the China mirror preset
#
# Or manually:
#   docker build -f docker/Dockerfile --build-arg LIBUIPC_SM=89 -t registry.agibot.com/genie-sim/geniesim4:latest .
#   docker build -f docker/Dockerfile \
#       --build-arg ISAACSIM_REGISTRY=nvcr.m.daocloud.io/nvidia \
#       --build-arg ISAACSIM_TAG=6.2.0.1 \
#       --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu \
#       --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#       --build-arg LIBUIPC_SM=89 -t registry.agibot.com/genie-sim/geniesim4:latest .
#
# Base: NVIDIA's published Isaac Sim 6.X image (Ubuntu 24.04).
# The image already ships Isaac Sim under /isaac-sim with
# /isaac-sim/python.sh wrapping a bundled CPython, plus the omniverse
# cache dirs owned by the canned "isaac-sim" user (uid:gid = 1234:1234).
#
# This image installs:
#   * ROS 2 Jazzy desktop on top of Ubuntu 24.04 (Noble) — apt-managed.
#   * A lightweight system-Python adapter to the existing /isaac-sim SDK.
#     python3 remains Ubuntu's Python 3.12 and uses Jazzy's rclpy. No second
#     Isaac SDK payload is installed.
#
# Host-user remap (1234 vs $(id -u)) is handled at *runtime* by
# entrypoint.sh — see the comments there. Build-time UID stays at 1234.

ARG ISAACSIM_REGISTRY=nvcr.io/nvidia
# Isaac Sim 6.2 is the Genie Sim 4.0 target. Keep these contracts aligned:
#   ISAACSIM_TAG         6.2.0.1   NGC base image respin
#   ISAACSIM_PIP_VERSION 6.2.0.0   adapter compatibility version
#   mujoco~=3.12.0                 source/geniesim_ros/pyproject.toml
# The adapter checks the bundled VERSION/ABI and derives external dependency
# pins from SDK metadata. A different major requires its own variant. Newton's
# MuJoCo requirements are guarded by docker/check_newton_pins.py.
ARG ISAACSIM_TAG=6.2.0.1

FROM scratch AS libuipc_assets
COPY docker/libuipc-artifacts.json /libuipc-artifacts.json

FROM ${ISAACSIM_REGISTRY}/isaac-sim:${ISAACSIM_TAG}

# Image-wide repository defaults; the CLI selects presets from docker/mirrors.toml.
ARG APT_MIRROR=upstream
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG ISAACSIM_PYPI_INDEX=https://pypi.nvidia.com
ARG ISAACSIM_PIP_VERSION=6.2.0.0

ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_CONFIG_FILE=/etc/pip.conf

USER root

# omni_python wrapper — exec the real script so $0 stays /isaac-sim/python.sh.
# Also open /isaac-sim to world-traverse (NVIDIA base ships it drwxr-x---).
RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/omni_python && \
    chmod +x /usr/local/bin/omni_python && \
    chmod o+rx /isaac-sim

COPY docker/configure_mirrors.py /usr/local/bin/geniesim-configure-mirrors
RUN chmod +x /usr/local/bin/geniesim-configure-mirrors && \
    omni_python /usr/local/bin/geniesim-configure-mirrors --apt "${APT_MIRROR}" --pip "${PIP_INDEX_URL}"

# ------------------------------------------------------------
# System dependencies + locale
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        wget \
        curl \
        git \
        git-lfs \
        sudo \
        gosu \
        lsb-release \
        unzip \
        xterm \
        build-essential \
        cmake \
        libeigen3-dev \
        ffmpeg \
        jq \
        vim \
        acl \
        locales \
        software-properties-common \
        ca-certificates \
        gnupg \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-numpy python3-scipy python3-opencv \
        iproute2 procps gettext-base \
        libvulkan1 vulkan-tools \
        libx11-6 libxrandr2 libxss1 libxcursor1 libxi6 libglu1-mesa \
        libgl1 libegl1 libnss3 libxcomposite1 libxdamage1 libxext6 \
        liburing2 liburing-dev \
        graphviz libgraphviz-dev && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ------------------------------------------------------------
# ROS 2 Jazzy
# https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
# ------------------------------------------------------------
RUN add-apt-repository universe -y && \
    geniesim-configure-mirrors --apt "${APT_MIRROR}" && \
    curl -fsSL -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-desktop \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-vision-msgs-rviz-plugins \
        ros-jazzy-joy \
        ros-jazzy-xacro ros-jazzy-cv-bridge ros-jazzy-gazebo-msgs \
        ros-dev-tools && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update --rosdistro jazzy --include-eol-distros 2>&1 | grep -v "as root is not recommended"

# ------------------------------------------------------------
# Register the bundled SDK for ordinary system python3 imports. The local
# wheel owns only its adapter/metadata; SDK files stay in /isaac-sim.
# ISAACSIM_PIP_VERSION retains the CLI's release-pair compatibility contract.
# ------------------------------------------------------------
ENV PIP_NO_COMPILE=1
COPY docker/bundled_isaacsim.py /opt/geniesim/docker/bundled_isaacsim.py
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --upgrade --ignore-installed pip && \
    python3 /opt/geniesim/docker/bundled_isaacsim.py \
        --version "${ISAACSIM_PIP_VERSION}" --output /tmp/isaac-adapter && \
    python3 -m pip install --no-deps /tmp/isaac-adapter/*.whl && \
    rm -rf /tmp/isaac-adapter && \
    python3 /opt/geniesim/docker/bundled_isaacsim.py --clean-bytecode /usr/local/lib/python3.12/dist-packages

# Pre-accept EULA for Isaac Sim 6.X (OMNI_KIT_ACCEPT_EULA path)
RUN touch /isaac-sim/kit/EULA_ACCEPTED && chmod o+w /isaac-sim/kit

# ------------------------------------------------------------
# Sudoers + workspace mountpoint
#
# The base image ships the "isaac-sim" user at uid:gid 1234:1234 and
# owns /isaac-sim. We grant it passwordless sudo so entrypoint.sh can
# do the runtime UID remap. /workspace is the canonical bind-mount
# target for the host repo.
# ------------------------------------------------------------
RUN id isaac-sim >/dev/null 2>&1 || useradd -m -u 1234 -s /bin/bash isaac-sim && \
    usermod -aG sudo isaac-sim && \
    echo "isaac-sim ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/isaac-sim && \
    chmod 0440 /etc/sudoers.d/isaac-sim && \
    mkdir -p /workspace && chown 1234:1234 /workspace

# ------------------------------------------------------------
# GeneSim module dependencies (extracted from **tier-1** pyproject.toml files).
# NOTE: collect_deps.py is shared across all variants — lives in docker/.
# This 6.2 image also includes teleop dependencies for Pico and Wuji.
# NumPy/SciPy remain SDK-owned globally; ROS controls use the apt overlay.
# ------------------------------------------------------------
COPY docker/collect_deps.py /tmp/collect_deps.py
COPY source/geniesim_cli/pyproject.toml /tmp/source/geniesim_cli/pyproject.toml
COPY source/geniesim/pyproject.toml /tmp/source/geniesim/pyproject.toml
COPY source/geniesim_benchmark/pyproject.toml /tmp/source/geniesim_benchmark/pyproject.toml
COPY source/geniesim_ros/pyproject.toml /tmp/source/geniesim_ros/pyproject.toml
COPY source/geniesim_teleop/pyproject.toml /tmp/source/geniesim_teleop/pyproject.toml
ARG WUJI_SDK_VERSION=2026.8.17
COPY docker/collect_runtime_constraints.py /opt/geniesim/docker/collect_runtime_constraints.py
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 /opt/geniesim/docker/bundled_isaacsim.py --torch-providers --output /tmp/torch-providers && \
    python3 /opt/geniesim/docker/collect_runtime_constraints.py \
        --constraints /opt/geniesim/docker/isaac-constraints.txt --repairs /tmp/runtime-repairs.txt && \
    python3 /tmp/collect_deps.py /tmp/source --include-peer geniesim_teleop > /tmp/geniesim_deps.txt && \
    python3 -m pip install --ignore-installed -r /tmp/geniesim_deps.txt -r /tmp/runtime-repairs.txt \
        /tmp/torch-providers/*.whl \
        "wuji-sdk==${WUJI_SDK_VERSION}" \
        --extra-index-url "${ISAACSIM_PYPI_INDEX}" \
        -c /opt/geniesim/docker/isaac-constraints.txt && \
    rm -rf /tmp/collect_deps.py /tmp/source /tmp/geniesim_deps.txt /tmp/runtime-repairs.txt /tmp/torch-providers && \
    python3 /opt/geniesim/docker/bundled_isaacsim.py --clean-bytecode /usr/local/lib/python3.12/dist-packages

# ------------------------------------------------------------
# R14 guard: verify Newton's optional sim requirements after the constrained
# dependency install. The adapter installs Newton's metadata in site-packages;
# this fails on mismatch and no-ops where Newton is not pip-visible.
# See docker/check_newton_pins.py.
# ------------------------------------------------------------
COPY docker/check_newton_pins.py /tmp/check_newton_pins.py
RUN python3 /tmp/check_newton_pins.py && rm -f /tmp/check_newton_pins.py

# Preinstall the selected CUDA 12.8 native artifact without a build GPU.
# The bundled SDK supplies its complete matching CUDA runtime provider.
ARG LIBUIPC_SM
COPY docker/install_libuipc.py docker/libuipc-artifacts.json /opt/geniesim/docker/
COPY 3rdparty/pyuipc-LICENSE 3rdparty/pyuipc-README.md /opt/geniesim/3rdparty/
ARG LIBUIPC_SOURCE=modelscope
ARG LIBUIPC_REVISION
ENV GENIESIM_LIBUIPC_CACHE=/opt/geniesim/3rdparty
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=from=libuipc_assets,target=/libuipc-assets/wheels,ro \
    GENIESIM_ASSETS_SRC=/libuipc-assets \
    GENIESIM_LIBUIPC_SOURCE="${LIBUIPC_SOURCE}" \
    GENIESIM_LIBUIPC_REVISION="${LIBUIPC_REVISION}" \
    python3 /opt/geniesim/docker/install_libuipc.py --build-sm "${LIBUIPC_SM}"

# Place the entrypoint at a stable absolute path so the image is
# self-sufficient (no requirement to bind-mount the repo just to find
# the entrypoint).
COPY docker/entrypoint.sh /usr/local/bin/geniesim-entrypoint
RUN chmod +x /usr/local/bin/geniesim-entrypoint

# ------------------------------------------------------------
# rosdep install for geniesim_ros packages
# ------------------------------------------------------------
COPY source/geniesim_ros/src/ros_ws/src /tmp/geniesim_ros_src
RUN apt-get update && \
    rosdep install --from-paths /tmp/geniesim_ros_src --ignore-src -y --rosdistro jazzy && \
    rm -rf /tmp/geniesim_ros_src /var/lib/apt/lists/*

# OVRTX registers this search path at renderer startup. The installed parent
# is root-owned, so create the directory before dropping to the remapped UID.
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/ovrtx/bin/mdl/omniverse_exts && \
    python3 -m pip check && \
    python3 /opt/geniesim/docker/bundled_isaacsim.py --clean-bytecode /usr/local/lib/python3.12/dist-packages

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/geniesim-entrypoint"]
CMD ["bash"]
