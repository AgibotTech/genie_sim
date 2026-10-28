# GenieSim Docker image — Isaac Sim 6.X + ROS 2 Jazzy + system Python 3.12.
#
# Build:
#   geniesim docker6.2 build
#   geniesim docker6.2 build --china      # use DaoCloud mirrors for nvcr/apt/pip
#
# Or manually:
#   docker build -f docker/Dockerfile -t registry.agibot.com/genie-sim/geniesim4:latest .
#   docker build -f docker/Dockerfile \
#       --build-arg ISAACSIM_REGISTRY=nvcr.m.daocloud.io/nvidia \
#       --build-arg ISAACSIM_TAG=6.2.0.1 \
#       --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu \
#       --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#       -t registry.agibot.com/genie-sim/geniesim4:latest .
#
# Base: NVIDIA's published Isaac Sim 6.X image (Ubuntu 24.04).
# The image already ships Isaac Sim under /isaac-sim with
# /isaac-sim/python.sh wrapping a bundled CPython, plus the omniverse
# cache dirs owned by the canned "isaac-sim" user (uid:gid = 1234:1234).
#
# This image installs:
#   * ROS 2 Jazzy desktop on top of Ubuntu 24.04 (Noble) — apt-managed.
#   * Isaac Sim 6.X PyPI bundle into the system Python 3.12 (NOT a venv,
#     per project policy: jazzy's rclpy lives in system python3.12, and
#     mujoco_ros_node.py uses sys.executable, so both must coexist in the
#     same interpreter).
#
# Host-user remap (1234 vs $(id -u)) is handled at *runtime* by
# entrypoint.sh — see the comments there. Build-time UID stays at 1234.

ARG ISAACSIM_REGISTRY=nvcr.io/nvidia
# Isaac Sim 6.2 is the Genie Sim 4.0 target. Keep the requested base
# respin paired with the published wheel; a future major needs its own variant.
#
# Three strings, one release — they MUST be bumped together:
#   ISAACSIM_TAG         6.2.0.1   NGC base image respin
#   ISAACSIM_PIP_VERSION 6.2.0.0   PyPI wheel version (4 components)
#   mujoco~=3.12.0                 source/geniesim_ros/pyproject.toml
# Leaving the wheel unpinned silently tracks whatever pypi.nvidia.com
# serves, which is how this image used to end up with wheels and base
# image naming different releases. Pin both or they disagree.
#
# The third string is outside this directory and is the one most easily
# forgotten. Isaac owns the Newton that consumes MuJoCo (6.2 ships Newton
# 1.6.1 with MuJoCo 3.12.0), and the dep layer below installs our pin
# under Isaac/colcon constraints, with a Newton metadata guard so a stale
# pin fails the build. See docker/check_newton_pins.py.
ARG ISAACSIM_TAG=6.2.0.1

FROM ${ISAACSIM_REGISTRY}/isaac-sim:${ISAACSIM_TAG}

# Build-arg knobs for behind-the-GFW builds. Defaults are upstream;
# `geniesim docker build --china` rewires all three to mainland mirrors.
ARG APT_MIRROR=""
ARG PIP_INDEX_URL=""
ARG ISAACSIM_PYPI_INDEX=https://pypi.nvidia.com
ARG ISAACSIM_PIP_VERSION=6.2.0.0

ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

USER root

# omni_python wrapper — exec the real script so $0 stays /isaac-sim/python.sh.
# Also open /isaac-sim to world-traverse (NVIDIA base ships it drwxr-x---).
RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/omni_python && \
    chmod +x /usr/local/bin/omni_python && \
    chmod o+rx /isaac-sim

# ------------------------------------------------------------
# Optional apt mirror rewrite (used when --china is passed).
# ------------------------------------------------------------
RUN if [ -n "${APT_MIRROR}" ]; then \
        echo "[build] rewriting apt sources -> ${APT_MIRROR}"; \
        sed -i -E "s|https?://[^ ]*archive.ubuntu.com/ubuntu|${APT_MIRROR}|g; \
                   s|https?://[^ ]*security.ubuntu.com/ubuntu|${APT_MIRROR}|g; \
                   s|https?://[^ ]*ports.ubuntu.com/ubuntu-ports|${APT_MIRROR}|g" \
            /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; \
        sed -i -E "s|https?://[^ ]*archive.ubuntu.com/ubuntu|${APT_MIRROR}|g; \
                   s|https?://[^ ]*security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
            /etc/apt/sources.list 2>/dev/null || true; \
    fi

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
ARG APT_MIRROR
RUN add-apt-repository universe -y && \
    if [ -n "${APT_MIRROR}" ]; then \
        sed -i -E "s|https?://[^ ]*archive.ubuntu.com/ubuntu|${APT_MIRROR}|g; \
                   s|https?://[^ ]*security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
            /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list 2>/dev/null || true; \
    fi && \
    curl -fsSL -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-desktop \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-vision-msgs-rviz-plugins \
        ros-jazzy-joy \
        ros-dev-tools && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update --rosdistro jazzy --include-eol-distros 2>&1 | grep -v "as root is not recommended"

# ------------------------------------------------------------
# Isaac Sim 6.X — install into system python3.12.
#
# Ubuntu 24.04 enforces PEP 668 ("externally-managed environment");
# PIP_BREAK_SYSTEM_PACKAGES=1 (set above) lifts that for the whole
# image. PIP_INDEX_URL (build-arg) optionally redirects the primary
# index to a mainland mirror; the NVIDIA index is always supplied as
# --extra-index-url because the isaacsim wheels live only there.
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ -n "${PIP_INDEX_URL}" ]; then \
        python3 -m pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi && \
    python3 -m pip install --upgrade --ignore-installed pip && \
    python3 -m pip install --upgrade --ignore-installed \
        "isaacsim[all,extscache]==${ISAACSIM_PIP_VERSION}" \
        --extra-index-url "${ISAACSIM_PYPI_INDEX}"

# Pre-accept EULA for Isaac Sim 6.X (OMNI_KIT_ACCEPT_EULA path)
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/isaacsim/kit && \
    touch /usr/local/lib/python3.12/dist-packages/isaacsim/kit/EULA_ACCEPTED && \
    chmod o+w /usr/local/lib/python3.12/dist-packages/isaacsim/kit

# # isaacsim 6.x's pip prebundle has an old typing_extensions without Sentinel.
# RUN curl -sSL "https://raw.githubusercontent.com/python/typing_extensions/main/src/typing_extensions.py" \
#     -o /usr/local/lib/python3.12/dist-packages/isaacsim/extscache/omni.services.pip_archive-0.18.6+109.0.4.lx64.cp312/pip_prebundle/typing_extensions.py

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
# Tier-2 peers (`geniesim_teleop`, `geniesim_scene_agent`, `geniesim_world`)
# are opt-in via the umbrella's `[teleop|scene_agent|world|all]` extras and
# not installed by default — only their pyproject.tomls would need to be
# COPY'd here if you want them baked in.
# ------------------------------------------------------------
COPY docker/collect_deps.py /tmp/collect_deps.py
COPY source/geniesim_cli/pyproject.toml /tmp/source/geniesim_cli/pyproject.toml
COPY source/geniesim/pyproject.toml /tmp/source/geniesim/pyproject.toml
COPY source/geniesim_benchmark/pyproject.toml /tmp/source/geniesim_benchmark/pyproject.toml
COPY source/geniesim_ros/pyproject.toml /tmp/source/geniesim_ros/pyproject.toml
COPY docker/collect_runtime_constraints.py /opt/geniesim/docker/collect_runtime_constraints.py
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 /opt/geniesim/docker/collect_runtime_constraints.py \
        --constraints /opt/geniesim/docker/isaac-constraints.txt --repairs /tmp/runtime-repairs.txt && \
    python3 /tmp/collect_deps.py /tmp/source > /tmp/geniesim_deps.txt && \
    python3 -m pip install -r /tmp/geniesim_deps.txt -r /tmp/runtime-repairs.txt \
        -c /opt/geniesim/docker/isaac-constraints.txt && \
    rm -rf /tmp/collect_deps.py /tmp/source /tmp/geniesim_deps.txt /tmp/runtime-repairs.txt

# ------------------------------------------------------------
# R14 guard: verify Newton's optional sim requirements after the constrained
# dependency install. Isaac Sim 6.X puts Newton's metadata in site-packages;
# this fails on mismatch and no-ops where Newton is not pip-visible.
# See docker/check_newton_pins.py.
# ------------------------------------------------------------
COPY docker/check_newton_pins.py /tmp/check_newton_pins.py
RUN python3 /tmp/check_newton_pins.py && rm -f /tmp/check_newton_pins.py

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

RUN python3 -m pip check

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/geniesim-entrypoint"]
CMD ["bash"]
