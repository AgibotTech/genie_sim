# Public collection/replay dependencies; build context is this module directory.
# Pass --build-context geniesim_mirrors=../../docker, or use the host CLI.
ARG ISAACSIM_REGISTRY=nvcr.io/nvidia
ARG ISAAC_SIM_BASE_IMAGE=${ISAACSIM_REGISTRY}/isaac-sim:5.1.0
FROM ${ISAAC_SIM_BASE_IMAGE} AS mirrors
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG APT_MIRROR
ARG PIP_INDEX_URL
ENV PIP_CONFIG_FILE=/etc/pip.conf
COPY --from=geniesim_mirrors configure_mirrors.py /usr/local/bin/geniesim-configure-mirrors
RUN chmod +x /usr/local/bin/geniesim-configure-mirrors \
    && /isaac-sim/python.sh /usr/local/bin/geniesim-configure-mirrors --apt "${APT_MIRROR}" --pip "${PIP_INDEX_URL}"

FROM mirrors AS runtime
ARG TORCH_CUDA_ARCH_LIST=8.9
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        acl ca-certificates cmake curl ffmpeg git gnupg libeigen3-dev \
        libgl1 libglib2.0-0 ninja-build build-essential sudo \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# System ROS uses Python 3.12. Recording subprocesses clear the Isaac
# environment before sourcing this setup; never source it into Isaac Python.
RUN curl -fsSL --retry 3 -o /tmp/ros2-apt-source.deb \
        https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base ros-jazzy-rosbag2 ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-image-transport ros-jazzy-image-transport-plugins \
    && rm -rf /var/lib/apt/lists/* /tmp/ros2-apt-source.deb

# Compiler/headers for cuRobo/Warp. Keep the bundled Torch and Isaac Sim.
RUN curl -fsSL --retry 3 -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 cuda-cccl-12-8 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

ENV CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    ROS_DISTRO=jazzy ROS_CMD_DISTRO=jazzy ROS_VERSION=2 ROS_PYTHON_VERSION=3 \
    ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ISAACSIM_HOME=/isaac-sim SIM_REPO_ROOT=/workspace \
    PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy \
    LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib:/usr/local/cuda-12.8/lib64

COPY requirements.txt /tmp/requirements.collection.txt
COPY replay/requirements.txt /tmp/requirements.replay.txt
# Preserve the SDK's declared setuptools-scm, USD, and lxml compatibility ranges.
RUN printf '%s\n' 'setuptools-scm>=8.0.0,<9.0.0' 'usd-core>=25.2.post1,<26.0' \
        'lxml>=4.9.2,<5.0.0' > /tmp/constraints.sdk.txt
RUN /isaac-sim/python.sh -m pip install --upgrade pip \
    && /isaac-sim/python.sh -m pip install \
        -r /tmp/requirements.collection.txt -r /tmp/requirements.replay.txt \
        -c /tmp/constraints.sdk.txt setuptools wheel "setuptools-scm>=8,<9" \
        "docstring-parser==0.16" "cv-bridge==1.13.0.post0" "cuda-core[cu12]>=0.7"

ARG CUROBO_REPO=https://github.com/NVlabs/curobo.git
ARG CUROBO_COMMIT=3fd54dc782a82e5500a771cfd47856ea499d5fef
RUN git init /opt/curobo_src \
    && git -C /opt/curobo_src remote add origin "${CUROBO_REPO}" \
    && git -C /opt/curobo_src fetch --depth 1 origin "${CUROBO_COMMIT}" \
    && git -C /opt/curobo_src checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/curobo_src rev-parse HEAD)" = "${CUROBO_COMMIT}"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.8.1
# The pinned revision declares cu12 extras, not an isaacsim extra.
# Reapply repository constraints to preserve NumPy and Warp versions.
RUN /isaac-sim/python.sh -m pip install --no-build-isolation \
        -c /tmp/requirements.collection.txt -c /tmp/requirements.replay.txt \
        -c /tmp/constraints.sdk.txt "/opt/curobo_src[cu12]" \
    && rm -rf /opt/curobo_src/.git /opt/curobo_src/docs \
        /opt/curobo_src/benchmark /opt/curobo_src/examples /opt/curobo_src/tests

# Restore the packaging target referenced by bundled Torch vendor symlinks.
RUN /isaac-sim/python.sh -m pip install --no-deps --upgrade \
        --target /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle \
        packaging

# Official UID 1234 is also used by the repository's host launch scripts.
RUN test "$(id -u isaac-sim)" = 1234 \
    && echo 'isaac-sim ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/isaac-sim \
    && chmod 0440 /etc/sudoers.d/isaac-sim \
    && mkdir -p /workspace /isaac-sim/.cache /isaac-sim/.nv/ComputeCache \
        /isaac-sim/.nvidia-omniverse/logs /isaac-sim/.nvidia-omniverse/config \
        /isaac-sim/.local/share/ov/data /isaac-sim/.local/share/ov/pkg \
    && chown -R 1234:1234 /workspace /isaac-sim/.cache /isaac-sim/.nv \
        /isaac-sim/.nvidia-omniverse /isaac-sim/.local/share/ov \
    && printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/omni_python \
    && chmod +x /usr/local/bin/omni_python

# Install missing dependencies declared by the bundled Torch and MSAL packages.
RUN /isaac-sim/python.sh -m pip install "triton==3.3.0" "PyJWT[crypto]>=1,<3"

# Missing application dependencies and ROS tooling must fail the image build.
RUN /isaac-sim/python.sh -c \
    'import torch, numpy, scipy, cv2, h5py, grpc, pinocchio, ruckig, warp, curobo, cv_bridge, rclpy; from sensor_msgs.msg import Image; from tf2_msgs.msg import TFMessage; assert numpy.__version__ == "1.26.4"; assert torch.__version__.startswith("2.7.")' \
    && env -u PYTHONPATH -u LD_LIBRARY_PATH bash -c \
        'source /opt/ros/jazzy/setup.bash && ros2 bag record --help >/dev/null && ros2 pkg executables image_transport | grep -q republish'


# Install repository-derived URDFs into the cuRobo asset directory.
# v2 urdf_path is relative to the cuRobo asset root, e.g. robot/G2/model.urdf.
# cuRobo resolves it under its installed content/assets directory.
# Copy each required URDF there so YAML loading can find it.
#
# Load v2 YAML from the repository config/curobo/v2 directory.
# The workspace copy contains YAML; no cuRobo-directory YAML copy is required.
# Configurations with load_meshes=false do not load URDF mesh references.
# Their URDF files alone suffice for this asset-loading path.
ENV CUROBO_ASSETS_ROBOT_DIR=/isaac-sim/kit/python/lib/python3.11/site-packages/curobo/content/assets/robot
COPY config/robot_cfg/G2/G2_crs_90D.urdf ${CUROBO_ASSETS_ROBOT_DIR}/G2/G2_crs_90D.urdf
COPY config/robot_cfg/G2/G2_crsB_90D_from_usd.urdf ${CUROBO_ASSETS_ROBOT_DIR}/G2/G2_crsB_90D_from_usd.urdf
COPY config/robot_cfg/G2/G2_crsB_omnipicker_from_usd.urdf ${CUROBO_ASSETS_ROBOT_DIR}/G2/G2_crsB_omnipicker_from_usd.urdf
COPY config/robot_cfg/G2/G2_omnipicker_from_usd.urdf ${CUROBO_ASSETS_ROBOT_DIR}/G2/G2_omnipicker_from_usd.urdf
COPY config/robot_cfg/Aloha/Aloha.urdf ${CUROBO_ASSETS_ROBOT_DIR}/aloha/robot.urdf
COPY config/robot_cfg/ARX/ARX.urdf ${CUROBO_ASSETS_ROBOT_DIR}/arx/robot.urdf
COPY config/robot_cfg/G1/G1_omnipicker.urdf ${CUROBO_ASSETS_ROBOT_DIR}/G1/G1_omnipicker.urdf

# Include the v2 planner configuration explicitly in every image build.
COPY --chown=1234:1234 config/curobo/v2 /workspace/config/curobo/v2

# Copy application code after dependencies to preserve the build cache.
WORKDIR /workspace
COPY --chown=1234:1234 . /workspace

# Fail the build if a planner configuration or its URDF is missing.
RUN /isaac-sim/python.sh - <<'PYTHON'
from pathlib import Path
import yaml

root = Path("/workspace/config/curobo/v2/configs/robot")
configs = sorted(root.glob("*.yml"))
assert configs, "No cuRobo v2 configuration was copied"
assets = Path("/isaac-sim/kit/python/lib/python3.11/site-packages/curobo/content/assets")
for config in configs:
    data = yaml.safe_load(config.read_text())
    urdf = assets / data["kinematics"]["urdf_path"]
    assert urdf.is_file(), f"{config.name}: missing {urdf}"
print(f"Validated {len(configs)} cuRobo v2 configurations and URDFs")
PYTHON

ARG GIT_COMMIT_SHA=unknown
ARG GIT_COMMIT_DATE=unknown
ARG GIT_BRANCH=unknown
RUN printf 'Git Commit SHA: %s\nGit Commit Date: %s\nGit Branch: %s\n' \
        "$GIT_COMMIT_SHA" "$GIT_COMMIT_DATE" "$GIT_BRANCH" \
        > /workspace/git_commit_info.txt

USER 1234:1234
# Clear NVIDIA's headless entrypoint so explicit collection/replay commands run.
ENTRYPOINT []
CMD ["bash"]
