# Public collection/replay dependencies; build context is this module directory.
# Pass --build-context geniesim_mirrors=../../docker, or use the host CLI.
ARG GENIESIM_BASE_IMAGE=registry.agibot.com/genie-sim/geniesim4:latest
FROM ${GENIESIM_BASE_IMAGE} AS mirrors
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG APT_MIRROR
ARG PIP_INDEX_URL
ENV PIP_CONFIG_FILE=/etc/pip.conf
COPY --from=geniesim_mirrors configure_mirrors.py /usr/local/bin/geniesim-configure-mirrors
RUN chmod +x /usr/local/bin/geniesim-configure-mirrors \
    && /isaac-sim/python.sh /usr/local/bin/geniesim-configure-mirrors --apt "${APT_MIRROR}" --pip "${PIP_INDEX_URL}"

FROM mirrors AS dependencies
ARG TORCH_CUDA_ARCH_LIST=8.9
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# The main image owns Isaac Sim, system build tools and ROS Jazzy. Fail on
# an incompatible parent rather than silently rebuilding a different stack.
RUN test -x /isaac-sim/python.sh \
    && grep -q '^6\.2\.' /isaac-sim/VERSION \
    && test -f /opt/ros/jazzy/setup.bash \
    && test "$(dpkg-query -W -f='${db:Status-Status}' ros-jazzy-ros-base)" = installed

# Only request missing collection tools; leave already-installed packages
# out of the explicit apt install list.
RUN missing=() \
    && for package in ninja-build ros-jazzy-rosbag2 ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-image-transport ros-jazzy-image-transport-plugins; do \
        if ! dpkg-query -W -f='${db:Status-Status}\n' "$package" 2>/dev/null | grep -qx installed; then \
            missing+=("$package"); \
        fi; \
    done \
    && if ((${#missing[@]})); then \
        apt-get update && apt-get install -y --no-install-recommends "${missing[@]}"; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Recording subprocesses clear the Isaac environment before sourcing the
# inherited system ROS setup; never source it into Isaac Python.

# cuRobo's NVRTC backend uses the CUDA libraries bundled with this SDK.
# Do not install a second compiler/runtime stack or replace the bundled Torch.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    ROS_DISTRO=jazzy ROS_CMD_DISTRO=jazzy ROS_VERSION=2 ROS_PYTHON_VERSION=3 \
    ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ISAACSIM_HOME=/isaac-sim SIM_REPO_ROOT=/workspace

COPY docker/prepare_bundled_sdk.py /opt/prepare_bundled_sdk.py
RUN /isaac-sim/python.sh /opt/prepare_bundled_sdk.py --constraints /opt/isaac-sdk/constraints.txt
ENV LD_LIBRARY_PATH=/opt/isaac-sdk/usd/bin:/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:/opt/ros/jazzy/lib:/opt/isaac-sdk/nvidia/nvidia/cuda_runtime/lib:/opt/isaac-sdk/nvidia/nvidia/cuda_nvrtc/lib

COPY requirements.txt /tmp/requirements.collection.txt
COPY replay/requirements.txt /tmp/requirements.replay.txt
RUN /isaac-sim/python.sh -m pip install \
        -c /opt/isaac-sdk/constraints.txt \
        -r /tmp/requirements.collection.txt -r /tmp/requirements.replay.txt \
        "cv-bridge==1.13.0.post0" "cuda-core[cu12]>=0.7"

# This pinned pure-Python bridge predates NumPy 2. Keep its ROS wire format,
# replacing only removed ndarray methods; exercise the actual SDK messages.
RUN /isaac-sim/python.sh - <<'PYTHON'
from pathlib import Path
import cv_bridge
import numpy as np

path = Path(cv_bridge.__file__).with_name("core.py")
source = path.read_text()
assert source.count(".tostring()") == 2, "Unexpected pinned cv_bridge source"
assert source.count("im.byteswap().newbyteorder()") == 1
source = source.replace(".tostring()", ".tobytes()")
source = source.replace("im.byteswap().newbyteorder()", "im.byteswap().view(im.dtype.newbyteorder())")
path.write_text(source)
PYTHON
RUN /isaac-sim/python.sh - <<'PYTHON'
import numpy as np
from cv_bridge import CvBridge

bridge = CvBridge()
for encoding, image in (
    ("rgb8", np.arange(36, dtype=np.uint8).reshape(3, 4, 3)),
    ("32FC1", np.arange(12, dtype=np.float32).reshape(3, 4)),
    ("32FC1", np.arange(12, dtype=">f4").reshape(3, 4)),
):
    message = bridge.cv2_to_imgmsg(image, encoding=encoding)
    restored = bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
    np.testing.assert_array_equal(restored, image)
    assert message.step == image.shape[1] * image.dtype.itemsize * (3 if encoding == "rgb8" else 1)
print("cv_bridge RGB/depth and endian roundtrips passed")
PYTHON

ARG CUROBO_REPO=https://github.com/NVlabs/curobo.git
ARG CUROBO_COMMIT=3fd54dc782a82e5500a771cfd47856ea499d5fef
RUN git init /opt/curobo_src \
    && git -C /opt/curobo_src remote add origin "${CUROBO_REPO}" \
    && git -C /opt/curobo_src sparse-checkout init --cone \
    && git -C /opt/curobo_src sparse-checkout set curobo \
    && git -C /opt/curobo_src fetch --filter=blob:none --depth 1 origin "${CUROBO_COMMIT}" \
    && git -C /opt/curobo_src checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/curobo_src rev-parse HEAD)" = "${CUROBO_COMMIT}"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.8.1
# Non-isolated builds need their build backend before preparing metadata.
# PyJWT satisfies the SDK's MSAL requirement without importing its unrelated
# service archive, whose optional AWS packages use a separate dependency set.
RUN /isaac-sim/python.sh -m pip install -c /opt/isaac-sdk/constraints.txt \
        "setuptools-scm>=6.2" "PyJWT[crypto]==2.13.0"
# Match the bundled Torch cu128 build and retain the SDK dependency versions.
# The default cuRobo backend uses NVRTC; CUROBO_USE_PYBIND remains disabled.
RUN /isaac-sim/python.sh -m pip install --no-build-isolation \
        -c /opt/isaac-sdk/constraints.txt "/opt/curobo_src[cu12-torch]" \
    && rm -rf /opt/curobo_src/.git /opt/curobo_src/docs \
        /opt/curobo_src/benchmark /opt/curobo_src/examples /opt/curobo_src/tests

# A successful import does not establish NVRTC kernel compilation readiness.
RUN /isaac-sim/python.sh - <<'PYTHON'
from pathlib import Path
from cuda import pathfinder

for library, filename in (("cudart", "cuda_runtime.h"), ("nvrtc", "nvrtc.h")):
    include = pathfinder.find_nvidia_header_directory(library)
    assert include and (Path(include) / filename).is_file(), f"Missing {library} header provider"
print("SDK CUDA runtime and NVRTC headers are discoverable")
PYTHON

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

# Application imports and system ROS tooling must both work at build time.
RUN /isaac-sim/python.sh -m pip install -c /opt/isaac-sdk/constraints.txt "coverage==7.6.1"
RUN /isaac-sim/python.sh -c \
    'import sys, torch, numpy, scipy, cv2, h5py, grpc, pinocchio, ruckig, warp, curobo, cv_bridge, rclpy, numba; from importlib.metadata import version; from pxr import Usd; from sensor_msgs.msg import Image; from tf2_msgs.msg import TFMessage; assert sys.version_info[:2] == (3, 12); assert torch.version.cuda == "12.8"; assert version("nvidia-curobo") == curobo.__version__ == "0.8.1"; print("Runtime:", numpy.__version__, torch.__version__, warp.__version__, Usd.GetVersion())' \
    && /isaac-sim/python.sh -m pip check \
    && env -u PYTHONPATH -u LD_LIBRARY_PATH bash -c \
        'source /opt/ros/jazzy/setup.bash && ros2 bag record --help >/dev/null && ros2 pkg executables image_transport | grep -q republish'

FROM dependencies AS runtime
WORKDIR /workspace
COPY --chown=1234:1234 . /workspace

# Resolve the installed planner asset root instead of hardcoding Python's ABI.
RUN /isaac-sim/python.sh - <<'PYTHON'
from pathlib import Path
import shutil
import yaml
from curobo.content import get_assets_path

assets = Path(get_assets_path())
source = Path("/workspace/config/robot_cfg")
urdfs = {
    "G2/G2_crs_90D.urdf": "G2/G2_crs_90D.urdf",
    "G2/G2_crsB_90D_from_usd.urdf": "G2/G2_crsB_90D_from_usd.urdf",
    "G2/G2_crsB_omnipicker_from_usd.urdf": "G2/G2_crsB_omnipicker_from_usd.urdf",
    "G2/G2_omnipicker_from_usd.urdf": "G2/G2_omnipicker_from_usd.urdf",
    "aloha/robot.urdf": "Aloha/Aloha.urdf",
    "arx/robot.urdf": "ARX/ARX.urdf",
    "G1/G1_omnipicker.urdf": "G1/G1_omnipicker.urdf",
}
for destination, relative_source in urdfs.items():
    target = assets / "robot" / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative_source, target)
configs = sorted(Path("/workspace/config/curobo/v2/configs/robot").glob("*.yml"))
assert configs, "No cuRobo v2 configuration was copied"
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
