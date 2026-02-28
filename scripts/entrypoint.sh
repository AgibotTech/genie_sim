#!/bin/bash

set -e

# env
export ROS_DISTRO=jazzy
export ISAACSIM_HOME=/isaac-sim

# user 1234 access
sudo setfacl -m u:1234:rwX /geniesim/main
sudo setfacl -m u:1234:rwX /geniesim/main/source
sudo setfacl -m u:1234:rwX /geniesim/main/source/geniesim/benchmark/saved_task
sudo setfacl -m u:1234:rwX /geniesim/main/source/teleop
sudo setfacl -m u:1234:rwX /geniesim/main/source/teleop/app/bin
sudo setfacl -m u:1234:rwX /geniesim/main/source/teleop/app/share
sudo mkdir -p /geniesim/main/source/teleop/app/bin/.cache
sudo mkdir -p /geniesim/main/source/teleop/app/bin/logs
sudo mkdir -p /geniesim/main/source/teleop/app/bin/logs/dylog
sudo chown -R 1234:1234 /geniesim/main/source/teleop/app/bin/.cache
sudo chown -R 1234:1234 /geniesim/main/source/teleop/app/bin/logs
sudo chown -R 1234:1234 /geniesim/main/source/teleop/app/share
sudo setfacl -m u:1234:rwX /geniesim/main/source/teleop/app/bin/.cache
sudo setfacl -m u:1234:rwX /geniesim/main/source/teleop/app/bin/logs
sudo setfacl -R -m u:1234:rwX /geniesim/main/source/teleop/app/share
sudo setfacl -m u:1234:rwX /isaac-sim/.cache
sudo setfacl -m u:1234:rwX /isaac-sim/.nv/ComputeCache
sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/logs
sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/config
sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/data
sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/pkg

# bashrc
echo "export SIM_REPO_ROOT=/geniesim/main" >>~/.bashrc
echo "export ENABLE_SIM=1" >>~/.bashrc
echo "export ROS_DISTRO=${ROS_DISTRO}" >>~/.bashrc
echo "export ROS_VERSION=2" >>~/.bashrc
echo "export ROS_PYTHON_VERSION=3" >>~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >>~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >>~/.bashrc
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ISAACSIM_HOME}/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib" >>~/.bashrc
echo "export ROS_CMD_DISTRO=jazzy" >>~/.bashrc
echo "source ${ISAACSIM_HOME}/setup_ros_env.sh" >>~/.bashrc

echo "alias omni_python='${ISAACSIM_HOME}/python.sh'" >>~/.bashrc
echo "alias isaacsim='${ISAACSIM_HOME}/runapp.sh'" >>~/.bashrc
echo "alias geniesim='omni_python /geniesim/main/source/geniesim/app/app.py'" >>~/.bashrc

sudo rm -rf /geniesim/main/source/GenieSim.egg-info
/isaac-sim/python.sh -m pip install /geniesim/main/3rdparty/ik_solver-0.4.3-cp311-cp311-linux_x86_64.whl
/isaac-sim/python.sh -m pip install -e /geniesim/main/source

exec "$@"
