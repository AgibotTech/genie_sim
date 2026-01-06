#!/bin/bash

set -e

# env
export ROS_DISTRO=jazzy
export ISAACSIM_HOME=/isaac-sim
export CUROBO_PATH=/tmp/curobo
export SIM_REPO_ROOT=/geniesim/main/data_collection

# user 1234 access
sudo setfacl -m u:1234:rwX /isaac-sim/.cache
sudo setfacl -m u:1234:rwX /isaac-sim/.nv/ComputeCache
sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/logs
sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/config
sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/data
sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/pkg

# bashrc
echo "export SIM_REPO_ROOT=/geniesim/main/data_collection" >>~/.bashrc
echo "export SIM_ASSETS=/geniesim/main/source/geniesim/assets" >>~/.bashrc
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
echo "alias source_ros_py311='unset LD_LIBRARY_PATH && source /opt/ros/jazzy/setup.bash'" >>~/.bashrc

sudo cp -r ${SIM_REPO_ROOT}/config/curobo/assets/robot $ISAACSIM_HOME/kit/python/lib/python3.11/site-packages/curobo/content/assets/
sudo cp -r ${SIM_REPO_ROOT}/config/curobo/configs $ISAACSIM_HOME/kit/python/lib/python3.11/site-packages/curobo/content/
exec "$@"
