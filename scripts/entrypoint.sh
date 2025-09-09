#!/bin/bash

set -e

# env
export ROS_DISTRO=humble

# bashrc
echo "alias omni_python='/isaac-sim/python.sh'" >>~/.bashrc
echo "alias run_server='omni_python source/geniesim/app/raise_standalone_sim.py'" >>~/.bashrc
echo "alias run_client='omni_python source/geniesim/benchmark/task_benchmark.py --task_name'" >>~/.bashrc
echo "alias run_teleop='omni_python source/geniesim/teleop/teleop.py --task_name'" >>~/.bashrc
echo "alias run_replay='omni_python source/geniesim/teleop/replay_state.py'" >>~/.bashrc

echo "source /opt/ros/$ROS_DISTRO/setup.bash" >>~/.bashrc

echo "export SIM_ASSETS=/root/assets" >>~/.bashrc
echo "export SIM_REPO_ROOT=/root/workspace/main" >>~/.bashrc

echo "export ROS_DISTRO=humble" >>~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >>~/.bashrc

# you can add more customized cmds here

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --
# exec "$@"
cp -r /root/workspace/main/source/geniesim/app/robot_cfg/* /isaac-sim/kit/python/lib/python3.10/site-packages/curobo/content/assets/robot
cp -r /root/workspace/main/source/geniesim/app/curobo/configs/robot/ /isaac-sim/kit/python/lib/python3.10/site-packages/curobo/content/configs/
#
/bin/bash
