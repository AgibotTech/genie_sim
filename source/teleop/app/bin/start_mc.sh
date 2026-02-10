#!/bin/bash

# Check if ROS environment is already sourced
setup_env() {

  SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
  SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
  INSTALL_DIR=$(dirname "${SCRIPT_DIR}")

  export ROS_LOCALHOST_ONLY=1
  local DEFAULT_ROS_DISTRO="humble"
  # Check if ROS_DISTRO environment variable exists
  if [ -z "${ROS_DISTRO}" ]; then
    # ROS environment not sourced, check if setup.bash exists
    if [ -f "/opt/ros/${DEFAULT_ROS_DISTRO}/setup.bash" ]; then
      echo "Sourcing ROS environment..."
      source /opt/ros/${DEFAULT_ROS_DISTRO}/setup.bash
      return 0
    else
      echo "Warning: /opt/ros/${DEFAULT_ROS_DISTRO}/setup.bash not found"
      return 1
    fi
  else
    echo "ROS environment already sourced (ROS_DISTRO=$ROS_DISTRO)"
    return 0
  fi
}

check_running() {
  # check if the binary is already running
  local binary_name=${1:-$SCRIPT_NAME}
  local other_pid=($(pgrep -f "/${binary_name}( |$)" | grep -v "$$" | head -n 1))
  if [[ -n "$other_pid" ]]; then
    # prompt user to stop the binary
    read -p "Another instance of '$binary_name' is already running [PID: $other_pid]. Do you want to stop it? (y/n): " answer
    if [[ "$answer" == "y" ]]; then
      kill -9 "$other_pid"
    else
      exit 1
    fi
  fi
}

launch_binary() {
  local binary_name=$1
  shift # Remove the first argument, leaving the rest for getopts
  local binary_path="${SCRIPT_DIR}/${binary_name}"

  if [ ! -f "$binary_path" ]; then
    echo "Binary file not found: $binary_path"
    exit 1
  fi

  launch_args=()

  load_options() {
    local robot_name=""
    local default_state=""
    local configuration_directory=""

    mkdir -p "${SCRIPT_DIR}/.cache"
    local robot_name_cache_file="${SCRIPT_DIR}/.cache/robot_name"
    local configuration_directory_cache_file="${SCRIPT_DIR}/.cache/mc_configuration_directory"

    # Detect system architecture and set default sim_mode
    local sim_mode=false
    system_arch=$(uname -m)
    if [ "$system_arch" = "x86_64" ]; then
      sim_mode=true
    elif [ "$system_arch" = "aarch64" ]; then
      sim_mode=true
    fi

    # Parse cache-related command line arguments
    while [[ $# -gt 0 ]]; do
      case "$1" in
      -C)
        if [ -z "$2" ]; then
          echo "Usage: $SCRIPT_NAME -C <configuration_directory>"
          exit 1
        fi
        configuration_directory="$2"
        shift 2
        ;;
      -s)
        rm -f "$robot_name_cache_file"
        rm -f "$configuration_directory_cache_file"
        shift
        ;;
      --sim)
        sim_mode=true
        shift
        ;;
      --state)
        default_state="$2"
        shift 2
        ;;
      --robot=*)
        robot_name="${1#*=}"
        shift
        ;;
      *)
        launch_args+=("$1")
        shift
        ;;
      esac
    done

    if [ -f "$robot_name_cache_file" ]; then
      robot_name=$(cat "$robot_name_cache_file")
    fi

    if [ -f "$configuration_directory_cache_file" ]; then
      configuration_directory=$(cat "$configuration_directory_cache_file")
    fi

    if [ -z "$configuration_directory" ]; then
      configuration_directory="${INSTALL_DIR}/configuration"
      echo "No configuration directory specified. Using default: $configuration_directory"
    fi

    echo "$configuration_directory" >"$configuration_directory_cache_file"

    local robot_directory="${configuration_directory}/robot"

    if [ -z "$robot_name" ]; then
      local directories=($(find "$robot_directory" -maxdepth 1 -mindepth 1 -type d | sort))
      local length=${#directories[@]}

      if [ $length -eq 0 ]; then
        echo "No robot directories found in $robot_directory. Exiting."
        exit 1
      elif [ $length -eq 1 ]; then
        local selected_robot_directory="${directories[0]}"
        robot_name=$(basename "$selected_robot_directory")
        echo "Selected robot: $robot_name"
      else
        echo "Please select a robot: "
        for i in "${!directories[@]}"; do
          local dir_name=$(basename "${directories[$i]}")
          echo -e "\e[1;31m$i\e[0m: \e[1;32m$dir_name\e[0m"
        done

        read -p "Please enter your choice: " choice
        if [[ "$choice" =~ ^[0-9]+$ && "$choice" -ge 0 && "$choice" -lt "${#directories[@]}" ]]; then
          local selected_robot_directory="${directories[$choice]}"
          robot_name=$(basename "$selected_robot_directory")
        else
          echo "Invalid choice. Exiting."
          exit 1
        fi

        echo "Selected robot: $robot_name"
      fi

      # Save robot name to cache
      echo "$robot_name" >"$robot_name_cache_file"
    fi

    # Return values through global variables
    ROBOT_NAME="$robot_name"
    DEFAULT_STATE="$default_state"
    CONFIGURATION_DIRECTORY="$configuration_directory"
    SIM_MODE="$sim_mode"
  }

  start_binary() {
    # Parse launch-related command line arguments (no debug helpers)
    local bypass_args=()

    while [[ $# -gt 0 ]]; do
      bypass_args+=("$1")
      shift
    done

    # Run binary
    bypass_args+=("--robot=$ROBOT_NAME")
    if [ -n "$DEFAULT_STATE" ]; then
      bypass_args+=("--state=$DEFAULT_STATE")
    fi
    bypass_args+=("--configuration=$CONFIGURATION_DIRECTORY")


    # set environment variables
    source "${INSTALL_DIR}/share/geniesim_msg/local_setup.bash"
    export DYLOG_log_dir="${INSTALL_DIR}/bin/logs/dylog"
    export LD_LIBRARY_PATH=${INSTALL_DIR}/bin:${INSTALL_DIR}/lib:${INSTALL_DIR}/vendors/lib:${INSTALL_DIR}/conan/lib:$LD_LIBRARY_PATH
    mkdir -p ${DYLOG_log_dir}

    # Launch binary directly (no gdb / valgrind wrapper)
    "$binary_path" "${bypass_args[@]}"
    echo -e "\n\x1b[32m$binary_name exited, you can launch it again by running:\x1b[0m"
    echo "source ${INSTALL_DIR}/share/geniesim_msg/local_setup.bash"
    echo "export LD_LIBRARY_PATH=${INSTALL_DIR}/bin:${INSTALL_DIR}/lib:${INSTALL_DIR}/vendors/lib:${INSTALL_DIR}/conan/lib:\$LD_LIBRARY_PATH"
    echo "$binary_path ${bypass_args[@]}"
  }

  # Load options and select robot
  load_options "$@"
  start_binary "${launch_args[@]}"
}

setup_env
check_running genie_motion_control
launch_binary genie_motion_control "$@"
