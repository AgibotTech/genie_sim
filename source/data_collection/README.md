# Data Collection

A data collection system for robotic simulation tasks using Isaac Sim and cuRobo.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Option 1: Docker Container (Recommended)](#option-1-docker-container-recommended)
    - [Building the Docker Image](#building-the-docker-image)
    - [One-Click Data Collection (Recommended)](#one-click-data-collection-recommended)
    - [Interactive Mode](#interactive-mode)
  - [Option 2: Local Deployment](#option-2-local-deployment)
- [Usage Examples](#usage-examples)
  - [Docker - Automated Data Collection](#docker-automated-data-collection)
  - [Docker - Interactive Development](#docker-interactive-development)
  - [Local - Full Control](#local-full-control)
- [Task Configuration Guide](#task-configuration-guide)

## Prerequisites

- Docker (for containerized deployment)
- NVIDIA GPU with CUDA support (40 series GPU recommended, for 50 series GPU (SM_120) may not be able to install cuRobo)
- Python 3.11
- Conda (for local deployment)

## Getting Started

**Note:** Before running, make sure to set the `SIM_ASSETS` environment variable:
```bash
export SIM_ASSETS={YOUR_ASSETS_PATH}
```

### Option 1: Docker Container (Recommended)

#### Building the Docker Image

First, build the Docker image:

assuming the image of benchmark `registry.agibot.com/genie-sim/open_source:latest` is built, run

```bash
docker build -f ./dockerfile -t registry.agibot.com/genie-sim/open_source-data-collection:latest .
```

**Note:**  For cuRobo installation, the Dockerfile is configured for RTX 4090D by default. If you're using a different GPU model, you need to modify the `TORCH_CUDA_ARCH_LIST` environment variable in the Dockerfile, 50 series GPU (SM_120) may not be able to install cuRobo, this needs a compatibility update by the cuRobo team.

#### One-Click Data Collection (Recommended)

Use `scripts/run_data_collection.sh` to start data collection in one command.

**Usage:**

```bash
./scripts/run_data_collection.sh [OPTIONS]
```

**Options:**
- `--headless` - Run in headless mode (default: false)
- `--no-record` - Disable recording (default: record enabled)
- `--task TASK_PATH` - Task template path (e.g. `tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json`)
- `--standalone` - Run in standalone mode (only save logs, no terminal output) (default: false)
- `--container-name NAME` - Container name (default: `data_collection_open_source`)
- `--help, -h` - Show help message

**Environment Variables:**
- `SIM_ASSETS` - Path to Isaac Sim assets (required)

**Examples:**

```bash
# Run with default task in GUI mode
./scripts/run_data_collection.sh

# Run in headless mode with custom task
./scripts/run_data_collection.sh --headless --task tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json

# Run in standalone headless mode (logs only, no terminal output)
./scripts/run_data_collection.sh --standalone --headless

# Run without recording
./scripts/run_data_collection.sh --no-record
```

**Logs:**
Logs are saved to `logs/{TASK_NAME}/` directory:
- `run_data_collection_sh.log` - Script output
- `container.log` - Container logs
- `data_collector_server.log` - Data collector server logs (if available)
- `run_data_collection.log` - data collection application logs (if available)

**Outputs:**
Outputs are save to `recording_data/[{TASK_NAME}_{INDEX}]/` directory
#### Interactive Mode

Use `scripts/start_gui.sh` to launch an interactive container for debugging or development.

**Usage:**

```bash
./scripts/start_gui.sh [ACTION] [CONTAINER_NAME]
```

**Actions:**
- `run` (default) - Create and run a new container
- `exec` - Enter an existing container
- `start` - Start a stopped container
- `restart` - Restart a container

**Parameters:**
- `ACTION` - One of: `exec`, `start`, `restart`, `run` (default: `run`)
- `CONTAINER_NAME` - Container name (default: `data_collection_open_source`)

**Examples:**

```bash
# Create and run a new container (default)
./scripts/start_gui.sh run my_container

# Enter an existing container
./scripts/start_gui.sh exec my_container

# Start a stopped container
./scripts/start_gui.sh start my_container

# Restart a container
./scripts/start_gui.sh restart my_container
```

**Running Services Inside Container:**

After entering the container using `exec`, you need to start two services in separate terminals:

**Terminal 1 - Start the container and run data collector server:**

```bash
# Enter the container
./scripts/start_gui.sh exec my_container

# Inside container, start data collector server
python scripts/data_collector_server.py --enable_physics --enable_curobo --publish_ros
```

**Terminal 2 - Enter the same container and run data collection application:**

```bash
# Enter the same container (in a new terminal)
./scripts/start_gui.sh exec my_container

# Inside container, run data collection application
python scripts/run_data_collection.py --task_template tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json --use_recording
```

**Note:** Both terminals need to `exec` into the same container. Make sure the container is running before executing these commands.

### Option 2: Local Deployment

#### 1. Create Conda Environment

```bash
conda create -n data_collect python=3.11
conda activate data_collect
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

#### 3. Install cuRobo

Clone [cuRobo](https://github.com/NVlabs/curobo) and install:

```bash
# Set CUROBO_DIR to your cuRobo installation directory
export CUROBO_DIR=/path/to/cuRobo

# Copy robot assets and configs
cp -r ${SIM_ASSETS}/robot/curobo_robot/assets/robot ${CUROBO_DIR}/src/curobo/content/assets
cp -r config/curobo/configs ${CUROBO_DIR}/src/curobo/content/

# Install cuRobo
cd ${CUROBO_DIR} && pip install -e ".[isaacsim]" --no-build-isolation
```

**Note:** Make sure to set `TORCH_CUDA_ARCH_LIST` according to your GPU architecture before installing cuRobo. For RTX 4090D, use:
```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

#### 4. Setup ROS2
Install ROS2 on your local, who should be located in either `/opt/ros/humble/` or `/opt/ros/jazzy/`.
Then set environment variables, which can also be insert to your ~/.bashrc:
```bash
export ROS_DISTRO=jazzy # or humble
export ROS_CMD_DISTRO=${ROS_DISTRO}
export CONDA_SITE_PACKAGES=YOUR_CONDA_ENV_SITE_PACKAGES_PATH # e.g. ~/anaconda3/envs/data_collect/lib/python3.11/site-packages/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_SITE_PACKAGES}/isaacsim/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```
#### 5. Start the Services

You need to run two services in separate terminals:

**Terminal 1 - Data Collector Server:**

```bash
python scripts/data_collector_server.py [OPTIONS]
```

**Options for `data_collector_server.py`:**
- `--headless` - Run in headless mode
- `--enable_physics` - Enable physics simulation
- `--enable_curobo` - Enable cuRobo motion planning
- `--publish_ros` - Publish ROS messages, MUST be set if recording is needed

**Terminal 2 - Data Collection Application:**

```bash
python scripts/run_data_collection.py [OPTIONS]
```

**Options for `run_data_collection.py`:**
- `--use_recording` - Use recording mode, MUST be set if recording is needed
- `--task_template` - Task template JSON file path

**Example:**

```bash
# Terminal 1
python scripts/data_collector_server.py --enable_physics --enable_curobo --publish_ros

# Terminal 2
python scripts/run_data_collection.py --task_template tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json --use_recording
```

## Usage Examples

### Docker - Automated Data Collection

```bash
# Set assets path
export SIM_ASSETS=~/assets

# Run data collection with custom task
./scripts/run_data_collection.sh \
  --headless \
  --task tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json
```

### Docker - Interactive Development

```bash
# Set assets path
export SIM_ASSETS=~/assets

# Start interactive container
./scripts/start_gui.sh run my_container

# Terminal 1: Enter container and start data collector server
./scripts/start_gui.sh exec my_container
# Inside container:
python scripts/data_collector_server.py --enable_physics --enable_curobo --publish_ros

# Terminal 2: Enter the same container and run data collection application
./scripts/start_gui.sh exec my_container
# Inside container:
python scripts/run_data_collection.py --task_template tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json --use_recording
```

### Local - Full Control

```bash
# Terminal 1: Start server with physics and cuRobo
python scripts/data_collector_server.py --enable_physics --enable_curobo --publish_ros

# Terminal 2: Run data collection
python scripts/run_data_collection.py \
  --task_template tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json \
  --use_recording
```

## Task Configuration Guide

To create and configure data collection tasks, refer to the [Task Configuration Guide](TASK_CONFIG_GUIDE.md). This comprehensive guide covers:

- Creating task configuration files from scratch
- Configuring scenes, robots, and objects
- Setting up task stages (pick, place, insert, rotate, reset)
- Configuring runtime checkers and data filter rules
- Understanding action parameters and workspace types
- Complete examples and best practices

The guide provides detailed explanations and examples for all configuration options, making it easy to create custom tasks for your data collection needs.
