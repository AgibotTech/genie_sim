# GenieSim post-training server

This package connects RLinf environments to GenieSim. It runs simulation and
benchmark success checking in a separate server. A `GenieSimVecEnv` client
maps policy joint targets and observations onto SHM or ZMQ transport. Pico
teleoperation provides an interactive way to inspect and debug the same tasks.

Training algorithms, policy models, and RLinf worker placement belong to RLinf.
This package supplies the simulation side of that boundary.

## Supported tasks

This release supports 26 tasks using the official G2op task/scene pairings
from the `g2op_if_*`, `g2op_manip_*`, and `g2op_spatial_*` benchmark configs.
`sorting_packages` and `sorting_packages_continuous` are excluded.
ZMQ task initialization and reload accept only these official
`task_name`/`sub_task_name` pairs, which the server's metadata advertises:

- **Instruction following (10):** `pick_billiards_color`, `pick_block_color`,
  `pick_block_number`, `pick_block_shape`, `pick_block_size`,
  `pick_common_sense`, `pick_follow_logic_or`, `pick_object_type`,
  `pick_specific_object`, `straighten_object`.
- **Manipulation (8):** `hold_pot`, `clean_the_desktop`, `open_door`,
  `place_block_into_box`, `pour_workpiece`, `scoop_popcorn`,
  `stock_and_straighten_shelf`, `take_wrong_item_shelf`.
- **Spatial reasoning (8):** `pick_object_absolute_position`,
  `pick_object_relative_position`, `place_beverage_to_anothers_position`,
  `place_object_relative_position`, `sort_cubes_by_size`, `sort_number`,
  `stack_bowls`, `stack_three_building_blocks`.

Validate task execution with the selected backend and assets using the
[runtime checks](tests/README.md).
Other tasks remain in the shared benchmark package outside this release scope.
Custom integrations can use the task loader and `build_handler` directly,
including SHM embedding; see [task selection](docs/interface.md#task-selection).

## Post-training workflow

1. On the simulation node, build the post-training image and start the server;
   see [server preparation](#server-preparation) and
   [start and initialize](#start-and-initialize).
2. On the training node, build the training image with the client installed and
   start training. The trainer connects to the server, selects the task and
   backends, and requests simulation over ZMQ. For RLinf, see the
   [RLinf GenieSim guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/geniesim.html).

[Pico teleoperation](TELEOP_USAGE.md) connects to the same server to debug tasks.

## Runtime layout

```text
RLinf environment / Pico teleop
    -> GenieSimVecEnv or SimClient
    -> SHM / ZMQ
    -> sim_server + GenieSimHandler
    -> physics backend + render bridge + benchmark checker
```

Physics backends include Newton MuJoCo Warp, AVBD, Featherstone, Isaac PhysX,
Isaac Newton, and fused Isaac PhysX. Render bridges include Isaac tiled, OVRTX,
and a null renderer. `isaac_fused` runs physics and rendering in one process
and must be selected for both backend fields. Availability depends on the
installed simulator and engine packages.

The `isaac_physx` and `isaac_newton` wrappers support one environment and
restoring its initial state. They reject seeded resets and instance switching.
Use `isaac_fused` for batched environments and benchmark instance resets.

ZMQ supports idle startup, task initialization and reload, metadata, and
randomization readback. SHM retains local step/reset communication for a task
built by the embedding application; it does not implement the ZMQ task control
plane. See [the interface](docs/interface.md).

## Client preparation

A client can connect to an existing server without installing ROS or a simulator.
With Python 3.10 or newer, run this command from the repository root in the
Python environment that runs RLinf:

```bash
python -m pip install ./source/rlinf_geniesim
```

This installs `rlinf-geniesim-client` with the existing `rlinf_geniesim` import
paths, including `rlinf_geniesim.envs.vec_env.GenieSimVecEnv`. After installation,
the client needs neither the checkout nor a `PYTHONPATH` entry. A client image
can install the package during its build and omit the checkout from the final
image. Align the installed source revision with the server deployment.

The default install provides ZMQ communication and NumPy observations using
NumPy, PyZMQ, and Pillow. RLinf supplies its own PyTorch environment. The `shm`
extra adds local SHM support, and `sensor` adds SciPy for sensor augmentations;
see [the client installation guide](CLIENT_README.md).

Server implementations, launch and debug scripts, simulation backends, benchmark
tasks, and assets remain in the server checkout. `requirements-client.txt`
retains the shared dependencies for source-based scripts; it is not the
standalone client's installation entry point. Pico teleoperation needs additional
dependencies; see [teleoperation](TELEOP_USAGE.md).

## Server preparation

Use the repository [installation instructions](../../README.md) for the
GenieSim base runtime, ROS workspace, simulator, and assets. This package is
maintained separately from the `geniesim` CLI peers. Asset files must be
available to the simulator under `GENIESIM_ASSETS_PATH`.

The server image adds this package's dependencies to an existing GenieSim
base image. The optional teleop image adds Pico/IK dependencies on top of the
server. Both use the shared GenieSim Docker CLI. The RLinf training/client
image is separate and installs only `rlinf-geniesim-client` from this checkout.

The simulation host's GPU and driver must satisfy the
[Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
for RTX rendering. Check the host driver with `nvidia-smi` before starting the server.
System-Python physics backends also require a driver that supports the base
image's CUDA 13 builds of Torch and Warp.

On the simulation host, outside containers, run from the repository root:

```bash
python3 -m pip install -e source/geniesim_cli
geniesim docker6.2 build
geniesim docker --profile post-train build
```

The first build creates the default base `registry.agibot.com/genie-sim/geniesim4:latest`
locally from NVIDIA's `nvcr.io/nvidia/isaac-sim:6.2.0.1`, paired with the
`6.2.0.0` system-Python Isaac Sim package, and keeps the physics libraries
supplied by Isaac Sim. The second build uses that local image as its default
base, so neither build needs access to the AgiBot registry.

A compatible custom base must provide system Python 3.12, ROS 2 Jazzy,
bundled Isaac Sim 6.2.x at `/isaac-sim` including `/isaac-sim/python.sh`,
and the matching Newton/MuJoCo/Warp stack. It must also provide the common
GenieSim entrypoint's `isaac-sim` user and runtime tools. The post-training
build checks the bundled version reported by `/isaac-sim/VERSION`.

The `isaac_fused` backend runs with `/isaac-sim/python.sh` and uses the
Isaac Sim 6.2 tiled camera API. Use `geniesim docker` or `geniesim docker6.2`
for its base. Isaac Sim 4.5, standalone 5.1 and the 5.1 + 6.2 combo are
deprecated (E.O.L.) and are not compatible parents for this backend.

On the simulation host, mount the checkout and assets and enter the server
container. The host only needs the CLI, Docker, and the NVIDIA container runtime;
it does not need the asset package's Python dependencies:

```bash
export GENIESIM_ASSETS_SRC=/path/to/geniesim_assets
geniesim docker --profile post-train up --headless
geniesim docker --profile post-train into
```

The container mounts this checkout at `/workspace` and assets at
`/geniesim_assets`, uses host networking and 8 GB shared memory, and runs commands
as the host UID/GID. The entrypoint installs the mounted packages without
changing the image's dependencies. The batch server launcher builds the selected
ROS packages as that same user, keeping the workspace and lock writable.
`up` prepares the container; the runtime CLI below starts the server processes
through the batch launcher.
The image does not supply missing assets. See the
[Docker profile reference](../../docker/README.md#post-training-profiles) and
[dependency and binary distribution boundaries](DEPENDENCIES.md).

For application-owned local servers, the optional
[`spawn_server` Python API](docs/interface.md#private-local-servers) starts one
private process per call in this prepared runtime. It retains the existing
client and backend interfaces; the manual workflow below remains available.

## Start and initialize

Inside the simulation server container, from `/workspace`:

```bash
geniesim post-train server start \
  --count 1 --base-port 5601 --devices 0
```

An RLinf client initializes the task and backends through `GenieSimVecEnv.init`.
For a standalone check or teleop session, initialize explicitly inside this
same server container:

```bash
source source/rlinf_geniesim/scripts/rlinf_env.sh
python3 source/rlinf_geniesim/scripts/init_task.py \
  --addr tcp://127.0.0.1:5601 --sub pick_block_color --envs 1
```

The launcher starts an idle ZMQ server. The client chooses the task and
backends at initialization. The first successful initialization locks the
backend settings for that server process. Restart the server to change them.
The default init command selects fused Isaac PhysX, which renders in the same
process. For Newton MuJoCo Warp with Isaac tiled rendering, initialize a fresh
server inside the server container with:

```bash
python3 source/rlinf_geniesim/scripts/init_task.py \
  --addr tcp://127.0.0.1:5601 --sub pick_block_color --envs 1 \
  --physics newton_mujoco_warp
```

The render backend and substep count follow `--physics` unless given.

`--config FILE` accepts additional task-init sections, such as `randomization`,
`rates`, and `observation`. Task identity and backend settings remain CLI
arguments. RLinf clients can instead initialize through `GenieSimVecEnv.init`.

Check or stop the recorded server for one port inside the server container:

```bash
geniesim post-train server status --count 1 --base-port 5601
geniesim post-train server logs --port 5601 --follow
geniesim post-train server stop --count 1 --base-port 5601
```

Starting an occupied slot fails. Stop it explicitly before restarting.
Logs show the last 100 lines; `--follow` follows new lines and log replacement
until Ctrl-C. `RLINF_RUN_ROOT` selects the same runtime directory for start,
status, stop, and logs (default `/tmp`). These commands operate in the current
runtime; enter the server container before using them for container processes.

To remove the container after stopping the servers, run on the simulation host:

```bash
geniesim docker --profile post-train down
```

## Multiple servers

Inside the simulation server container, start contiguous ports serially,
distributing servers across the listed GPUs:

```bash
geniesim post-train server start \
  --count 2 --base-port 5601 --devices 0,1 --dry-run
geniesim post-train server start \
  --count 2 --base-port 5601 --devices 0,1
```

For noncontiguous ports or explicit ROS domains, copy
[sim_servers.example.conf](config/sim_servers.example.conf), then pass that
file after the action. Inside the server container:

```bash
geniesim post-train server start /path/to/servers.conf
geniesim post-train server status /path/to/servers.conf
geniesim post-train server stop /path/to/servers.conf --only 5601,5602
```

`--dry-run` validates the complete plan before
launching. Each slot has its own port, ROS domain, process record, logs, and
runtime files. If a batch launch fails, it stops only servers started by that
invocation. Use the same plan or count/base-port selection with `status` and
`stop`; `--only PORTS` narrows that selection.

The CLI locates the checkout from `GENIESIM_REPO_ROOT`, then
`GENIESIM_REPO_PATH` / `SIM_REPO_ROOT`, then the current directory's ancestors.
Container profiles set `SIM_REPO_ROOT=/workspace`. On a prepared bare-metal
runtime, run from the checkout or set `GENIESIM_REPO_ROOT` explicitly.
The existing `start_sim_servers.sh` and `start_sim_server.sh` scripts remain
available for integrations that already invoke them directly.

## Verify and debug

Run a finite model-free chunk/reset check against the initialized server:

```bash
PYTHONPATH=source:source/geniesim_benchmark/src \
  python3 source/rlinf_geniesim/scripts/smoke_test.py \
  --address tcp://127.0.0.1:5601 --attach pick_block_color
```

The check holds the reset joint targets, verifies observations and auto-reset
boundaries, then detaches without stopping the server. It uses the server's
resolved robot layout. `--task-init FILE` initializes from a complete payload
instead of attaching; [the example](config/task_init_pick_block_color.yaml)
uses a null renderer.

For interactive task inspection, see [Pico teleoperation](TELEOP_USAGE.md).
For simulator integration checks, see [runtime validation](tests/README.md).

## Pre-release testing

Before delivering or publishing the server, follow the
[pre-release testing guide](docs/release_testing.md) on a prepared simulation
host. It covers building the post-training images, starting the server on real
assets with the default backends, and checking task switching and reset
through Pico teleoperation, with and without a headset.

## Reference

- [Client API and lifecycle](docs/interface.md)
- [Environment variables](ENV_VARS.md)
- [Randomization configuration](docs/GENERALIZATION_DESIGN.md)
- [Randomization readback](docs/RANDOMIZATION_READBACK_API.md)
- [Fused backend](docs/ISAAC_FUSED_PHYSX_DESIGN.md)
- [Per-instance body poses and rest-state bake](docs/INSTANCE_BODY_POSES.md)

## License

Source files use [MPL-2.0](LICENSE), subject to file-specific notices.
Dependencies, simulator binaries, and assets retain their own terms; see
[DEPENDENCIES.md](DEPENDENCIES.md).
