# Data Collection

Collect robot task episodes and replay recorded trajectories with the module's
Docker image, based on Isaac Sim 6.2 and cuRobo v2. Robot and object assets are
provided separately. Supported robot families include G1, G2, ARX, and Aloha.

There are two task workflows:

| Task type | How to create it | How to run it |
| --- | --- | --- |
| **LLM task** | Use [scene-agent](../geniesim_scene_agent/README.md) to prepare scenes and author `data_gen.py`. | `geniesim data-collection run <TASK>` |
| **JSON task** | Write a declarative template using the [Task Configuration Guide](TASK_CONFIG_GUIDE.md). | `geniesim data-collection run --task-json <JSON_OR_TEMPLATE>` |

An LLM task's `task_info.json` is metadata for its Python task code; it is not
the declarative JSON task format.

## Contents

- [Requirements](#requirements)
- [Docker installation](#docker-installation)
- [Collect data](#collect-data)
- [Outputs and logs](#outputs-and-logs)
- [Replay recordings](#replay-recordings)
- [Configuration](#configuration)
- [Advanced Docker usage](#advanced-docker-usage)
- [License and third-party notices](#license-and-third-party-notices)

## Requirements

- An NVIDIA GPU and compatible driver, Docker, and NVIDIA Container Toolkit.
- A checkout containing this module and the shared `docker/` build resources.
- An external asset bundle containing `robot/`, `objects/`, and the asset index
  required by the chosen tasks.
- The main repository's `geniesim` host CLI for the commands below; see the
  [CLI guide](../geniesim_cli/README.md). It manages Docker and locates resources.

The image inherits Isaac Sim 6.2, system tools and ROS 2 Jazzy from the main
GenieSim image, then adds collection/replay dependencies and cuRobo. The
default build targets GPU architecture `8.9`; adjust
`TORCH_CUDA_ARCH_LIST` for another GPU and check compatibility with the pinned
cuRobo revision.

All host commands below run from the **main repository root**, unless a block
explicitly changes directory.

## Docker installation

### Configure assets and tasks

```bash
export GENIESIM_ASSETS_PATH=/path/to/assets
geniesim data-collection config
```

The asset root must include its `__init__.py` and attached Python scripts and
helpers. Collection and replay expose this mounted tree as `geniesim_assets`
for scripts that import asset-owned behavior helpers.

The task library defaults to the sibling
`source/geniesim_scene_agent/examples/tasks/`. To use another library:

```bash
export GENIESIM_LLM_TASK_ROOT=/path/to/task-library
```

Each LLM task contains `data_gen.py`, `task_info.json`, and scene directories
with `scene.usda` and `scene_info.json`. See the
[example library](../geniesim_scene_agent/examples/README.md).

### Build the image

Build or obtain the main repository's Isaac Sim 6.2 image first; see the
[main image recipe](../../docker/Dockerfile). The default parent is
`registry.agibot.com/genie-sim/geniesim4:latest`.

```bash
geniesim data-collection build
```

To use a specific main-image tag or digest:

```bash
# Set this to your existing Isaac Sim 6.2 main-image reference:
geniesim data-collection build --build-arg GENIESIM_BASE_IMAGE="$GENIESIM_BASE_IMAGE"
```

The incremental build reuses the parent's ROS installation and repositories;
it requests only missing recording tools instead of reinstalling ROS.
Isaac Sim rendering and collection retain the
bundled SDK interpreter and numerical dependencies.
Mirror presets below apply to the incremental installs. Select the parent
registry through `GENIESIM_BASE_IMAGE`, or use the main image's own mirror
options when building that parent.

The default image is
`registry.agibot.com/genie-sim/geniesim4-data-collection:latest`.
Use `DATA_COLLECTION_IMAGE` or the CLI's `--image TAG` option to select another
image. The default container name is `geniesim4-data-collection`.

Build using shared mirror presets or custom download sources:

```bash
geniesim data-collection build --china
geniesim data-collection build --mirrors upstream --dry-run
# Set these variables to your own repository URLs before running:
geniesim data-collection build --china \
  --build-arg APT_MIRROR="$APT_MIRROR" \
  --build-arg PIP_INDEX_URL="$PIP_INDEX_URL"
```

`--china` is an alias for `--mirrors china`. Presets come from the checkout's
[`docker/mirrors.toml`](../../docker/mirrors.toml). Explicit build arguments
win over the preset. Without a preset, omitted or empty APT/pip values inherit
the parent image's settings. The image saves its selected APT repositories and
pip configuration for subsequent installs; source and SDK version pins remain
unchanged. See [build options](#build-options) for direct Docker builds.

## Collect data

### LLM tasks

To create a new LLM task, use
[scene-agent](../geniesim_scene_agent/README.md#automatic-collection-task-orchestration)
for scene preparation and task orchestration. It produces the `data_gen.py`,
`task_info.json`, `scene.usda`, and `scene_info.json` files consumed here.
See the [task-orchestration skill](../geniesim_scene_agent/skills/geniesim-task-orchestration/SKILL.md)
for the authoring workflow. Existing task libraries can be executed directly
with this collection image.

List tasks and inspect a task's available scene IDs:

```bash
geniesim data-collection llm list --robot g2
geniesim data-collection llm show place_long_block_across_two_supports_g2
```

Collect one episode, including camera rendering:

```bash
geniesim data-collection run place_long_block_across_two_supports_g2 \
  --scene 2026121696_166a0404 --render
```

The host CLI starts the collection server and client inside the Docker image.
It copies the task into the run directory and mounts that copy at `/run/tasks`;
assets are mounted read-only at `/sim_assets`.

| Option | Behavior |
| --- | --- |
| `--scene ID` | Choose a fixed scene; default `random` samples a scene for each episode. |
| `--run-times N` | Collect N episodes serially; default 1. See the limits below. |
| `--render` | Enable collection camera rendering. |
| `--no-record` | Execute without saving an episode. Recording is enabled by default. |
| `--no-headless` | Show the GUI. Execution is headless by default. |
| `--task-root PATH`, `--assets-root PATH` | Override resource roots for this command. |
| `--output-dir PATH` | Select a new run directory. |
| `--image TAG`, `--container-name NAME` | Override the Docker image or container name. |
| `--dry-run` | Print the command without creating a run or starting Docker. |

**Single-run limits:** tasks with deformable objects or USD-attached behavior
scripts require `--run-times 1`. Start a new CLI run for another episode.
The CLI and container entrypoints reject multiple runs when task/scene metadata
sets `enable_deformable: true`; random scene selection checks all candidates.
USD-attached behavior scripts are not automatically inspected by the host CLI,
so check those assets before requesting multiple runs. These limits also apply
to LLM jobs launched through `--task-json`.

### JSON tasks

JSON tasks describe scene generation and action stages in a declarative
configuration. Create or adapt a template using the
[Task Configuration Guide](TASK_CONFIG_GUIDE.md), then run it with the explicit
JSON selector:

```bash
geniesim data-collection run --task-json sort_the_fruit_into_the_box_apple_g2 \
  --headless --standalone
```

The guide documents task fields, robot and scene settings, action stages,
and complete examples. JSON tasks use the configured template rather than
an LLM task's `data_gen.py`.

On GPU, JSON collection prepares each episode's objects during scene reset,
before recording starts. Creating new rigid bodies while GPU physics is already
running is unsupported; include those objects in the episode setup instead.

The JSON and interactive launchers mount the task library read/write at
`/llm_tasks` because they update scene USD references. Use a working copy when
those source files must be preserved.

## Outputs and logs

Default LLM collection runs are saved under the **data_collection module**:

```text
source/data_collection/recording_data/llm/<task>/<run>/
├── tasks/                  # Isolated task copy
├── episodes/               # Recorded episodes
├── report/                 # Server/client logs and per-episode results
├── run.log
├── run_manifest.json
└── run_result.json
```

`--output-dir` selects a different run directory. Check `run_result.json` and the
per-episode results to determine success. Camera videos require `--render`.
Each episode also saves `report/server_config_<N>.json`, recording the metadata
source, resolved server settings, and command, including startup failures.

JSON tasks write episodes under `source/data_collection/recording_data/` and
launcher logs under `source/data_collection/logs/`.
The direct launcher retains its script, container, server, and client logs in
the reported task log directory.

## Replay recordings

Replay the latest successful recorded CLI episode for a task, or supply an
explicit episode directory:

```bash
geniesim data-collection replay place_long_block_across_two_supports_g2
geniesim data-collection replay /path/to/episode --no-fuzz --depth ALL
```

The CLI copies the input before replay and runs the Docker image. Default preset
`data_collection` uses three environments and one round. Use `--preset teleop`
for teleoperation data, `--config FILE` for custom JSON, or
`--max-concurrent-envs N` and `--num-rounds N` to change replay coverage.
`--render-hz` defaults to 30 and controls simulation advance; timestamps come from
the original recording. `--no-fuzz` disables appearance randomization.

Replay runs are saved under
`source/data_collection/recording_data/replay_runs/<episode>/<run>/`.
Rendered episodes are in `output/`; `output_check.json` reports conversion,
HDF5, and video checks. An image/container override and `--dry-run` are available
as for collection.

For direct Docker use, mount an **episode working copy** and an output directory
writable by UID 1234:

```bash
docker run --rm --name geniesim4-data-collection --gpus all \
  -e ACCEPT_EULA=Y -e SIM_ASSETS=/sim_assets \
  -v /path/to/assets:/sim_assets:ro \
  -v /path/to/episode-copy:/input:rw \
  -v /path/to/replay-output:/output:rw \
  registry.agibot.com/genie-sim/geniesim4-data-collection:latest \
  /isaac-sim/python.sh -m replay --input_dir /input --recording_root /output
```

The [Replay guide](replay/README.md) documents required input files, direct
container arguments, presets, HDR assets, output formats, and completion checks.
Host CLI options such as `--depth` and `--no-fuzz` are converted into a replay
configuration; the direct `python -m replay` entry uses that JSON configuration.

## Configuration

### Resource roots

| Setting | Purpose |
| --- | --- |
| `GENIESIM_ASSETS_PATH` | Asset bundle; legacy `SIM_ASSETS` is also supported. |
| `GENIESIM_LLM_TASK_ROOT` | External task library; legacy `LLM_TASK_ROOT` is also supported. |
| `DATA_COLLECTION_IMAGE` | Collection/replay Docker image. |
| `GENIESIM_DATA_COLLECTION_CONFIG` | Alternative saved-settings file. |

Explicit environment roots take precedence over saved settings in
`~/.config/geniesim/data-collection.json`. Without configured roots, discovery
checks installed packages, sibling scene-agent examples, and nearby resource
layouts. `geniesim data-collection config` shows saved settings and the effective
task-library path; it does not resolve or validate the effective asset root.
To inspect the effective asset root from the monorepo root, run:

```bash
python source/data_collection/scripts/resolve_assets.py
```

### LLM task server settings

These settings apply to `geniesim data-collection run <TASK>` and
`geniesim data-collection llm run <TASK>`. These entrypoints use the selected
scene's `task_info.json` when present, otherwise the task-level file.
A scene-specific file must contain the required task and robot metadata as
well as any server settings. The [LLM JSON wrapper](#llm-jobs-through-the-json-launcher)
uses a different startup path with the limitations described below.

| Task metadata field | Server argument | Default |
| --- | --- | --- |
| `backend` | `--backend` | `cpu` |
| `physics_frequency` | `--physics_step` | `60` Hz |
| `enable_deformable` | `--enable_deformable` when true | `false` |
| `curobo_batch_size` | `--curobo_batch_size` | `10` |

Recording frequency is configured separately by `recording_setting.fps`.
For example, `fold_double_layer_t_shirt_g2` selects GPU physics, 360 Hz stepping,
and deformable support through its metadata. Run it with `--run-times 1`.
Robot definitions and planner configurations are in `config/robot_cfg/` and
`config/curobo/v2/` relative to this module. For JSON task settings, use the
[Task Configuration Guide](TASK_CONFIG_GUIDE.md).

## Advanced Docker usage

### Interactive container

```bash
geniesim data-collection up
```

`up` creates and enters the container. From another host terminal,
`geniesim data-collection into` opens a second shell;
`geniesim data-collection down` stops and removes the container when finished.
Use a distinct `--container-name NAME` when needed. Inside the container, run
application commands with `/isaac-sim/python.sh` or `omni_python`.

Alternatively, `scripts/start_gui.sh run` starts an interactive container from
the module directory. To launch a JSON job directly instead:

```bash
cd source/data_collection
./scripts/run_data_collection.sh \
  --task tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json \
  --headless --standalone
```

Use different container names or stop an existing container before starting a
launcher with the same name.

### LLM jobs through the JSON launcher

For integrations that already supply job JSON, `--task-json` also accepts an
LLM job wrapper with `mode: "llm"`, `task_name`, `scene_dir`, and `run_times`.
It still executes the existing LLM task's `data_gen.py`; the wrapper is a job
selector, not a declarative JSON task template. Use a working task-library copy
because this launcher mounts it read/write at `/llm_tasks`.

This launcher starts the server with GPU physics, 60 Hz stepping, deformable
support disabled, and a cuRobo batch size of 10. It does not apply the server
settings from task or scene metadata, and its client reads only the task-level
`task_info.json`. Use `geniesim data-collection run <TASK>` or `llm run <TASK>`
for tasks that need metadata-driven server settings or scene-specific metadata,
including deformable tasks such as `fold_double_layer_t_shirt_g2`.

### Build options

An equivalent direct build from the module directory is:

```bash
cd source/data_collection
docker build -f docker/Dockerfile \
  --build-context geniesim_mirrors=../../docker \
  -t registry.agibot.com/genie-sim/geniesim4-data-collection:latest .
```

The named context supplies the shared mirror helper. Keep that directory with
this module when exporting sources. Direct Docker builds accept mirror URLs
through `--build-arg`; preset flags `--china` and `--mirrors` belong to `geniesim`.
Keep external asset/task libraries outside the module's build context.

<details>
<summary>Docker build arguments and runtime layout</summary>

| Build argument | Default | Purpose |
| --- | --- | --- |
| `GENIESIM_BASE_IMAGE` | `registry.agibot.com/genie-sim/geniesim4:latest` | Main GenieSim image with Isaac Sim 6.2 and ROS Jazzy; accepts a tag or digest. |
| `APT_MIRROR`, `PIP_INDEX_URL` | Inherit parent | Persistent APT/pip download sources. |
| `TORCH_CUDA_ARCH_LIST` | `8.9` | GPU architecture for extension builds. |
| `CUROBO_REPO` | `https://github.com/NVlabs/curobo.git` | cuRobo source repository. |
| `CUROBO_COMMIT` | `3fd54dc782a82e5500a771cfd47856ea499d5fef` | Pinned cuRobo v2 revision. |
| `GIT_COMMIT_SHA`, `GIT_COMMIT_DATE`, `GIT_BRANCH` | `unknown` | Optional source metadata stored in the image. |

The child preserves the parent's bundled SDK Torch, NumPy, Warp, and CUDA
library versions. cuRobo uses NVRTC from that SDK without installing a second
CUDA compiler stack. Both Isaac Sim and system ROS 2 Jazzy use Python 3.12,
but recording subprocesses still use a separate system ROS environment.
The launcher keeps the SDK bridge and system ROS libraries separate.
The build copies and validates seven cuRobo v2 planner configurations and their
URDFs. Recording conversion uses the repository URDFs for forward kinematics.

</details>

## License and third-party notices

Original AgiBot code is provided under the [Mozilla Public License 2.0](LICENSE).
Third-party components and robot assets retain their own licenses and notices;
see [third-party licenses](THIRD_PARTY_LICENSES/).
`server/controllers/parallel_gripper.py` follows NVIDIA's Apache-2.0 release;
`server/ui_builder.py` follows AgiBot's published MPL-2.0 version.

The Docker build fetches cuRobo v2 from
[NVIDIA's repository](https://github.com/NVlabs/curobo) at commit
`3fd54dc782a82e5500a771cfd47856ea499d5fef`, under the retained
[Apache-2.0 license](THIRD_PARTY_LICENSES/LICENSE.CUROBO-V2).
