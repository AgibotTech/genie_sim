# data_collection — Agent Development Guide

## Scope and provenance

This module contains the data collection and replay implementation imported
from the standalone open-source branch at commit
`d36a74d6b07cbfeca1fcf1ba4000fb2ad5835905`.
The standalone test files are excluded from this monorepo import.

Read [README.md](README.md), [TASK_CONFIG_GUIDE.md](TASK_CONFIG_GUIDE.md),
and [the replay guide](replay/README.md)
for Docker installation and runtime contracts. LLM tasks are authored with
scene-agent and execute `data_gen.py`; the Task Configuration Guide covers
separate declarative JSON templates. An LLM task's `task_info.json` is metadata,
not a declarative task template.
See `RobotAPI.control_gripper` in [robot_api.py](client/llm_gen/robot_api.py)
for the opt-in native `ControlGripper` command. Keep legacy gripper, pick, move, and trajectory
semantics unchanged; do not apply task-specific control patches globally.

## Runtime

- Isaac Sim 5.1 and Python 3.11.
- cuRobo v2, pinned to `3fd54dc782a82e5500a771cfd47856ea499d5fef`.
- Assets are acquired separately; use `GENIESIM_ASSETS_PATH`/`SIM_ASSETS`, installed-package lookup, or nearby relative directories.
- Default application image: `registry.agibot.com/genie-sim/geniesim3-data-collection:latest`; override with `DATA_COLLECTION_IMAGE`.
- Default container name: `geniesim3-data-collection`; override through the launch arguments.
- Supported robot families: G1, G2 (including CRS and 90D), ARX, and Aloha.
- Recording conversion uses repository URDFs for forward kinematics.
- cuRobo configuration lives only under `config/curobo/v2`; the Dockerfile
  copies it explicitly and checks every referenced planner URDF.
- Keep `config/robot_cfg`: it supplies runtime robot definitions and recording metadata.
- No `sim_data_format` package or separate `ik_solver` wheel is required.
- Perception-only replay exports, independent LeRobot exporters, and data
  quality validation are outside this module's open-source scope.

## Entrypoints and outputs

Run the scripts with this module as the working directory:

Container bootstrap scripts do not define a `geniesim` shell alias; the
command is owned by `geniesim_cli`, including benchmark and collection dispatch.

| Workflow | Entry |
|---|---|
| Automated collection | `scripts/run_data_collection.sh --task <JSON> --headless --standalone` |
| Collection server | `scripts/data_collector_server.py` |
| Collection client | `scripts/run_data_collection.py` |
| Isolated LLM task collection | `scripts/collect_llm_task.py`; shared lifecycle and episode execution in `scripts/collection_runtime.py` |
| Interactive container | `scripts/start_gui.sh` |
| Replay | `python -m replay`; consult `replay/README.md` for arguments |

Collection writes episodes under `recording_data/` and logs under `logs/`.
Replay output paths are selected through the replay configuration and arguments.
Keep both sides of the gRPC interface consistent when editing it.

## Monorepo integration

`pyproject.toml` and `__init__.py` retain the lightweight locator package
`geniesim-data-collection`. It exposes the module directory without importing
Isaac Sim, ROS, or the collection runtime. `geniesim data-collection install`
registers it editable and saves resolved task/asset roots in the user settings
file; environment overrides take precedence. `run <TASK>` defaults to LLM tasks;
JSON jobs use `run --task-json PATH`. `llm list/show` inspects the external library,
and `replay <TASK_OR_EPISODE>` renders a saved recording. LLM tasks and replay
inputs use isolated working copies. Runtime installation remains in `data-collection build`.
Data collection is a host CLI verb and must not trigger automatic tier-1 bootstrap.
`geniesim data-collection config` reports saved settings and the effective task
root. To inspect the effective asset root, run `python scripts/resolve_assets.py`
from this module; `config` does not resolve or validate that path.

The `geniesim data-collection` CLI discovers tasks, builds the image, and launches
collection using the same independent `scripts/resolve_assets.py` locator as
both host scripts. It accepts the main repository asset setting, legacy
settings, installed asset packages, and nearby relative asset directories.
Resolved assets are bind-mounted and `SIM_ASSETS` is set inside the container.
The single `docker/Dockerfile` builds from the official Isaac Sim 5.1
container and installs the ROS CLI, CUDA compiler, collection/replay dependencies,
robot URDFs, and application code. Do not restore cuRobo 0.7.6 or internal image defaults.

`geniesim data-collection build --china` / `--mirrors NAME` select the checkout's
`docker/mirrors.toml`; explicit URL build arguments override presets. Without a
preset, APT/pip settings inherit the parent. Direct builds must supply BuildKit's
`--build-context geniesim_mirrors=../../docker` while keeping this module as the
main context. The shared helper runs before dependencies and persists
`PIP_CONFIG_FILE=/etc/pip.conf` for later installs. Keep third-party repositories,
SDK constraints and pinned dependencies unchanged when modifying mirror support.

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES/).
Repository licensing does not replace third-party code and asset licenses.

## External LLM task libraries

`scripts/resolve_llm_task_root.py` is the standard-library source of truth.
Precedence: `GENIESIM_LLM_TASK_ROOT`, legacy `LLM_TASK_ROOT`, sibling scene-agent
examples, installed scene-agent examples, local task libraries, asset-pack tasks.
The default LLM CLI copies tasks into its run directory and mounts the working
library at `/run/tasks`. JSON/GUI launchers mount the selected root read/write
at `/llm_tasks` and pass the container path through both variables. The client
uses the same resolver. Keep external asset/task libraries outside the image
build context. Scene USD binding writes to the task mount; use a working copy
when the original must be preserved.
`--task` takes a job JSON with `mode=llm`, `task_name`, `scene_dir`, and `run_times`,
not the robot metadata in a library's `task_info.json`.

The default `geniesim data-collection run <TASK>` and `llm run <TASK>` entrypoints
read the selected scene's `task_info.json`, falling back to task metadata, for
backend, physics frequency, deformable support, and planner batch size. Their
clients use the same scene-first metadata selection. LLM wrappers launched
through `--task-json` instead use GPU physics, 60 Hz stepping, deformable support
disabled, and a cuRobo batch size of 10; these server settings are not read from
metadata, and the client reads only task-level `task_info.json`. Use the default
LLM entrypoints for metadata-driven server settings or scene-specific metadata.

`scripts/llm_task_policy.py` rejects multi-run deformable collection
in host and container entrypoints. Tasks with USD-attached behavior scripts
also require a single run; this asset condition is documented rather than
automatically inspected by the lightweight host CLI.
