# genie_sim_client

Shared Newton subprocess/ROS client in geniesim_ros. Read [README.md](README.md),
[DESIGN.md](DESIGN.md) before changing its contract.
Benchmark integration and CLI work are caller-owned; see [BENCHMARK_HANDOFF.md](BENCHMARK_HANDOFF.md).

- `genie_sim_client/config.py`: launcher YAML → immutable argv; manifest → scene YAML.
- `genie_sim_client/client.py`: exclusive lifecycle/step/reset, named layout and snapshot contract.
- `genie_sim_client/_ros_bridge.py`: lazy ROS imports and atomic cache/credentials.
- `genie_sim_client/_process.py`: isolated process-group teardown.
- `genie_sim_client/examples/newton_joint_step.py`: standalone reset/step/state example; no application imports.
- `tests/`: ROS/GPU-free client protocol, failure, example and dependency tests.

Never import either application, engine backend, Isaac Sim, Newton or Warp here.
The `genie_sim_common` edge is build glue only. Engine installation and assembled
scene assets are supplied by the caller. Preserve wire topic names `/rlinf/*`
as part of the current engine protocol; names do not imply an application dependency.
No physics config via Python kwargs, environment variables or sentinel files.
No reward/done/auto-reset or checker/task asset lookup in this package.
The bridge subscribes only to joint/body state. Keep unused scene-joint/AABB
mirrors and individual cache getters out; consumers read state atomically.
Scope independence tests to the shared client and its example. Application
adapters, task/config loaders and application integration tests belong in the
owning application. Keep the shared example application-independent.

arrival_v1 has no command-id acknowledgement and cannot implement exact
committed stepping. Do not strengthen that claim without changing the engine
protocol and tests. Engine ABC changes require review under the core package contract.

Run `python3 -m pytest source/geniesim_ros/src/ros_ws/src/genie_sim_client/tests`
from the repository root with the package directory on PYTHONPATH. Format Python
with black at 120 columns. CMake registers the same tests with ament pytest.
