# genie_sim_engine_newton

The Newton-standalone physics backend. Kit-free: it drives body and
joint state directly through Newton, with no Isaac Sim runtime in the
process. The Wuji evaluation launcher uses a separately owned Isaac Sim 6
RTX worker for observations and video; Newton remains the only physics owner.

## What's in here

- **One backend**, `physics_engine:=newton` — supports Newton cloth and soft body alongside rigid.
- **Four solver adapters**, picked by `physics_solver:=` —
  MuJoCo-Warp (single-world and replicated), Featherstone, and AVBD.
- **One entry point**: `scripts/genie_sim_engine_newton_node.py`.
- **The `newton_extras` assemble stage** (`assemble/`), which adds
  cloth and soft-body entries to the scene the core pipeline built.
- **Standalone checks** under `tools/`, which run without a ROS
  runtime or a GPU.

The C++ ROS 2 bridge that publishes `/clock`, `/joint_states`, `/tf` and
`/odom` is owned by `genie_sim_engine_core`. Paused RGB batches are published
by this backend on its existing lifecycle ROS node after render/readback.
This backend implements core's `PhysicsEngine` ABC and consumes the robot
assets produced by the shared URDF→USD assemble pipeline.

## Quick check

```bash
ros2 pkg list | grep genie_sim_engine_newton
```

The engine is launched indirectly — via `genie_sim_bringup`'s
`app.launch.py` with a `launcher_config:=…`.

## When you'd touch this package

- Tuning physics, drives, contact, mass, friction, or mimic-joint
  behaviour on the newton path.
- Adding or changing a solver adapter.
- Adding a debug publisher or a visualizer, or extending the build
  pipeline (`engine/newton/setup/`).
- Working on cloth / soft-body solver wiring.

Physics values are not edited here: they are homed in the profile named
by `physics_params_file`, or in the launcher YAML's
`scene_yaml_defaults:` block.

## Mechanics & deep-dives

- [AGENTS.md](AGENTS.md) — package layout, adapter table, config surface.
- [docs/newton_quirks.md](docs/newton_quirks.md) — Newton / MuJoCo-Warp behaviours worth knowing first.
- [docs/ovrtx_sync.md](docs/ovrtx_sync.md) — the OVRtx in-process visualizer design.
- [docs/isaacsim_observation.md](docs/isaacsim_observation.md) — the default Wuji evaluation path: complete Newton mesh snapshots, a synchronous Isaac Sim worker, paused RGB batches and simulation-clock observer MP4 recording. Requires the actual Isaac Sim 6 runtime; CPU checks are not GPU acceptance.
- [docs/gl_observation.md](docs/gl_observation.md) — optional Newton GL observations and recording, selected with `physics_engine_visualizer: newton` in a local launcher copy.
- Cross-package deep-dives live in
  [`../genie_sim_engine_core/docs/`](../genie_sim_engine_core/docs/):
  [engines.md](../genie_sim_engine_core/docs/engines.md),
  [pipeline.md](../genie_sim_engine_core/docs/pipeline.md),
  [asset_consumption.md](../genie_sim_engine_core/docs/asset_consumption.md),
  [param_injection_eval.md](../genie_sim_engine_core/docs/param_injection_eval.md),
  [perf.md](../genie_sim_engine_core/docs/perf.md),
  [physics_knobs.md](../genie_sim_engine_core/docs/physics_knobs.md).

Generic scene-plugin and particle-render functionality is available for Newton scenes.

Scene recording copies particles to host memory only while a snapshot consumer
is active. Plugins with `on_record_snapshot` may expose
`record_snapshot_enabled()` to report their live state. Chef uses the existing
`newton.chef.food_state.enabled` configuration and stops requesting snapshots
if its publisher is unavailable. See [the recording contract](AGENTS.md#recording-snapshot-consumers)
for compatibility and reset behavior.
