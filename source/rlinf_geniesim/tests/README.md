# Runtime validation

Run simulator checks in the prepared server runtime with the selected physics,
render backend and real asset package. Follow the [pre-release testing guide](../docs/release_testing.md)
for image construction, server startup, task switching and Pico teleoperation.
The [module README](../README.md) describes the server and client setup.

## Runtime smoke checks

- Build both post-training image layers with `--no-cache`, then start fresh
  containers with empty runtime caches and the host UID/GID.
- Build the selected ROS packages from an empty workspace and exercise
  the server CLI's start, status, logs and stop commands.
- Initialize a supported task with one environment and compatible physics and
  render backends. Check finite reset state and rendered camera images over ZMQ.
- Connect from a separate teleop container. Check the web console, MJPEG stream,
  received-input reporting and UDP port release on close, then exercise the
  headset and robot controls.

For a finite, robot-layout-aware client check, use `scripts/smoke_test.py` as
shown in the [module README](../README.md). Retain the checkout and asset
revisions, runtime settings, logs and outputs for each run.

## Pending runtime validation

Run `python3 -m pip check` in the base, server and teleop images before runtime
validation. Check ROS images through `RosRenderBridge`, which reads message
buffers directly. The base image's NumPy 1.x `cv_bridge` binary is incompatible
with NumPy 2 and is not used by the post-training image path.

Complete the following checks in the prepared runtime before release:

- [ ] Validate the server-to-RLinf workflow: observations, actions and chunks,
  rewards, termination, reset and task/instance reload; complete a short policy
  evaluation and training run.
- [ ] Load at least one instance of all 26 released tasks with real assets and
  sample instance switches.
- [ ] Exercise the supported physics/render backends and SHM/ZMQ combinations,
  multiple environments and servers, disconnect/reconnect, shutdown and cleanup.
- [ ] Validate Pico with the headset and robot runtime: arm and gripper control,
  pause/resume, reset and task/instance reload.

## Simulator integration checks

`integration/` contains explicitly invoked checks against a running server or
simulator node:

| Tool | Purpose |
| --- | --- |
| [fused_smoke.py](integration/fused_smoke.py) | Launch the fused node and check schema, state, physics motion and rendered frames |
| [fused_integration_check.py](integration/fused_integration_check.py) | Exercise the fused backend against the real node |
| [omnigraph_env_check.py](integration/omnigraph_env_check.py) | Check per-environment OmniGraph behavior during simulation |
| [fake_policy_rollout.py](integration/fake_policy_rollout.py) | Send synthetic G2 action chunks to a real ZMQ server and validate episodes |

These commands require their documented simulator dependencies. Read each
command's help or module documentation before running it. The synthetic policy
rollout targets a real server; use a model policy for model evaluation.

## USD asset maintenance

[hoist_body_local_offsets.py](tools/hoist_body_local_offsets.py) moves authored
rigid-body offsets while checking that composed geometry remains unchanged.
It reports proposed changes by default; inspect that output before using
`--apply` to rewrite scene assets.
