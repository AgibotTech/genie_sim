# genie_sim_client

An independent ROS 2 Python client for a Newton subprocess, joint/body state
transport, lockstep requests and reset confirmation. It has no benchmark or
RLinf application dependency.

Read the [design and protocol limits](DESIGN.md),
[application integration guide](BENCHMARK_HANDOFF.md) and
[maintainer guide](AGENTS.md).

## Build and run

From the repository root inside the prepared ROS runtime:

```bash
geniesim ros build dev
source devel/setup.bash
```

Install the engine separately and use the assembly pipeline to create a scene
manifest. Import `genie_sim_client` directly; importing the outer `geniesim_ros`
distribution invokes its wheel/runtime bootstrap.

```python
from genie_sim_client import ClientConfig, SimulationClient

config = ClientConfig.from_files("/absolute/launcher.yaml", "/absolute/stage/manifest.json")
with SimulationClient(config) as client:
    initial = client.snapshot()
    targets = initial.joint_positions.copy()  # Full (N, J) position targets
    result = client.step(targets, render=False)
    after_reset = client.reset([0], seeds=[42])
```

The application-independent example resets once, issues ten explicit hold
requests, closes the client and prints the final state as JSON:

```bash
python3 -m genie_sim_client.examples.newton_joint_step \
  --launcher /absolute/launcher.yaml \
  --manifest /absolute/stage/manifest.json --steps 10
```

Add `--joint ACTUAL_ENGINE_JOINT --target 0.1` to command a joint in radians or
metres. Task scoring and report writing belong to the caller.

## Configuration and API

The [example launcher](config/launcher_client_newton.yaml) selects the stock
MuJoCo-Warp profile.

| Setting / API | Source and meaning |
| --- | --- |
| Solver, profile, `physics_hz`, `num_envs` | Launcher `genie_sim_engine.ros__parameters`; absent `num_envs` uses the engine default of 1 |
| `step_mode`, `step_lockstep_ticks` | Explicit `lockstep` and positive integer K; no separate control rate or implicit rounding |
| `use_tf_render` | Must be true; the client waits for body poses |
| `scene_yaml_defaults` | Serialized to the engine and merged underneath scene YAML |
| `stage_manifest` | Assembly artifact referencing scene YAML; no third physics configuration channel |
| `headless` | Display switch on `from_files`, default true |
| `spawn` | Default true; false attaches to a compatible externally started engine during startup grace |
| `ready_timeout_s` | Default 600 s; readiness requires three distinct joint messages with a stable layout and body state |
| `step_timeout_s` | Default 5 s for each state, reset epoch/body and publisher-discovery wait |
| `log_path` | Unique log by default; explicit path supported and included in errors |
| `renderer` | Optional `RenderBridge`; `step(render=True)` obtains physics state before rendering |

`WorldSnapshot` owns immutable joint/body arrays, per-world names, message
sequences, publisher wall-clock stamps and reset epoch. Stamps are not simulation
time, and `observation_id` is not a physics tick. `config.step_duration` is the
requested K/physics_hz duration, not measured completed time.

Completion is `arrival_v1`: a new joint message followed by sufficiently fresh
body state. Startup free-running, separate command latching and asynchronous
host mirrors prevent proof of exact command ownership or committed K ticks.
`completion="committed_v2"` is unsupported and raises before sending.

Use one engine, control producer and reset source per ROS domain/topic set.
`/rlinf/reset_envs*` names are engine protocol topics, not an application
requirement. The client preserves shared rclpy context ownership and does not
change CUDA or pacing environment variables. An after-send timeout fails the
session: close/start rather than automatically resending an uncertain command.
External engine recovery belongs to its owner. Batched layout support does not
establish N>1 physics support in the current Newton backend.

## Verification

From the repository root, without ROS or GPU:

```bash
python3 -m pytest -q source/geniesim_ros/src/ros_ws/src/genie_sim_client/tests
```

CMake also registers ament pytest. An optional local rclpy loopback uses a fake
engine, requires no Newton/GPU, and needs an unused ROS domain:

```bash
python3 source/geniesim_ros/src/ros_ws/src/genie_sim_client/tests/ros_transport_smoke.py --domain-id 173
```

This checks rclpy APIs, message types, QoS, actions, seeded reset and shared-context
lifetime. Same-process loopback does not replace cross-process DDS, Newton GPU
or renderer acceptance.

## Lifecycle pause barrier

`client.pause()` waits for INACTIVE and `resume()` for ACTIVE through the existing lifecycle services. Acknowledging an enqueued service request is insufficient; GetState must confirm it. Step/reset reject paused sessions. Pause remains available after an uncertain failed action, without reviving that session. The application separately joins fresh state/images/progress; arrival_v1 remains unchanged.


## IPC action and observation client

`IPCClient` and `IPCClientConfig` provide transport for the independent libuipc
backend. They do not start a world or subprocess. The caller selects scene and
launcher YAML, owns normal bringup, and uses correlated actions and snapshots.
See [the IPC protocol](IPC_CLIENT.md) for completion semantics and synchronized
images. The Newton `SimulationClient` API and protocol remain unchanged.
