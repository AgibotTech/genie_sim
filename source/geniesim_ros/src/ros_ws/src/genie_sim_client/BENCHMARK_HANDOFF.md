# Application integration with the Newton client

This guide uses the implemented `genie_sim_client` API. The caller owns action
and observation adaptation, episodes, policies, checkers, scoring and reports.
The package supplies a standalone stepping example, not a benchmark runner or
CLI integration.

Read [README.md](README.md) for startup and [DESIGN.md](DESIGN.md) for protocol
limits. Import the client directly; neither benchmark nor RLinf is a dependency.

## 1. Verify the client before integrating

From the repository root, run the client tests without ROS/Newton/GPU:

```bash
python3 -m pytest -q source/geniesim_ros/src/ros_ws/src/genie_sim_client/tests
```

These tests cover configuration, state ownership/layouts, command serialization,
reset epoch/body freshness, timeout/no-replay, process teardown and running the
standalone example while both applications and ROS/GPU imports are blocked.
They do not establish Newton physical correctness or benchmark task success.

For a real scene, first source the matching ROS environment and install the
standalone Newton engine. Build/source the client from the ROS workspace:

```bash
cd source/geniesim_ros/src/ros_ws
colcon build --packages-up-to genie_sim_client
source install/setup.bash
```

Use the existing scene assembly workflow to produce a valid manifest and assets.
The launcher must select `genie_sim_engine_newton` and explicitly set
`step_mode: lockstep`, positive `step_lockstep_ticks`, positive `physics_hz` and
`use_tf_render: true`. Start benchmark acceptance with `num_envs: 1`; batched
wire-layout tests do not establish multi-world support in the current engine.
See [the example launcher](config/launcher_client_newton.yaml).

Run the application-independent transport/state example:

```bash
python3 -m genie_sim_client.examples.newton_joint_step \
  --launcher /absolute/launcher.yaml \
  --manifest /absolute/stage/manifest.json --steps 10
```

It resets once, issues ten hold requests, closes the client and prints final
joint state as JSON. Add `--joint ACTUAL_ENGINE_JOINT --target 0.1` to command a
joint in native units: radians for revolute joints, metres for prismatic joints.
It produces no benchmark verdict or report. Test commands here still need live
Newton/GPU acceptance in a matching engine environment.

## 2. API and ownership

| API | Behavior |
|---|---|
| `ClientConfig.from_files(launcher, manifest, headless=True)` | Reads the launcher and manifest's scene reference; produces the engine command and requested pacing metadata |
| `SimulationClient(config, ...)` | Owns one serialized session; normally launches an isolated engine process group |
| `with client:` | Starts the session and closes it on context exit; client startup failures release resources |
| `client.snapshot()` | Reads immutable cached state without sending a physics command |
| `client.step(targets, render=False)` | Sends one full `(N,J)` joint-position target array and returns `StepResult` |
| `client.reset([0], seeds=[7], instances=None)` | Requests engine reset and waits for reset epoch and sufficiently fresh body state |
| `client.close()` | Releases the bridge and owned process group; leaves the host's shared ROS context alive |

Solver/profile, contacts, physics rate and ticks per request belong exclusively
to launcher/scene YAML. The manifest is an assembled artifact, not another
physics configuration route. Episode counts, policy selection, step limits,
scoring and report paths belong to benchmark.

Use actual `snapshot.schema.joint_names` and `body_paths`. Joint positions and
velocities are immutable float32 `(N,J)` arrays; body poses are `(N,B,7)` with
xyz followed by quaternion WXYZ. Camera images, contacts, AABBs, object classes
and task semantics are not supplied. Policies/checkers needing those inputs
require explicit application/engine integration.

For keyed arm/gripper/waist actions, flatten all groups into one full target
array and call `step` once. Validate all names, dimensions, duplicate assignments
and finite values before publishing. To hold omitted joints, retain the previous
commanded targets; initialize that buffer from the confirmed reset snapshot.
Observation/checker calls must not send an extra command. Settling requires
explicit hold requests rather than sleeping while the lockstep world is paused.

## 3. Suggested caller-side loop

The following is illustrative benchmark-owned code, not an additional client API.
It accepts direct joint-name/scalar actions for a single world. The caller supplies group/observation adaptation, policy reset, evaluation,
rendering and reports.

```python
from genie_sim_client import ClientConfig, SimulationClient


def drive_episode(client, next_action, *, max_steps=300, seed=7):
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if client.config.num_envs != 1:
        raise ValueError("start benchmark acceptance with one world")
    state = client.reset([0], seeds=[seed])
    indices = {name: i for i, name in enumerate(state.schema.joint_names)}
    held = state.joint_positions.copy()
    steps = 0
    for _ in range(max_steps):
        action = next_action(state)  # {} holds; None ends without a step
        if action is None:
            return state, steps, False  # not truncated
        candidate = held.copy()
        for name, value in action.items():
            candidate[0, indices[name]] = float(value)
        # Client rejects nonfinite/malformed targets before publishing.
        state = client.step(candidate).snapshot
        held = candidate
        steps += 1
    return state, steps, True  # cap reached; no additional policy call
```

The caller creates `ClientConfig`, enters `with SimulationClient(config) as client`,
and calls this function with its policy. The returned state is available for
evaluation without another physics request. The caller owns the policy/checker
and their resources. Treat truncation separately from a passing pose checker;
a truncated episode must not become successful merely because its final pose
passes. Choose explicit semantics for completion exactly at the step cap.

## 4. Rendering, reset and failures

Supply a benchmark-owned `RenderBridge` through the client constructor when
needed. `client.step(targets, render=True)` calls its `render(snapshot)` after
receiving physics state and returns the payload in `result.rendered`. The renderer
may apply poses and render/wait for images; it must not integrate physics. For
render-only observations, call the renderer with `client.snapshot()` directly.
Missing images or geometry must not silently count as success.

Engine reset is distinct from visual teleports. Seed/instance vectors match the
selected world IDs; `-1` retains the engine's no-change meaning. `reset(None)`
selects all worlds. Engines lacking `reset_epoch` are rejected. Reset restores
state but does not provide arbitrary scene mutation or new topology assembly.

Use one caller thread and an exclusive engine/control/reset source per ROS
domain/topic set. Absolute `/rlinf/*` reset topics are existing engine protocol
names. `spawn=False` attaches to an externally managed engine with matching
configuration during startup grace; the external owner is responsible for recovery.

After-send timeout or invalid state can mean physics has already advanced.
Stop the run, close the client and propagate the error; never replay an uncertain
action automatically. Rendering failure also cannot undo physics. An adapter
that starts a client and then reads an initial snapshot must close the client
if that read fails, including a failed adapter `__enter__`. Report partial runs
as execution failures and preserve previous completed outputs.

Completion is `arrival_v1`: new joint state followed by a sufficiently fresh body
message, with no command-ID acknowledgement. Startup free-running, separate
command latching and asynchronous GPU mirrors prevent proving exact command
ownership or committed K ticks. Publisher stamps are wall clock, not simulation
time; `observation_id` is a client read sequence. `config.step_duration` is the
requested K/physics_hz duration. `completion="committed_v2"` is unsupported.

## 5. Acceptance boundaries

Validate application adapters against cross-process DDS and the real Newton
engine, renderer inputs and task results. Test malformed actions before send,
seeded resets, held targets, policy-call limits, startup/transport/render/checker
failure cleanup and preservation of partial reports.

Dynamic scene mutation, arbitrary teleports, geometry and camera queries require
explicit engine or renderer support. Do not integrate an additional physics world
alongside client-driven Newton. Exact committed stepping is unsupported; the
[roadmap](../../../../docs/ROADMAP.md) records the protocol requirements.
