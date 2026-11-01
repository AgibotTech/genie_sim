# Client interface

## Ownership and initialization

The caller owns an injected `SimClient`. Closing its `GenieSimVecEnv` detaches
that wrapper; the caller still closes the connection. Neither operation stops
the remote server. Use the server launcher for process lifecycle management.

```python
from rlinf_geniesim.comm import make_client
from rlinf_geniesim.envs.vec_env import GenieSimVecEnv

client = make_client("zmq", address="tcp://127.0.0.1:5601", timeout_s=900)
try:
    client.set_backends({
        "physics": "isaac_fused",
        "physics_solver": "physx",
        "physics_solver_substep": 4,
        "render": "isaac_fused",
        "device": "cuda:0",
    })
    env = GenieSimVecEnv(client, to_torch=False, max_episode_steps=200)
    try:
        env.init({
            "task_name": "table_task_1_g2_op",
            "sub_task_name": "pick_block_color",
            "num_envs": 1,
        })
        obs, info = env.reset(seeds=[0])
    finally:
        env.close()
finally:
    client.close()
```

Constructing the wrapper against an idle server is valid. Initialize before
resetting or stepping. Backend settings lock at the first successful init;
`num_envs` is a workload setting in the task argument, not `set_backends`.
Additional init sections go in `env.init(task, config={...})`; `comm`, `task`,
and `backends` are not allowed inside that config mapping.

`env.attach_task(sub_task_name)` adopts an already initialized task without
rebuilding it. `env.init(...)` also switches tasks and refreshes schema, joint
indices, and instruction state. Do not reload through the raw client while a
wrapper still holds the old task layout.

If init or attachment fails after contacting the server, the wrapper disables
its cached layout. Complete init or attachment successfully before resetting
or stepping again.

## Private local servers

On Linux with pidfd support (kernel 5.3 or newer), `spawn_server` starts a private server in a prepared simulation
runtime. Each call owns one server process and returns its connected client.
The server runs under a separate interpreter, so the caller does not need ROS
in its Python environment. The mounted server checkout still needs the ROS
workspace, simulator dependencies, and assets described in the
[server preparation guide](../README.md#server-preparation).

```python
from rlinf_geniesim.spawn import spawn_server
from rlinf_geniesim.envs.vec_env import GenieSimVecEnv

with spawn_server(
    "stage0",
    ros_domain=215,
    code_dir="/workspace",
    assets_dir="/geniesim_assets",
    python="/usr/bin/python3",
) as server:
    server.client.set_backends({
        "physics": "isaac_fused",
        "render": "isaac_fused",
        "device": "cuda:0",
        "physics_solver_substep": 4,
    })
    env = GenieSimVecEnv(server.client, to_torch=False)
    try:
        env.init({
            "task_name": "table_task_1_g2_op",
            "sub_task_name": "pick_block_color",
            "num_envs": 1,
        })
        obs, info = env.reset(seeds=[0])
    finally:
        env.close()
```

Use the same task, reset, and step interfaces as for an external server.
Backend support and task restrictions are unchanged. `spawn_server` returns
when the idle control loop responds; task initialization has its own timeout.
It does not build the ROS workspace or install dependencies. Source locations
come from `code_dir` / `SIM_REPO_ROOT`, `RLINF_ROS_SETUP` (default
`/opt/ros/$ROS_DISTRO/setup.bash`, with `jazzy` as the fallback), and
`RLINF_WS_BUILD` / `GENIESIM_ENGINE_WS` (default `/tmp/ros_ws_build`). The server
interpreter defaults to `RLINF_SERVER_PYTHON` or `/usr/bin/python3` and must be
an absolute path to an executable interpreter; script wrappers such as
`python.sh` are rejected so the owned PID remains the server PID. This restriction
applies only to the server interpreter; backend launchers are unchanged.
The child retains `CUDA_VISIBLE_DEVICES`, removes the caller's Python/Conda/venv
environment, and sources ROS and the built workspace.
The caller's environment is unchanged.

Choose distinct `ros_domain` values in **215–232** for servers on the same ROS
network. This range is separate from the manual launcher's 0–101 range and
avoids Linux's default ephemeral ports; custom host port ranges require review
against the [ROS domain guidance](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Domain-ID.html).
All spawn callers on one host must share `run_root` (`RLINF_RUN_ROOT`, default
`/tmp`) for domain locking. The child holds the lock until it exits. The lock
does not coordinate other hosts, containers with separate filesystems, or ROS
applications outside this launcher. Each call generates a unique tag, runtime
directory, and Linux abstract ZMQ socket, even when callers reuse a stage label.
No TCP port or shared bootstrap file is used.

`server.stop()` is idempotent, closes the client, and waits for the server
and its setup helpers to exit, escalating to SIGKILL after `stop_timeout_s`
(default 15 seconds). Context-manager exit calls it automatically. Startup failures and interruptions
also stop the child. Once running, the server watches its owner's PID and start
time; owner loss requests normal cleanup, with a 15-second forced-exit fallback.
Backend cleanup and parent-death binding use the existing implementation.
A forced exit cannot guarantee graceful backend resource cleanup; `stop()`
also runs the existing PID-scoped shared-memory cleanup when the owner is alive.
No automatic server restart is attempted.
Before the server watchdog starts, startup cleanup depends on the caller
remaining alive. Abrupt caller termination during a hung startup may leave
setup processes and the domain lock requiring external cleanup. This includes
ROS sourcing, Python imports, and transport construction. A successful return
from `spawn_server` means the watchdog is already running.

Keep each client on its creating thread. The server's log and comm configuration
remain under `server.run_dir`; `log_dir` can redirect the server log. Backend
logs retain their existing locations. `ready_timeout_s` defaults to 900 seconds,
`request_timeout_s` to 60 seconds, and `jpeg_quality` to 85 (0 disables JPEG).
The standalone client wheel includes this launcher, but no simulation runtime.
The legacy `GenieSimVecEnvConfig.spawn_server` flag remains unsupported; call
this function and inject `server.client` instead.

External servers retain their original lifecycle: closing a client or vector
environment does not stop the server. The manual launchers do not set the owner
variables and do not enable this watchdog.

## Task selection

ZMQ task metadata advertises the [26 released G2op tasks](../README.md#supported-tasks)
under `sub_tasks` and the corresponding `g2op_if`, `g2op_manip`, and
`g2op_spatial` entries under `suites`. Use a suite entry's `task_name` and
`sub_task_name` together when initializing: they select the official G2op
robot/background pairing. ZMQ initialization and reload reject any pair outside
this catalog before releasing the active task. This includes other robot
variants, alternate backgrounds, and the two package-sorting tasks.

Custom integrations can use the `task_loader` helpers and call
`scripts.sim_server.build_handler` directly, then embed the resulting handler
with SHM. These lower-level APIs retain the shared benchmark's broader
task-loading capabilities. Runtime randomization remains available for the
released tasks.

## Actions and observations

`env.step(actions)` accepts joint-position targets shaped `(N, chunk, A)` and
returns `(obs, rewards, terminations, truncations, infos)`.

| Value | Contract |
| --- | --- |
| `A` | `env.action_dim`: left/right arms, grippers, and optional waist |
| `obs["states"]` | `(N, env.state_dim)` float32: arms, grippers, waist, head |
| Camera observations | `(N, H, W, 3)` uint8 when enabled; keys follow `image_keys` |
| `obs["task_descriptions"]` | One current instruction per environment |
| Rewards | `(N, chunk)` per-step rewards |
| Terminations/truncations | `(N, chunk)` boolean flags |
| `infos["chunk_success"]` | `(N, chunk)` success flags before chunk-boundary reset |

Joint counts come from the resolved robot schema. Do not assume every robot
has G2's state dimension. Uncontrolled joints hold the reset target.
`to_torch=True` converts numeric outputs to tensors on the requested device.

The default path returns the chunk's last observation and renders its last
frame. Compatible ZMQ clients send a chunk in one RPC; other clients use
individual steps. `return_per_step_obs=True` requests all observations.
With auto-reset enabled, a done flag is latched through the chunk and reported
in its last column. Reset occurs once at the boundary. The returned observation
is the reset observation, with the terminal frame in `infos["final_observation"]`
and terminal metrics in `infos["final_info"]`; the corresponding `_final_*`
masks identify affected environments.

## Reset and randomization

`env.reset(...)` returns `(obs, infos)` synchronously with the selected
environments ready to act. The compatibility `block` argument is accepted but
has no effect.

- `env_ids` selects a subset; omitted means all environments.
- `seeds` assigns explicit per-environment seeds. `seed` derives distinct seeds
  from one scalar; use `seeds` when a group needs identical seeds.
- Seed `-1` uses authored values without seeded perturbations. A nonnegative
  seed enables the configured randomization.
- `instance_ids` switches within a compatible instance group; `-1` retains
  the current instance. Before the first reset, the wrapper uses its configured
  `instance_id` (default `0`) unless `instance_ids` explicitly selects another.
- `instruction_ids` pins instruction IDs; omitted or `-1` entries are sampled
  from that instance's instruction table.
- Per-environment arrays can align with the selected subset or the full batch.
  When their lengths are equal, they follow the order of `env_ids`.

If a reset request fails or returns invalid state, step remains blocked until
the affected environments have been reset successfully. Input validation
errors raised before sending the request leave the session usable.

See [randomization](GENERALIZATION_DESIGN.md), [readback](RANDOMIZATION_READBACK_API.md),
and [instance pose/reset contracts](INSTANCE_BODY_POSES.md).

## Transports

SHM and ZMQ retain the same local numeric step/reset contract. SHM uses fixed
shared-memory layouts and semaphores; it has no task initialization/reload or
metadata/readback verb. An embedding application can construct a handler and
serve it with `make_transport("shm", ...)`.

After an SHM timeout, the pending request still owns the shared buffers. The
client, including a replacement connection, waits for that request to finish
before sending another. A timeout does not cancel simulation work.

ZMQ adds task control-plane requests and can cross machine boundaries. A
server owns one active task world. Clients must coordinate access to that
world; a connection is not a separate environment allocation.

The legacy `GenieSimVecEnvConfig` constructor, 2D action promotion, and
`chunk_step` alias remain for existing callers and emit deprecation warnings.
New code should use an injected client and 3D action chunks.
