# rlinf_geniesim — Agent Development Guide

> Maintenance contract: when the client API, the ZMQ/SHM wire contract, the
> server CLI, the supported task set or the Docker profiles change, update this
> guide, [README.md](README.md) and the affected reference page in the same diff.

## 1. What this is

`rlinf_geniesim` is the simulation side of RLinf post-training. A ZMQ or SHM
server runs a GenieSim task with its physics backend, render bridge and
benchmark success checker; `GenieSimVecEnv` maps RLinf joint targets and
observations onto it. Pico teleoperation drives the same server for
interactive debugging. Training algorithms, policies and RLinf worker placement
belong to RLinf.

It is a separately maintained module (see [source/AGENTS.md](../AGENTS.md)):
`geniesim bootstrap` does not install it. The same directory ships two
distributions: the lightweight `rlinf-geniesim-client` wheel for RLinf
environments, and the server that runs from the checkout inside the
post-training image.

## 2. Layout

```text
rlinf_geniesim/
├── envs/vec_env.py       # GenieSimVecEnv: client-side batched environment
├── comm/                 # client/transport pairs: zmq/ (task control plane), shm/ (local step/reset)
├── sim_handler.py        # GenieSimHandler: step/reset/checker orchestration
├── scripts/sim_server.py # server entry point, task init/reload, backends lock
├── physics/backends/     # Newton (MuJoCo Warp, AVBD, Featherstone), Isaac PhysX/Newton, fused Isaac
├── render/bridges/       # Isaac tiled, OVRTX, null renderer
├── child_lifetime.py     # binds backend child processes to the server's lifetime
├── spawn.py              # optional private local server launcher
├── owner_watchdog.py     # opt-in lifetime binding to the spawning process
├── task_loader.py        # task/scene manifests
├── task_pairing.py       # official background/robot pairing from the released suites
├── randomize/            # domain randomization plans and task defaults
├── eval/                 # benchmark checker runner and Kit stubs
├── teleop/               # Pico signal, IK bridge, web console
├── scripts/              # launchers, init_task.py, smoke_test.py, teleop entry point
├── config/               # comm, server plans, task-init examples
├── docs/                 # interface, design notes, pre-release testing
└── tests/                # explicit simulator integration checks and USD maintenance tools
```

## 3. CLI and API surface

| Surface | Purpose |
| --- | --- |
| `geniesim docker --profile post-train\|post-train-teleop build\|up\|into\|down` | Server and teleop images and containers (host) |
| `geniesim post-train server start\|status\|stop\|logs` | Batch server launcher inside the server container |
| `geniesim post-train teleop` | Pico teleop client and web console |
| `scripts/init_task.py` | Initialize or reload a task on an idle server |
| `scripts/smoke_test.py` | Finite model-free chunk/reset check against a server |
| `GenieSimVecEnv`, `make_client("zmq"\|"shm")` | Client API used by RLinf; see [docs/interface.md](docs/interface.md) |
| `spawn_server` | Own a private local server in a prepared runtime; see [lifecycle and limits](docs/interface.md#private-local-servers) |
| `make_transport`, `build_handler_from_init` | Server-side embedding, including SHM |

Environment variables are listed in [ENV_VARS.md](ENV_VARS.md).

## 4. Architectural rules

1. The client wheel imports only the modules listed in `pyproject.toml`. Do not
   import server, ROS, simulator or PyTorch dependencies from them.
2. ZMQ owns the task control plane: idle startup, init, reload, metadata and
   randomization readback. SHM carries only local step/reset for a handler the
   embedding application built.
3. The first successful init locks the backend settings for that server
   process. Changing backends requires a new server.
4. `isaac_fused` is selected for both the physics and the render field.
   `isaac_physx` and `isaac_newton` support one environment and reject seeded
   resets and instance switching.
5. Init deep-merges onto the active configuration. Callers that switch tasks
   send every field they rely on (instances, superset, pairing) explicitly
   rather than omitting it.
6. Only the official G2op task/scene pairings listed in the README are accepted
   by ZMQ init and reload. Keep the README task list and the server metadata in
   agreement.
7. Backend child processes are launched through `child_lifetime.bind_to_server`
   from the main thread, so they exit when the server is killed.
8. Keep the client wheel revision aligned with the server deployment.

## 5. Skills index

This module ships no `skills/` yet. Operator workflows live in the guides below.

## 6. Common workflows

Post-training runs on two nodes, in order:

1. **Simulation node**: build the post-training image and start the server
   ([server preparation](README.md#server-preparation),
   [start and initialize](README.md#start-and-initialize)).
2. **Training node**: build the training image, which installs the client, and
   start training against the server. For RLinf, see the
   [RLinf GenieSim guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/geniesim.html).

Other workflows, as needed:

| Workflow | Guide |
| --- | --- |
| Install the client | [CLIENT_README.md](CLIENT_README.md) |
| Debug a task with Pico teleoperation | [TELEOP_USAGE.md](TELEOP_USAGE.md) |
| Pre-release acceptance | [docs/release_testing.md](docs/release_testing.md) |
| Randomization | [docs/GENERALIZATION_DESIGN.md](docs/GENERALIZATION_DESIGN.md) |

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| Init rejected with a backend lock error | The server was initialized with other backends; start a fresh server |
| Init rejected for a task pairing | Use a pairing from the server's `task_metadata()['suites']` |
| First init takes minutes | Rest-state bake and kernel compilation on first use; later inits reuse the cache |
| `barrier stuck` or step timeouts | Inspect `/tmp/rlinf_isaac_tiled_*.log` or the fused log for render times |
| Teleop stepping stays paused | A reload or reset failed; reload successfully to rebuild the cached layout |

## 8. Verification recipes

| Scope | Recipe |
| --- | --- |
| Simulator integration and USD maintenance | [tests/README.md](tests/README.md) |
| Image build, server startup and teleop on a simulation host | [docs/release_testing.md](docs/release_testing.md) |
| One running server | `scripts/smoke_test.py --address ... --attach <task>` |

Validate physics and render backends with the simulation-host runtime checks.

## 9. Do not

- Add server or simulator imports to the client modules.
- Rely on omitted fields when reloading a task.
- Change shared base-image Torch or physics pins for this profile; see
  [docker/AGENTS.md](../../docker/AGENTS.md).
- Start a second server on a port that is already in use.
