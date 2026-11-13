# genie_sim_engine_libuipc — Agent Development Guide

Update this file and DESIGN.md when configuration or lifecycle behavior changes.

## Ownership and layout

This package owns IPC native stepping, scenes and CPU/feedback inputs and
implements the existing core PhysicsEngine ABI.

| Path | Responsibility |
| --- | --- |
| engine.py, episode_sequence.py | Fixed steps, native recovery and recorded episode ownership |
| driver.py, config.py | SDK-to-native stepping and solver settings |
| scene.py, geometry.py, body_bindings.py, usd.py | Objects, authored mappings and USD policy |
| cpu/ | CPU MuJoCo model, worker, collision policy and optional viewer |
| feedback.py | Measured-joint validation and stale-input selection |
| deformable_publisher.py | Existing topology/frame producer |
| solver/ | Vendored SolverIPC, license and provenance |
| session.py, scripts/ | Core session adapter and executable |
| assemble/ | Local asset pipeline, importer and scene implementations; no sibling backend imports |
| examples/ | Launcher-configured particle capture; no Docker or installer orchestration |
| test/ | Explicit native_smoke.py GPU probe |

## Entry points

Engine token: libuipc. Solver: ipc. One fixed substep and one world.
Executable: genie_sim_engine_libuipc_node.py.
Assembly module/pipeline: genie_sim_engine_libuipc.assemble / libuipc.
Scene block: libuipc:. Launcher parameter names follow the existing core ABI.

## Invariants

1. Never import genie_sim_engine_newton or genie_sim_engine_isaacsim, including
   from assembly subprocesses. Copy required helpers locally and record source
   history/adaptations in DESIGN.md. Upstream newton SDK dependencies are intentional.
2. Never move backend runtime imports or IPC fields into core contracts.
3. Native World owns dynamic state; SDK array writes alone cannot reset it.
4. Preserve q/qd indices, authored mesh/body offsets and quaternion conventions.
5. Physics configuration comes from launcher/scene YAML and reachable profiles.
6. Preserve vendor license/provenance. Format Python with black --line-length 120.
7. Keep SDK upgrades separate from extraction.
8. Include this package in ordinary ROS builds; select it through launcher/scene
   configuration.
9. Native GPU smoke remains an explicit invocation, not a test_*.py collected by
   normal pytest/CTest. Examples read engine settings from launcher YAML, own
   fresh output paths and close engine resources on failure.

## Verification

See [examples and capture](examples/README.md) for the explicit native probe.
Runtime acceptance and current limits are described in [DESIGN.md](DESIGN.md).

## Recorded in-process sequences

The three Wuji benchmark tasks use one engine, renderer and CPU MuJoCo worker
across their material layouts. `libuipc.episode_sequence` in the startup scene
YAML preloads each episode's fresh `state_recording` and `material` mapping.
The first entry must match the ordinary initial recording/material settings.
Only a configured index crosses the wire; clients cannot supply physics settings,
paths or material plans at runtime. Geometry/solver changes remain unsupported.

The owner serializes begin/finish/reset between native steps. Finish pauses
physics and verifies/closes the archive while retaining the world. Reset requires
finish, recovers native World frame 0, restores CPU MuJoCo integration/controls
without respawning its worker, rotates the archive and advances the source epoch.
The next episode stays paused until begin; physics runs normally during that
episode's capture and inference. Every archive starts at tick/time 0.
See [DESIGN.md](DESIGN.md) and the [client wire contract](../genie_sim_client/IPC_CLIENT.md).
CPU tests live in `test/test_episode_sequence.py` and `test/test_cpu_reset.py`;
the explicit end-to-end GPU probe is
`source/geniesim_benchmark/tests/native_libuipc_sequence_smoke.py` from the repo root.

Preloaded `robot_base_pose` entries may vary x/y only. Material-only entries keep
native recover(0); a changed x/y explicitly reconstructs the native World inside
the same engine process, validates unchanged geometry/topology, and reloads the
YAML-derived CPU model inside the same CPU worker PID. The deformable transport,
ROS owner and renderer remain attached. Robot base z/orientation, mesh geometry,
solver profiles and joint contracts remain fixed. This extension requires the
CPU viewer to be disabled. Its full native/renderer acceptance must be checked
with an explicit diagnostic before batch deployment.
