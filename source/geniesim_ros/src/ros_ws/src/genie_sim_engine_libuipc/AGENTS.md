# genie_sim_engine_libuipc — Agent Development Guide

Update this file and DESIGN.md when configuration or lifecycle behavior changes.

## Ownership and layout

This package owns IPC native stepping, scenes and CPU/feedback inputs and
implements the existing core PhysicsEngine ABI.

| Path | Responsibility |
| --- | --- |
| engine.py | Registration, fixed steps, readback and lifecycle |
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
