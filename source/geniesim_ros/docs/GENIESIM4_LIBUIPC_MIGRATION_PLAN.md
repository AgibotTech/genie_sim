# Libuipc roadmap

This roadmap records remaining acceptance and extension work for the independent
`genie_sim_engine_libuipc` backend. Entries are not release promises. Current
architecture, configuration and lifecycle belong in the
[backend design](../src/ros_ws/src/genie_sim_engine_libuipc/DESIGN.md).

## Current baseline

The package implements the existing core engine/session ABI and owns local
assembly, IPC stepping, CPU/measurement inputs and deformation publication.
Neither sibling backend package is a dependency. Normal workspace builds include
it; scene and launcher selection determines whether it runs. Native GPU contacts
and camera rendering require runtime qualification.

## Acceptance gates

| Gate | Work | Required evidence |
| --- | --- | --- |
| Compatible native environment | Select supported Python, SDK, CUDA, native artifact and GPU | Native cloth/rigid/mixed smoke passes with recorded environment identity |
| Runtime selection | Build/install, select libuipc launchers and regenerate manifests | Package/executable discovery and clean single-world startup |
| Four task scenes | Run hanger, cloth, paper and bag with declared profiles | Finite states, expected particle ranges, initial clearance and correct contact |
| Lifecycle | Repeated reset, pause/resume, worker failure and shutdown | Native state/targets recover, epochs advance only on success, resources close |
| Renderers | OVRtx and Isaac separate renderers with deformation | Completed state matches mesh/image output, including after reset |
| Client integration | Use shared contracts with application-owned task orchestration | Verified observations, task semantics and explicit CLI dispatch |
| Deterministic CPU coupling | Define CPU tick ownership and coordinated reset | Equal action/seed sequences meet trajectory tolerances across request delays |
| Multiple worlds / performance | Add world isolation and partial reset before batching | Independent state/reset/rendering plus measured throughput and memory |

## Verification entry points

On a compatible native runtime:

```bash
python3 source/geniesim_ros/src/ros_ws/src/genie_sim_engine_libuipc/test/native_smoke.py
```

The explicit native probe exercises `World.init/advance/retrieve/recover(0)`.
Importing `uipc` is insufficient, and passing this probe alone does not certify
the complete backend. See [examples and capture](../src/ros_ws/src/genie_sim_engine_libuipc/examples/README.md)
for launcher-based inputs and per-run outputs.
