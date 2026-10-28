# Engine comparison

Launchers select the backend package/executable and its `physics_engine` token.
Solver settings belong to launcher YAML. Core owns the
[engine factory and contracts](../genie_sim_engine_core/engine/base.py), session,
run loop, pacing, manifest schema and ROS transport.

## Engine paths

| Engine token | Implementation | Physics | Support |
| --- | --- | --- | --- |
| `isaac_physx` | [IsaacPhysXEngine](../../genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/kit/isaac_physx.py) | Kit PhysX rigid simulation | Tier 1 G2 path |
| `isaac_newton` | [IsaacNewtonEngine](../../genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/kit/isaac_newton.py) | Kit Newton MuJoCo-Warp wrapper, rigid only | Experimental, startup-gated |
| `newton` | [NewtonHeadlessEngine](../../genie_sim_engine_newton/genie_sim_engine_newton/engine/newton/engine.py) | Kit-free Newton and solver adapters | MuJoCo-Warp rigid Tier 1; deformable combinations experimental |

`launcher_newton_mjwarp` selects MuJoCo-Warp. Other Newton launchers select the
Featherstone/VBD, AVBD, MuJoCo/VBD or MuJoCo/XPBD compositions. A solver name does
not establish that a scene is supported; use the
[scene/backend matrix](../../genie_sim_bringup/README.md#scene-testing).

## Control and state

| Concern | Isaac PhysX | Isaac Newton wrapper | Standalone Newton |
| --- | --- | --- | --- |
| Articulation | Isaac articulation/tensor view | Newton-backed Isaac view | Newton model, control and state arrays |
| Joint commands | Articulation targets | Wrapper targets | Adapter control buffers; MuJoCo-Warp PD or solver-specific control |
| Joint state | Live articulation readback | Wrapper view readback | Completed state joint arrays |
| Body state | Composed stage/tensor poses | Wrapper body poses | Completed `body_q` |
| Sysid precedence | After class tuning | After class writes | After class/mimic fallback, before compile |
| Rendering | Separate OVRtx or Isaac renderer | Separate renderer | GL/RViz/inline OVRtx or separate renderer, according to launcher |

The maintained PhysX/standalone MuJoCo-Warp paths apply selected SI joint values.
Isaac PhysX passive overlays require 6.2 native joint-axis support. See
[joint parameters](../../../../../docs/JOINT_PARAMS.md) for the field matrix,
unit conversion, mimic-follower constraints and unsupported consumers.

## Shared and backend-owned behavior

Core `EngineSession` loads the launcher-selected profile and scene/manifest,
initializes transport, creates the engine and runs `EngineRunLoop`. Publishers
and pacing follow shared contracts. Post-step hooks extend session behavior
without moving backend runtime logic into core.

Backends own solver construction/stepping, initial state, reset mechanics,
state extraction, deformables and renderer integration. Newton adapters own
solver-specific controls and graph capture. Kit runtime setup is shared by
PhysX and the experimental wrapper.

[Asset consumption](asset_consumption.md), [parameter application](param_injection_eval.md)
and [step modes](../../../../../docs/STEP_MODES.md) describe the relevant
boundaries in detail.
