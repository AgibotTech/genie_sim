# genie_sim_engine_isaacsim

Kit-hosted PhysX and Newton backends, plus URDF conversion and scene assembly.
Newton standalone and native IPC are separate backend packages.

| Runtime | PhysX | Isaac Newton |
|---|---|---|
| Isaac Sim 6.2 | AS3 conversion and isolated G2 live smoke passed | Public config/scene APIs and isolated G2 live smoke passed |
| Isaac Sim 5.1 / 4.5 and 5.1 + 6.2 combo | Deprecated (E.O.L.); flat-asset code retained | Outside current qualification |

AS3 output must select `Physics=physx` before overlays or body/joint discovery.
That variant includes the common physics payload while retaining DriveAPI.
Assembly persists the selection in the generated robot root layer. Runtime
selects it on the scene reference before the first robot app update, without
editing the source asset. Flat assets with no variant set remain supported.
Legacy assets with `Physics=PhysX` are also supported; unknown physics variants
and variants that compose no rigid bodies are rejected.

Conversion compares movable joints with the source URDF, normalizing the
existing `idxNN_` prefixes. Missing or duplicate joints, missing physics
payloads, and incomplete derived pad/coupler annotations are hard errors.
A conversion that writes zero required overlays cannot pass as a valid asset.

For Isaac Newton, launcher `physics_solver_substep` configures
`NewtonConfig.num_substeps`; `physics_solver_iterations` authors
`mjc:option:iterations` on the PhysicsScene before reset. Zero keeps the upstream
substep default or the authored scene iteration value. Solver constructor
options are applied to a fresh `MuJoCoSolverConfig` instance before the stage is
created and verified through `get_newton_config`. Dataclass defaults are never
modified. The live startup banner reports the resulting solver and options.
Spherical-joint axes retain their imported gains and force limits; the scalar
robot-joint tuning profiles do not apply to them.

See [AGENTS.md](AGENTS.md) for implementation and verification, and the
[platform roadmap](../../../../../../docs/ROADMAP.md) for remaining release gates.
