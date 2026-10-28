# genie_sim_engine_newton

The Newton-standalone physics backend: a `PhysicsEngine` subclass, its
solver adapters, and the Kit-free build pipeline that gets a scene from
USD to a stepping model.

The C++ ROS 2 bridge and the `PhysicsEngine` ABC live in
[`genie_sim_engine_core`](../genie_sim_engine_core), which this package
depends on. The Isaac Sim backends live in
[`genie_sim_engine_isaacsim`](../genie_sim_engine_isaacsim).

## Entry point

| Script | When to use |
|---|---|
| `scripts/genie_sim_engine_newton_node.py` | Newton-standalone (`physics_engine:=newton`), Kit-free |

Launched indirectly — `genie_sim_bringup`'s `app.launch.py` resolves the
package/executable pair from the launcher YAML's
`launcher.physics.engines.<key>` block.

## Engine

| `physics_engine` | Class | Notes |
|---|---|---|
| `newton` | `NewtonHeadlessEngine` (`engine/newton/engine.py`) | Newton-direct, Kit-free. Newton cloth-capable backend. Solver picked by `physics_solver:=`. |

The installed Newton/MuJoCo versions follow the parent workspace pin.
Control uses the official `joint_target_q` / `joint_target_qd` attributes.
Model construction selects Newton 1.6.1's DOF target layout so command
and reset indices share the same layout.
The layout selection is scoped to model construction and restores the
upstream global default afterwards.

For an empty `scene:`, the assembler's `/World/GroundPlane` is a visual-only
USD mesh at z=0. Model import excludes that exact mesh because Newton adds its
own visible collision plane at z=0; drawing both causes black-line flicker in
ViewerGL. Authored worlds and physics-bearing floors remain imported, and the
USD stays intact for Isaac/OVRtx. Coverage: `tests/test_generated_ground.py`.

MuJoCo-Warp adapters share the selective `hand_[lr]_` self-collision policy
in `common/hand_collision.py`. Before `put_model`
builds its collision pair list, contact-enabled hand geoms receive all affinity
bits and distant hand-related excludes are cleared. Non-hand masks/excludes,
visual-only geoms, and parent/grandparent exclusions stay intact. Apply the same
policy to the MJCF dump because Newton exports the original MjSpec. This policy
does not change drive or contact-material parameters.

### Solver adapters

`engine/newton/adapters/` — one `SolverAdapter` (`adapters/base.py`) per
solver, absorbing solver-specific PD prep, construction, and target
buffers so `engine/newton/lifecycle.py` stays solver-agnostic.

| module | solver |
|---|---|
| `mujoco_warp.py` | MuJoCo-Warp, single world |
| `mujoco_warp_multi_env.py` | MuJoCo-Warp, replicated worlds |
| `featherstone.py` | Featherstone, velocity injection |
| `avbd.py` | AVBD |

## Package layout

```
genie_sim_engine_newton/
├── assemble/        # newton_extras stage of the core assemble pipeline
├── common/          # gripper/joint/object classification, MJCF post-process, session
├── engine/newton/   # the backend itself
│   ├── engine.py        # NewtonHeadlessEngine — the only concrete subclass
│   ├── engine_base.py   # _NewtonStandaloneBase: mixin composition + shared init
│   ├── lifecycle.py     # build, reset, teleport, episode_*
│   ├── adapters/        # SolverAdapter per solver (see table above)
│   ├── setup/           # build-phase mixins, file order = execution order
│   └── visualizers/     # RViz markers + InlineOvrtxVisualizer
├── tools/           # standalone checks, run without a ROS runtime
└── warmup.py
scripts/
└── genie_sim_engine_newton_node.py
```

Build phases compose into `_NewtonStandaloneBase` via plain multiple
inheritance — no metaclasses, no auto-discovery. See
`engine/newton/setup/__init__.py` for the table-of-contents.

## Config surface

The engine accepts a launcher YAML and a scene YAML, and nothing else —
see [`geniesim_ros/AGENTS.md`](../../../../AGENTS.md), "one knob, one
route". Physics tuning goes through the profile named by the
`physics_params_file` ROS parameter, not through module constants or
environment variables.

### Per-instance reset poses

RLinf task artifacts must carry `instance_switch.body_pose_version: 1` and
`instance_body_poses["<instance>"]["<slot>/<body-path>"] = [x,y,z,qw,qx,qy,qz]`.
These are composed USD **body** poses in meters, local to each Newton world;
the key is the exact body label below `/World/Objects/` (the slot itself is
also valid when it owns the rigid body). `instance_poses` remains locator
metadata and is never combined with a build-instance asset offset. Every
object FREE joint needs a row; extra fixed/kinematic/articulated-body rows
are allowed. Startup and reset reject legacy artifacts, missing rows,
nonfinite poses and nonunit quaternions with a re-bake error. An omitted
instance or `-1` resets that world's current instance, initially the build
instance; only successful resets update the per-world selection.

Newton FK uses `body = X_p * joint_q * inverse(X_c)`, so the consumer writes
`inverse(X_p) * body * X_c`, converting wxyz to Newton xyzw. A CPU check with
the installed USD importer confirmed ordinary free bodies have world parent
and identity anchors; a nested offset composed to `(0.9, 2.2, 3.0)` correctly.
A second real `newton.eval_fk` check with translated/rotated parent and child
anchors recovered the requested `(3, 4, 5)` identity body pose exactly.
Object anchor metadata is cached per model; no live physics pose is cached
as an asset offset. Native replication has zero spacing, so this path adds
no renderer/GridCloner origin. The regression suite is
`source/rlinf_geniesim/tests/test_engine_instance_robot_pose.py` plus
`test_engine_reset*.py`; neither requires a GPU.

## Detailed docs

- [docs/scene_newton_keys.md](docs/scene_newton_keys.md) — scene YAML `newton` keys: robot contact policy and static world-AABB publishing.

- [docs/newton_quirks.md](docs/newton_quirks.md) — Newton/MuJoCo-Warp behaviours worth knowing before debugging
- [docs/ovrtx_sync.md](docs/ovrtx_sync.md) — the OVRtx in-process visualizer design
- [../genie_sim_engine_core/docs/pipeline.md](../genie_sim_engine_core/docs/pipeline.md) — assemble pipeline, stage cache, robot routes, manifest, collision policy, material overrides, the base pin (`pin_base_to_world` / `pin_base_mode`), mimic joints
- [../genie_sim_engine_core/docs/engines.md](../genie_sim_engine_core/docs/engines.md) — engine comparison: solver, joint control, state readback, cloth, diagnostics
- [../genie_sim_engine_core/docs/asset_consumption.md](../genie_sim_engine_core/docs/asset_consumption.md) — how each engine path consumes the assembled `robot.usda` + payloads
- [../genie_sim_engine_core/docs/param_injection_eval.md](../genie_sim_engine_core/docs/param_injection_eval.md) — do `isaac_newton` and newton-standalone share the mujoco-warp param injection? (no — two implementations; the USD overlay is the only shared bridge)
- [../genie_sim_engine_core/docs/perf.md](../genie_sim_engine_core/docs/perf.md) — stats glossary, bottleneck signatures, tuning levers
- [../genie_sim_engine_core/docs/physics_knobs.md](../genie_sim_engine_core/docs/physics_knobs.md) — every physics parameter and where it is homed

Generic scene-plugin and particle-render functionality is available for Newton scenes.

Backend independence is checked by
`genie_sim_engine_newton/tools/test_backend_independence.py`: it rejects IPC
solver routing and imports without requiring libuipc to be installed.

### Recording snapshot consumers

Particle host mirroring is opt-in through a callable scene-plugin
`on_record_snapshot`. An optional `record_snapshot_enabled()` query can disable
copies and callbacks dynamically; without it, existing snapshot plugins remain
opted in. Newton allocates the particle mirror on the first active step. On
disable it waits for pending copies before releasing the pinned buffers; a
later activation starts a fresh mirror. This transition-only wait does not run
on ordinary active or inactive steps. Reset/teleport flushes an existing active
particle mirror, while joint/body mirror handling is unchanged.

Chef reports whether its YAML-configured food publisher is live. Recording
disabled, publisher initialization failure, or publisher shutdown therefore
requires no particle host copies or snapshot-only body readback. Rendering and
other debug publishers keep their own paths. CPU coverage lives in
`genie_sim_engine_newton/tools/test_record_snapshot.py`.

## Runtime sysid precedence

`common/joint_params.py` resolves explicit scene `robot.joint_overlays` package
URIs, then uses core's `JointParamResolver`. The stock and multi-env MuJoCo-Warp
adapters apply name-keyed values after every class/mimic fallback and before
solver construction. `joint_damping`, `joint_friction` and `joint_armature` are
SI arrays: `joint_damping` needs no degree compensation. Preserve
explicit zeros and keep mimic followers undriven. Fields absent from overlays
retain class/importer behavior. Featherstone/AVBD reject selected overlays because they do not implement this application contract. Regression coverage is in
`../genie_sim_engine_isaacsim/tests/test_runtime_joint_params.py` and its explicit
6.2 smoke runner.

## Frozen evaluation observations

The optional Isaac snapshot visualizer captures real Newton geometry while physics is paused. `pause()` flushes final joint/body host mirrors before invoking a render-only callback, including repeated pauses; it never steps physics. Dynamic recording mirrors retain their existing enable/disable behavior. The node owns image/video/worker resources and closes them on startup failure or exit. See [snapshot observations](docs/isaacsim_observation.md).

`wok_evaluation_json` explicitly enables a read-only task observer on the existing lifecycle node. It scores each completed physics tick, publishes versioned progress and fails closed on geometry/contact/transport errors. The dedicated procedural evaluation scene supplies the required geometry. Earned points remain local to evaluation and output; no checker truth is sent to the model. These hooks do not add command-correlated step acknowledgements to arrival_v1.

The snapshot worker is an optional caller-supplied `genie_sim_render` installation, resolved only when selected. Bringup already includes that renderer. Do not add an unconditional Newton-to-render dependency: render depends on the Isaac package, which composes Newton assembly and would create a package cycle.
