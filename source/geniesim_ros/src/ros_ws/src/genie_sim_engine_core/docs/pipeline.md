# Prepare-stage pipeline

Before any engine script is launched, an **assemble pipeline** owned by
:mod:`genie_sim_engine_core.assemble` prepares the robot USD, the
Newton sidecar, and the scene manifest. The physics node itself is a
pure consumer — it only reads `manifest.json` and references the
already-prepared `robot.usda` into the stage.

## Pipeline overview

One process invokes the core CLI, which dispatches the appropriate
per-backend pipeline. The `isaac_newton` pipeline composes three
`AssembleStage` subclasses registered via `@Pipeline.register` in the
backend packages:

```
python -m genie_sim_engine_core.assemble \
   --scene <scene.yaml> --output-dir <stage_dir> --base-path <assets>
   --physics-engine isaac_newton
   --pipeline-module genie_sim_engine_newton.assemble
   --pipeline-module genie_sim_engine_isaacsim.assemble
```

Stage sequence inside the pipeline runner (`PipelineRunner.run` in
:mod:`genie_sim_engine_core.assemble.pipeline`):

```
[urdf_to_usd]     -->  [isaac_scene]     -->  [newton_extras]     -->  [genie_sim_engine_newton]
 UrdfToUsdStage         IsaacSceneStage        NewtonExtrasStage        (physics loop; consumer)
 → robot.usda           → manifest.json        → newton_solvers.json    reads manifest.json
                        → render_layer.usda    (skipped if no cloth)
```

- `UrdfToUsdStage` lives in `genie_sim_engine_isaacsim.assemble` and
  spawns a fresh subprocess for Isaac Sim's URDF importer (Kit's
  `SimulationApp` is a per-process singleton — the subprocess isolates
  its lifecycle from the pipeline runner).
- `IsaacSceneStage` lives in `genie_sim_engine_isaacsim.assemble` and
  spawns another fresh subprocess to compose `render_layer.usda`. Its
  manifest dict is forwarded through
  :data:`~genie_sim_engine_core.assemble.RAW_MANIFEST_SENTINEL` so core
  writes the file atomically.
- `NewtonExtrasStage` lives in `genie_sim_engine_newton.assemble`;
  self-skips when the scene YAML has no `newton.entries`.

Stages inherit skip behaviour from
:class:`~genie_sim_engine_core.assemble.stage.AssembleStage`:

- `UrdfToUsdStage.skip_reason` returns non-`None` when
  `robot.robot_source.urdf` is absent (pre-baked USD route).
- Cache-hit skip for the URDF stage is enforced by the launcher passing
  `--skip-stage urdf_to_usd` when `<stage_dir>/urdf.sha256` matches.

The launcher scaffold is
`make_assemble_pipeline` in
[genie_sim_bringup/launch/utils.py](../../genie_sim_bringup/launch/utils.py),
called from `physics_backend.launch.py`,
which is composed by [app.launch.py](../../genie_sim_bringup/launch/app.launch.py).

All stages share a `stage_dir` derived from the scene YAML's filename:

```
./assets/scenes/<scene_yaml_stem>/
    robot.urdf         <- intermediate (UrdfToUsdStage output)
    robot.usda         <- physics-grade robot USD
    urdf.sha256        <- cache stamp; enables --skip-stage urdf_to_usd
    render_layer.usda  <- cameras / RenderProducts
    manifest.json      <- what the engine scripts actually read
    newton_solvers.json <- cloth/soft/particle sidecar (optional)
    pkg/<pkg>/...      <- symlinks for package:// URI rewriting
```

## Two routes: scene YAML decides

The branch is selected by a **single** signal — presence of
`robot.robot_source.urdf` in the scene YAML
(see `needs_assemble_robot()` in
[utils.py](../../genie_sim_bringup/launch/utils.py)):

| Condition in `robot.robot_source` | Path taken | Where `robot.usda` comes from |
| --- | --- | --- |
| `urdf:` key **absent** | **Pre-baked USD** | Pre-baked asset at `./assets/robot/<robot_name>/robot.usda`. `assemble_robot` is a no-op. |
| `urdf:` key **present** (value may be `{}`) | **URDF -> USD** | Generated from xacro by the URDF-import stage. |

Reserved keys on `robot.robot_source`:

- `package` (default: `genie_sim_robot_model`) — ROS 2 package providing the
  xacro (URDF route) or the pre-baked USD assets.
- `pin_base_to_world` (default: `false`) — whether the URDF root link is held
  to world. Decided at **runtime**, not at bake time — see
  [the base pin](#base-pin-runtime-toggle).
- `pin_base_mode` (default: `""` — take the default, which is `weld`) — *how* a
  pinned base is held. Same section.

The `urdf:` block accepts these reserved nested keys (URDF route only):

- `xacro_entry` (default: `xacro/robot.xacro`) — entry path inside the package share dir.
- `mimic` — propagated to the manifest for `PhysxMimicJointAPI`, **not** forwarded to xacro.

Any other string keys under `urdf:` are forwarded as additional xacro mappings.

## Stage cache

Before either assemble stage runs, the launcher inspects `manifest.json`:

- **HIT** — skip both assemble stages. Reuse the existing `robot.usda`.
- **MISS** — run the assemble stages (and `mkdir -p stage_dir`).
- **OVERRIDE** — `always_regenerate_robot_usd:=true` forces re-run even on a hit.

```bash
# force a fresh build
rm -rf assets/scenes/<scene_stem>/
# or
ros2 launch genie_sim_bringup app.launch.py always_regenerate_robot_usd:=true
```

## Debug snapshots written every launch

- **`<stage_dir>/scene.yaml`** — verbatim copy of the input scene yaml, written before
  cache hit/miss decision. Use `diff` against the source yaml to reveal operator-edited drift.
- **`<stage_dir>/robot_runtime.usda`** — thin override layer exported by `IsaacSimStage`
  after `_init_articulation`. Debug-only; the Isaac scene stage still references `robot.usda`.

## `UrdfToUsdStage` — URDF → USD

Defined in
:mod:`genie_sim_engine_isaacsim.assemble.urdf_to_usd_stage`. Its impl
body lives in :mod:`genie_sim_engine_isaacsim.assemble._robot_impl`
and executes this flow:

1. **Read scene YAML.** If `robot.robot_source.urdf` is absent, log and exit (no-op).
2. **Build xacro mappings** from `robot_source`: `robot_model`, `arm`, `body`, plus
   `gripper` → `variant`. Extra string keys under `urdf:` are forwarded as-is (except
   reserved `xacro_entry`, `mimic`).
3. **Process xacro** → writes `robot.urdf` into `stage_dir`.
4. **Rewrite `package://` URIs** → symlinks under `<stage_dir>/pkg/<pkg>`, rewrites all URIs.
5. **Drive Isaac Sim's URDF importer** (`_convert_urdf_to_usd`):
   - **6.x** → `urdf_usd_converter.Converter` + `importer_utils` post-passes + AS3 transformer.
   - **4.x / 5.x** → boots `SimulationApp({"headless": True})`, uses `URDFParseAndImportFile`.
   - Both converge on `<stage_dir>/robot.usda`.

### Cross-version invariants

Three design rules keep all three pipelines working with the same runtime code:

1. **Body discovery is hierarchy-walked.** `_collect_body_paths` in
   `genie_sim_engine_isaacsim.kit.stage` walks the full subtree via `Usd.PrimRange`
   and filters by `HasAPI(RigidBodyAPI)` — layout-agnostic.

2. **`/tf_render` carries absolute prim paths and local transforms.** `snapshot_body_transforms`
   emits `prim.GetPath().pathString` as each `child_frame_id`, and the pose is local relative
   to the USD parent. World transforms would compose twice on AS3. `/tf_scene`
   carries world poses for scene-body transport.

3. **`snapshot_odom` resolves `base_link` lazily.** Walks the subtree on first call, caches the
   resolved path; re-resolves only if the prefix changes.

### URDF/xacro authoring as the source of truth

The 6.x converter's ghost-root path can author invalid joint parents for PhysX.
Keep the kinematic root massless for KDL; if it has no visual or collision,
author the minimum root visual described in the
[robot-model guide](../../genie_sim_robot_model/AGENTS.md#non-ghost-root-link-pattern).
For non-root load-bearing links, supply justified inertials. Use
[diagnose_urdf.py](../../genie_sim_robot_model/scripts/diagnose_urdf.py) to inspect
missing inertials and ghost roots before conversion. Placeholder inertials add
real mass and are simulation approximations, including under Newton.

### Default selective collision policy

Implemented by `_apply_post_transformer_collision_policy` in
[assemble/_robot_impl.py](../../genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/assemble/_robot_impl.py).
Runs after the AS3 transformer on the final `robot.usda`. The baked `robot.usda` is the single
source of truth for all three engines.

| Link kind | Authored `<collision>`? | Action |
|---|---|---|
| Gripper (`gripper`/`finger`/`jaw`/`knuckle`/`palm`) | yes | `MeshCollisionAPI` approx `sdf` |
| Gripper | no | Promote visual mesh with `CollisionAPI` + `MeshCollisionAPI` approx `sdf` |
| Wheel (`wheel`/`caster`/`castor`) | yes | `MeshCollisionAPI` approx `convexHull` |
| Wheel | no | No collider |
| Anything else | yes | **DISABLED** — `physics:collisionEnabled = False` |
| Anything else | no | No-op |

SDF for gripper (preserves concave grip geometry), convex hull for wheel (stable rolling contact).

To add contact on a specific link, add its name pattern to `_COLL_GRIPPER_RE` /
`_COLL_WHEEL_RE` in `genie_sim_engine_isaacsim.assemble._robot_impl`.

```
`[collision] post-transformer policy` applied: gripper-authored(SDF)=N, ...
```

### `<material_override>`: roughness / metallic on embedded materials

Authored as a child of `<visual>` to patch PBR scalars the DAE pipeline can't carry:

```xml
<link name="chassis_link">
  <visual>
    <geometry><mesh filename="${mesh_dir}/tracer_base.dae"/></geometry>
    <material_override>
      <roughness>0.30</roughness>
      <metallic>0.85</metallic>
    </material_override>
  </visual>
</link>
```

This extension is implemented by the **Isaac Sim 6.2 AS3 assembly path**.
URDF consumers such as RSP and MoveIt ignore the extension. Assembly preserves
`robot_raw.urdf`, strips the blocks from the converter copy, and gives each
selected visual a stable, collision-free name. This distinguishes repeated mesh
files and primitive shapes without changing the authored xacro.

Overrides are applied to the **final `robot.usda`, after the AS3 transformer**.
Only the selected visual's instances are expanded for binding edits. Private
material references and bindings are authored in the generated root layer;
shared payload materials, albedo/texture networks, omitted fields and other
visuals remain unchanged. No mesh-stem or whole-link fallback is used. Missing
visual targets fail assembly and do not write a successful conversion stamp.

Finite values outside `[0, 1]` warn and clamp; malformed or non-finite values
warn and leave that field unchanged. Either field is optional. Regenerate old
robot caches with `always_regenerate_robot_usd:=true` after adopting this
implementation; old assets do not have the converter's stable visual names.
Isaac 4.5/5.1's AS2 importer is unchanged and does not apply this extension.
Use OVRtx or Isaac rendering for visual acceptance; Newton GL does not show PBR.

## The base pin: runtime toggle {#base-pin-runtime-toggle}

`robot.robot_source.pin_base_to_world` is a **runtime** flag, not a bake-time
one, and `pin_base_mode` refines it. The cached `robot.usda` always carries a
world-weld joint; the runtime decides per scene whether it stays active.

> The scene-yaml key is `pin_base_to_world`. `fix_base` is the *internal* arg
> name inside `IsaacSimStage` and the name of the Isaac-side policy function,
> and it reads naturally in prose — but it is **not** a yaml key any more. A
> yaml that still sets `fix_base` is read by nothing at runtime; the assemble
> stage prints it and moves on.

### Two fields, three resolved modes

Declared once, as `genie_sim_engine_core.schema.scene.PinBaseMode` — the schema
validator and every routing consumer read that one declaration rather than
re-deriving the rule:

| `pin_base_to_world` | `pin_base_mode` | Resolved | What holds the base |
|---|---|---|---|
| `false` | *(any — the boolean outranks it, and the engine warns)* | `free` | nothing; the base is under ordinary physics (mobile scenes: wheel-on-floor contact drives it) |
| `true` | omitted or `weld` | `weld` | the URDF importer's `root_joint` FixedJoint, left active |
| `true` | `mocap` | `mocap` | **not a FixedJoint.** The root joint stays FREE and the solver clamps it to a settable reference every substep, so a reset can re-place the base |

Worth knowing when reading a scene yaml: under `mocap` a true
`pin_base_to_world` does **not** mean a weld exists — "pinned" means *held*,
not *welded*. Only `weld` keeps the `root_joint`; `free` and `mocap` both
deactivate it (`setup/stage.py::_deactivate_root_joint`).

`mocap` needs a solver that implements the clamp, declared as
`SolverAdapter.supports_base_pin` — today only `mujoco_warp_multi_env`. Asking
for it anywhere else is caught, not ignored, and the two backends differ by
consequence rather than convenience:

- **Newton refuses.** Two `RuntimeError`s: one in `engine_base.__init__` if the
  adapter cannot clamp, one before the first step if it could but registered no
  root free joint. Without them the weld comes off and nothing replaces it — the
  base falls under full gravity while the log still reads `pin_base_mode=mocap`.
- **Isaac warns and downgrades to `weld`.** There `fix_base` still *holds* the
  base; all that is lost is reset-time re-placement.

> **The asymmetry is deliberate — please don't "fix" it for consistency.**
> Two backends handling the same bad input two different ways looks like an
> oversight, so here is the rule it follows: *refuse when the fallback is
> unsafe, downgrade when the fallback is merely weaker.*
>
> On Newton there is no fallback. `mocap` deactivates the `root_joint`, and if
> the adapter cannot clamp, nothing takes over — the base falls under full
> gravity, through the floor, while the log still reads
> `pin_base_mode=mocap (explicit)`. A warning there would be a warning about a
> simulation that is already wrong. On Isaac the weld is still in place and
> still holding; the *only* casualty is reset-time re-placement, which the
> scene may well never use. Raising there would refuse to run a scene that
> would have run correctly for everything else it asks for.
>
> Making them symmetric costs something either way. Symmetric-refuse turns a
> recoverable capability gap into a hard stop on the backend most scenes
> actually use. Symmetric-downgrade is the one that is actually dangerous: it
> would reintroduce exactly the silent fall this gate was added to close.
> Both sites carry a comment saying so, so the next reader does not have to
> re-derive it.

### When `mocap` is actually the right answer

A mocap body is held at a pose the host writes every step rather than by
a fixed constraint. Use it only with an adapter that supports this clamp.

Reach for it when **both** halves are true:

1. The base must be *held* — it should not fall, drift, or react to contacts, so
   `free` is wrong; and
2. the pose it is held at must **change between resets** — domain
   randomization, a curriculum that moves the robot, or a replayed base
   trajectory. That is what `weld` cannot do: the FixedJoint anchor is authored
   at build time, and reset-time re-placement under `weld` is the known gap
   (`lifecycle.py`).

It is not a coincidence that the only adapter implementing the clamp is
`mujoco_warp_multi_env` — multi-env rollout is where per-reset re-placement
earns its cost. For a single stationary arm, `weld` is simpler and strictly
cheaper (a constraint the solver handles once vs. a write every substep), and
for anything that should respond to the floor, `free` is correct.

So: `weld` for a stationary base, `free` for a base that physics should drive,
`mocap` only for a *scripted* base — one whose pose is commanded rather than
simulated or fixed.

### How the flag reaches the stage

```
scene.yaml
    robot.robot_source:
      pin_base_to_world: true|false
      pin_base_mode: ""|weld|mocap        # refuses anything else at parse time
        │
        ▼
genie_sim_engine_core.schema.scene.RobotSource.from_dict
    normalizes the mode once (case + whitespace) and rejects junk
        │
        ▼
UrdfToUsdStage (genie_sim_engine_isaacsim.assemble._robot_impl)
    converter always authors a world-weld (PhysicsFixedJoint "root_joint")
        │
        ▼
IsaacSceneStage (genie_sim_engine_isaacsim.assemble._scene_impl)
    writes manifest["scene_yaml"] = <original yaml path>
        │
        ▼
the engine node (via genie_sim_engine_core.common.session.EngineSession)
    reads manifest["scene_yaml"] live, re-parses both fields from the yaml,
    and forwards them as engine kwargs
        │
        ├─ newton_standalone ─────────────────────────────────────────
        │     PinBaseMode.resolve(...) -> engine_base._pin_base_mode
        │     setup/stage.py        keeps or deactivates root_joint
        │     engine_base           gates `mocap` on the adapter, picks
        │                           the substep regime
        │
        └─ isaac_physx / isaac_newton ────────────────────────────────
              fix_base=pin_base_to_world -> IsaacSimStage
              genie_sim_engine_isaacsim.kit.stage._apply_fix_base_policy
                  walks the robot subtree for the world-weld
                  (FixedJoint with body0=articulation root)
                  sets physics:jointEnabled = fix_base
```

Detection is name-agnostic (works for both 6.x AS3 and flat 4.x/5.x naming).
Default when `scene_yaml` is missing: unpinned (`pin_base_to_world=False`) —
the schema default, reached because the parse runs against an empty mapping. A
launcher's `scene_yaml_defaults` block can still supply both fields; it merges
*beneath* the scene, so the scene's own keys win.

## Mimic joints

URDF mimic joints are authored as schemas by the URDF→USD importer:

| Pipeline | Schema | Enforced by |
|---|---|---|
| `isaac_physx` | `PhysxMimicJointAPI:<axis>` | PhysX solver — rigid kinematic constraint |
| `isaac_newton` | `NewtonMimicAPI` | Wrapper constraints and shared target expansion |
| `newton` | `NewtonMimicAPI` | Imported equalities and shared target expansion |
| 4.x / 5.x | neither | not enforced |

Target expansion is shared through
[core engine/_mimic.py](../genie_sim_engine_core/engine/_mimic.py).
Newton MuJoCo-Warp enforces imported mimic equalities; follower drives are
stripped so the equality transmits master motion without a second actuator.
`kit.stage._configure_drives` identifies followers by their mimic schemas.
See [joint parameters](../../../../../docs/JOINT_PARAMS.md) for passive follower
properties and rejection of nonzero follower drive overrides.

There is **no scene-yaml mimic block** — a mimic relation is robot geometry, lives in USD only.

## `IsaacSceneStage` — bind the robot USD into the manifest

Defined in :mod:`genie_sim_engine_isaacsim.assemble.isaac_scene_stage`
with the body in :mod:`genie_sim_engine_isaacsim.assemble._scene_impl`.
Picks **which** USD to advertise (`urdf:` present → staged `robot.usda`; absent → pre-baked asset).
Writes `manifest.json` with paths **relative to `base_path`** so the cache survives workspace moves.

The only artifacts it produces:

1. **`render_layer.usda`** — mirrors camera prims from the robot stage under `/RenderOVRTX`,
   defines `RenderProduct` nodes, appends a synthetic `FreeCam`.

2. **`manifest.json`** — the engine's contract:

   | Field | Source | Purpose |
   | --- | --- | --- |
   | `base_path` | CLI `--base-path` | Anchor for relative paths. |
   | `scene_usda` | `config["scene"]` | World USD opened at runtime. |
   | `robot_usda` | resolved staged/pre-baked path | Robot USD referenced into scene. |
   | `render_layer_usda` | path to just-written layer | Optional render-product layer. |
   | `robot_prefix` | `robot.robot_prefix` | Stage prim namespace (`/<robot_prefix>`). |
   | `robot_from_urdf` | derived from `urdf:` presence | Classification path hint for `kit.stage`. |
   | `scene_yaml` | absolute path to **original** input yaml | Engine reads this live for `pin_base_to_world` / `pin_base_mode` / `convert_joints_to_fixed` etc. |
   | `cameras[]` | per-camera fan-out | Per-camera intrinsics + render product paths. |

### Where the scene+robot bond happens

At runtime, inside `IsaacSimStage.__init__` → `_open_scene_with_references`
([kit/stage.py](../../genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/kit/stage.py)):

1. `open_stage(scene_usda)`
2. `add_reference_to_stage(robot_usda, "/<robot_prefix>")`
3. `add_reference_to_stage(render_layer_usda, "/RenderOVRTX")`
4. `simulation_app.update()` pumped after each reference to let composition settle.

## The engine is a pure consumer

The physics node only:

1. `wait_for_manifest(manifest_path)` — blocks until JSON exists.
2. Re-anchors paths against `base_path` via a local `_abs()` helper.
3. Opens stage + references robot via `_open_scene_with_references`, configures drives +
   articulation gains from `config/physics_params.yaml`.

It does **not** touch the URDF, run xacro, or invoke the URDF importer.

## TL;DR

```
scene_*.yaml
    │
    ▼
python -m genie_sim_engine_core.assemble --physics-engine isaac_newton
    │
    ├─► UrdfToUsdStage       → robot.urdf, robot.usda, urdf.sha256
    ├─► IsaacSceneStage      → render_layer.usda + manifest.json
    └─► NewtonExtrasStage    → newton_solvers.json  (only if scene has cloth entries)
                                            │
                                            ▼
        genie_sim_engine_isaacsim_node.py / genie_sim_engine_newton_node.py
              reads manifest.json and references robot.usda into the stage
```

- Force a rebuild: `always_regenerate_robot_usd:=true` or `rm -rf assets/scenes/<scene_stem>/`.
- Switch routes: add or remove `robot.robot_source.urdf` in the scene YAML.
