# genie_sim_robot_model

Robot description package — xacro/URDF sources and pre-baked USD assets for
all robots supported by the GenieSim stack.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE) (per-robot upstream licenses — see [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES))
Third-party notices: [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)

**Maintenance contract**: when you add a robot, update `robots/`, `xacro/`,
`THIRD_PARTY_NOTICES`, and this file in the same diff.

---

## Support tiers

The GenieSim stack actively develops and validates **AgiBot Genie G2 only**.
Every other robot description in `robots/` is community-grade reference
material, kept buildable but not exercised against the full physics /
launch / fold-task pipelines.

| Tier | Robots | Meaning |
|---|---|---|
| **Tier 1 — G2** | G2 crs/omnipicker, crsB/Swiftpicker and crsB/Wuji | Primary maintained descriptions and prioritized physics/launch test scenarios; runtime behavior must be checked against the selected scene and backend. |
| **Tier 2 — reference** | `agilex/aloha`, `agilex/piper`, `arx/x5`, `arx/acone`, `franka/fr3`, `universal_robots/ur5` | Reference descriptions; gains, contacts and end-effector behavior require platform-specific validation. |

See [bringup](../genie_sim_bringup/README.md) for launch configuration. A
support tier does not establish that every scene/backend combination passes. Bug reports need the scene, launcher,
exact command, asset/runtime identity, and expected versus observed behavior.

---

## Layout

```
genie_sim_robot_model/
├── xacro/
│   └── robot.xacro                ← universal entry point (dispatches by robot_model)
├── robots/
│   ├── agilex/                    ← AgileX Robotics (aloha, piper)            [Tier 2]
│   ├── arx/                       ← ARX Robotics (x5, acone)                  [Tier 2]
│   ├── franka/                    ← Franka Robotics (fr3)                     [Tier 2]
│   ├── genie/                     ← AgiBot Genie mobile manipulation platform [Tier 1]
│   │   └── g2/                    ← G2 variants: crs/crsB, Omnipicker/Swiftpicker/Wuji/no_gripper
│   ├── universal_robots/          ← Universal Robots (ur5)                    [Tier 2]
│   ├── */*/overlays/              ← per-run identification overlays (sparse YAML)
│   ├── OVERLAYS.md                ← overlay conventions + what ships (data side)
│   └── SYSID_PARAMS.md            ← every identified value, provenance, caveats (handover)
├── scripts/
│   ├── normalize_obj_names.py     ← OBJ object-name normalizer (mesh-developer tool)
│   └── diagnose_urdf.py           ← URDF/xacro authoring diagnostics (detect-only)
├── urdf/                          ← generated URDF outputs (not committed)
├── rviz/                          ← per-robot RViz2 configs
└── launch/                        ← robot_state_publisher launch files
```

---

## Mesh-developer tools

### `scripts/normalize_obj_names.py` — defeat 6.x converter mesh de-duplication

The Isaac Sim 6.x URDF→USD converter (`urdf_usd_converter._impl.mesh`) keys
each authored mesh prim under `Geometries/<name>` where `<name>` is the
.obj file's `o <name>` directive, and de-duplicates by name across files.
DCC tools (3ds Max, Blender, Maya) commonly emit generic default object
names — `Cylinder001`, `Box001`, `s`, `2`, `1` — that collide aggressively
across a multi-link robot. When a collision happens, the first occurrence
wins and the rest get a `_N` suffix or are silently dropped, leaving the
assembled robot USD with **missing or swapped link visuals**.

ARX acone is the canonical case: 14+ visual meshes share four distinct
internal names. Without normalization the assembled USD is missing
`arm_r_*` and half the gripper visuals. UR5 doesn't trip it because its
`.dae` meshes carry per-link names baked in by the DCC tool.

**The tool** rewrites every `.obj` in place to carry exactly one
`o <file_stem>` directive (file stem = filename without `.obj`) and zero
`g` directives. Vertex/normal/texture/face/material lines are preserved
unmodified. The result is one unique object name per file, derived
deterministically from the filename → one unique
`Geometries/<file_stem>` prim in the converted USD, no cross-file
collisions. The tool is idempotent.

#### Usage

```bash
# Diagnose only — list every .obj, report cross-file name collisions
# and which files would be rewritten. No file is modified.
python3 scripts/normalize_obj_names.py --dry-run [PATH ...]

# Auto-fix — rewrite every .obj in place. Re-run after each batch of
# mesh updates from the DCC tool.
python3 scripts/normalize_obj_names.py [PATH ...]
```

If no PATH is given, the tool scans the current working directory
recursively. Pass directories or individual `.obj` files to limit scope.

`--dry-run` exits non-zero if any file needs normalization (so it doubles
as a CI gate); the auto-fix run exits zero if all writes succeed.

#### Mesh-developer workflow

1. Export new `.obj` meshes from the DCC tool.
2. Drop them under `robots/<vendor>/<robot>/meshes/`.
3. Run `python3 scripts/normalize_obj_names.py --dry-run robots/<vendor>` to
   see what's about to change.
4. Run `python3 scripts/normalize_obj_names.py robots/<vendor>` to apply
   the rewrite.
5. Commit the normalized `.obj` files.
6. Delete any stale `assets/scenes/<scene_stem>/` cache so the next launch
   re-runs the `urdf_to_usd` assemble stage against the fixed meshes.

The `urdf_to_usd` URDF→USD pipeline does **not** carry a runtime
fallback for non-normalized .obj files — that responsibility lives with
the mesh-source-of-truth here.

### `scripts/diagnose_urdf.py` — URDF/xacro authoring diagnostics

Reports authoring issues without rewriting files. Review findings in context:
empty kinematic frames are valid on Isaac Sim 6.2, while dynamic bodies need
real mass and inertia. Do not turn a frame into a near-massless body by adding
placeholder inertials.

#### Checks

| Finding | Author action |
|---|---|
| Non-root link with neither inertial nor collision, excluding `_inertia` frame partners | Decide whether it is a frame or a physical body; author measured mass/inertia only for the latter. |
| Root with no inertial, visual, or collision | Review the [non-ghost root pattern](#non-ghost-root-link-pattern). |
| Wheel/tire-named link without a cylinder collider | Measure the mesh and author suitable collision geometry. |
| Non-diagonal inertia or near-degenerate diagonal inertia | Review the principal axes and converter behavior; use the recompute tools where appropriate. |
| Resolved mesh bounds beyond 5 m | Check mesh units; `fix_dae_units.py` handles COLLADA conversion. |

#### Operates on xacro source

Pass source xacros to retain corrections when generating URDFs. The tool scans
text, including templated link names; it does not evaluate the xacro tree.
Generated `.urdf` files can be checked separately to verify the composed model.

#### Usage

```bash
# Strict gate: exits 1 for any finding, including missing inertials on frames.
python3 scripts/diagnose_urdf.py --dry-run [PATH ...]

# Report only: missing-inertial findings alone do not fail the command.
python3 scripts/diagnose_urdf.py [PATH ...]
```

PATH may name files or directories; the default is a recursive scan of the
current directory. Both modes leave inputs unchanged. The default mode exits
1 for root, wheel, tensor, or mesh-unit findings. Pass `--mesh-root` when a
xacro uses `${mesh_dir}` so mesh-unit checks can resolve the files.

#### Mesh-developer workflow

1. Run diagnostics on changed source files and review each finding.
2. Correct physical inertials, root structure, colliders, and mesh units in
   their source files. Keep frame-only links massless.
3. Regenerate URDFs and validate the resulting trees and mesh references.
4. Reassemble robot assets and check the selected runtime.

---

## xacro entry point

`xacro/robot.xacro` is the **single entry point** for all robots. It dispatches
by the `robot_model` xacro arg. Additional args (`body`, `arm`, `gripper`,
`variant`) select sub-configurations within a robot family.

The current bringup launch path builds the description using `xacro.process_file`:
```python
xacro.process_file("share/genie_sim_robot_model/xacro/robot.xacro",
                   mappings={"robot_model": ..., "arm": ..., "body": ..., "gripper": ...})
```

Any string key under `robot.robot_source.urdf:` in the scene YAML (except
reserved keys `robot_model`, `body`, `arm`, `gripper`, `mimic`,
`package`, `xacro_relpath`) is forwarded as an additional xacro
mapping.

Reserved xacro mappings (set by the launch system):

| Key | Source | Purpose |
|---|---|---|
| `robot_model` | scene YAML / CLI | Selects the robot family |
| `body` | scene YAML / CLI | Body variant |
| `arm` | scene YAML / CLI | Arm variant |
| `gripper` | scene YAML / CLI | Gripper choice (consistent arg name across piper / G2 / UR5) |

---

## Adding a robot

1. Create `robots/<robot_name>/` with xacro, meshes, and collision geometry.
2. Add a dispatch branch in `xacro/robot.xacro`.
3. Add third-party license info to `THIRD_PARTY_NOTICES` if the robot
   description originates from an upstream vendor.
4. **Run mesh-developer tools** if any link references `.obj` meshes:
   - `.obj` → `python3 scripts/normalize_obj_names.py robots/<vendor>`
   - `.fbx` must be converted to `.dae` beforehand using an external tool
     (e.g. Blender, Maya). The URDF→USD pipeline only supports
     `.stl` / `.obj` / `.dae`. Update the xacro to reference `.dae`.
   - `.dae`/`.stl`-only robots need no mesh-dev preprocessing.
5. **Run the URDF/xacro diagnose tool**:
   `python3 scripts/diagnose_urdf.py --dry-run robots/<vendor>`. Review
   every finding, author real mass/inertia for physical bodies, and leave
   empty kinematic frames massless.
6. Test: `xacro xacro/robot.xacro robot_model:=<name>` must produce valid URDF.
7. Test the URDF→USD pipeline:
   `ros2 launch genie_sim_bringup app.launch.py scene:=<scene_with_new_robot> ...`

---

## URDF→USD pipeline (upstream context)

This package is a **pure description provider**. The current bringup path
uses [`genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/assemble/_robot_impl.py`](../genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/assemble/_robot_impl.py)
for URDF→USD conversion. See the
[engine pipeline](../genie_sim_engine_core/docs/pipeline.md) for the launch and
asset-cache contract.

In this launch path, `robot.robot_source.fix_base` defaults to `true` and
controls the runtime world weld. Set it to `false` for a mobile base whose
wheel contacts should translate the chassis. The cached robot USD is built
with a fixed root and the runtime applies the selected policy.

---

## Design patterns

Conventions every robot in this package follows. New robots SHOULD
adopt these; deviations need a comment in the xacro explaining why.

### Non-ghost root link pattern

**Problem**

The Isaac Sim 6.x URDF→USD converter
(`urdf_usd_converter._impl.link.physics_joints`) branches on a single
boolean — whether the URDF root link has any `<inertial>`, `<visual>`,
or `<collision>` child:

```python
is_ghost_link = (root_link.inertial is None
                 and len(root_link.visuals) == 0
                 and len(root_link.collisions) == 0)
```

When `is_ghost_link=True` (the URDF root is a pure kinematic frame,
which is the standard ROS convention for `base_link` / `world` / etc.),
the converter takes a Newton-only authoring branch: every joint —
including every fixed joint deep in the kinematic tree — gets
`body0 = default_prim` (the articulation root Xform), not the actual
parent link. PhysX cannot construct an articulation from such USD:
every joint appears anchored to the world origin simultaneously, the
robot collapses through the floor on the first physics step, and no
runtime patch can fix the topology after the fact.

The source descriptions carry the minimum root visual explicitly; verify
converted joint body references when changing this structure.

**KDL constraint**

Adding `<inertial>` to the root would be the obvious fix but
**breaks KDL** (`urdf_parser`'s kinematics library, used by
`robot_state_publisher` and MoveIt): KDL emits a "Root link X has
inertial properties..." warning and refuses to compute kinematics for
downstream consumers.

**The pattern**

Add a minimum-impact `<visual>` to the URDF root: a 1mm sphere. This
flips `is_ghost_link → False` without affecting KDL (which only
forbids `<inertial>` on the root), without affecting RViz / MoveIt
appearance (1mm at the origin is invisible at any practical zoom), and
without affecting collision (it has no `<collision>` block).

```xml
<link name="base_link">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry><sphere radius="0.001"/></geometry>
  </visual>
</link>
```

Any robot following the UR-style frame + `_inertia` sibling pattern
(empty `base_link` → fixed-joint sibling `base_link_inertia` carrying
the actual mass) needs this — that's most robots in this package.

The tradeoff: `<visual>` on the root link is technically observable via
RViz and `robot_state_publisher`'s `/tf` graph, but a 1mm sphere is
invisible at any normal viewing distance and adds no meaningful TF
graph noise.

**Robots applying this pattern**

| Robot | Root link | Reason root is a kinematic frame |
|---|---|---|
| `agilex/aloha` | `base_link` | UR-style frame + `base_link_inertia` sibling, plus mobile platform with chassis & wheels as siblings |
| `agilex/piper` | `base_link` | UR-style frame + `base_link_inertia` sibling |
| `arx/acone` | `base_link` | UR-style frame + `base_link_inertia` sibling; bimanual ("AC1") |
| `arx/x5` | `base_link` | UR-style frame + `base_link_inertia` sibling |
| `franka/fr3` | `base_link` | UR-style frame + `base_link_inertia` (carries upstream `link0`) |
| `universal_robots/ur5` | `${tf_prefix}base_link` | UR-style frame, sibling `arm_base_link` for UR-internal frame, sibling `base_link_inertia` |

The `genie/g2/*` xacros aren't ghost-rooted (their root links carry
visual/collision directly), so they don't need this fix.

**How to check**

Run the diagnose tool. A clean tree should show no
`[ghost-root]` warnings:

```bash
python3 scripts/diagnose_urdf.py --dry-run robots/
```

If a new robot you're adding shows up in `[ghost-root]`, apply this
pattern in the xacro. Each affected xacro carries an inline comment
referencing this AGENTS.md section, so the design intent is greppable
from any robot file.

---

## Identification overlays

Measured parameters arrive as **overlays**: sparse YAML under
`robots/<vendor>/<model>/overlays/`, one file per identification run,
carrying only the fields that run actually measured. The generated URDF
is the baseline. `robots/OVERLAYS.md` documents the conventions and
lists what ships. The schema and resolver live in
[`genie_sim_engine_core/schema/joint_params.py`](../genie_sim_engine_core/genie_sim_engine_core/schema/joint_params.py).

Four rules bind anything editing this package:

- **Never hand-edit a `kind: sysid` overlay.** It is the only record of
  what a run measured. To change a value for an experiment, append a
  `kind: operator` overlay carrying that one field; to adopt a re-run,
  add a new dated file.
- **Never pad an overlay to look complete.** Absence means not
  measured. Copying a baseline value into a gap destroys that distinction.
  Name the gap in `meta.not_identified` instead.
- **Joint and link keys are URDF names verbatim**, `idxNN_` prefixes
  included. An unknown name is a hard load error by design; do not
  "fix" a validator failure by renaming the key to something that
  resolves.
- **A `*_sysid.urdf` under `urdf/` alone does not select runtime overlays.**
  Compatible scenes also select `robot.joint_overlays`; Isaac PhysX 6.2 and
  Newton MuJoCo-Warp apply those joint values after class defaults. Armature
  has no standard URDF slot. Inspect initialized solver values to verify delivery.

Link inertials are selected through xacro `inertia_source` / `param_source`
arguments. For a complete baseline comparison, select xacro `default` and
`robot.joint_overlays: []`. The core schema CLI validates and inspects overlays
without configuring a solver. See the [runtime reference](../../../../docs/JOINT_PARAMS.md)
and [backend matrix](robots/SYSID_PARAMS.md#2-selection-and-runtime-application)
for supported fields, precedence and remaining gaps.

---

## Authoring checks

Use [diagnose_urdf.py](scripts/diagnose_urdf.py) to check descriptions before
conversion. The current checks and package patterns are:

| Check | Package contract |
| --- | --- |
| Root inertial | Keep the kinematic root massless for KDL consumers. |
| Frame/inertia siblings | A fixed sibling such as `base_link_inertia` carries physical mass; do not auto-inject mass into its frame partner. |
| Ghost root | A root with no inertial, visual, or collision needs the minimum visual described above for the Isaac converter. |
| Wheel colliders | Use simple cylinder collision geometry where appropriate for wheel contact. |
| Names and joints | Preserve URDF names and valid parent/child topology; overlays use those names verbatim. |

Frame naming follows [REP-105](https://www.ros.org/reps/rep-0105.html) for
mobile bases and [REP-120](https://www.ros.org/reps/rep-0120.html) for humanoid
extensions. These frame conventions do not define inertial-authoring patterns.
For the URDF grammar and parser, see the [URDF XML specification](http://wiki.ros.org/urdf/XML)
and [`urdfdom`](https://github.com/ros/urdfdom).

Keep name-based heuristics conservative: flag ambiguous cases for review
instead of automatically changing an unrelated link.

---

## Routing rules

- Universal xacro entry → `xacro/robot.xacro`
- Per-robot description files → `robots/<robot_name>/`
- Identified parameters (sysid output) → `robots/<vendor>/<model>/overlays/`
- Overlay conventions / what ships → `robots/OVERLAYS.md`
- What a given number is, where it came from, how far to trust it → `robots/SYSID_PARAMS.md`
- Overlay schema + resolution → `../genie_sim_engine_core/genie_sim_engine_core/schema/joint_params.py`
- OBJ mesh-name normalizer → `scripts/normalize_obj_names.py`
- URDF/xacro diagnostics → `scripts/diagnose_urdf.py`
- Third-party license notices → `THIRD_PARTY_NOTICES`
- URDF→USD conversion logic → `../genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/assemble/_robot_impl.py`
- Scene YAML robot block parsing → `../genie_sim_bringup/launch/utils.py`
