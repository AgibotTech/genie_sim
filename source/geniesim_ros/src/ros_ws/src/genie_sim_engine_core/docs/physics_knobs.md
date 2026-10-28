# Physics configuration and render diagnostics

Physics profiles are version-controlled YAML selected with `physics_params_file`.
Scene-specific contact and collision settings belong in the scene YAML. Selected
joint overlays apply after class defaults; see [joint parameters](../../../../../docs/JOINT_PARAMS.md).
Environment variables below control rendering diagnostics, not physics tuning.

A development workspace using `--symlink-install` reads profile edits without a
rebuild. A deployment can select its own profile through the launcher.

## A. Config entries

| Source | Meaning |
| --- | --- |
| Built-in default | Value used when a key is absent; zero disables the corresponding optional override, not the solver itself |
| Base profile | [physics_params.yaml](../config/physics_params.yaml), the shared runtime profile |
| RLinf profile | [physics_params_rlinf.yaml](../config/physics_params_rlinf.yaml), selected by `launcher_newton_mjwarp_multi_env.yaml` |

The reader determines whether a setting applies. Both MuJoCo-Warp adapters
consume the rows marked **all mjwarp**. Featherstone and AVBD use different
solver APIs; `warn_inert_params` reports configured adapter-only settings they
cannot consume. Pad decomposition runs in the shared Newton model builder and
also applies to Featherstone. The reader-scope regression is
`genie_sim_engine_newton/tools/test_knob_reader_scope.py`.

| Config entry | read by | shipped | shared profile | RLinf profile | What it controls |
| --- | --- | --- | --- | --- | --- |
| `solver.mjc_iterations` | *all mjwarp* | `0` | `0` | `0` | `SolverMuJoCo`'s rigid constraint-solver iteration count. `0` leaves SolverMuJoCo on its own default, which is MuJoCo's `100`. **This is not the launcher's `physics_solver_iterations`** — see below |
| `solver.impratio` | *all mjwarp* | `0.0` | `0.0` | `30.0` | mjwarp `opt.impratio` + elliptic friction cone, as one key. `0` preserves the default cone; positive values select elliptic friction with the requested impedance ratio. |
| `newton.mujoco_enable_multiccd` | *all mjwarp* | (adapter) | — | — | **Scene yaml, not this file.** MuJoCo's multi-point convex contact manifolds. Tri-state: absent keeps each adapter's own default (multi-env on, stock off), `true`/`false` override. Scene-homed because MultiCCD needs zero-margin convex pairs, which the scene's geometry either has or does not |
| `gripper_contact.gripper_master_force_cap` | *all mjwarp* | `0.0` | (inherits) | `30.0` | forcerange cap on the gripper master actuator (N*m). **Mind the priority order** — see below. |
| `gripper_contact.pad_decomposition.enabled` | *all mjwarp* | `false` | (inherits) | `true` | CoACD convex decomposition of the gripper pads. Off leaves each pad as a single hull, degrading contact from a face to a point |
| `gripper_contact.pad_decomposition.threshold` | *all mjwarp* | `0.01` | (inherits) | `0.01` | ditto, decomposition threshold |
| `gripper_contact.pad_decomposition.max_hulls` | *all mjwarp* | `16` | (inherits) | `16` | ditto, maximum hull count |
| `usd_drive_api.gripper.master_stiffness` | *all mjwarp* | `10000.0` | `2000.0` | `100.0` | PD stiffness of the gripper master |
| `arm_contact.arm_max_torque` | *all mjwarp* | `0.0` | (inherits) | `100.0` | forcerange cap (N*m) applied after compile to every arm position-servo actuator, whatever the URDF or `articulation_view_runtime.arm_*.max_effort` authored (that table can only raise a joint's limit, never lower it). `0` preserves the compiled actuator range. `100` matches the URDF's own `<limit effort="100">` per arm joint. |

### Per-class contact parameters

`newton.mjc_contact.<class>.{solref,solimp,friction,condim}` in the scene YAML
merges onto `MJC_CONTACT_DEFAULTS`. Classes are `robot`, `floor`, `passive`,
`static_prop`, and `other`. Omit `condim` to preserve MuJoCo's default of 3.
The optional `gripper_pad` refinement inherits `robot` and overrides named keys:

```yaml
newton:
  mjc_contact:
    robot:
      friction: [1.5, 0.5, 0.2]
      condim: 3          # arm and body links
    gripper_pad:
      condim: 6          # fingertips only
```

Membership is not in the yaml, deliberately. The pads are the robot shapes
whose body carries the assemble-time `newton:isGripperPad` annotation — the
same set `pad_decomposition` already runs CoACD over
(`common/gripper_annotations.py`, with a gripper-classified `link4` fallback
when annotations are absent). The yaml supplies the parameters, the asset
supplies which bodies are pads, so a differently-named gripper needs no yaml
edit and no scene can name a body that does not exist.

The refinement is opt-in on the presence of the block: without it the pads stay
in `robot`, and `newton.mjc_contact.robot.condim: 6` applies to **every** robot
shape — additional contact dimensions on links that do not grip. If a `gripper_pad:` block is present
and no pad body can be identified, the engine warns by name at bring-up and
leaves everything on `robot` rather than pretending it applied.

The multi-env launcher supplies contact defaults under `scene_yaml_defaults`.
Scene values take precedence through a deep merge. `test_h1_h2_pads.py` checks
that launcher defaults and contact classification agree.

### Independent solver iteration counts

`physics_solver_iterations` controls the particle solver. `solver.mjc_iterations`
controls the rigid MuJoCo solver in both adapters, including `mjxpbd` and
`mjvbd`. Zero preserves the MuJoCo default of 100 iterations. Particle profiles
can use 4 or 5 iterations with many substeps without reducing rigid convergence.
Increasing rigid iterations costs time on every substep; measure contact
convergence and step time before changing the count.

### MuJoCo contact and effort settings

`solver.impratio > 0` selects the elliptic cone and its friction impedance ratio
in both MuJoCo-Warp adapters. The RLinf profile uses 30. Contact tuning should
measure both static creep and vibration-induced slip; see [contact tuning](../../../../../docs/CONTACT_TUNING.md).

`newton.mujoco_enable_multiccd` belongs in the scene YAML because multi-point
convex manifolds depend on collision geometry and zero margins. It is tri-state:
absent preserves each adapter's default (multi-env on, single-env off), while
explicit `true` or `false` overrides that default.

A positive `usd_drive_api.gripper.master_max_force` takes precedence over
`gripper_contact.gripper_master_force_cap`. The base profile selects 18 N*m;
the RLinf profile selects 50 N*m, so its secondary 30 N*m setting is inactive.
These are simulation drive clamps, not manufacturer torque ratings. The
Swiftpicker identification overlay records its 18 N*m capture condition and
publishes no identified effort limit. See [sysid evidence](../../genie_sim_robot_model/robots/SYSID_PARAMS.md).
The engine reports the selected cap and its `source=yaml` or `source=default`.
Selected joint effort overlays are applied after class defaults; confirm final
actuator ranges when combining them with adapter contact caps.

### Launcher-owned MuJoCo capacity knobs

`mujoco_njmax` and `mujoco_nconmax` size solver buffers and belong in launcher
YAML. The single-env adapter defaults to 1024 constraint rows and 512 contacts
when absent or zero. Dense scenes need enough collision and constraint capacity;
`launcher_newton_mjxpbd.yaml` selects larger buffers. Confirm resolved capacities
on the adapter startup line and check for overflow during scene testing.

## A2. Collider construction (the `scene_colliders:` block)

> For the full background behind these knobs — what collision shapes lose going from USD to mujoco_warp,
> why the compensation logic falls into three classes, and whether direct alignment with USD is
> possible — see [`collision_pipeline.md`](collision_pipeline.md) in the same directory.

These decide **what shape each collider enters the solver as**, and changing them is directly reflected
in contact behavior. The whole block is off in the factory profile (off = identical to an engine without
these two passes); the RLinf profile turns them on. The numbers in the table are geometric measurements,
not hand-tuned physical constants — a long-term change should change the calibration basis behind these
thresholds, not the thresholds themselves on a whim.

Background: three settings of USD's `physics:approximation` end up sending a concave object into the
solver as a **single convex hull** — `none`/`sdf` (PhysX's exact-mesh mode, not in newton's table),
`convexDecomposition` (newton hardcodes the CoACD threshold to 0.5, too loose, often yielding just 1
block), and `meshSimplification` (requires `fast_simplification`; when unavailable, the logged
"Falling back to bounding_box" path uses a solid rectangular box). The
consequences are: door frames seal the doorway shut, shopping carts and shelves swallow the bottles
inside them, and parts bins and block sorting boxes turn solid. So the criterion is based on the
**result**, not on what the asset declares.

| Config item | Factory default (`physics_params.yaml`) | RLinf profile | What it controls |
| --- | --- | --- | --- |
| `scene_colliders.decomposition.enabled` | `false` | `true` | Master switch. When off, concave colliders fall back to newton's single convex hull |
| `scene_colliders.decomposition.max_solidity` | `0.5` | `0.5` | Ceiling on solidity (mesh volume / convex-hull volume). Only used to filter out things where "the convex hull is basically already the mesh": floor 0.863, bottle body 0.697 |
| `scene_colliders.decomposition.min_void` | `0.001` | `0.001` | Floor on cavity volume (convex hull − mesh, m³). This is the real criterion for "will it swallow something" |
| `scene_colliders.decomposition.max_verts` | `200000` | `200000` | Vertex cap, guards against pathological meshes. The largest real collider (the shopping cart) has 101,682 |
| `scene_colliders.decomposition.threshold` | `0.05` | `0.05` | The concavity threshold for our own CoACD call (newton's hardcoded 0.5 is too loose) |
| `scene_colliders.decomposition.max_hulls` | `32` | `32` | Maximum number of convex hulls per collider |
| `scene_colliders.box_cover.enabled` | `false` | `true` | Box covering. When off, concave colliders always go through CoACD |
| `scene_colliders.box_cover.pitch` | `0` (= the algorithm's built-in constant 0.004) | `0` | Voxel edge length (meters). Roughly the accuracy with which a hole's diameter is recovered |
| `scene_colliders.box_cover.max_boxes` | `0` (= the algorithm's built-in constant 400) | `0` | How many boxes can cover one collider at most before falling back to CoACD |
| `scene_colliders.background_decor_static` | `false` | `true` | Demotes free-floating decor under `/Background` **with no joint reference at all** to a static collider; collision geometry is kept, only the dynamics are removed |

**Why box covering on top of convex decomposition — isn't decomposition enough**: every piece from convex
decomposition is a **circumscribing** approximation, so a ring around a hole's edge inevitably gets eaten
away. The block sorting box (`place_block_into_box`) is broken exactly this way: the hole's true inscribed
diameter is 39.4mm, but after CoACD only 25.3mm is left, so none of the three cylindrical blocks (diameter
33.9 / 36.4 / 37.5) can fit through — what the user sees is "the blocks won't drop through".

Box covering takes a different approach: voxelize the mesh, then use a greedy algorithm to merge the
filled voxels into a batch of boxes. Empty voxels can't grow anything, so **holes are preserved by
construction**; boxes are also MuJoCo's cheapest primitive. Measured on this box: 239-327 boxes
(voxelization isn't bit-for-bit deterministic, so the box count varies each time, but the three measured
hole diameters came out identical); hole diameters 37.9 / 31.2 / 28.3mm (true values 39.4 / 31.2 / 30.5),
all three blocks fit (3/3), and stepping cost only goes up 1%.

**Four gates — fail any one and it falls back to CoACD**:

1. **Voxel budget** of 500,000 cells, plus a **total-work budget** of "voxel cells × face count" at 300
   million pairs. `trimesh.contains` is ray-based, costing by **points × faces**: measured at about 70
   bytes/pair, 1.4e7 pairs/second. A fixed chunk size is unsafe here — a chunk of 100,000 points against a
   54,397-face mesh (a potted plant) is already 380 GB. Chunking is instead sized adaptively by face count
   (each chunk capped at 0.5 GB), which keeps peak memory at 14.3 GB on the largest real task.
2. **Volume fidelity**: the material volume measured from voxels must stay within 1.5x of the mesh's own
   volume, **checked on both sides**. Too much means the ray method misjudged some exterior space as solid
   on a degenerate mesh; too little means the wall is thinner than the voxel size and the covering wall
   leaks. The parts bin (`sft_plastic_box`) has a 2mm wall, and 4mm voxels only recover 0.28-0.42 of the
   material — this is exactly what this gate catches and falls back to CoACD for (the block sorting box
   measures 1.07 on the same scale). Falling back to CoACD costs the parts bin nothing,
   since it's an open box — what a convex piece eats away there is a corner, not a passage.
3. **Box-count budget** of 400 boxes.

The order can't be reversed: once CoACD has run, the hole has already been eaten away, and adding boxes
afterward can't bring it back. So box covering is tried first, and CoACD only as a fallback. The result of
either path is disk-cached by mesh content + parameters (the runtime temporary cache), and even the
conclusion "this collider doesn't suit box covering" is cached — voxelization itself takes seconds, but
the model has to be rebuilt on every reload.

**How the two thresholds were calibrated** (don't change these on a hunch). Must be
cut: shopping-cart solidity 0.012, shelf 0.038, parts bin 0.042, door frame 0.103, block sorting box 0.294;
cavity volume — door frame 0.263 m³, parts bin 0.0079, block sorting box 0.00216. Must be kept: door-handle
solidity 0.141 / cavity 0.000243, latch housing 0.170 / 0.000240, bottle label 0.342 / 0.000231, floor
0.863. The two groups differ by 8.9x on the **cavity** axis but only 1.2x on solidity — so the primary
criterion is cavity volume, and solidity is only a backstop: convex-hull volume alone measures "how big
the object is", not "how much cavity it encloses", and undercounts both the shopping cart and the parts
bin.

**The criterion for `scene_colliders.background_decor_static` is based on joints, not on "is it background"**: `open_door`'s
door is also in the background, but its door panel/latch/handle/frame are all referenced by joints, so the
whole door stays dynamic; the 70 bottles and jars in `laboratory_2` have no joints at all, which is why
they get demoted. As long as one part of the same asset (the same `.../entity` root) carries a joint, the
whole thing stays dynamic, avoiding a mechanism getting half-frozen. The cost of not demoting: measured on
`pour_workpiece`, nv=520, 298 of 311 startup interpenetration pairs are background props interpenetrating
each other, physics runs at 1.5 Hz, and reset can't return within the client's 60s limit.

**The cost and limit of tuning finer**: measured on the block sorting box, `TH=0.02 MAXH=64` recovers the
three top-face holes to 40.6/31.9/30.8 mm (true values 43.4/37.4/35.5) with no measurable change in step
cost; going finer still to `TH=0.01 MAXH=128` makes CoACD crash three times in a row (`exited -11`, an
OpenMP race inside CoACD), and the collider silently falls back to a
single convex hull — so finer isn't always better.


## B. Render diagnostics

These environment variables are read by the render package. They do not replace
physics profile settings. Some render optimizations are enabled by default.

| Variable | Default | Effect |
| --- | --- | --- |
| `RLINF_TILED_FABRIC` | `1` | Writes world poses into Fabric; `0` uses USD xform operations. Automatic fallback warns when `usdrt` cannot attach; inspect RGB for valid content. |
| `RLINF_TILED_DISABLE_VIEWPORT` | `1` | Disables the unused headless viewport. `0` enables it; observation cameras remain separate. |
| `RLINF_TILED_RT_SUBFRAMES` | unset | Sets a fixed RTX subframe count instead of adapting to elapsed physics steps. |
| `RLINF_TILED_RT_SUBFRAMES_MAX` | `8` | Caps the adaptive subframe count. |
| `RLINF_TILED_RT_SUBFRAMES_RESET` | `16` | Sets subframes for the first frame after reset. |
| `RLINF_TILED_SUBFRAME_LOG` | unset | A non-empty value logs the subframes used for each frame. |
| `RLINF_FUSED_RENDER_SWEEP` | unset | `1` probes render-driving methods at fused-node startup and reports pixel ranges. |

For render ownership and diagnostics, see [the render guide](../../genie_sim_render/AGENTS.md).

## Where the config file lives

The shared profiles are installed from `genie_sim_engine_core/config/`.

### Selecting a profile

Set `physics_params_file` under `genie_sim_engine.ros__parameters` in launcher YAML:

| Value | Resolution |
| --- | --- |
| Unset or empty | Base profile |
| Bare filename, such as `physics_params_rlinf.yaml` | File under the core package's `config/` directory |
| Absolute path | That profile file |

Use bare filenames for portable package profiles. Profile selection is logged;
a named file that does not exist is an error. `tests/params/test_profile_selection.py`
checks resolution and launcher selection.

## How to confirm a change actually took effect

Compare the selected configuration with startup readback and final solver state:

```text
[mjwarp-adapter] robot solver: SolverMuJoCo (iterations=default(100), substeps=24, ...)
[mjwarp-adapter] elliptic cone + impratio=30 (anti-creep: stiffer tangential friction)
[mjwarp-adapter] master forcerange capped to +-50 N*m on 2 actuator(s)
[mjc] gripper master_max_force=50 N*m (source=yaml)
```

For contact compliance, inspect `mjc_contact.py`'s post-finalize per-shape
`ke`/`kd` readback. The requested `solref` alone does not establish what the
solver built. Validate contact behavior and performance with the active scene.
