# How collision shapes travel from USD to mujoco_warp: why there is so much compensation logic in between

**One line summary**: the collision shape written in an asset passes through three layers of conversion before it reaches the solver, and each layer loses something.
The compensation logic that looks scattered is not a pile of patches — it falls into **three distinct kinds of gap**, and each kind needs a different fix. "Just align directly with USD" is not achievable; the gap is not in mujoco_warp, it is in newton.

## Why this document exists

The reasoning behind this compensation logic is scattered across comments in about a dozen functions — `disable_swallowing_room_shells` talks about convex-hulling, `_decompose_scene_colliders` talks about the CoACD threshold, `depenetrate_spawn` talks about the difference between PhysX and mjwarp. Each comment is correct on its own, but none of them answers "why do we need this much". This document collects the full picture in one place, instead of leaving the next reader to grep through all of them and ask the same question again.

## 1. The pipeline: what each layer can express

```
USD / PhysX  →  newton's USD import  →  newton ModelBuilder shapes  →  mjwarp / MuJoCo
```

| Layer | Collision shapes it can express |
| --- | --- |
| **USD / PhysX** | exact triangle mesh (static), SDF, convex hull, convex decomposition, bounding box / bounding sphere, mesh simplification |
| **newton** | `BOX CAPSULE CONE CONVEX_MESH CYLINDER ELLIPSOID HFIELD MESH NONE PLANE SPHERE` |
| **MuJoCo / mjwarp** | all of the above, **plus `mjGEOM_SDF`** (mjwarp has an implementation of it in `collision_sdf.py`) |

Two hard constraints follow from this:

1. **newton's type table has no SDF.** MuJoCo has it, and mjwarp implements it too, but newton has no way to represent it, so it can't be passed downstream. The `physics:approximation = "sdf"` written in an asset (`open_door`'s door frame and `place_block_into_box`'s block sorting box both use it) **is already lost before it reaches the solver** — this is not something mujoco_warp fails to do.
2. **MuJoCo's `mesh` geom is convex when colliding.** For a concave object to be "hollow", it has to be split into multiple convex pieces.

## 2. Three kinds of gap (this is the real answer to "why so much")

### Class A: mesh representation and import

newton's `approximation_to_remeshing_method` (`newton/_src/utils/import_usd.py`) maps `physics:approximation` to a remeshing method, and it has three concrete problems:

1. **The table only has 5 values; anything not in it is skipped entirely** — the mesh goes into the solver unchanged and gets convex-hulled. And the "Unknown physics:approximation" message is only printed when `verbose=True` — the engine calls `add_usd` with `False`, so it's **completely silent**. Both `sdf` and `none` degrade this way.
2. **The CoACD threshold for `convexDecomposition` is hardcoded to 0.5** — too loose, so even badly concave objects often come out as a single block. Measured on the door frame: 0.5 gives 1 block (the doorway stays sealed), 0.1 gives 8 blocks (it opens up).
3. **`meshSimplification` depends on the optional package `fast_simplification`**, when it is unavailable, the log says `Falling back to bounding_box` — the shelf turns straight into a **solid rectangular box**, which is worse than a convex hull.

There's one more thing that isn't a defect but needs to be known: **`approximate_meshes` replaces the shape's mesh with its result**. Once it has run, the shopping cart is left in the builder with only a 390-vertex convex hull — the original 101,682 vertices are gone. So "decompose it afterward" can't recover it — the fix has to happen **before** `add_usd` runs.

### Class B: solver contact semantics

PhysX **clamps depenetration velocity** — the millimeter-scale overlaps an author leaves in place get quietly absorbed. mjwarp doesn't clamp, and sends the object flying instead. This class is a behavioral difference between the two engines, not a mistake on anyone's part.

### Class C: asset authoring assumptions

Assets were authored around PhysX, and some of those assumptions don't hold under mjwarp: decor is built as dynamic rigid bodies, props overlap each other, and objects' initial poses aren't resting states.

### List of compensation logic

| Logic | Where | Class | What it compensates for |
| --- | --- | --- | --- |
| `_collect_scene_colliders_to_decompose` | `engine/newton/setup/stage.py` | A | Takes concave colliders back from newton (blocks `approximation` so newton leaves them alone) |
| `_decompose_scene_colliders` | `engine/newton/setup/model.py` | A | Runs CoACD with **our own threshold**, replacing the single convex hull |
| `_decompose_gripper_pads` | `engine/newton/setup/model.py` | A | Convex decomposition of the gripper fingertip pads, so pinching is multi-point contact instead of a single point |
| `_strip_disabled_physics_schemas` | `engine/newton/setup/model.py` | A | Clears `approximation` off disabled colliders, otherwise the GL view draws the approximated shape instead of the mesh |
| `depenetrate_spawn` | `rlinf_geniesim/task_loader.py` | B | Objects sink a few millimeters into the supporting surface on spawn (measured 4.2mm) → lifted to leave a 2mm gap |
| `depenetrate_instance_poses` | `rlinf_geniesim/task_loader.py` | B | Same as above, but applied to **the pose table written back on every reset** — fixing only the scene only covers one episode |
| `disable_swallowing_room_shells` | `rlinf_geniesim/task_loader.py` | B | Room shells are concave; convex-hulling turns them into a solid brick that swallows the robot → their collision is turned off |
| Contact parameters split by geometry class | `engine/newton/mjc_contact.py` | B | Objects no longer tip over when they land on the table, without affecting grasping |
| `impratio` / elliptic friction cone | `common/params.py` | B | mjwarp's pyramidal cone has no true static friction, so a grasped object slowly creeps out |
| `_demote_background_decor` | `engine/newton/setup/stage.py` | C | Free-floating decor under `/Background` **with no joint reference at all** is demoted to a static collider |
| `_demote_kinematic_bodies` | `engine/newton/setup/stage.py` | C | Scene furniture with `kinematicEnabled=true` is demoted to a static collider |
| Object initial-pose baking | benchmark scene usda | C | Bakes placement poses into resting-state poses, cutting landing jitter by two to three orders of magnitude |

The layers need separate verification: a correct hull does not establish valid
spawn placement, and a stable spawn does not establish grasp contact.

## 3. Current collider construction

The scene stage blocks approximation for selected colliders before `add_usd`.
The model stage then constructs box covers or CoACD hulls with configured
thresholds. MJCF postprocessing runs after this import and cannot recover mesh
vertices already replaced by a hull. General SDF import is not implemented in
this path; extension work belongs in the [roadmap](../../../../../docs/ROADMAP.md).

How close this gets has been measured on the block sorting box (three holes on the top face, true equivalent diameters 43.4 / 37.4 / 35.5 mm):

| Setting | Through-holes on top face | Over-filled volume as a fraction of the true cavity |
| --- | --- | --- |
| newton default (threshold 0.5) | 1 block, hole completely sealed | — |
| our default `TH=0.05 MAXH=32` | 2-3 holes, 30-37 mm | 19% |
| `TH=0.02 MAXH=64` | 3 holes, 40.6 / 31.9 / 30.8 | 7.5% |
| `TH=0.01 MAXH=128` | 3 holes, 42.3 / 35.7 / 32.8 | 4.9% |

The finest setting **can't actually be used**: CoACD crashes three times in a row inside the engine (`exited -11`, the OpenMP race condition noted elsewhere in the `_run_coacd_isolated` comment), and the collider silently falls back to a single convex hull. So finer isn't always better.

> **Measure holes using the inscribed circle, not the equivalent diameter.** The table above uses equivalent diameter (area converted back to a circle), which makes an irregular hole sound larger than what can actually pass through it. What actually determines "can the block fall through" is the hole's **inscribed circle** — for the same box, the equivalent diameters are 43.4 / 37.4 / 35.5mm, but the inscribed diameters are only 39.4 / 31.2 / 30.5mm. The box-cover measurements below are all inscribed diameters: a CoACD result can look "big enough" by equivalent diameter while none of the three cylinders can actually fit through it.

### Voxelized box covering for shells
Even using fine CoACD settings still isn't enough — the true inscribed diameter of the circular hole is 39.4mm, but CoACD's default settings leave only 25.3mm, so none of the three cylindrical blocks (diameter 33.9 / 36.4 / 37.5) can fit through. The root cause is the method itself: every piece from convex decomposition is a **circumscribing** approximation, so a ring around the hole's edge inevitably gets eaten away.

So shell-type colliders take a different path: voxelize the mesh (4mm), then use a greedy algorithm to merge the filled voxels into a batch of axis-aligned boxes. Empty voxels can't grow anything, so holes are preserved **by construction**; boxes are also MuJoCo's cheapest primitive. Measured on the block sorting box this produces 239-327 boxes (voxelization on a non-watertight mesh isn't bit-for-bit deterministic, so the box count varies each time, but the three measured hole diameters came out identical). Once compiled into MuJoCo, the measured holes are 37.9 / 31.2 / 28.3mm, all three cylindrical blocks fit (3/3), and stepping cost only goes up by 1%.

The cost is that it only holds for **shells where the wall is thicker than the voxel size and the face count is not too high**, so there are four gates (voxel budget of 500,000 cells, total-work budget of "voxels × faces" at 300 million pairs, volume-fidelity within 1.5x on both sides, box-count cap of 400) — fail any one and it falls back to CoACD:

* The parts bin has a 2mm wall, and 4mm voxels only recover 0.28-0.42 of the material, so the covering wall would leak → caught by the volume-fidelity gate;
* The potted plant has 54,397 faces × 220,000 cells = 1.2e10 pairs, and the ray-casting method would take 862 seconds → caught by the total-work gate.
  Ray-query work scales with points × faces. Adaptive face-count chunking bounds
  memory; the recorded `clean_the_desktop` run used 14.3 GB at its measured peak.
Thresholds and calibration are in the A2 section of [`physics_knobs.md`](physics_knobs.md).

## 4. Why the criterion is "the measured shape", not "the declared value"

We don't look at what an asset declares, only measure what shape it **actually ends up as** — because three layers of distortion-prone conversion sit between the declaration and the final shape (see section 1 and Class A), and only measuring the result is trustworthy.

Two measures are used:

* **Solidity** = mesh volume / convex-hull volume. Measures "how far off a single convex hull is".
* **Cavity volume** = convex-hull volume − mesh volume. Measures "how much empty space it encloses" — **this is the quantity that actually decides whether a collider will swallow another object**.

Calibration data (units m³):

| | Solidity | Cavity volume |
| --- | --- | --- |
| Shopping cart | 0.012 | ~0.31 |
| Supermarket shelf | 0.038 | ~0.79 |
| Parts bin | 0.042 | 0.0079 |
| Door frame | 0.103 | 0.263 |
| **Block sorting box (smallest true container)** | **0.294** | **0.00216** |
| **Door handle (largest "should not be cut")** | 0.141 | **0.000243** |
| Latch housing | 0.170 | 0.000240 |
| Bottle label groove | 0.342 | 0.000231 |
| Room floor | 0.863 | very large |

The two groups differ by 8.9x on the **cavity** axis but only 1.2x on solidity. So the primary criterion is cavity volume (floor 0.001 m³); solidity (ceiling 0.5) is only there to catch the case where "the convex hull is basically already the mesh" — floor 0.863, bottle body 0.697. See the A2 section of `physics_knobs.md` for the knob names.

Convex-hull volume alone is the wrong dimension to threshold on: it measures "how big the object is", not "how much cavity it encloses", and undercounts both the shopping cart and the parts bin — cavity volume is what actually decides whether a collider swallows another object.

## 5. Calibration observations

These retained measurements explain the geometry thresholds; they are not a
current-image task acceptance matrix. Re-run checks for the selected runtime
and asset revision.

| Task | Recorded result with the collider policy |
| --- | --- |
| `open_door` | Doorway clear at four height levels |
| `stock_and_straighten_shelf` | Deepest sink 12.2 mm, zero NaN bodies, physics warm-up 354 ms |
| `pour_workpiece` | nv=100, 41.8 Hz, reset in 2.9 s |
| `take_wrong_item_shelf` | Spawn interpenetration 0.55 mm, zero objects falling over in the measured reset check |
| `place_block_into_box` | 269-box cover; measured inscribed hole diameters 37.9/31.2/28.3 mm; three blocks passed |

## 6. Verification

All of these criteria can be measured without relying on "does it look right":

* **Is the doorway / interior cavity open**: scatter probe points in the object's own coordinate frame and check, point by point, whether each falls inside **any** collision block's convex hull. Use the **local frame** — if the object is rotated, scattering points by world-space AABB will land some inside the wall and produce false positives.
* **How much of the hole got blocked**: after voxelizing, compare point by point "where the real mesh is solid" (`trimesh.contains`) against "where the decomposed blocks cover", then for each face compute the connected component that runs clear through, and take its **inscribed circle** (distance-transform peak × 2) as the hole diameter. Don't use equivalent diameter — it makes an irregular hole sound bigger than what can actually pass (see the callout box in section 3). The most convincing version **measures on the compiled MuJoCo model**: read every geom of that object straight out of `robot_runtime.xml` (mesh vertices / box corners) — that's the shape the solver actually sees, so nothing can hide behind some intermediate layer going wrong.
* **Spawn interpenetration**: run `mj_forward` once on the compiled MJCF, count the contact pairs with `dist < 0`, and classify them as "static×dynamic / dynamic×dynamic / robot self-contact" — only after classifying do you know which layer to fix.
* **Is it still sinking**: send the **correct** hold-pose command for a thousand-plus steps. Note that `states` is interleaved as `[q0, qd0, q1, qd1, …]`; `states[0::2]` gives the joint positions. Taking `states[:N]` instead gives a garbled command, and the robot will collapse — which looks like a physics problem but isn't one.
