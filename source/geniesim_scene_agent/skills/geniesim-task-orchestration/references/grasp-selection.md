# Inspect and select grasps

## Locate annotated candidates

Resolve the actual asset ID from `scene_info.layout[object_id].usd`. The usual
external asset layout is:

```text
<assets_root>/interaction/<asset_id>/
  interaction.json
  grasp_pose/
    grasp_pose.pkl
```

Read `interaction.passive.grasp` in `interaction.json`:

```json
{"interaction": {"passive": {"grasp": {
  "default": ["grasp_pose/grasp_pose.pkl"],
  "handle": ["grasp_pose/handle.pkl"]
}}}}
```

Resolve the declared filenames relative to that asset's interaction directory;
there can be several labels and several files per label. A label can also
contain a single filename string. Do not assume the filename, that a directory
implies a usable pose, or that all annotated poses match the requested task.
The runtime resolves interaction files using the asset model directory basename;
normally it equals the asset ID. If an asset uses a different directory name,
check its `ASSETS_INDEX.url` and the corresponding interaction directory.

A grasp pickle normally contains `grasp_pose` as an `(N,4,4)` array and `width`
as an `(N,)` array in meters. Missing files, empty pose arrays, malformed
transforms, and mismatched widths are distinct failure cases. In the inspected
annotated loader, missing width uses 0.02 m per candidate. For **custom**
`GraspConfig.grasp_pose`, omitted width instead uses 0.08 m. These are fallback
values, not evidence of a suitable opening for the current object/gripper.

## Bundled inspection tool

Run from the repository root:

```bash
python skills/geniesim-task-orchestration/scripts/inspect_grasps.py \
  --assets-root "$GENIESIM_ASSETS" --asset-id benchmark_building_blocks_007

python skills/geniesim-task-orchestration/scripts/inspect_grasps.py \
  --assets-root "$GENIESIM_ASSETS" \
  --scene-info <instance>/scene_info.json --object-id <object_id> --load-poses
```

Without `--load-poses`, the tool reads JSON and checks file existence. With it,
the tool loads **trusted asset pickles** (pickle loading executes serialized
Python), validates finite homogeneous transforms and rotations, and reports
pose counts, translation bounds, and width ranges by label/file. It imports
NumPy only for pose loading. It writes no files and requires no RobotAPI,
service, or asset catalog. Inspect per-file status even when another file is
available. Counts are stored candidates, not a deduplicated pool or an IK result.

The report distinguishes `missing_annotation`, `no_grasp_labels`,
`missing_files`, `files_present_unchecked`, `empty`, `invalid`, and `available`.
`available` means structurally valid stored candidates exist. Geometry,
task constraints, collision, and reachability still need verification.

## Choose the strategy

| Condition | Strategy |
| --- | --- |
| Annotated poses, no specific contact/direction requirement | Let `pick_actor` select from compatible annotations |
| Specific contact region, direction, or articulated part | Check annotations against it; filter valid candidates or compute custom poses |
| Missing, empty, or unusable annotations | Infer grasp regions from mesh and generate custom candidates |

Filtering supports `GraspConfig.grasp_pose_filter(poses_world, object_pose,
widths) -> (filtered_poses, filtered_widths)`. Return aligned arrays, not a
boolean mask alone. Other filters include percentile bounds and axis/upside-down
options; their meaning depends on object axes and robot gripper conventions.

## Geometry-derived grasps

1. Inspect mesh bounds and surfaces using `scene_agent_assets inspect-usd`.
   Confirm the composed transform, scale, collision geometry, hollow space,
   handle/part geometry, and graspable thickness. Rendering helps verify a
   region but does not measure its grasp width.
2. Select opposing contact regions a parallel gripper can span. Avoid rims,
   sharp tips, occupied interiors, and surfaces that obstruct the finger sweep.
   Determine required contact region and orientation from the task and the
   next motion: pulling a drawer, holding a pen, or pouring a bottle may need
   different grasps of the same asset.
3. Build several **robot-gripper** target transforms with a valid rotation,
   contact translation, and opening within the actual gripper range. Do not
   assume a grasp transform is just the mesh center or that a generic world
   down-facing rotation is valid for every robot profile.
4. Check the pre-grasp approach, close motion, post-grasp lift, and downstream
   motion for clearance. Limit variation to contact regions that still satisfy
   the task. For an articulation, target the intended part rather than the
   stationary body.
5. Solve IK on gripper targets and verify the planned approach and physical
   grasp in simulation. Empty or unreachable pools fail with a reason; adjust
   the contact region, approach, robot/layout, or permitted candidates.

```python
grasp = GraspConfig(
    grasp_pose=robot_gripper_candidates,       # (N, 4, 4)
    grasp_pose_frame="world",                 # or "relative"
    grasp_width=opening_widths_m,              # (N,)
    flip_grasp=False,                         # preserve a task-required direction
    action_description={"zh": "grasp the handle", "en": "grasp the handle"},
)
success = robot.pick_actor(object_id, arm=arm, grasp_config=grasp, part=part)
```

`relative` uses the manipulated object's reference frame; with `part=` use
that part's reference frame. `world` uses the desired robot-gripper pose in
world coordinates. Custom candidates bypass the annotated orientation
conversion, but still undergo filters and IK. Set flip/upright filters
deliberately; they can modify or reject an otherwise task-specific candidate.

## Stored annotation frame is not the custom gripper frame

The runtime loads asset annotation poses, applies asset up-axis alignment,
annotation transforms/scales where defined, and converts the annotated
`grasp_gripper` orientation to `robot_gripper` before composing world poses.
Custom poses are already robot-gripper orientations, so that conversion is
skipped. Do not feed raw stored pickle matrices to the custom API without
performing and verifying those frame conversions. Prefer letting the runtime
load/filter annotations when reusing them.

**Annotated grasp poses** describe contact. A **pre-grasp pose** is an approach
pose derived from the contact pose and approach offset/vector. The directory
`grasp_pose/` does not promise a separate array of pre-grasp poses.
`pre_grasp_offset` controls approach offset; special world-direction approach
parameters are documented in the full API reference. Check the approach path
as well as the contact endpoint.
