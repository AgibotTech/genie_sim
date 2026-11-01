# Automatic collection task contract

## Files and responsibilities

```text
<task_root>/<english_task_name>/
  data_gen.py
  task_info.json
  manifest.json                 # bundled example inventory and interface tags
  README.md                     # bundled example description and interface tags
  <instance>/
    scene.usda
    scene_info.json
```

New generated scene instances use numeric instance directories and the scene
generation skill's additional generation, validation, and render files.
Retained examples may have opaque instance names. Keep task names consistent
between the directory, `task_info.task_name`, manifest, and accompanying docs.

The task runner selects the instance, constructs RobotAPI, manages recording,
calls `execute_task(robot)`, sends final status, and exits. The task function
selects targets, describes and executes the task, checks physical completion,
and returns a Python boolean. It should not construct another RobotAPI or
start/stop recording independently.

Robot execution requires an installed compatible collection runtime exposing
`client.llm_gen.robot_api` and an Isaac Sim RPC server. The skill and its
Python annotation inspector do not require a second project checkout.

## Asset configuration

- `GENIESIM_ASSETS`: external asset filesystem root used by this repository's
  Python authoring/inspection tools and example binding.
- `SIM_ASSETS`: the runtime's importable asset package directory. Its
  `__init__.py` must export `ASSETS_INDEX` and `ASSETS_PATH`.
- These may be the same root when that root implements the runtime asset
  package. A directory of USD files alone need not satisfy `SIM_ASSETS`.

Bind example USD references with `python examples/bind_assets.py`; it discovers the external asset root through configuration, the installed asset package, or relative directories. Use `--assets-root <assets>` to override it.
Do not write catalogs, grasp candidates, or renders into the source assets.

## Scene metadata

```json
{
  "scene_id": "example_task",
  "seed": 7,
  "layout": {
    "block_instance": {
      "usd": "benchmark_building_blocks_007",
      "xyz": [0.0, 0.0, 0.8],
      "xyzw": [0.0, 0.0, 0.0, 1.0],
      "keywords": ["block", "target_object"],
      "tags": ["object_group", "block_instance"]
    }
  }
}
```

The layout key is the **instance object ID**. `usd` is the **asset ID** used
to resolve `ASSETS_INDEX` and interaction annotations. A prim path is a third
identifier: `/World/Objects/<object_id>/entity`, optionally with a part suffix.
RobotAPI does not recognize every arbitrary prim as an indexed object.

Use semantic roles/keywords to locate instances. Use tags and graph ancestry
to disambiguate creation groups, then geometry to verify spatial claims.
The relation graph can use `links` or `edges`; its hierarchy does not prove
physical containment. Do not depend on UUIDs, particular object counts, or
the order of dictionary iteration. Validate required cardinalities and use
an explicit rule when several objects satisfy a selector.

`xyz`/`xyzw` are initial metadata, not live simulation state. Read current
poses after grasping, moving a support/container, or other contact. Keep the
same semantic role vocabulary across variants; instance IDs may change.
Read task-specific fields such as `target_letter` as part of the contract.
Do not override a scene-defined goal with a new random value at execution.

## Coordinates and geometry

- Distances: meters. World +Z is up; +X/+Y define forward/left for the canonical
  scene, while the robot base may be translated or rotated.
- Metadata quaternion: xyzw. Robot initial quaternion and runtime object
  pose quaternion: wxyz. `quat2mat` takes xyzw; `quat2mat_wxyz` takes wxyz.
- Homogeneous transforms are `(4,4)`, with translation in `T[:3,3]`.
- Pose target candidates normally have shape `(N,4,4)`; the target refers to
  the actor/object, or the gripper when `actor="gripper"`.
- Trajectories contain ordered xyz+wxyz waypoints `(N,7)` or alternative
  ordered paths `(M,N,7)`. Do not interchange these with pose candidate arrays.

`get_object_pose` reads a prim transform, not a measured box center. For an
articulated indexed asset it defaults to `body` if no part is supplied. Its
reference point can differ from the mesh center, bottom, and grasp annotation
origin. Read actual mesh bounds/offsets or verified support metadata before
using `position.z + half_height`.

The world size query constructs an AABB from indexed local dimensions and
orientation; it does not scan the composed mesh or necessarily account for
an instance scale. The local size query is also asset metadata based. For
scaled assets, non-centered geometry, hollow interiors, shelves, and precise
contact tasks use the repository's mesh/support tools and verify in simulation.

## Interaction data

`robot.interactions` is keyed by **asset ID**, usually with direct `active`
and `passive` sections. On disk `interaction.json` usually wraps them in an
`interaction` field. Normalize that wrapper when reading files.

- `passive.grasp`: how this object can be grasped.
- `active.place`: how this manipulated object participates in placement.
- `passive.place`: receiving points on the target.

Use labels actually present in the current asset. Leave a primitive unspecified
to allow the runtime to traverse compatible annotations when the task does
not impose a particular label. Use explicit object target poses when receiving
annotations are absent or represent the wrong task relationship. A missing
grasp label can be replaced by a geometry-derived custom grasp.

## Descriptions and recording settings

`task_info.json` contains these runtime fields:

| Field | Type / default | Meaning |
| --- | --- | --- |
| `task_name` | string / empty | English task identifier, consistent with its directory |
| `task_description` | object / empty | `task_name`, `english_task_name`, `init_scene_text` text fields |
| `robot` | object / runtime G2 profile | `robot_cfg`, initial pose, joint initialization/noise, optional anchor links |
| `auto_record` | boolean / false | Recording request; the runner may override it |
| `recording_setting` | object / empty | Camera list, fps, semantic-render flag and additional recorder options |
| `task_metric` | object / empty | Runtime-supported quality/filter configuration |

Preserve a compatible robot profile from the chosen example; robot joint
names and camera prims differ by profile. `robot_init_pose.position` is xyz
in meters and `.quaternion` is wxyz. Optional `.random.delta_position` samples
independent symmetric uniform position offsets. `init_arm_pose` maps joint
names to runtime units, and `init_arm_pose_noise` uses the runtime's supported
noise schema. `gripper_anchor_link` may map `left`/`right` to custom anchor prims.

The recording settings consumed directly by RobotAPI are `fps` (default 30),
`camera_list` (default empty), and `render_semantic` (default false). Depth is
disabled in this constructor path. The complete settings object is also
forwarded as additional recording parameters; unknown keys are not guaranteed
to be supported. `task_metric` is forwarded as supplied, not validated by this
skill. Derive metric rule names from the installed runtime or compatible
examples instead of inventing them.

Before the first action, resolve the actual goal and set:

```python
robot.client.set_task_task_basic_info({
    "task_name": task_text,
    "english_task_name": english_task_text,
    "init_scene_text": initial_scene_text,
})
```

Both text variables may be English for English-only tasks. Keep
`init_scene_text` English. For actions, provide a nonempty description with
`zh`/`en` keys; language keys are schema, not control parameters. Describe the
action without an arm prefix, which is usually added by the runtime. Keep
the instruction truthful about object selection and the full task sequence.

## Bundled template geometry contract

The container-placement template is an executable pattern for an explicitly
measured task contract, not a universal interior detector. It uses exactly one
`target_object` and one `target_container` keyword match by default; override
`target_role`/`receiver_role` in `scene_info.task_contract` for other roles.

Its required `task_contract` fields are `object_bounds_local` and
`receiver_interior_bounds_local`: each is `[min_xyz,max_xyz]`, measured in the
same local reference frames returned by the runtime pose queries. Include
actual instance scale when measuring. This box-interior recipe only applies
when that box describes a usable interior region; use task-specific geometry
for other shapes. Optional fields are `containment_clearance_m` (nonnegative,
default 0), `settle_seconds` (default 0.5), and `specific_grasp_required` (default
false). With a custom grasp, supply `custom_grasp.poses` `(N,4,4)`,
`.widths_m` `(N,)`, and `.frame` (default relative). These are template fields,
not built-in RobotAPI metadata requirements. Generate/recompute world grasps
at runtime when initial-state variability changes their contact frame.

The template refuses an unspecified task-specific grasp or missing grasp
annotation, so the author must implement geometry inference or supply verified
custom candidates. Its final checker transforms all measured object-box
corners into the current receiver frame and checks interior bounds after
release/reset/settling. Adapt it for support, articulation, contact, or other
task completion instead of reusing a container check indiscriminately.

## Local execution

For the default user demonstration, apply
[data collection](../../geniesim-data-collection/SKILL.md) and its bundled
[CLI reference](../../geniesim-data-collection/references/cli.md). The host
`geniesim data-collection run` workflow manages the server/client, copies tasks
before rebinding USD references, and records configured camera videos directly
with `--render`. Recording is enabled by default in this high-level workflow.

The lower-level installed runtime exposes:

```python
from client.llm_gen.launcher import run_llm_task

run_llm_task(
    task_root_dir="examples/tasks",
    task_name="open_microwave_door_g2",
    scene_dir="<chosen_instance>",
    run_times=1,
    use_recording=False,
    client_host="localhost:50051",
)
```

Always supply an explicit task root/name; avoid runtime machine-specific
defaults. A fixed scene is best for initial diagnosis. `"random"` selects a
scene per run. `run_times` counts executions, not generated scene instances.
The lower-level runner preserves initial robot pose/noise information between
repeated runs via `from_robot`; scene selection randomness and robot pose
randomness are different mechanisms. Reproducibility requires recording which
controls were actually seeded, not merely the scene metadata seed.

The launcher modifies USD asset references to match `SIM_ASSETS`, overrides
`auto_record` with `use_recording`, calls the task, stops recording, sends
`success` and `[stage_id,-1]`, and exits after its loop. In the inspected current version its CLI
defaults recording off; pass `--use_recording` explicitly for that path. It lacks an encompassing cleanup handler if task code
raises and caches the `data_gen` module import. Use separate processes for
different tasks and a writable test copy when required. The current high-level collection CLI uses isolated
episode processes with explicit module loading and cleanup. Prefer it over
copying these legacy limitations into new task code.
