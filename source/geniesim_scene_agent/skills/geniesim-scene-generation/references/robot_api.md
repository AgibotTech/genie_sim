# RobotAPI quick reference for scene authors

This is a concise reference for making a generated scene usable by a `data_gen.py` task. Read the bundled [complete RobotAPI parameter reference](../../geniesim-task-orchestration/references/robot_api_reference.md) for full signatures, defaults, constraints and behavior. Scene generation alone does not require `RobotAPI`.

## Runtime contract

`run_llm_task(task_root_dir, task_name, scene_dir="random", run_times=2, use_recording=False, client_host="localhost:50051")` looks for `<task_root_dir>/<task_name>/<scene_dir>/scene.usda` and `scene_info.json`. The task directory also needs `task_info.json` and `data_gen.py` with `execute_task(robot) -> bool`. `RobotAPIConfig.from_json(task_info.json)` reads `task_name`, `task_description`, `robot`, `recording_setting`, and `task_metric`.

`SIM_ASSETS` must point to the importable assets package exporting `ASSETS_INDEX` and `ASSETS_PATH`. Each manipulable object's `scene_info.layout[object_id].usd` should resolve in `ASSETS_INDEX`; unknown IDs are skipped during object registration. `RobotAPI` addresses loaded objects at `/World/Objects/<object_id>/entity`. Verify this prim exists in the composed scene. Keep semantic roles and `keywords` stable across scene variants; instance IDs may change, so task code should select objects by role.

`scene_info.layout[object_id]` commonly includes `usd`, `xyz` in meters, `xyzw` quaternion, `keywords`, `tags`, and `description`. The scene metadata quaternion is **xyzw**. Robot initial pose in `task_info.robot.robot_init_pose.quaternion` is **wxyz**. `get_object_pose` returns **wxyz**. Convert explicitly when moving between them.

The pose getter reads a prim world transform, not necessarily a bounding-box
center. Articulated assets default to the body part. Size getters use asset
metadata; use actual mesh/support geometry for shifted origins, instance
scales, hollow interiors, shelves, and precise contacts.

## Frequently used methods

| Purpose | Call | Result / note |
| --- | --- | --- |
| Find scene objects | `robot.scene_info["layout"]` | Map of object IDs to scene metadata; filter on `keywords`. |
| Resolve a prim | `robot.get_object_prim_path(object_id, articulation_part=None)` | `/World/Objects/<id>/entity[/<part>]`. |
| Read object pose | `robot.get_object_pose(object_id, articulation_part=None)` | `(position_xyz, quaternion_wxyz)` arrays. |
| Read object size | `robot.get_object_size_in_world_frame(object_id)` | World-frame dimensions or `None`. |
| Choose an arm | `robot.judge_arm_to_use(position_xyz)` | `"left"` or `"right"`; near the middle it may sample randomly. |
| Pick | `robot.pick_actor(object_id, arm="right", grasp_config=GraspConfig())` | `bool`; `part=` supports articulated parts. |
| Place on another object | `robot.place_actor_to_target_actor(object_id, arm, target_id, place_config=PlaceConfig())` | `bool`; needs a suitable target interaction. |
| Place at explicit pose | `robot.place_actor_to_target_pose(object_id, arm, target_pose, place_config=PlaceConfig())` | `bool`; `target_pose` is a batch of 4×4 world transforms. |
| Pick and place | `robot.pick_and_place_actor_to_target_actor(...)` / `robot.pick_and_place_actor_to_target_pose(...)` | `bool`; combines the stages. |
| Move to a pose | `robot.move_actor_to_target_pose(actor, target_pose, arm="right", move_config=MoveConfig(), part=None)` | `bool`; `actor="gripper"` moves the free gripper. |
| Open articulation | `robot.open(object_id, joint_name=None, open_config=OpenConfig(), arm="right")` | `bool`; requires an articulated asset and an action description. |
| Joint motion | `robot.joint_to({joint_name: target}, steps=50, delta=False)` | `bool`; target values are absolute unless `delta=True`. |
| Arm reset / wait | `robot.back_to_origin(arm="")`, `robot.wait(seconds)` | `bool`; empty arm resets both arms. |
| Synchronized arms | `robot.dual_act(left=DualPick(...), right=DualPick(...), action_description={"zh": ..., "en": ...})` | `(success, store_name)`; also supports `DualPlaceToActor`, `DualPlaceToPose`, `DualMoveToPose`, `DualRotate`. |

`GraspConfig`, `PlaceConfig`, `MoveConfig`, `OpenConfig`, and the dual specs live in the same module. Supply `action_description={"zh": "...", "en": "..."}` for planned actions; describe the action without an arm subject. Use each Config's `unmovable_objects` and `extra_checkers` for physical success checks. For pose placement, use multiple reachable candidate transforms where possible. `robot.open()` needs the asset's actual revolute joint and should be checked against the observed joint state.

## Task authoring and grasps

Use [task orchestration](../../geniesim-task-orchestration/SKILL.md) and its
[task template](../../geniesim-task-orchestration/assets/data_gen_template.py).
Read the template's measured-geometry contract before adapting it. Use
compatible existing grasp annotations when there is no special requirement;
filter or derive custom contact poses for required directions/positions or
missing annotations. The [grasp guide](../../geniesim-task-orchestration/references/grasp-selection.md)
documents `<assets_root>/interaction/<asset_id>/interaction.json` and the
declared `grasp_pose/*.pkl` files, frame conversions, and an inspection tool.

Actor-to-actor placement requires suitable receiving annotations. For a target
without a matching place interaction, use explicit reachable object transforms.
Read [task validation](../../geniesim-task-validation/SKILL.md) before treating
the scene and task as ready for physical collection.
