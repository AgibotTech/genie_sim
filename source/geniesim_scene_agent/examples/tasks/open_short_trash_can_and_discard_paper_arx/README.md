# Open a short trash can and discard a paper ball

Task: `open_short_trash_can_and_discard_paper_arx`. Robot: `ARX`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.close_gripper` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_local_frame` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.place_actor_to_target_pose`

### Client RPC

`robot.client.remove_objs_from_obstacle` `robot.client.run_step` `robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `open_action` `pick_and_place` `table_with_articulation` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`left` `paper_ball` `right` `table` `trash` `trash_can` `on_table` `short` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `scene_269` | [scene_info.json](scene_269/scene_info.json) | [scene.usda](scene_269/scene.usda) |
| `scene_476` | [scene_info.json](scene_476/scene_info.json) | [scene.usda](scene_476/scene.usda) |
| `scene_500` | [scene_info.json](scene_500/scene_info.json) | [scene.usda](scene_500/scene.usda) |
| `scene_533` | [scene_info.json](scene_533/scene_info.json) | [scene.usda](scene_533/scene.usda) |
| `scene_557` | [scene_info.json](scene_557/scene_info.json) | [scene.usda](scene_557/scene.usda) |
| `scene_644` | [scene_info.json](scene_644/scene_info.json) | [scene.usda](scene_644/scene.usda) |
| `scene_718` | [scene_info.json](scene_718/scene_info.json) | [scene.usda](scene_718/scene.usda) |
| `scene_760` | [scene_info.json](scene_760/scene_info.json) | [scene.usda](scene_760/scene.usda) |
| `scene_939` | [scene_info.json](scene_939/scene_info.json) | [scene.usda](scene_939/scene.usda) |
| `scene_998` | [scene_info.json](scene_998/scene_info.json) | [scene.usda](scene_998/scene.usda) |
