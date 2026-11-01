# Open a trash can and discard garbage

Task: `open_trash_can_and_discard_garbage_g2`. Robot: `G2`.

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

`robot.client.remove_objs_from_obstacle` `robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `open_action` `pick_and_place` `table_with_articulation` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`table` `trash` `trash_can` `black` `cylinder` `desktop_with_legs` `litter` `on_table` `waste` `waste_container` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1142874582_1292401d` | [scene_info.json](1142874582_1292401d/scene_info.json) | [scene.usda](1142874582_1292401d/scene.usda) |
| `1419267295_1292401d` | [scene_info.json](1419267295_1292401d/scene_info.json) | [scene.usda](1419267295_1292401d/scene.usda) |
| `1513848659_1292401d` | [scene_info.json](1513848659_1292401d/scene_info.json) | [scene.usda](1513848659_1292401d/scene.usda) |
| `1683305026_1292401d` | [scene_info.json](1683305026_1292401d/scene_info.json) | [scene.usda](1683305026_1292401d/scene.usda) |
| `1707866049_1292401d` | [scene_info.json](1707866049_1292401d/scene_info.json) | [scene.usda](1707866049_1292401d/scene.usda) |
| `2451160132_1292401d` | [scene_info.json](2451160132_1292401d/scene_info.json) | [scene.usda](2451160132_1292401d/scene.usda) |
| `2482628179_1292401d` | [scene_info.json](2482628179_1292401d/scene_info.json) | [scene.usda](2482628179_1292401d/scene.usda) |
| `2547838482_1292401d` | [scene_info.json](2547838482_1292401d/scene_info.json) | [scene.usda](2547838482_1292401d/scene.usda) |
| `2560860927_1292401d` | [scene_info.json](2560860927_1292401d/scene_info.json) | [scene.usda](2560860927_1292401d/scene.usda) |
| `3490158150_1292401d` | [scene_info.json](3490158150_1292401d/scene_info.json) | [scene.usda](3490158150_1292401d/scene.usda) |
