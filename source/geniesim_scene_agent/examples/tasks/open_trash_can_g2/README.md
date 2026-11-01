# Open a trash can

Task: `open_trash_can_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.close_gripper` `robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_prim_path` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose`

### Client RPC

`robot.client.remove_objs_from_obstacle` `robot.client.set_task_task_basic_info`

### Configuration classes

`MoveConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `open_action` `table_with_articulation` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`table` `trash_can` `black` `cylinder` `desktop_with_legs` `waste_container` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1079539515_872e3622` | [scene_info.json](1079539515_872e3622/scene_info.json) | [scene.usda](1079539515_872e3622/scene.usda) |
| `1689102168_872e3622` | [scene_info.json](1689102168_872e3622/scene_info.json) | [scene.usda](1689102168_872e3622/scene.usda) |
| `1700329999_872e3622` | [scene_info.json](1700329999_872e3622/scene_info.json) | [scene.usda](1700329999_872e3622/scene.usda) |
| `2012696365_872e3622` | [scene_info.json](2012696365_872e3622/scene_info.json) | [scene.usda](2012696365_872e3622/scene.usda) |
| `2151993547_872e3622` | [scene_info.json](2151993547_872e3622/scene_info.json) | [scene.usda](2151993547_872e3622/scene.usda) |
| `3231711098_872e3622` | [scene_info.json](3231711098_872e3622/scene_info.json) | [scene.usda](3231711098_872e3622/scene.usda) |
| `3429772938_872e3622` | [scene_info.json](3429772938_872e3622/scene_info.json) | [scene.usda](3429772938_872e3622/scene.usda) |
| `3594868316_872e3622` | [scene_info.json](3594868316_872e3622/scene_info.json) | [scene.usda](3594868316_872e3622/scene.usda) |
| `4294921594_872e3622` | [scene_info.json](4294921594_872e3622/scene_info.json) | [scene.usda](4294921594_872e3622/scene.usda) |
| `554059594_872e3622` | [scene_info.json](554059594_872e3622/scene_info.json) | [scene.usda](554059594_872e3622/scene.usda) |
