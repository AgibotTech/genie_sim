# Open a trash can

Task: `open_trash_can_aloha`. Robot: `Aloha`.

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
| `1054812535_872e3622` | [scene_info.json](1054812535_872e3622/scene_info.json) | [scene.usda](1054812535_872e3622/scene.usda) |
| `156103270_872e3622` | [scene_info.json](156103270_872e3622/scene_info.json) | [scene.usda](156103270_872e3622/scene.usda) |
| `1657207123_872e3622` | [scene_info.json](1657207123_872e3622/scene_info.json) | [scene.usda](1657207123_872e3622/scene.usda) |
| `1730453883_872e3622` | [scene_info.json](1730453883_872e3622/scene_info.json) | [scene.usda](1730453883_872e3622/scene.usda) |
| `1831197395_872e3622` | [scene_info.json](1831197395_872e3622/scene_info.json) | [scene.usda](1831197395_872e3622/scene.usda) |
| `2971377856_872e3622` | [scene_info.json](2971377856_872e3622/scene_info.json) | [scene.usda](2971377856_872e3622/scene.usda) |
| `4155782372_872e3622` | [scene_info.json](4155782372_872e3622/scene_info.json) | [scene.usda](4155782372_872e3622/scene.usda) |
| `46549495_872e3622` | [scene_info.json](46549495_872e3622/scene_info.json) | [scene.usda](46549495_872e3622/scene.usda) |
| `495124220_872e3622` | [scene_info.json](495124220_872e3622/scene_info.json) | [scene.usda](495124220_872e3622/scene.usda) |
| `711434317_872e3622` | [scene_info.json](711434317_872e3622/scene_info.json) | [scene.usda](711434317_872e3622/scene.usda) |
