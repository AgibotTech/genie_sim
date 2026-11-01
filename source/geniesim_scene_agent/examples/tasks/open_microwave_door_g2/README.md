# Open a microwave door

Task: `open_microwave_door_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_articulation_joint_info` `robot.get_object_pose` `robot.get_object_prim_path` `robot.move_actor_to_target_pose` `robot.open` `robot.remove_objs_from_obstacle`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`MoveConfig` `OpenConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `open_action` `table_only` `table_with_articulation`

## Scene keyword sample

`kitchen_appliance` `microwave` `desktop_with_legs` `gray` `rectangular` `table` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1242762849_49b042e3` | [scene_info.json](1242762849_49b042e3/scene_info.json) | [scene.usda](1242762849_49b042e3/scene.usda) |
| `2385251767_49b042e3` | [scene_info.json](2385251767_49b042e3/scene_info.json) | [scene.usda](2385251767_49b042e3/scene.usda) |
| `2400873466_49b042e3` | [scene_info.json](2400873466_49b042e3/scene_info.json) | [scene.usda](2400873466_49b042e3/scene.usda) |
| `2696235817_49b042e3` | [scene_info.json](2696235817_49b042e3/scene_info.json) | [scene.usda](2696235817_49b042e3/scene.usda) |
| `2794206328_49b042e3` | [scene_info.json](2794206328_49b042e3/scene_info.json) | [scene.usda](2794206328_49b042e3/scene.usda) |
| `3679029264_49b042e3` | [scene_info.json](3679029264_49b042e3/scene_info.json) | [scene.usda](3679029264_49b042e3/scene.usda) |
| `3865650719_49b042e3` | [scene_info.json](3865650719_49b042e3/scene_info.json) | [scene.usda](3865650719_49b042e3/scene.usda) |
| `4071959889_49b042e3` | [scene_info.json](4071959889_49b042e3/scene_info.json) | [scene.usda](4071959889_49b042e3/scene.usda) |
| `4085176620_49b042e3` | [scene_info.json](4085176620_49b042e3/scene_info.json) | [scene.usda](4085176620_49b042e3/scene.usda) |
| `4127542583_49b042e3` | [scene_info.json](4127542583_49b042e3/scene_info.json) | [scene.usda](4127542583_49b042e3/scene.usda) |
