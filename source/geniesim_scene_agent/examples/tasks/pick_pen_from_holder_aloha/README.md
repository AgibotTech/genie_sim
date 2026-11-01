# Pick a pen from its holder

Task: `pick_pen_from_holder_aloha`. Robot: `Aloha`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig`

## Scene and action tags

`keywords` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`pen` `pen_holder` `table` `container` `cylinder` `desktop_with_legs` `long` `stationery` `white` `workspace_table` `writing_tool`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1164896803_7a34fd7b` | [scene_info.json](1164896803_7a34fd7b/scene_info.json) | [scene.usda](1164896803_7a34fd7b/scene.usda) |
| `1174459548_7a34fd7b` | [scene_info.json](1174459548_7a34fd7b/scene_info.json) | [scene.usda](1174459548_7a34fd7b/scene.usda) |
| `119956052_7a34fd7b` | [scene_info.json](119956052_7a34fd7b/scene_info.json) | [scene.usda](119956052_7a34fd7b/scene.usda) |
| `1523598545_7a34fd7b` | [scene_info.json](1523598545_7a34fd7b/scene_info.json) | [scene.usda](1523598545_7a34fd7b/scene.usda) |
| `2187162832_7a34fd7b` | [scene_info.json](2187162832_7a34fd7b/scene_info.json) | [scene.usda](2187162832_7a34fd7b/scene.usda) |
| `2244168767_7a34fd7b` | [scene_info.json](2244168767_7a34fd7b/scene_info.json) | [scene.usda](2244168767_7a34fd7b/scene.usda) |
| `2572712346_7a34fd7b` | [scene_info.json](2572712346_7a34fd7b/scene_info.json) | [scene.usda](2572712346_7a34fd7b/scene.usda) |
| `2844227931_7a34fd7b` | [scene_info.json](2844227931_7a34fd7b/scene_info.json) | [scene.usda](2844227931_7a34fd7b/scene.usda) |
| `3305959883_7a34fd7b` | [scene_info.json](3305959883_7a34fd7b/scene_info.json) | [scene.usda](3305959883_7a34fd7b/scene.usda) |
| `3328603511_7a34fd7b` | [scene_info.json](3328603511_7a34fd7b/scene_info.json) | [scene.usda](3328603511_7a34fd7b/scene.usda) |
