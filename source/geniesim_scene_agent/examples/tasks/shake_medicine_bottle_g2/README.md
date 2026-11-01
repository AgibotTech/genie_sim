# Shake a medicine bottle

Task: `shake_medicine_bottle_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_ee_pose` `robot.get_object_pose` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.shake`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `ShakeConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `pick` `shake` `table_only` `unmovable_objects`

## Scene keyword sample

`medicine_bottle` `table` `left` `bottle` `cylindrical` `desktop_with_legs` `medical` `on_desk` `wood` `workspace_table` `right`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `118006308_f26ad8fd` | [scene_info.json](118006308_f26ad8fd/scene_info.json) | [scene.usda](118006308_f26ad8fd/scene.usda) |
| `1200150437_f26ad8fd` | [scene_info.json](1200150437_f26ad8fd/scene_info.json) | [scene.usda](1200150437_f26ad8fd/scene.usda) |
| `1304660867_f26ad8fd` | [scene_info.json](1304660867_f26ad8fd/scene_info.json) | [scene.usda](1304660867_f26ad8fd/scene.usda) |
| `1584751185_f26ad8fd` | [scene_info.json](1584751185_f26ad8fd/scene_info.json) | [scene.usda](1584751185_f26ad8fd/scene.usda) |
| `1752136928_f26ad8fd` | [scene_info.json](1752136928_f26ad8fd/scene_info.json) | [scene.usda](1752136928_f26ad8fd/scene.usda) |
| `2516741038_f26ad8fd` | [scene_info.json](2516741038_f26ad8fd/scene_info.json) | [scene.usda](2516741038_f26ad8fd/scene.usda) |
| `2715958854_f26ad8fd` | [scene_info.json](2715958854_f26ad8fd/scene_info.json) | [scene.usda](2715958854_f26ad8fd/scene.usda) |
| `3197186235_f26ad8fd` | [scene_info.json](3197186235_f26ad8fd/scene_info.json) | [scene.usda](3197186235_f26ad8fd/scene.usda) |
| `3554271359_f26ad8fd` | [scene_info.json](3554271359_f26ad8fd/scene_info.json) | [scene.usda](3554271359_f26ad8fd/scene.usda) |
| `942930563_f26ad8fd` | [scene_info.json](942930563_f26ad8fd/scene_info.json) | [scene.usda](942930563_f26ad8fd/scene.usda) |
