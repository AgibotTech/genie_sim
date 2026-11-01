# Arrange cosmetics in a line

Task: `arrange_cosmetics_in_line_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`cosmetics` `table` `cosmetic_0` `cosmetic_1` `cosmetic_2` `furniture` `left` `on_desk` `right` `workspace` `cosmetic_3`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1054407681_3177c826` | [scene_info.json](1054407681_3177c826/scene_info.json) | [scene.usda](1054407681_3177c826/scene.usda) |
| `1129432213_3177c826` | [scene_info.json](1129432213_3177c826/scene_info.json) | [scene.usda](1129432213_3177c826/scene.usda) |
| `152178252_3177c826` | [scene_info.json](152178252_3177c826/scene_info.json) | [scene.usda](152178252_3177c826/scene.usda) |
| `1674184299_3177c826` | [scene_info.json](1674184299_3177c826/scene_info.json) | [scene.usda](1674184299_3177c826/scene.usda) |
| `2409108852_3177c826` | [scene_info.json](2409108852_3177c826/scene_info.json) | [scene.usda](2409108852_3177c826/scene.usda) |
| `51460454_3177c826` | [scene_info.json](51460454_3177c826/scene_info.json) | [scene.usda](51460454_3177c826/scene.usda) |
| `664909543_3177c826` | [scene_info.json](664909543_3177c826/scene_info.json) | [scene.usda](664909543_3177c826/scene.usda) |
| `812091175_3177c826` | [scene_info.json](812091175_3177c826/scene_info.json) | [scene.usda](812091175_3177c826/scene.usda) |
| `824053860_3177c826` | [scene_info.json](824053860_3177c826/scene_info.json) | [scene.usda](824053860_3177c826/scene.usda) |
| `960341929_3177c826` | [scene_info.json](960341929_3177c826/scene_info.json) | [scene.usda](960341929_3177c826/scene.usda) |
