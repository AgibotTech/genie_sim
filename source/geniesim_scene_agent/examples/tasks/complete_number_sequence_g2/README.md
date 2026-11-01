# Complete a number sequence

Task: `complete_number_sequence_g2`. Robot: `G2`.

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

`keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`scattered` `table` `building_block` `candidate` `digit_2` `digit_3` `digit_4` `in_sequence` `left` `workspace_table` `digit_1` `right` `digit_0` `digit_5` `digit_8` `digit_7`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `148041596_eed3ce0a` | [scene_info.json](148041596_eed3ce0a/scene_info.json) | [scene.usda](148041596_eed3ce0a/scene.usda) |
| `1964656000_eed3ce0a` | [scene_info.json](1964656000_eed3ce0a/scene_info.json) | [scene.usda](1964656000_eed3ce0a/scene.usda) |
| `2063803066_eed3ce0a` | [scene_info.json](2063803066_eed3ce0a/scene_info.json) | [scene.usda](2063803066_eed3ce0a/scene.usda) |
| `2604127814_eed3ce0a` | [scene_info.json](2604127814_eed3ce0a/scene_info.json) | [scene.usda](2604127814_eed3ce0a/scene.usda) |
| `3295276277_eed3ce0a` | [scene_info.json](3295276277_eed3ce0a/scene_info.json) | [scene.usda](3295276277_eed3ce0a/scene.usda) |
| `3453196601_eed3ce0a` | [scene_info.json](3453196601_eed3ce0a/scene_info.json) | [scene.usda](3453196601_eed3ce0a/scene.usda) |
| `3814863304_eed3ce0a` | [scene_info.json](3814863304_eed3ce0a/scene_info.json) | [scene.usda](3814863304_eed3ce0a/scene.usda) |
| `4117695138_eed3ce0a` | [scene_info.json](4117695138_eed3ce0a/scene_info.json) | [scene.usda](4117695138_eed3ce0a/scene.usda) |
| `4252543393_eed3ce0a` | [scene_info.json](4252543393_eed3ce0a/scene_info.json) | [scene.usda](4252543393_eed3ce0a/scene.usda) |
| `945408773_eed3ce0a` | [scene_info.json](945408773_eed3ce0a/scene_info.json) | [scene.usda](945408773_eed3ce0a/scene.usda) |
