# Solve arithmetic with number blocks

Task: `solve_arithmetic_with_number_blocks_g2`. Robot: `G2`.

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

`table` `block_0` `block_1` `block_2` `block_3` `building_block` `colored` `desktop_with_legs` `white` `workspace_table` `digit_8` `number_block_8` `digit_3` `digit_6` `number_block_3` `number_block_6` `digit_0` `digit_1` `digit_4` `number_block_0`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1040613359_22feebc7` | [scene_info.json](1040613359_22feebc7/scene_info.json) | [scene.usda](1040613359_22feebc7/scene.usda) |
| `1869961590_22feebc7` | [scene_info.json](1869961590_22feebc7/scene_info.json) | [scene.usda](1869961590_22feebc7/scene.usda) |
| `2034248385_22feebc7` | [scene_info.json](2034248385_22feebc7/scene_info.json) | [scene.usda](2034248385_22feebc7/scene.usda) |
| `2057790164_22feebc7` | [scene_info.json](2057790164_22feebc7/scene_info.json) | [scene.usda](2057790164_22feebc7/scene.usda) |
| `2293694900_22feebc7` | [scene_info.json](2293694900_22feebc7/scene_info.json) | [scene.usda](2293694900_22feebc7/scene.usda) |
| `2737099704_22feebc7` | [scene_info.json](2737099704_22feebc7/scene_info.json) | [scene.usda](2737099704_22feebc7/scene.usda) |
| `376899710_22feebc7` | [scene_info.json](376899710_22feebc7/scene_info.json) | [scene.usda](376899710_22feebc7/scene.usda) |
| `89255232_22feebc7` | [scene_info.json](89255232_22feebc7/scene_info.json) | [scene.usda](89255232_22feebc7/scene.usda) |
| `922465434_22feebc7` | [scene_info.json](922465434_22feebc7/scene_info.json) | [scene.usda](922465434_22feebc7/scene.usda) |
| `9850775_22feebc7` | [scene_info.json](9850775_22feebc7/scene_info.json) | [scene.usda](9850775_22feebc7/scene.usda) |
