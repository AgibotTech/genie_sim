# Sort number blocks in ascending order

Task: `sort_number_blocks_ascending_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `sort` `table_only` `unmovable_objects`

## Scene keyword sample

`table` `building_block` `educational_toy` `plastic` `rectangular` `small` `study_desk` `white` `workspace_table` `digit_5` `digit_7` `digit_3` `digit_6` `digit_0` `digit_1` `digit_4` `digit_8` `digit_9` `digit_2`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `2218023272_b6712dca` | [scene_info.json](2218023272_b6712dca/scene_info.json) | [scene.usda](2218023272_b6712dca/scene.usda) |
| `2291432974_b6712dca` | [scene_info.json](2291432974_b6712dca/scene_info.json) | [scene.usda](2291432974_b6712dca/scene.usda) |
| `2883251851_b6712dca` | [scene_info.json](2883251851_b6712dca/scene_info.json) | [scene.usda](2883251851_b6712dca/scene.usda) |
| `3075108520_b6712dca` | [scene_info.json](3075108520_b6712dca/scene_info.json) | [scene.usda](3075108520_b6712dca/scene.usda) |
| `3297860651_b6712dca` | [scene_info.json](3297860651_b6712dca/scene_info.json) | [scene.usda](3297860651_b6712dca/scene.usda) |
| `3581008725_b6712dca` | [scene_info.json](3581008725_b6712dca/scene_info.json) | [scene.usda](3581008725_b6712dca/scene.usda) |
| `3688693653_b6712dca` | [scene_info.json](3688693653_b6712dca/scene_info.json) | [scene.usda](3688693653_b6712dca/scene.usda) |
| `595360455_b6712dca` | [scene_info.json](595360455_b6712dca/scene_info.json) | [scene.usda](595360455_b6712dca/scene.usda) |
| `700840624_b6712dca` | [scene_info.json](700840624_b6712dca/scene_info.json) | [scene.usda](700840624_b6712dca/scene.usda) |
| `882933754_b6712dca` | [scene_info.json](882933754_b6712dca/scene_info.json) | [scene.usda](882933754_b6712dca/scene.usda) |
