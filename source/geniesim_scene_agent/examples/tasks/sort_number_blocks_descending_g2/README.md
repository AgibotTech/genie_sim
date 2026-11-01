# Sort number blocks in descending order

Task: `sort_number_blocks_descending_g2`. Robot: `G2`.

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

`table` `building_block` `educational_toy` `plastic` `rectangular` `small` `study_desk` `white` `workspace_table` `digit_2` `digit_4` `digit_0` `digit_3` `digit_5` `digit_7` `digit_6` `digit_9` `digit_1` `digit_8`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1215959342_b6712dca` | [scene_info.json](1215959342_b6712dca/scene_info.json) | [scene.usda](1215959342_b6712dca/scene.usda) |
| `1777343724_b6712dca` | [scene_info.json](1777343724_b6712dca/scene_info.json) | [scene.usda](1777343724_b6712dca/scene.usda) |
| `2012546980_b6712dca` | [scene_info.json](2012546980_b6712dca/scene_info.json) | [scene.usda](2012546980_b6712dca/scene.usda) |
| `2032783734_b6712dca` | [scene_info.json](2032783734_b6712dca/scene_info.json) | [scene.usda](2032783734_b6712dca/scene.usda) |
| `2673512053_b6712dca` | [scene_info.json](2673512053_b6712dca/scene_info.json) | [scene.usda](2673512053_b6712dca/scene.usda) |
| `3245902762_b6712dca` | [scene_info.json](3245902762_b6712dca/scene_info.json) | [scene.usda](3245902762_b6712dca/scene.usda) |
| `3422892037_b6712dca` | [scene_info.json](3422892037_b6712dca/scene_info.json) | [scene.usda](3422892037_b6712dca/scene.usda) |
| `3973091440_b6712dca` | [scene_info.json](3973091440_b6712dca/scene_info.json) | [scene.usda](3973091440_b6712dca/scene.usda) |
| `4103960331_b6712dca` | [scene_info.json](4103960331_b6712dca/scene_info.json) | [scene.usda](4103960331_b6712dca/scene.usda) |
| `4110343444_b6712dca` | [scene_info.json](4110343444_b6712dca/scene_info.json) | [scene.usda](4110343444_b6712dca/scene.usda) |
