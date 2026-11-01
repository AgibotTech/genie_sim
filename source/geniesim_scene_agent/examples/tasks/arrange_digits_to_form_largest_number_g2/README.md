# Arrange digit blocks to form the largest number

Task: `arrange_digits_to_form_largest_number_g2`. Robot: `G2`.

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

`table` `building_block` `educational_toy` `plastic` `rectangular` `small` `study_desk` `white` `workspace_table` `digit_0` `digit_1` `digit_3` `digit_4` `digit_5` `digit_8` `digit_7` `digit_2`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1408844627_2278a825` | [scene_info.json](1408844627_2278a825/scene_info.json) | [scene.usda](1408844627_2278a825/scene.usda) |
| `149597638_2278a825` | [scene_info.json](149597638_2278a825/scene_info.json) | [scene.usda](149597638_2278a825/scene.usda) |
| `1871807915_2278a825` | [scene_info.json](1871807915_2278a825/scene_info.json) | [scene.usda](1871807915_2278a825/scene.usda) |
| `1978950285_2278a825` | [scene_info.json](1978950285_2278a825/scene_info.json) | [scene.usda](1978950285_2278a825/scene.usda) |
| `2043838181_2278a825` | [scene_info.json](2043838181_2278a825/scene_info.json) | [scene.usda](2043838181_2278a825/scene.usda) |
| `2154229148_2278a825` | [scene_info.json](2154229148_2278a825/scene_info.json) | [scene.usda](2154229148_2278a825/scene.usda) |
| `3204232883_2278a825` | [scene_info.json](3204232883_2278a825/scene_info.json) | [scene.usda](3204232883_2278a825/scene.usda) |
| `4138184920_2278a825` | [scene_info.json](4138184920_2278a825/scene_info.json) | [scene.usda](4138184920_2278a825/scene.usda) |
| `46150973_2278a825` | [scene_info.json](46150973_2278a825/scene_info.json) | [scene.usda](46150973_2278a825/scene.usda) |
| `873829996_2278a825` | [scene_info.json](873829996_2278a825/scene_info.json) | [scene.usda](873829996_2278a825/scene.usda) |
