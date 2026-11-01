# Pick the largest or smallest number block

Task: `pick_largest_or_smallest_number_arx`. Robot: `ARX`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig`

## Scene and action tags

`keywords` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`building_block` `table` `colored` `desktop_with_legs` `for_number_blocks` `numbered` `white` `workspace_table` `digit_5` `digit_5_block` `digit_2` `digit_2_block` `digit_6` `digit_6_block` `digit_9` `digit_9_block` `digit_1` `digit_1_block` `digit_3` `digit_3_block`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1261178633_d486440f` | [scene_info.json](1261178633_d486440f/scene_info.json) | [scene.usda](1261178633_d486440f/scene.usda) |
| `1414892885_d486440f` | [scene_info.json](1414892885_d486440f/scene_info.json) | [scene.usda](1414892885_d486440f/scene.usda) |
| `1550602827_d486440f` | [scene_info.json](1550602827_d486440f/scene_info.json) | [scene.usda](1550602827_d486440f/scene.usda) |
| `1754015860_d486440f` | [scene_info.json](1754015860_d486440f/scene_info.json) | [scene.usda](1754015860_d486440f/scene.usda) |
| `2022738965_d486440f` | [scene_info.json](2022738965_d486440f/scene_info.json) | [scene.usda](2022738965_d486440f/scene.usda) |
| `2024314788_d486440f` | [scene_info.json](2024314788_d486440f/scene_info.json) | [scene.usda](2024314788_d486440f/scene.usda) |
| `2167630538_d486440f` | [scene_info.json](2167630538_d486440f/scene_info.json) | [scene.usda](2167630538_d486440f/scene.usda) |
| `2208369447_d486440f` | [scene_info.json](2208369447_d486440f/scene_info.json) | [scene.usda](2208369447_d486440f/scene.usda) |
| `2967257042_d486440f` | [scene_info.json](2967257042_d486440f/scene_info.json) | [scene.usda](2967257042_d486440f/scene.usda) |
| `806223255_d486440f` | [scene_info.json](806223255_d486440f/scene_info.json) | [scene.usda](806223255_d486440f/scene.usda) |
