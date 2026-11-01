# Pick the largest or smallest number block

Task: `pick_largest_or_smallest_number_g1`. Robot: `G1`.

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

`building_block` `table` `colored` `desktop_with_legs` `for_number_blocks` `numbered` `white` `workspace_table` `digit_0` `digit_0_block` `digit_1` `digit_1_block` `digit_2` `digit_2_block` `digit_4` `digit_4_block` `digit_8` `digit_8_block` `digit_3` `digit_3_block`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1676082592_d486440f` | [scene_info.json](1676082592_d486440f/scene_info.json) | [scene.usda](1676082592_d486440f/scene.usda) |
| `1759962033_d486440f` | [scene_info.json](1759962033_d486440f/scene_info.json) | [scene.usda](1759962033_d486440f/scene.usda) |
| `1873407152_d486440f` | [scene_info.json](1873407152_d486440f/scene_info.json) | [scene.usda](1873407152_d486440f/scene.usda) |
| `2622015211_d486440f` | [scene_info.json](2622015211_d486440f/scene_info.json) | [scene.usda](2622015211_d486440f/scene.usda) |
| `2993037434_d486440f` | [scene_info.json](2993037434_d486440f/scene_info.json) | [scene.usda](2993037434_d486440f/scene.usda) |
| `3218241283_d486440f` | [scene_info.json](3218241283_d486440f/scene_info.json) | [scene.usda](3218241283_d486440f/scene.usda) |
| `3495977382_d486440f` | [scene_info.json](3495977382_d486440f/scene_info.json) | [scene.usda](3495977382_d486440f/scene.usda) |
| `3650253035_d486440f` | [scene_info.json](3650253035_d486440f/scene_info.json) | [scene.usda](3650253035_d486440f/scene.usda) |
| `3836975646_d486440f` | [scene_info.json](3836975646_d486440f/scene_info.json) | [scene.usda](3836975646_d486440f/scene.usda) |
| `3986184116_d486440f` | [scene_info.json](3986184116_d486440f/scene_info.json) | [scene.usda](3986184116_d486440f/scene.usda) |
