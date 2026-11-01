# Pick the largest or smallest number block

Task: `pick_largest_or_smallest_number_g2`. Robot: `G2`.

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

`building_block` `table` `colored` `desktop_with_legs` `for_number_blocks` `numbered` `white` `workspace_table` `digit_1` `digit_1_block` `digit_3` `digit_3_block` `digit_6` `digit_6_block` `digit_2` `digit_2_block` `digit_5` `digit_5_block` `digit_8` `digit_8_block`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1754015860_d486440f` | [scene_info.json](1754015860_d486440f/scene_info.json) | [scene.usda](1754015860_d486440f/scene.usda) |
| `2022398358_d486440f` | [scene_info.json](2022398358_d486440f/scene_info.json) | [scene.usda](2022398358_d486440f/scene.usda) |
| `224091041_d486440f` | [scene_info.json](224091041_d486440f/scene_info.json) | [scene.usda](224091041_d486440f/scene.usda) |
| `2437697766_d486440f` | [scene_info.json](2437697766_d486440f/scene_info.json) | [scene.usda](2437697766_d486440f/scene.usda) |
| `262781369_d486440f` | [scene_info.json](262781369_d486440f/scene_info.json) | [scene.usda](262781369_d486440f/scene.usda) |
| `2848942432_d486440f` | [scene_info.json](2848942432_d486440f/scene_info.json) | [scene.usda](2848942432_d486440f/scene.usda) |
| `3099568228_d486440f` | [scene_info.json](3099568228_d486440f/scene_info.json) | [scene.usda](3099568228_d486440f/scene.usda) |
| `3594824469_d486440f` | [scene_info.json](3594824469_d486440f/scene_info.json) | [scene.usda](3594824469_d486440f/scene.usda) |
| `4169585675_d486440f` | [scene_info.json](4169585675_d486440f/scene_info.json) | [scene.usda](4169585675_d486440f/scene.usda) |
| `600264277_d486440f` | [scene_info.json](600264277_d486440f/scene_info.json) | [scene.usda](600264277_d486440f/scene.usda) |
