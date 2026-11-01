# Pick the largest or smallest number block

Task: `pick_largest_or_smallest_number_aloha`. Robot: `Aloha`.

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

`building_block` `table` `colored` `desktop_with_legs` `for_number_blocks` `numbered` `white` `workspace_table` `digit_0` `digit_0_block` `digit_3` `digit_3_block` `digit_8` `digit_8_block` `digit_5` `digit_5_block` `digit_9` `digit_9_block` `digit_1` `digit_1_block`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `15208296_d486440f` | [scene_info.json](15208296_d486440f/scene_info.json) | [scene.usda](15208296_d486440f/scene.usda) |
| `1984741664_d486440f` | [scene_info.json](1984741664_d486440f/scene_info.json) | [scene.usda](1984741664_d486440f/scene.usda) |
| `2010563240_d486440f` | [scene_info.json](2010563240_d486440f/scene_info.json) | [scene.usda](2010563240_d486440f/scene.usda) |
| `2814623862_d486440f` | [scene_info.json](2814623862_d486440f/scene_info.json) | [scene.usda](2814623862_d486440f/scene.usda) |
| `3303354454_d486440f` | [scene_info.json](3303354454_d486440f/scene_info.json) | [scene.usda](3303354454_d486440f/scene.usda) |
| `352609510_d486440f` | [scene_info.json](352609510_d486440f/scene_info.json) | [scene.usda](352609510_d486440f/scene.usda) |
| `3575427661_d486440f` | [scene_info.json](3575427661_d486440f/scene_info.json) | [scene.usda](3575427661_d486440f/scene.usda) |
| `4168299912_d486440f` | [scene_info.json](4168299912_d486440f/scene_info.json) | [scene.usda](4168299912_d486440f/scene.usda) |
| `657488537_d486440f` | [scene_info.json](657488537_d486440f/scene_info.json) | [scene.usda](657488537_d486440f/scene.usda) |
| `946140265_d486440f` | [scene_info.json](946140265_d486440f/scene_info.json) | [scene.usda](946140265_d486440f/scene.usda) |
