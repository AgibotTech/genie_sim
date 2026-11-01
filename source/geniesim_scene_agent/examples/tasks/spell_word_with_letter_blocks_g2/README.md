# Spell a word with letter blocks

Task: `spell_word_with_letter_blocks_g2`. Robot: `G2`.

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

`table` `alphabet` `desktop_with_legs` `educational` `plastic` `white` `workspace_table` `letter_A` `letter_R` `letter_D` `letter_B` `letter_E` `letter_F` `letter_I` `letter_C` `letter_K` `letter_N` `letter_O` `letter_T` `letter_U`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1380657831_13d0e496` | [scene_info.json](1380657831_13d0e496/scene_info.json) | [scene.usda](1380657831_13d0e496/scene.usda) |
| `1558123570_13d0e496` | [scene_info.json](1558123570_13d0e496/scene_info.json) | [scene.usda](1558123570_13d0e496/scene.usda) |
| `1737633141_13d0e496` | [scene_info.json](1737633141_13d0e496/scene_info.json) | [scene.usda](1737633141_13d0e496/scene.usda) |
| `1922228904_13d0e496` | [scene_info.json](1922228904_13d0e496/scene_info.json) | [scene.usda](1922228904_13d0e496/scene.usda) |
| `3082502683_13d0e496` | [scene_info.json](3082502683_13d0e496/scene_info.json) | [scene.usda](3082502683_13d0e496/scene.usda) |
| `3284419949_13d0e496` | [scene_info.json](3284419949_13d0e496/scene_info.json) | [scene.usda](3284419949_13d0e496/scene.usda) |
| `3619908481_13d0e496` | [scene_info.json](3619908481_13d0e496/scene_info.json) | [scene.usda](3619908481_13d0e496/scene.usda) |
| `3691469246_13d0e496` | [scene_info.json](3691469246_13d0e496/scene_info.json) | [scene.usda](3691469246_13d0e496/scene.usda) |
| `489307385_13d0e496` | [scene_info.json](489307385_13d0e496/scene_info.json) | [scene.usda](489307385_13d0e496/scene.usda) |
| `802177966_13d0e496` | [scene_info.json](802177966_13d0e496/scene_info.json) | [scene.usda](802177966_13d0e496/scene.usda) |
