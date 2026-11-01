# Pick a letter by its position in the alphabet

Task: `pick_letter_by_alphabet_position_g2`. Robot: `G2`.

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

`table` `digit_block` `furniture` `left` `letter_block` `number_hint` `right` `target` `uppercase` `digit_1` `digit_4` `letter_A` `letter_D` `letter_B` `letter_C` `letter_E` `letter_J` `letter_L` `letter_P` `letter_Q`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1039362262_8d80b28c` | [scene_info.json](1039362262_8d80b28c/scene_info.json) | [scene.usda](1039362262_8d80b28c/scene.usda) |
| `1064416386_8d80b28c` | [scene_info.json](1064416386_8d80b28c/scene_info.json) | [scene.usda](1064416386_8d80b28c/scene.usda) |
| `1190920773_8d80b28c` | [scene_info.json](1190920773_8d80b28c/scene_info.json) | [scene.usda](1190920773_8d80b28c/scene.usda) |
| `227706595_8d80b28c` | [scene_info.json](227706595_8d80b28c/scene_info.json) | [scene.usda](227706595_8d80b28c/scene.usda) |
| `2521755128_8d80b28c` | [scene_info.json](2521755128_8d80b28c/scene_info.json) | [scene.usda](2521755128_8d80b28c/scene.usda) |
| `3056644391_8d80b28c` | [scene_info.json](3056644391_8d80b28c/scene_info.json) | [scene.usda](3056644391_8d80b28c/scene.usda) |
| `3597876150_8d80b28c` | [scene_info.json](3597876150_8d80b28c/scene_info.json) | [scene.usda](3597876150_8d80b28c/scene.usda) |
| `3773490390_8d80b28c` | [scene_info.json](3773490390_8d80b28c/scene_info.json) | [scene.usda](3773490390_8d80b28c/scene.usda) |
| `3942186889_8d80b28c` | [scene_info.json](3942186889_8d80b28c/scene_info.json) | [scene.usda](3942186889_8d80b28c/scene.usda) |
| `4089824452_8d80b28c` | [scene_info.json](4089824452_8d80b28c/scene_info.json) | [scene.usda](4089824452_8d80b28c/scene.usda) |
