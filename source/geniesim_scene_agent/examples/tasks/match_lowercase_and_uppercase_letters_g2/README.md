# Match lowercase and uppercase letter blocks

Task: `match_lowercase_and_uppercase_letters_g2`. Robot: `G2`.

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

`letter` `lowercase` `table` `uppercase` `D` `H` `A` `E` `J` `K` `U` `V` `Y` `a` `B` `C` `F` `G` `I` `Q`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1239538906_7f3e0804` | [scene_info.json](1239538906_7f3e0804/scene_info.json) | [scene.usda](1239538906_7f3e0804/scene.usda) |
| `1309199314_7f3e0804` | [scene_info.json](1309199314_7f3e0804/scene_info.json) | [scene.usda](1309199314_7f3e0804/scene.usda) |
| `1461424000_7f3e0804` | [scene_info.json](1461424000_7f3e0804/scene_info.json) | [scene.usda](1461424000_7f3e0804/scene.usda) |
| `2347108564_7f3e0804` | [scene_info.json](2347108564_7f3e0804/scene_info.json) | [scene.usda](2347108564_7f3e0804/scene.usda) |
| `2496238015_7f3e0804` | [scene_info.json](2496238015_7f3e0804/scene_info.json) | [scene.usda](2496238015_7f3e0804/scene.usda) |
| `3487490601_7f3e0804` | [scene_info.json](3487490601_7f3e0804/scene_info.json) | [scene.usda](3487490601_7f3e0804/scene.usda) |
| `376550017_7f3e0804` | [scene_info.json](376550017_7f3e0804/scene_info.json) | [scene.usda](376550017_7f3e0804/scene.usda) |
| `4112967111_7f3e0804` | [scene_info.json](4112967111_7f3e0804/scene_info.json) | [scene.usda](4112967111_7f3e0804/scene.usda) |
| `630768296_7f3e0804` | [scene_info.json](630768296_7f3e0804/scene_info.json) | [scene.usda](630768296_7f3e0804/scene.usda) |
| `683231406_7f3e0804` | [scene_info.json](683231406_7f3e0804/scene_info.json) | [scene.usda](683231406_7f3e0804/scene.usda) |
