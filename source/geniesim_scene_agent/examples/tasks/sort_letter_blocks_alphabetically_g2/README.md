# Sort letter blocks alphabetically

Task: `sort_letter_blocks_alphabetically_g2`. Robot: `G2`.

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

`table` `alphabet` `desktop_with_legs` `educational` `plastic` `white` `workspace_table` `letter_Q` `letter_A` `letter_D` `letter_I` `letter_K` `letter_O` `letter_U` `letter_W` `letter_Z` `letter_B` `letter_C` `letter_E` `letter_H`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1478592755_1b3ddb4c` | [scene_info.json](1478592755_1b3ddb4c/scene_info.json) | [scene.usda](1478592755_1b3ddb4c/scene.usda) |
| `1633906522_1b3ddb4c` | [scene_info.json](1633906522_1b3ddb4c/scene_info.json) | [scene.usda](1633906522_1b3ddb4c/scene.usda) |
| `24935857_1b3ddb4c` | [scene_info.json](24935857_1b3ddb4c/scene_info.json) | [scene.usda](24935857_1b3ddb4c/scene.usda) |
| `2940222360_1b3ddb4c` | [scene_info.json](2940222360_1b3ddb4c/scene_info.json) | [scene.usda](2940222360_1b3ddb4c/scene.usda) |
| `3029540015_1b3ddb4c` | [scene_info.json](3029540015_1b3ddb4c/scene_info.json) | [scene.usda](3029540015_1b3ddb4c/scene.usda) |
| `3454980650_1b3ddb4c` | [scene_info.json](3454980650_1b3ddb4c/scene_info.json) | [scene.usda](3454980650_1b3ddb4c/scene.usda) |
| `3952552784_1b3ddb4c` | [scene_info.json](3952552784_1b3ddb4c/scene_info.json) | [scene.usda](3952552784_1b3ddb4c/scene.usda) |
| `4047864062_1b3ddb4c` | [scene_info.json](4047864062_1b3ddb4c/scene_info.json) | [scene.usda](4047864062_1b3ddb4c/scene.usda) |
| `491982309_1b3ddb4c` | [scene_info.json](491982309_1b3ddb4c/scene_info.json) | [scene.usda](491982309_1b3ddb4c/scene.usda) |
| `745612936_1b3ddb4c` | [scene_info.json](745612936_1b3ddb4c/scene_info.json) | [scene.usda](745612936_1b3ddb4c/scene.usda) |
