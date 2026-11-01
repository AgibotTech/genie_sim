# Pick odd or even number blocks

Task: `pick_odd_or_even_number_blocks_g2`. Robot: `G2`.

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

`table` `building_block` `educational_toy` `plastic` `rectangular` `small` `study_desk` `white` `workspace_table` `digit_5` `digit_0` `digit_2` `digit_3` `digit_8` `digit_7` `digit_4` `digit_1`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1184208457_b3f965c3` | [scene_info.json](1184208457_b3f965c3/scene_info.json) | [scene.usda](1184208457_b3f965c3/scene.usda) |
| `145412925_b3f965c3` | [scene_info.json](145412925_b3f965c3/scene_info.json) | [scene.usda](145412925_b3f965c3/scene.usda) |
| `2199084595_b3f965c3` | [scene_info.json](2199084595_b3f965c3/scene_info.json) | [scene.usda](2199084595_b3f965c3/scene.usda) |
| `2383098358_b3f965c3` | [scene_info.json](2383098358_b3f965c3/scene_info.json) | [scene.usda](2383098358_b3f965c3/scene.usda) |
| `2516544333_b3f965c3` | [scene_info.json](2516544333_b3f965c3/scene_info.json) | [scene.usda](2516544333_b3f965c3/scene.usda) |
| `2542096373_b3f965c3` | [scene_info.json](2542096373_b3f965c3/scene_info.json) | [scene.usda](2542096373_b3f965c3/scene.usda) |
| `2556839959_b3f965c3` | [scene_info.json](2556839959_b3f965c3/scene_info.json) | [scene.usda](2556839959_b3f965c3/scene.usda) |
| `2912679979_b3f965c3` | [scene_info.json](2912679979_b3f965c3/scene_info.json) | [scene.usda](2912679979_b3f965c3/scene.usda) |
| `4151799249_b3f965c3` | [scene_info.json](4151799249_b3f965c3/scene_info.json) | [scene.usda](4151799249_b3f965c3/scene.usda) |
| `530016570_b3f965c3` | [scene_info.json](530016570_b3f965c3/scene_info.json) | [scene.usda](530016570_b3f965c3/scene.usda) |
