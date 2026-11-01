# Stack trash cans

Task: `stack_trash_cans_g2`. Robot: `G2`.

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

`robot.client.run_step` `robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `stack` `table_with_articulation` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`table` `cylinder` `desktop_with_legs` `gray` `position_left` `small` `white` `workspace_table` `outer_trash_can_left` `inner_trash_can_left` `position_right` `inner_trash_can_right` `outer_trash_can_right`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `2471276903_0afb2c89` | [scene_info.json](2471276903_0afb2c89/scene_info.json) | [scene.usda](2471276903_0afb2c89/scene.usda) |
| `2744833184_0afb2c89` | [scene_info.json](2744833184_0afb2c89/scene_info.json) | [scene.usda](2744833184_0afb2c89/scene.usda) |
| `2752960244_0afb2c89` | [scene_info.json](2752960244_0afb2c89/scene_info.json) | [scene.usda](2752960244_0afb2c89/scene.usda) |
| `2799263161_0afb2c89` | [scene_info.json](2799263161_0afb2c89/scene_info.json) | [scene.usda](2799263161_0afb2c89/scene.usda) |
| `2993261706_0afb2c89` | [scene_info.json](2993261706_0afb2c89/scene_info.json) | [scene.usda](2993261706_0afb2c89/scene.usda) |
| `3276345931_0afb2c89` | [scene_info.json](3276345931_0afb2c89/scene_info.json) | [scene.usda](3276345931_0afb2c89/scene.usda) |
| `3340834839_0afb2c89` | [scene_info.json](3340834839_0afb2c89/scene_info.json) | [scene.usda](3340834839_0afb2c89/scene.usda) |
| `512221943_0afb2c89` | [scene_info.json](512221943_0afb2c89/scene_info.json) | [scene.usda](512221943_0afb2c89/scene.usda) |
| `53610256_0afb2c89` | [scene_info.json](53610256_0afb2c89/scene_info.json) | [scene.usda](53610256_0afb2c89/scene.usda) |
| `877060411_0afb2c89` | [scene_info.json](877060411_0afb2c89/scene_info.json) | [scene.usda](877060411_0afb2c89/scene.usda) |
