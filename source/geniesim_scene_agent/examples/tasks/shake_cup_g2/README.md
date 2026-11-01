# Shake a cup

Task: `shake_cup_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_ee_pose` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.pick_actor` `robot.shake`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig` `ShakeConfig`

## Scene and action tags

`extra_checkers` `intermediate_motion` `keywords` `pick` `shake` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`cup` `benchmark_beer_mug_007` `benchmark_cup_043` `benchmark_beer_mug_001` `benchmark_beer_mug_002` `desktop_with_legs` `random_selection` `table` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1526575523_a7e32264` | [scene_info.json](1526575523_a7e32264/scene_info.json) | [scene.usda](1526575523_a7e32264/scene.usda) |
| `1664373985_a7e32264` | [scene_info.json](1664373985_a7e32264/scene_info.json) | [scene.usda](1664373985_a7e32264/scene.usda) |
| `1693291958_a7e32264` | [scene_info.json](1693291958_a7e32264/scene_info.json) | [scene.usda](1693291958_a7e32264/scene.usda) |
| `1744150060_a7e32264` | [scene_info.json](1744150060_a7e32264/scene_info.json) | [scene.usda](1744150060_a7e32264/scene.usda) |
| `1850941555_a7e32264` | [scene_info.json](1850941555_a7e32264/scene_info.json) | [scene.usda](1850941555_a7e32264/scene.usda) |
| `1897316972_a7e32264` | [scene_info.json](1897316972_a7e32264/scene_info.json) | [scene.usda](1897316972_a7e32264/scene.usda) |
| `2270388071_a7e32264` | [scene_info.json](2270388071_a7e32264/scene_info.json) | [scene.usda](2270388071_a7e32264/scene.usda) |
| `3970350293_a7e32264` | [scene_info.json](3970350293_a7e32264/scene_info.json) | [scene.usda](3970350293_a7e32264/scene.usda) |
| `398287595_a7e32264` | [scene_info.json](398287595_a7e32264/scene_info.json) | [scene.usda](398287595_a7e32264/scene.usda) |
| `744370991_a7e32264` | [scene_info.json](744370991_a7e32264/scene_info.json) | [scene.usda](744370991_a7e32264/scene.usda) |
