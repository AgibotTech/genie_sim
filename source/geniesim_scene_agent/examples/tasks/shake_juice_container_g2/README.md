# Shake a juice container

Task: `shake_juice_container_g2`. Robot: `G2`.

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

`extra_checkers` `intermediate_motion` `keywords` `pick` `shake` `table_only` `unmovable_objects`

## Scene keyword sample

`beverage` `juice_bottle` `table` `bottle` `desktop_with_legs` `glass` `transparent` `wood` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `144423173_d96197b4` | [scene_info.json](144423173_d96197b4/scene_info.json) | [scene.usda](144423173_d96197b4/scene.usda) |
| `1538292281_d96197b4` | [scene_info.json](1538292281_d96197b4/scene_info.json) | [scene.usda](1538292281_d96197b4/scene.usda) |
| `1607550832_d96197b4` | [scene_info.json](1607550832_d96197b4/scene_info.json) | [scene.usda](1607550832_d96197b4/scene.usda) |
| `2074409827_d96197b4` | [scene_info.json](2074409827_d96197b4/scene_info.json) | [scene.usda](2074409827_d96197b4/scene.usda) |
| `2320876971_d96197b4` | [scene_info.json](2320876971_d96197b4/scene_info.json) | [scene.usda](2320876971_d96197b4/scene.usda) |
| `2662088017_d96197b4` | [scene_info.json](2662088017_d96197b4/scene_info.json) | [scene.usda](2662088017_d96197b4/scene.usda) |
| `2873635023_d96197b4` | [scene_info.json](2873635023_d96197b4/scene_info.json) | [scene.usda](2873635023_d96197b4/scene.usda) |
| `3031317147_d96197b4` | [scene_info.json](3031317147_d96197b4/scene_info.json) | [scene.usda](3031317147_d96197b4/scene.usda) |
| `3344552238_d96197b4` | [scene_info.json](3344552238_d96197b4/scene_info.json) | [scene.usda](3344552238_d96197b4/scene.usda) |
| `4236178313_d96197b4` | [scene_info.json](4236178313_d96197b4/scene_info.json) | [scene.usda](4236178313_d96197b4/scene.usda) |
