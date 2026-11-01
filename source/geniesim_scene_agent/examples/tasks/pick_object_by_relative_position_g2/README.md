# Pick an object by its relative position

Task: `pick_object_by_relative_position_g2`. Robot: `G2`.

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

`keywords` `pick` `relative_position` `table_with_containers`

## Scene keyword sample

`object` `table` `target` `face cleanser` `water bottle` `apple` `peach` `bowl sponge` `charmander figure` `building blocks` `clock` `coke cola` `grape juice` `green apple` `lemon` `medicine bottle` `mouse` `oolong tea beverage` `orange` `paper cup`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1269402123_5d5daab8` | [scene_info.json](1269402123_5d5daab8/scene_info.json) | [scene.usda](1269402123_5d5daab8/scene.usda) |
| `1930973656_5d5daab8` | [scene_info.json](1930973656_5d5daab8/scene_info.json) | [scene.usda](1930973656_5d5daab8/scene.usda) |
| `2627123243_5d5daab8` | [scene_info.json](2627123243_5d5daab8/scene_info.json) | [scene.usda](2627123243_5d5daab8/scene.usda) |
| `2952884494_5d5daab8` | [scene_info.json](2952884494_5d5daab8/scene_info.json) | [scene.usda](2952884494_5d5daab8/scene.usda) |
| `3646434049_5d5daab8` | [scene_info.json](3646434049_5d5daab8/scene_info.json) | [scene.usda](3646434049_5d5daab8/scene.usda) |
| `3936603161_5d5daab8` | [scene_info.json](3936603161_5d5daab8/scene_info.json) | [scene.usda](3936603161_5d5daab8/scene.usda) |
| `4276211065_5d5daab8` | [scene_info.json](4276211065_5d5daab8/scene_info.json) | [scene.usda](4276211065_5d5daab8/scene.usda) |
| `724362167_5d5daab8` | [scene_info.json](724362167_5d5daab8/scene_info.json) | [scene.usda](724362167_5d5daab8/scene.usda) |
| `846498237_5d5daab8` | [scene_info.json](846498237_5d5daab8/scene_info.json) | [scene.usda](846498237_5d5daab8/scene.usda) |
| `994930425_5d5daab8` | [scene_info.json](994930425_5d5daab8/scene_info.json) | [scene.usda](994930425_5d5daab8/scene.usda) |
