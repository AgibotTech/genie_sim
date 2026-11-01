# Pick an object by its relative position

Task: `pick_object_by_relative_position_aloha`. Robot: `Aloha`.

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

`object` `table` `target` `oolong tea beverage` `orange` `building blocks` `charmander figure` `face cleanser` `grape juice` `green apple` `medicine bottle` `peach` `toy car` `toy plane` `water bottle` `clock` `paper cup` `tape measure` `yellow stapler` `desktop_with_legs`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1622437050_5d5daab8` | [scene_info.json](1622437050_5d5daab8/scene_info.json) | [scene.usda](1622437050_5d5daab8/scene.usda) |
| `2294939122_5d5daab8` | [scene_info.json](2294939122_5d5daab8/scene_info.json) | [scene.usda](2294939122_5d5daab8/scene.usda) |
| `2365088929_5d5daab8` | [scene_info.json](2365088929_5d5daab8/scene_info.json) | [scene.usda](2365088929_5d5daab8/scene.usda) |
| `2929080261_5d5daab8` | [scene_info.json](2929080261_5d5daab8/scene_info.json) | [scene.usda](2929080261_5d5daab8/scene.usda) |
| `2936097941_5d5daab8` | [scene_info.json](2936097941_5d5daab8/scene_info.json) | [scene.usda](2936097941_5d5daab8/scene.usda) |
| `3083417022_5d5daab8` | [scene_info.json](3083417022_5d5daab8/scene_info.json) | [scene.usda](3083417022_5d5daab8/scene.usda) |
| `3317332914_5d5daab8` | [scene_info.json](3317332914_5d5daab8/scene_info.json) | [scene.usda](3317332914_5d5daab8/scene.usda) |
| `3750252761_5d5daab8` | [scene_info.json](3750252761_5d5daab8/scene_info.json) | [scene.usda](3750252761_5d5daab8/scene.usda) |
| `556395472_5d5daab8` | [scene_info.json](556395472_5d5daab8/scene_info.json) | [scene.usda](556395472_5d5daab8/scene.usda) |
| `671876887_5d5daab8` | [scene_info.json](671876887_5d5daab8/scene_info.json) | [scene.usda](671876887_5d5daab8/scene.usda) |
