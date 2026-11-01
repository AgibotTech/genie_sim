# Pick an object by its relative position

Task: `pick_object_by_relative_position_g1`. Robot: `G1`.

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

`object` `table` `target` `clock` `yellow stapler` `charmander figure` `face cleanser` `fanta` `green apple` `lemon` `medicine bottle` `oolong tea beverage` `orange` `tape measure` `water bottle` `apple` `bowl sponge` `building blocks` `coke cola` `mouse`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1284582291_5d5daab8` | [scene_info.json](1284582291_5d5daab8/scene_info.json) | [scene.usda](1284582291_5d5daab8/scene.usda) |
| `1688268838_5d5daab8` | [scene_info.json](1688268838_5d5daab8/scene_info.json) | [scene.usda](1688268838_5d5daab8/scene.usda) |
| `1745854064_5d5daab8` | [scene_info.json](1745854064_5d5daab8/scene_info.json) | [scene.usda](1745854064_5d5daab8/scene.usda) |
| `2463866842_5d5daab8` | [scene_info.json](2463866842_5d5daab8/scene_info.json) | [scene.usda](2463866842_5d5daab8/scene.usda) |
| `3170251076_5d5daab8` | [scene_info.json](3170251076_5d5daab8/scene_info.json) | [scene.usda](3170251076_5d5daab8/scene.usda) |
| `3552049445_5d5daab8` | [scene_info.json](3552049445_5d5daab8/scene_info.json) | [scene.usda](3552049445_5d5daab8/scene.usda) |
| `489011585_5d5daab8` | [scene_info.json](489011585_5d5daab8/scene_info.json) | [scene.usda](489011585_5d5daab8/scene.usda) |
| `590939737_5d5daab8` | [scene_info.json](590939737_5d5daab8/scene_info.json) | [scene.usda](590939737_5d5daab8/scene.usda) |
| `814480616_5d5daab8` | [scene_info.json](814480616_5d5daab8/scene_info.json) | [scene.usda](814480616_5d5daab8/scene.usda) |
| `910758360_5d5daab8` | [scene_info.json](910758360_5d5daab8/scene_info.json) | [scene.usda](910758360_5d5daab8/scene.usda) |
