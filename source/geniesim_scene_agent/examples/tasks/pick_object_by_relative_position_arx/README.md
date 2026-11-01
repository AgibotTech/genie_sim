# Pick an object by its relative position

Task: `pick_object_by_relative_position_arx`. Robot: `ARX`.

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

`object` `table` `target` `coke cola` `face cleanser` `orange` `clock` `fanta` `green apple` `lemon` `medicine bottle` `oolong tea beverage` `rubik cube` `tape measure` `toy car` `toy plane` `water bottle` `building blocks` `grape juice` `mouse`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1154662924_5d5daab8` | [scene_info.json](1154662924_5d5daab8/scene_info.json) | [scene.usda](1154662924_5d5daab8/scene.usda) |
| `1175414364_5d5daab8` | [scene_info.json](1175414364_5d5daab8/scene_info.json) | [scene.usda](1175414364_5d5daab8/scene.usda) |
| `1779336046_5d5daab8` | [scene_info.json](1779336046_5d5daab8/scene_info.json) | [scene.usda](1779336046_5d5daab8/scene.usda) |
| `1855087579_5d5daab8` | [scene_info.json](1855087579_5d5daab8/scene_info.json) | [scene.usda](1855087579_5d5daab8/scene.usda) |
| `2190491975_5d5daab8` | [scene_info.json](2190491975_5d5daab8/scene_info.json) | [scene.usda](2190491975_5d5daab8/scene.usda) |
| `2593014789_5d5daab8` | [scene_info.json](2593014789_5d5daab8/scene_info.json) | [scene.usda](2593014789_5d5daab8/scene.usda) |
| `3591476208_5d5daab8` | [scene_info.json](3591476208_5d5daab8/scene_info.json) | [scene.usda](3591476208_5d5daab8/scene.usda) |
| `401681976_5d5daab8` | [scene_info.json](401681976_5d5daab8/scene_info.json) | [scene.usda](401681976_5d5daab8/scene.usda) |
| `4179860217_5d5daab8` | [scene_info.json](4179860217_5d5daab8/scene_info.json) | [scene.usda](4179860217_5d5daab8/scene.usda) |
| `460789771_5d5daab8` | [scene_info.json](460789771_5d5daab8/scene_info.json) | [scene.usda](460789771_5d5daab8/scene.usda) |
