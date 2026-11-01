# Pick a pen from its holder

Task: `pick_pen_from_holder_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig`

## Scene and action tags

`keywords` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`pen` `pen_holder` `table` `container` `cylinder` `desktop_with_legs` `long` `stationery` `white` `workspace_table` `writing_tool`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1174459548_7a34fd7b` | [scene_info.json](1174459548_7a34fd7b/scene_info.json) | [scene.usda](1174459548_7a34fd7b/scene.usda) |
| `2152638697_7a34fd7b` | [scene_info.json](2152638697_7a34fd7b/scene_info.json) | [scene.usda](2152638697_7a34fd7b/scene.usda) |
| `2366526378_7a34fd7b` | [scene_info.json](2366526378_7a34fd7b/scene_info.json) | [scene.usda](2366526378_7a34fd7b/scene.usda) |
| `25212940_7a34fd7b` | [scene_info.json](25212940_7a34fd7b/scene_info.json) | [scene.usda](25212940_7a34fd7b/scene.usda) |
| `256230739_7a34fd7b` | [scene_info.json](256230739_7a34fd7b/scene_info.json) | [scene.usda](256230739_7a34fd7b/scene.usda) |
| `2811888939_7a34fd7b` | [scene_info.json](2811888939_7a34fd7b/scene_info.json) | [scene.usda](2811888939_7a34fd7b/scene.usda) |
| `3414593073_7a34fd7b` | [scene_info.json](3414593073_7a34fd7b/scene_info.json) | [scene.usda](3414593073_7a34fd7b/scene.usda) |
| `3835105929_7a34fd7b` | [scene_info.json](3835105929_7a34fd7b/scene_info.json) | [scene.usda](3835105929_7a34fd7b/scene.usda) |
| `387247766_7a34fd7b` | [scene_info.json](387247766_7a34fd7b/scene_info.json) | [scene.usda](387247766_7a34fd7b/scene.usda) |
| `880681385_7a34fd7b` | [scene_info.json](880681385_7a34fd7b/scene_info.json) | [scene.usda](880681385_7a34fd7b/scene.usda) |
