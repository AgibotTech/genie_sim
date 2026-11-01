# Remove the odd one out

Task: `remove_odd_one_out_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig`

## Scene and action tags

`keywords` `pick` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`left` `normal` `odd` `on_desk` `right` `table` `fruit` `building_block` `pen` `pomegranate` `billiard` `cup` `green_apple` `orange` `snack` `desktop_with_legs` `white` `workspace_table` `blue` `purple`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1069324168_7d4beecf` | [scene_info.json](1069324168_7d4beecf/scene_info.json) | [scene.usda](1069324168_7d4beecf/scene.usda) |
| `131717240_7d4beecf` | [scene_info.json](131717240_7d4beecf/scene_info.json) | [scene.usda](131717240_7d4beecf/scene.usda) |
| `2091674108_7d4beecf` | [scene_info.json](2091674108_7d4beecf/scene_info.json) | [scene.usda](2091674108_7d4beecf/scene.usda) |
| `2458945642_7d4beecf` | [scene_info.json](2458945642_7d4beecf/scene_info.json) | [scene.usda](2458945642_7d4beecf/scene.usda) |
| `256372925_7d4beecf` | [scene_info.json](256372925_7d4beecf/scene_info.json) | [scene.usda](256372925_7d4beecf/scene.usda) |
| `2781701000_7d4beecf` | [scene_info.json](2781701000_7d4beecf/scene_info.json) | [scene.usda](2781701000_7d4beecf/scene.usda) |
| `3061394664_7d4beecf` | [scene_info.json](3061394664_7d4beecf/scene_info.json) | [scene.usda](3061394664_7d4beecf/scene.usda) |
| `3887826112_7d4beecf` | [scene_info.json](3887826112_7d4beecf/scene_info.json) | [scene.usda](3887826112_7d4beecf/scene.usda) |
| `4059470929_7d4beecf` | [scene_info.json](4059470929_7d4beecf/scene_info.json) | [scene.usda](4059470929_7d4beecf/scene.usda) |
| `994372013_7d4beecf` | [scene_info.json](994372013_7d4beecf/scene_info.json) | [scene.usda](994372013_7d4beecf/scene.usda) |
