# Disassemble a gear or bearing

Task: `disassemble_gear_or_bearing_g2`. Robot: `G2`.

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

`disassembly` `extra_checkers` `keywords` `pick` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`gearbox_center` `main_table` `mechanical_part` `on_cylinder` `table_000` `fixed` `furniture` `mechanical` `right_cylinder` `mounted_benchmark_bearing_001` `mounted_benchmark_gear_000` `back_cylinder` `small_cylinder` `mounted_benchmark_bearing_000` `mounted_benchmark_gear_001`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1333844047_9f21479c` | [scene_info.json](1333844047_9f21479c/scene_info.json) | [scene.usda](1333844047_9f21479c/scene.usda) |
| `1457831248_9f21479c` | [scene_info.json](1457831248_9f21479c/scene_info.json) | [scene.usda](1457831248_9f21479c/scene.usda) |
| `2132247079_9f21479c` | [scene_info.json](2132247079_9f21479c/scene_info.json) | [scene.usda](2132247079_9f21479c/scene.usda) |
| `2156946717_9f21479c` | [scene_info.json](2156946717_9f21479c/scene_info.json) | [scene.usda](2156946717_9f21479c/scene.usda) |
| `251984099_9f21479c` | [scene_info.json](251984099_9f21479c/scene_info.json) | [scene.usda](251984099_9f21479c/scene.usda) |
| `3396579150_9f21479c` | [scene_info.json](3396579150_9f21479c/scene_info.json) | [scene.usda](3396579150_9f21479c/scene.usda) |
| `3931217938_9f21479c` | [scene_info.json](3931217938_9f21479c/scene_info.json) | [scene.usda](3931217938_9f21479c/scene.usda) |
| `3934687359_9f21479c` | [scene_info.json](3934687359_9f21479c/scene_info.json) | [scene.usda](3934687359_9f21479c/scene.usda) |
| `849591289_9f21479c` | [scene_info.json](849591289_9f21479c/scene_info.json) | [scene.usda](849591289_9f21479c/scene.usda) |
| `958064924_9f21479c` | [scene_info.json](958064924_9f21479c/scene_info.json) | [scene.usda](958064924_9f21479c/scene.usda) |
