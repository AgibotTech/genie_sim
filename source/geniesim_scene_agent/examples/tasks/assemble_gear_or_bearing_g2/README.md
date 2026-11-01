# Assemble a gear or bearing

Task: `assemble_gear_or_bearing_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`assembly` `extra_checkers` `keywords` `pick_and_place` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`furniture` `gearbox_center` `main_table` `mechanical` `mechanical_part` `placed_middle` `random_placement` `side_item` `table_000` `part_benchmark_gear_000` `part_benchmark_bearing_000` `part_benchmark_bearing_001` `part_benchmark_gear_001`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1449213536_ec239348` | [scene_info.json](1449213536_ec239348/scene_info.json) | [scene.usda](1449213536_ec239348/scene.usda) |
| `1729508093_ec239348` | [scene_info.json](1729508093_ec239348/scene_info.json) | [scene.usda](1729508093_ec239348/scene.usda) |
| `2508483052_ec239348` | [scene_info.json](2508483052_ec239348/scene_info.json) | [scene.usda](2508483052_ec239348/scene.usda) |
| `2670571099_ec239348` | [scene_info.json](2670571099_ec239348/scene_info.json) | [scene.usda](2670571099_ec239348/scene.usda) |
| `2771302358_ec239348` | [scene_info.json](2771302358_ec239348/scene_info.json) | [scene.usda](2771302358_ec239348/scene.usda) |
| `2959827622_ec239348` | [scene_info.json](2959827622_ec239348/scene_info.json) | [scene.usda](2959827622_ec239348/scene.usda) |
| `3026037309_ec239348` | [scene_info.json](3026037309_ec239348/scene_info.json) | [scene.usda](3026037309_ec239348/scene.usda) |
| `3696367908_ec239348` | [scene_info.json](3696367908_ec239348/scene_info.json) | [scene.usda](3696367908_ec239348/scene.usda) |
| `4160099307_ec239348` | [scene_info.json](4160099307_ec239348/scene_info.json) | [scene.usda](4160099307_ec239348/scene.usda) |
| `426055824_ec239348` | [scene_info.json](426055824_ec239348/scene_info.json) | [scene.usda](426055824_ec239348/scene.usda) |
