# Swap two beverages

Task: `swap_beverages_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_actor` `robot.place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `swap` `table_only` `unmovable_objects`

## Scene keyword sample

`beverage` `coca_cola` `fanta` `sprite` `wang_lao_ji` `can` `desktop_with_legs` `drink` `table` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1089266073_05e928cc` | [scene_info.json](1089266073_05e928cc/scene_info.json) | [scene.usda](1089266073_05e928cc/scene.usda) |
| `1215270212_05e928cc` | [scene_info.json](1215270212_05e928cc/scene_info.json) | [scene.usda](1215270212_05e928cc/scene.usda) |
| `1333464637_05e928cc` | [scene_info.json](1333464637_05e928cc/scene_info.json) | [scene.usda](1333464637_05e928cc/scene.usda) |
| `1421751660_05e928cc` | [scene_info.json](1421751660_05e928cc/scene_info.json) | [scene.usda](1421751660_05e928cc/scene.usda) |
| `1700059176_05e928cc` | [scene_info.json](1700059176_05e928cc/scene_info.json) | [scene.usda](1700059176_05e928cc/scene.usda) |
| `267208817_05e928cc` | [scene_info.json](267208817_05e928cc/scene_info.json) | [scene.usda](267208817_05e928cc/scene.usda) |
| `329674820_05e928cc` | [scene_info.json](329674820_05e928cc/scene_info.json) | [scene.usda](329674820_05e928cc/scene.usda) |
| `647663496_05e928cc` | [scene_info.json](647663496_05e928cc/scene_info.json) | [scene.usda](647663496_05e928cc/scene.usda) |
| `934121528_05e928cc` | [scene_info.json](934121528_05e928cc/scene_info.json) | [scene.usda](934121528_05e928cc/scene.usda) |
| `94847147_05e928cc` | [scene_info.json](94847147_05e928cc/scene_info.json) | [scene.usda](94847147_05e928cc/scene.usda) |
