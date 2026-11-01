# Arrange digit blocks to form the smallest number

Task: `arrange_digits_to_form_smallest_number_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`table` `building_block` `educational_toy` `plastic` `rectangular` `small` `study_desk` `white` `workspace_table` `digit_2` `digit_3` `digit_4` `digit_5` `digit_0` `digit_1` `digit_7` `digit_8`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1106533813_39d2ec0d` | [scene_info.json](1106533813_39d2ec0d/scene_info.json) | [scene.usda](1106533813_39d2ec0d/scene.usda) |
| `1181174823_39d2ec0d` | [scene_info.json](1181174823_39d2ec0d/scene_info.json) | [scene.usda](1181174823_39d2ec0d/scene.usda) |
| `1279675284_39d2ec0d` | [scene_info.json](1279675284_39d2ec0d/scene_info.json) | [scene.usda](1279675284_39d2ec0d/scene.usda) |
| `1401362064_39d2ec0d` | [scene_info.json](1401362064_39d2ec0d/scene_info.json) | [scene.usda](1401362064_39d2ec0d/scene.usda) |
| `1906919599_39d2ec0d` | [scene_info.json](1906919599_39d2ec0d/scene_info.json) | [scene.usda](1906919599_39d2ec0d/scene.usda) |
| `2266499317_39d2ec0d` | [scene_info.json](2266499317_39d2ec0d/scene_info.json) | [scene.usda](2266499317_39d2ec0d/scene.usda) |
| `2882473124_39d2ec0d` | [scene_info.json](2882473124_39d2ec0d/scene_info.json) | [scene.usda](2882473124_39d2ec0d/scene.usda) |
| `2995323122_39d2ec0d` | [scene_info.json](2995323122_39d2ec0d/scene_info.json) | [scene.usda](2995323122_39d2ec0d/scene.usda) |
| `627184682_39d2ec0d` | [scene_info.json](627184682_39d2ec0d/scene_info.json) | [scene.usda](627184682_39d2ec0d/scene.usda) |
| `993458404_39d2ec0d` | [scene_info.json](993458404_39d2ec0d/scene_info.json) | [scene.usda](993458404_39d2ec0d/scene.usda) |
