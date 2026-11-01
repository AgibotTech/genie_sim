# Stack paper cups

Task: `stack_paper_cups_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `stack` `table_with_containers` `unmovable_objects`

## Scene keyword sample

`dining_table` `table` `beverage` `cylinder` `disposable` `furniture` `left` `paper_cup_0` `paper_cup_1` `paper_cup_2` `rectangular` `white` `wood` `right` `paper_cup_3`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `105301124_e09852ba` | [scene_info.json](105301124_e09852ba/scene_info.json) | [scene.usda](105301124_e09852ba/scene.usda) |
| `1194032312_e09852ba` | [scene_info.json](1194032312_e09852ba/scene_info.json) | [scene.usda](1194032312_e09852ba/scene.usda) |
| `1441153333_e09852ba` | [scene_info.json](1441153333_e09852ba/scene_info.json) | [scene.usda](1441153333_e09852ba/scene.usda) |
| `1549688816_e09852ba` | [scene_info.json](1549688816_e09852ba/scene_info.json) | [scene.usda](1549688816_e09852ba/scene.usda) |
| `2516803297_e09852ba` | [scene_info.json](2516803297_e09852ba/scene_info.json) | [scene.usda](2516803297_e09852ba/scene.usda) |
| `2672654168_e09852ba` | [scene_info.json](2672654168_e09852ba/scene_info.json) | [scene.usda](2672654168_e09852ba/scene.usda) |
| `2887217263_e09852ba` | [scene_info.json](2887217263_e09852ba/scene_info.json) | [scene.usda](2887217263_e09852ba/scene.usda) |
| `3247064064_e09852ba` | [scene_info.json](3247064064_e09852ba/scene_info.json) | [scene.usda](3247064064_e09852ba/scene.usda) |
| `3316158327_e09852ba` | [scene_info.json](3316158327_e09852ba/scene_info.json) | [scene.usda](3316158327_e09852ba/scene.usda) |
| `725354860_e09852ba` | [scene_info.json](725354860_e09852ba/scene_info.json) | [scene.usda](725354860_e09852ba/scene.usda) |
