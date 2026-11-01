# Move a beverage to another beverage's position

Task: `move_beverage_to_another_beverages_position_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`beverage` `coca_cola` `fanta` `sprite` `wang_lao_ji` `can` `desktop_with_legs` `drink` `table` `white` `workspace_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `16143683_05e928cc` | [scene_info.json](16143683_05e928cc/scene_info.json) | [scene.usda](16143683_05e928cc/scene.usda) |
| `2203739479_05e928cc` | [scene_info.json](2203739479_05e928cc/scene_info.json) | [scene.usda](2203739479_05e928cc/scene.usda) |
| `2345737209_05e928cc` | [scene_info.json](2345737209_05e928cc/scene_info.json) | [scene.usda](2345737209_05e928cc/scene.usda) |
| `2490451072_05e928cc` | [scene_info.json](2490451072_05e928cc/scene_info.json) | [scene.usda](2490451072_05e928cc/scene.usda) |
| `2579919997_05e928cc` | [scene_info.json](2579919997_05e928cc/scene_info.json) | [scene.usda](2579919997_05e928cc/scene.usda) |
| `2599088719_05e928cc` | [scene_info.json](2599088719_05e928cc/scene_info.json) | [scene.usda](2599088719_05e928cc/scene.usda) |
| `2909393819_05e928cc` | [scene_info.json](2909393819_05e928cc/scene_info.json) | [scene.usda](2909393819_05e928cc/scene.usda) |
| `3193101860_05e928cc` | [scene_info.json](3193101860_05e928cc/scene_info.json) | [scene.usda](3193101860_05e928cc/scene.usda) |
| `3798786876_05e928cc` | [scene_info.json](3798786876_05e928cc/scene_info.json) | [scene.usda](3798786876_05e928cc/scene.usda) |
| `583111424_05e928cc` | [scene_info.json](583111424_05e928cc/scene_info.json) | [scene.usda](583111424_05e928cc/scene.usda) |
