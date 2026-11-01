# Move a beverage from the top shelf to the middle shelf

Task: `move_beverage_from_top_to_middle_shelf_g2`. Robot: `G2`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_local_frame` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.move_actor_to_target_pose` `robot.open_gripper` `robot.pick_actor` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `MoveConfig`

## Scene and action tags

`intermediate_motion` `keywords` `pick` `shelf_scene` `unmovable_objects`

## Scene keyword sample

`beverage` `drink` `shelf` `target_beverage` `target_shelf` `bottle` `container` `left` `on_top_layer` `storage_furniture` `supermarket_display` `target` `wood`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1180972529_8064ca9b` | [scene_info.json](1180972529_8064ca9b/scene_info.json) | [scene.usda](1180972529_8064ca9b/scene.usda) |
| `1923204850_8064ca9b` | [scene_info.json](1923204850_8064ca9b/scene_info.json) | [scene.usda](1923204850_8064ca9b/scene.usda) |
| `202744950_8064ca9b` | [scene_info.json](202744950_8064ca9b/scene_info.json) | [scene.usda](202744950_8064ca9b/scene.usda) |
| `2663079037_8064ca9b` | [scene_info.json](2663079037_8064ca9b/scene_info.json) | [scene.usda](2663079037_8064ca9b/scene.usda) |
| `3072179794_8064ca9b` | [scene_info.json](3072179794_8064ca9b/scene_info.json) | [scene.usda](3072179794_8064ca9b/scene.usda) |
| `3786164880_8064ca9b` | [scene_info.json](3786164880_8064ca9b/scene_info.json) | [scene.usda](3786164880_8064ca9b/scene.usda) |
| `3924405843_8064ca9b` | [scene_info.json](3924405843_8064ca9b/scene_info.json) | [scene.usda](3924405843_8064ca9b/scene.usda) |
| `4265143746_8064ca9b` | [scene_info.json](4265143746_8064ca9b/scene_info.json) | [scene.usda](4265143746_8064ca9b/scene.usda) |
| `445215649_8064ca9b` | [scene_info.json](445215649_8064ca9b/scene_info.json) | [scene.usda](445215649_8064ca9b/scene.usda) |
| `636551613_8064ca9b` | [scene_info.json](636551613_8064ca9b/scene_info.json) | [scene.usda](636551613_8064ca9b/scene.usda) |
