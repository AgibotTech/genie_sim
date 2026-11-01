# Store chess pieces in a box

Task: `store_chess_pieces_in_box_arx`. Robot: `ARX`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_local_frame` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_pose`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`extra_checkers` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`chess_piece` `chessboard` `container` `furniture` `highlighted` `table` `target` `white` `wooden` `workplace` `internal` `random` `position_x_-0.003_y_-0.174` `position_x_-0.012_y_-0.063` `position_x_-0.051_y_0.111` `position_x_-0.056_y_-0.122` `position_x_0.004_y_-0.041` `position_x_0.036_y_-0.038` `position_x_0.038_y_0.076` `position_x_0.049_y_-0.172`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1429366552_b6c36556` | [scene_info.json](1429366552_b6c36556/scene_info.json) | [scene.usda](1429366552_b6c36556/scene.usda) |
| `1859645629_b6c36556` | [scene_info.json](1859645629_b6c36556/scene_info.json) | [scene.usda](1859645629_b6c36556/scene.usda) |
| `2008872691_b6c36556` | [scene_info.json](2008872691_b6c36556/scene_info.json) | [scene.usda](2008872691_b6c36556/scene.usda) |
| `246678918_b6c36556` | [scene_info.json](246678918_b6c36556/scene_info.json) | [scene.usda](246678918_b6c36556/scene.usda) |
| `2510170373_b6c36556` | [scene_info.json](2510170373_b6c36556/scene_info.json) | [scene.usda](2510170373_b6c36556/scene.usda) |
| `2699377617_b6c36556` | [scene_info.json](2699377617_b6c36556/scene_info.json) | [scene.usda](2699377617_b6c36556/scene.usda) |
| `2797518975_b6c36556` | [scene_info.json](2797518975_b6c36556/scene_info.json) | [scene.usda](2797518975_b6c36556/scene.usda) |
| `2937939380_b6c36556` | [scene_info.json](2937939380_b6c36556/scene_info.json) | [scene.usda](2937939380_b6c36556/scene.usda) |
| `3655331220_b6c36556` | [scene_info.json](3655331220_b6c36556/scene_info.json) | [scene.usda](3655331220_b6c36556/scene.usda) |
| `4127500959_b6c36556` | [scene_info.json](4127500959_b6c36556/scene_info.json) | [scene.usda](4127500959_b6c36556/scene.usda) |
