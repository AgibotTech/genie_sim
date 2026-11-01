# Store chess pieces in a box

Task: `store_chess_pieces_in_box_aloha`. Robot: `Aloha`.

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

`chess_piece` `chessboard` `container` `furniture` `highlighted` `table` `target` `white` `wooden` `workplace` `internal` `random` `position_x_-0.017_y_0.006` `position_x_-0.024_y_0.049` `position_x_-0.042_y_-0.054` `position_x_-0.088_y_0.004` `position_x_-0.090_y_-0.003` `position_x_-0.127_y_-0.038` `position_x_-0.156_y_0.093` `position_x_0.019_y_-0.014`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1796156592_b6c36556` | [scene_info.json](1796156592_b6c36556/scene_info.json) | [scene.usda](1796156592_b6c36556/scene.usda) |
| `2054357159_b6c36556` | [scene_info.json](2054357159_b6c36556/scene_info.json) | [scene.usda](2054357159_b6c36556/scene.usda) |
| `2130693161_b6c36556` | [scene_info.json](2130693161_b6c36556/scene_info.json) | [scene.usda](2130693161_b6c36556/scene.usda) |
| `2448950437_b6c36556` | [scene_info.json](2448950437_b6c36556/scene_info.json) | [scene.usda](2448950437_b6c36556/scene.usda) |
| `2867355567_b6c36556` | [scene_info.json](2867355567_b6c36556/scene_info.json) | [scene.usda](2867355567_b6c36556/scene.usda) |
| `3255264419_b6c36556` | [scene_info.json](3255264419_b6c36556/scene_info.json) | [scene.usda](3255264419_b6c36556/scene.usda) |
| `3607189814_b6c36556` | [scene_info.json](3607189814_b6c36556/scene_info.json) | [scene.usda](3607189814_b6c36556/scene.usda) |
| `4160843558_b6c36556` | [scene_info.json](4160843558_b6c36556/scene_info.json) | [scene.usda](4160843558_b6c36556/scene.usda) |
| `4192849785_b6c36556` | [scene_info.json](4192849785_b6c36556/scene_info.json) | [scene.usda](4192849785_b6c36556/scene.usda) |
| `594690063_b6c36556` | [scene_info.json](594690063_b6c36556/scene_info.json) | [scene.usda](594690063_b6c36556/scene.usda) |
