# Store chess pieces in a box

Task: `store_chess_pieces_in_box_g1`. Robot: `G1`.

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

`chess_piece` `chessboard` `container` `furniture` `highlighted` `table` `target` `white` `wooden` `workplace` `internal` `random` `position_x_-0.012_y_0.032` `position_x_-0.065_y_-0.027` `position_x_-0.070_y_-0.054` `position_x_-0.084_y_-0.055` `position_x_-0.098_y_-0.084` `position_x_0.020_y_-0.050` `position_x_0.080_y_-0.096` `position_x_0.081_y_0.106`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1064926500_b6c36556` | [scene_info.json](1064926500_b6c36556/scene_info.json) | [scene.usda](1064926500_b6c36556/scene.usda) |
| `1215345097_b6c36556` | [scene_info.json](1215345097_b6c36556/scene_info.json) | [scene.usda](1215345097_b6c36556/scene.usda) |
| `1261134926_b6c36556` | [scene_info.json](1261134926_b6c36556/scene_info.json) | [scene.usda](1261134926_b6c36556/scene.usda) |
| `1423728778_b6c36556` | [scene_info.json](1423728778_b6c36556/scene_info.json) | [scene.usda](1423728778_b6c36556/scene.usda) |
| `1709992245_b6c36556` | [scene_info.json](1709992245_b6c36556/scene_info.json) | [scene.usda](1709992245_b6c36556/scene.usda) |
| `2245258233_b6c36556` | [scene_info.json](2245258233_b6c36556/scene_info.json) | [scene.usda](2245258233_b6c36556/scene.usda) |
| `2694959510_b6c36556` | [scene_info.json](2694959510_b6c36556/scene_info.json) | [scene.usda](2694959510_b6c36556/scene.usda) |
| `3610238004_b6c36556` | [scene_info.json](3610238004_b6c36556/scene_info.json) | [scene.usda](3610238004_b6c36556/scene.usda) |
| `415177214_b6c36556` | [scene_info.json](415177214_b6c36556/scene_info.json) | [scene.usda](415177214_b6c36556/scene.usda) |
| `887006686_b6c36556` | [scene_info.json](887006686_b6c36556/scene_info.json) | [scene.usda](887006686_b6c36556/scene.usda) |
