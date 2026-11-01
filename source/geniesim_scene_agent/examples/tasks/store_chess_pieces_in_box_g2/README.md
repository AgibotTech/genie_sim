# Store chess pieces in a box

Task: `store_chess_pieces_in_box_g2`. Robot: `G2`.

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

`chess_piece` `chessboard` `container` `furniture` `highlighted` `table` `target` `white` `wooden` `workplace` `internal` `random` `position_x_-0.009_y_0.047` `position_x_-0.021_y_0.095` `position_x_-0.039_y_-0.144` `position_x_-0.040_y_-0.015` `position_x_-0.069_y_0.143` `position_x_-0.077_y_0.031` `position_x_-0.178_y_0.014` `position_x_0.012_y_0.082`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1005026865_b6c36556` | [scene_info.json](1005026865_b6c36556/scene_info.json) | [scene.usda](1005026865_b6c36556/scene.usda) |
| `1192539314_b6c36556` | [scene_info.json](1192539314_b6c36556/scene_info.json) | [scene.usda](1192539314_b6c36556/scene.usda) |
| `1674626850_b6c36556` | [scene_info.json](1674626850_b6c36556/scene_info.json) | [scene.usda](1674626850_b6c36556/scene.usda) |
| `1817741159_b6c36556` | [scene_info.json](1817741159_b6c36556/scene_info.json) | [scene.usda](1817741159_b6c36556/scene.usda) |
| `2292623891_b6c36556` | [scene_info.json](2292623891_b6c36556/scene_info.json) | [scene.usda](2292623891_b6c36556/scene.usda) |
| `2786297674_b6c36556` | [scene_info.json](2786297674_b6c36556/scene_info.json) | [scene.usda](2786297674_b6c36556/scene.usda) |
| `2973420410_b6c36556` | [scene_info.json](2973420410_b6c36556/scene_info.json) | [scene.usda](2973420410_b6c36556/scene.usda) |
| `349036933_b6c36556` | [scene_info.json](349036933_b6c36556/scene_info.json) | [scene.usda](349036933_b6c36556/scene.usda) |
| `359496688_b6c36556` | [scene_info.json](359496688_b6c36556/scene_info.json) | [scene.usda](359496688_b6c36556/scene.usda) |
| `726540386_b6c36556` | [scene_info.json](726540386_b6c36556/scene_info.json) | [scene.usda](726540386_b6c36556/scene.usda) |
