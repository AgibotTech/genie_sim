# Capture a chess piece

Task: `capture_chess_piece_arx`. Robot: `ARX`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.back_to_origin` `robot.get_object_pose` `robot.get_object_size_in_world_frame` `robot.judge_arm_to_use` `robot.pick_and_place_actor_to_target_pose` `robot.wait`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig` `PlaceConfig`

## Scene and action tags

`capture` `extra_checkers` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`black` `chess_piece` `chessboard` `king_black` `king_white` `square` `table` `white` `queen_black` `rook_white` `queen_white` `bishop_white` `pawn_black` `bishop_black` `rook_black` `knight_white` `pawn_white` `knight_black` `brown` `centered_on_desk`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `111056403_c4de999d` | [scene_info.json](111056403_c4de999d/scene_info.json) | [scene.usda](111056403_c4de999d/scene.usda) |
| `1305828190_c4de999d` | [scene_info.json](1305828190_c4de999d/scene_info.json) | [scene.usda](1305828190_c4de999d/scene.usda) |
| `1464745571_c4de999d` | [scene_info.json](1464745571_c4de999d/scene_info.json) | [scene.usda](1464745571_c4de999d/scene.usda) |
| `1523869086_c4de999d` | [scene_info.json](1523869086_c4de999d/scene_info.json) | [scene.usda](1523869086_c4de999d/scene.usda) |
| `260882956_c4de999d` | [scene_info.json](260882956_c4de999d/scene_info.json) | [scene.usda](260882956_c4de999d/scene.usda) |
| `2688972735_c4de999d` | [scene_info.json](2688972735_c4de999d/scene_info.json) | [scene.usda](2688972735_c4de999d/scene.usda) |
| `3021193309_c4de999d` | [scene_info.json](3021193309_c4de999d/scene_info.json) | [scene.usda](3021193309_c4de999d/scene.usda) |
| `3314461558_c4de999d` | [scene_info.json](3314461558_c4de999d/scene_info.json) | [scene.usda](3314461558_c4de999d/scene.usda) |
| `3332045614_c4de999d` | [scene_info.json](3332045614_c4de999d/scene_info.json) | [scene.usda](3332045614_c4de999d/scene.usda) |
| `3714050975_c4de999d` | [scene_info.json](3714050975_c4de999d/scene_info.json) | [scene.usda](3714050975_c4de999d/scene.usda) |
