# Move a chess piece to threaten the opposing king

Task: `move_chess_piece_to_threaten_king_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`black` `chess_piece` `chessboard` `king_black` `king_white` `square` `white` `bishop_black` `rook_black` `queen_white` `bishop_white` `knight_white` `pawn_white` `queen_black` `rook_white` `brown` `centered_on_desk` `dining_table` `four_legs` `game_board`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1017461999_e56198e1` | [scene_info.json](1017461999_e56198e1/scene_info.json) | [scene.usda](1017461999_e56198e1/scene.usda) |
| `1155451812_e56198e1` | [scene_info.json](1155451812_e56198e1/scene_info.json) | [scene.usda](1155451812_e56198e1/scene.usda) |
| `1425258349_e56198e1` | [scene_info.json](1425258349_e56198e1/scene_info.json) | [scene.usda](1425258349_e56198e1/scene.usda) |
| `144782939_e56198e1` | [scene_info.json](144782939_e56198e1/scene_info.json) | [scene.usda](144782939_e56198e1/scene.usda) |
| `1852463090_e56198e1` | [scene_info.json](1852463090_e56198e1/scene_info.json) | [scene.usda](1852463090_e56198e1/scene.usda) |
| `2335601079_e56198e1` | [scene_info.json](2335601079_e56198e1/scene_info.json) | [scene.usda](2335601079_e56198e1/scene.usda) |
| `2770989771_e56198e1` | [scene_info.json](2770989771_e56198e1/scene_info.json) | [scene.usda](2770989771_e56198e1/scene.usda) |
| `3130465258_e56198e1` | [scene_info.json](3130465258_e56198e1/scene_info.json) | [scene.usda](3130465258_e56198e1/scene.usda) |
| `3697696048_e56198e1` | [scene_info.json](3697696048_e56198e1/scene_info.json) | [scene.usda](3697696048_e56198e1/scene.usda) |
| `826855371_e56198e1` | [scene_info.json](826855371_e56198e1/scene_info.json) | [scene.usda](826855371_e56198e1/scene.usda) |
