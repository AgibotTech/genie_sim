# Move a chess piece to a named square

Task: `move_chess_piece_to_named_square_arx`. Robot: `ARX`.

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

`black` `chess_piece` `chessboard` `king_black` `king_white` `square` `white` `queen_white` `bishop_white` `bishop_black` `knight_white` `pawn_white` `rook_white` `knight_black` `pawn_black` `queen_black` `rook_black` `brown` `centered_on_desk` `dining_table`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1093116326_76ba563d` | [scene_info.json](1093116326_76ba563d/scene_info.json) | [scene.usda](1093116326_76ba563d/scene.usda) |
| `2337433797_76ba563d` | [scene_info.json](2337433797_76ba563d/scene_info.json) | [scene.usda](2337433797_76ba563d/scene.usda) |
| `2614867064_76ba563d` | [scene_info.json](2614867064_76ba563d/scene_info.json) | [scene.usda](2614867064_76ba563d/scene.usda) |
| `286106809_76ba563d` | [scene_info.json](286106809_76ba563d/scene_info.json) | [scene.usda](286106809_76ba563d/scene.usda) |
| `3236288228_76ba563d` | [scene_info.json](3236288228_76ba563d/scene_info.json) | [scene.usda](3236288228_76ba563d/scene.usda) |
| `3470406658_76ba563d` | [scene_info.json](3470406658_76ba563d/scene_info.json) | [scene.usda](3470406658_76ba563d/scene.usda) |
| `372099486_76ba563d` | [scene_info.json](372099486_76ba563d/scene_info.json) | [scene.usda](372099486_76ba563d/scene.usda) |
| `4258066601_76ba563d` | [scene_info.json](4258066601_76ba563d/scene_info.json) | [scene.usda](4258066601_76ba563d/scene.usda) |
| `606496448_76ba563d` | [scene_info.json](606496448_76ba563d/scene_info.json) | [scene.usda](606496448_76ba563d/scene.usda) |
| `975719750_76ba563d` | [scene_info.json](975719750_76ba563d/scene_info.json) | [scene.usda](975719750_76ba563d/scene.usda) |
