# Move a chess piece to a named square

Task: `move_chess_piece_to_named_square_g2`. Robot: `G2`.

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

`black` `chess_piece` `chessboard` `king_black` `king_white` `square` `white` `queen_white` `bishop_white` `pawn_black` `rook_black` `rook_white` `knight_white` `queen_black` `knight_black` `pawn_white` `brown` `centered_on_desk` `dining_table` `four_legs`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1505274453_76ba563d` | [scene_info.json](1505274453_76ba563d/scene_info.json) | [scene.usda](1505274453_76ba563d/scene.usda) |
| `1618256421_76ba563d` | [scene_info.json](1618256421_76ba563d/scene_info.json) | [scene.usda](1618256421_76ba563d/scene.usda) |
| `2162783935_76ba563d` | [scene_info.json](2162783935_76ba563d/scene_info.json) | [scene.usda](2162783935_76ba563d/scene.usda) |
| `2398581235_76ba563d` | [scene_info.json](2398581235_76ba563d/scene_info.json) | [scene.usda](2398581235_76ba563d/scene.usda) |
| `2591589021_76ba563d` | [scene_info.json](2591589021_76ba563d/scene_info.json) | [scene.usda](2591589021_76ba563d/scene.usda) |
| `3420176425_76ba563d` | [scene_info.json](3420176425_76ba563d/scene_info.json) | [scene.usda](3420176425_76ba563d/scene.usda) |
| `3802855807_76ba563d` | [scene_info.json](3802855807_76ba563d/scene_info.json) | [scene.usda](3802855807_76ba563d/scene.usda) |
| `3980864799_76ba563d` | [scene_info.json](3980864799_76ba563d/scene_info.json) | [scene.usda](3980864799_76ba563d/scene.usda) |
| `4125749490_76ba563d` | [scene_info.json](4125749490_76ba563d/scene_info.json) | [scene.usda](4125749490_76ba563d/scene.usda) |
| `796647424_76ba563d` | [scene_info.json](796647424_76ba563d/scene_info.json) | [scene.usda](796647424_76ba563d/scene.usda) |
