# Pick the chess piece giving check

Task: `pick_chess_piece_giving_check_g2`. Robot: `G2`.

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

`keywords` `pick_and_place` `table_only` `unmovable_objects`

## Scene keyword sample

`black` `chessboard` `king_black` `king_white` `square` `table` `white` `bishop_black` `knight_white` `bishop_white` `rook_black` `rook_white` `knight_black` `brown` `centered_on_desk` `chess_piece` `dining_table` `four_legs` `game_board` `wood`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1042940665_17bb57ba` | [scene_info.json](1042940665_17bb57ba/scene_info.json) | [scene.usda](1042940665_17bb57ba/scene.usda) |
| `1541264763_17bb57ba` | [scene_info.json](1541264763_17bb57ba/scene_info.json) | [scene.usda](1541264763_17bb57ba/scene.usda) |
| `1927793975_17bb57ba` | [scene_info.json](1927793975_17bb57ba/scene_info.json) | [scene.usda](1927793975_17bb57ba/scene.usda) |
| `2572094729_17bb57ba` | [scene_info.json](2572094729_17bb57ba/scene_info.json) | [scene.usda](2572094729_17bb57ba/scene.usda) |
| `2590856131_17bb57ba` | [scene_info.json](2590856131_17bb57ba/scene_info.json) | [scene.usda](2590856131_17bb57ba/scene.usda) |
| `2720836419_17bb57ba` | [scene_info.json](2720836419_17bb57ba/scene_info.json) | [scene.usda](2720836419_17bb57ba/scene.usda) |
| `3541564166_17bb57ba` | [scene_info.json](3541564166_17bb57ba/scene_info.json) | [scene.usda](3541564166_17bb57ba/scene.usda) |
| `4144858609_17bb57ba` | [scene_info.json](4144858609_17bb57ba/scene_info.json) | [scene.usda](4144858609_17bb57ba/scene.usda) |
| `437805910_17bb57ba` | [scene_info.json](437805910_17bb57ba/scene_info.json) | [scene.usda](437805910_17bb57ba/scene.usda) |
| `84330234_17bb57ba` | [scene_info.json](84330234_17bb57ba/scene_info.json) | [scene.usda](84330234_17bb57ba/scene.usda) |
