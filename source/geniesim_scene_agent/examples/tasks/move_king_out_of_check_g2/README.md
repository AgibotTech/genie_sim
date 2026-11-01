# Move the king out of check

Task: `move_king_out_of_check_g2`. Robot: `G2`.

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

`black` `chessboard` `king_black` `king_white` `square` `table` `white` `knight_black` `pawn_white` `queen_black` `bishop_white` `pawn_black` `queen_white` `brown` `centered_on_desk` `chess_piece` `dining_table` `four_legs` `game_board` `wood`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1500513522_17bb57ba` | [scene_info.json](1500513522_17bb57ba/scene_info.json) | [scene.usda](1500513522_17bb57ba/scene.usda) |
| `1877266214_17bb57ba` | [scene_info.json](1877266214_17bb57ba/scene_info.json) | [scene.usda](1877266214_17bb57ba/scene.usda) |
| `2098266743_17bb57ba` | [scene_info.json](2098266743_17bb57ba/scene_info.json) | [scene.usda](2098266743_17bb57ba/scene.usda) |
| `2211061446_17bb57ba` | [scene_info.json](2211061446_17bb57ba/scene_info.json) | [scene.usda](2211061446_17bb57ba/scene.usda) |
| `2510285256_17bb57ba` | [scene_info.json](2510285256_17bb57ba/scene_info.json) | [scene.usda](2510285256_17bb57ba/scene.usda) |
| `2863007596_17bb57ba` | [scene_info.json](2863007596_17bb57ba/scene_info.json) | [scene.usda](2863007596_17bb57ba/scene.usda) |
| `320539372_17bb57ba` | [scene_info.json](320539372_17bb57ba/scene_info.json) | [scene.usda](320539372_17bb57ba/scene.usda) |
| `3847335851_17bb57ba` | [scene_info.json](3847335851_17bb57ba/scene_info.json) | [scene.usda](3847335851_17bb57ba/scene.usda) |
| `386572289_17bb57ba` | [scene_info.json](386572289_17bb57ba/scene_info.json) | [scene.usda](386572289_17bb57ba/scene.usda) |
| `405302629_17bb57ba` | [scene_info.json](405302629_17bb57ba/scene_info.json) | [scene.usda](405302629_17bb57ba/scene.usda) |
