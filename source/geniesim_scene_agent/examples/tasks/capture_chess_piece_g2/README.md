# Capture a chess piece

Task: `capture_chess_piece_g2`. Robot: `G2`.

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

`black` `chess_piece` `chessboard` `king_black` `king_white` `square` `table` `white` `bishop_black` `rook_white` `bishop_white` `queen_black` `rook_black` `knight_black` `pawn_black` `pawn_white` `knight_white` `queen_white` `brown` `centered_on_desk`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1160909934_c4de999d` | [scene_info.json](1160909934_c4de999d/scene_info.json) | [scene.usda](1160909934_c4de999d/scene.usda) |
| `133652017_c4de999d` | [scene_info.json](133652017_c4de999d/scene_info.json) | [scene.usda](133652017_c4de999d/scene.usda) |
| `2023284570_c4de999d` | [scene_info.json](2023284570_c4de999d/scene_info.json) | [scene.usda](2023284570_c4de999d/scene.usda) |
| `2715624707_c4de999d` | [scene_info.json](2715624707_c4de999d/scene_info.json) | [scene.usda](2715624707_c4de999d/scene.usda) |
| `2899683076_c4de999d` | [scene_info.json](2899683076_c4de999d/scene_info.json) | [scene.usda](2899683076_c4de999d/scene.usda) |
| `3021193309_c4de999d` | [scene_info.json](3021193309_c4de999d/scene_info.json) | [scene.usda](3021193309_c4de999d/scene.usda) |
| `3167414328_c4de999d` | [scene_info.json](3167414328_c4de999d/scene_info.json) | [scene.usda](3167414328_c4de999d/scene.usda) |
| `3548581845_c4de999d` | [scene_info.json](3548581845_c4de999d/scene_info.json) | [scene.usda](3548581845_c4de999d/scene.usda) |
| `380311263_c4de999d` | [scene_info.json](380311263_c4de999d/scene_info.json) | [scene.usda](380311263_c4de999d/scene.usda) |
| `599576995_c4de999d` | [scene_info.json](599576995_c4de999d/scene_info.json) | [scene.usda](599576995_c4de999d/scene.usda) |
