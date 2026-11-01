# Sort chess pieces by color

Task: `sort_chess_pieces_by_color_g2`. Robot: `G2`.

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

`extra_checkers` `keywords` `pick_and_place` `sort` `table_only` `unmovable_objects`

## Scene keyword sample

`chess_piece` `table` `workspace_table` `white` `black` `knight` `queen` `queen_white` `knight_white` `rook` `bishop` `knight_black` `pawn` `pawn_white` `rook_white` `bishop_black` `bishop_white` `queen_black` `rook_black` `on_desk`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1289596779_b876748c` | [scene_info.json](1289596779_b876748c/scene_info.json) | [scene.usda](1289596779_b876748c/scene.usda) |
| `2230916950_b876748c` | [scene_info.json](2230916950_b876748c/scene_info.json) | [scene.usda](2230916950_b876748c/scene.usda) |
| `2361945732_b876748c` | [scene_info.json](2361945732_b876748c/scene_info.json) | [scene.usda](2361945732_b876748c/scene.usda) |
| `3244146839_b876748c` | [scene_info.json](3244146839_b876748c/scene_info.json) | [scene.usda](3244146839_b876748c/scene.usda) |
| `3276598109_b876748c` | [scene_info.json](3276598109_b876748c/scene_info.json) | [scene.usda](3276598109_b876748c/scene.usda) |
| `3682967762_b876748c` | [scene_info.json](3682967762_b876748c/scene_info.json) | [scene.usda](3682967762_b876748c/scene.usda) |
| `3697892899_b876748c` | [scene_info.json](3697892899_b876748c/scene_info.json) | [scene.usda](3697892899_b876748c/scene.usda) |
| `3860469078_b876748c` | [scene_info.json](3860469078_b876748c/scene_info.json) | [scene.usda](3860469078_b876748c/scene.usda) |
| `4128543003_b876748c` | [scene_info.json](4128543003_b876748c/scene_info.json) | [scene.usda](4128543003_b876748c/scene.usda) |
| `50673739_ce7255ca` | [scene_info.json](50673739_ce7255ca/scene_info.json) | [scene.usda](50673739_ce7255ca/scene.usda) |
