# Complete a word with a letter block

Task: `complete_word_with_letter_block_g2`. Robot: `G2`.

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

`table` `alphabet` `correct_answer` `desktop_with_legs` `distractor` `educational` `plastic` `white` `word_sequence` `workspace_table` `letter_D` `correct_answer_A` `distractor_Q` `letter_E` `distractor_C` `distractor_G` `distractor_J` `letter_A` `letter_M` `distractor_A`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1305874587_dacde46d` | [scene_info.json](1305874587_dacde46d/scene_info.json) | [scene.usda](1305874587_dacde46d/scene.usda) |
| `1522908314_dacde46d` | [scene_info.json](1522908314_dacde46d/scene_info.json) | [scene.usda](1522908314_dacde46d/scene.usda) |
| `1535205523_dacde46d` | [scene_info.json](1535205523_dacde46d/scene_info.json) | [scene.usda](1535205523_dacde46d/scene.usda) |
| `1674631863_dacde46d` | [scene_info.json](1674631863_dacde46d/scene_info.json) | [scene.usda](1674631863_dacde46d/scene.usda) |
| `1775405776_dacde46d` | [scene_info.json](1775405776_dacde46d/scene_info.json) | [scene.usda](1775405776_dacde46d/scene.usda) |
| `2609924422_dacde46d` | [scene_info.json](2609924422_dacde46d/scene_info.json) | [scene.usda](2609924422_dacde46d/scene.usda) |
| `3653269151_dacde46d` | [scene_info.json](3653269151_dacde46d/scene_info.json) | [scene.usda](3653269151_dacde46d/scene.usda) |
| `3980765473_dacde46d` | [scene_info.json](3980765473_dacde46d/scene_info.json) | [scene.usda](3980765473_dacde46d/scene.usda) |
| `429193923_dacde46d` | [scene_info.json](429193923_dacde46d/scene_info.json) | [scene.usda](429193923_dacde46d/scene.usda) |
| `493092131_dacde46d` | [scene_info.json](493092131_dacde46d/scene_info.json) | [scene.usda](493092131_dacde46d/scene.usda) |
