# Spell a common-sense answer with letter blocks

Task: `spell_common_sense_answer_with_letter_blocks_g2`. Robot: `G2`.

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

`table` `alphabet` `desktop_with_legs` `educational` `plastic` `white` `workspace_table` `letter_E` `letter_I` `letter_R` `letter_T` `letter_B` `letter_C` `letter_D` `letter_N` `letter_A` `letter_F` `letter_H` `letter_J` `letter_L`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1285482612_9c751109` | [scene_info.json](1285482612_9c751109/scene_info.json) | [scene.usda](1285482612_9c751109/scene.usda) |
| `1736929667_9c751109` | [scene_info.json](1736929667_9c751109/scene_info.json) | [scene.usda](1736929667_9c751109/scene.usda) |
| `1875981657_9c751109` | [scene_info.json](1875981657_9c751109/scene_info.json) | [scene.usda](1875981657_9c751109/scene.usda) |
| `2444480645_9c751109` | [scene_info.json](2444480645_9c751109/scene_info.json) | [scene.usda](2444480645_9c751109/scene.usda) |
| `2516804253_9c751109` | [scene_info.json](2516804253_9c751109/scene_info.json) | [scene.usda](2516804253_9c751109/scene.usda) |
| `3266876095_9c751109` | [scene_info.json](3266876095_9c751109/scene_info.json) | [scene.usda](3266876095_9c751109/scene.usda) |
| `3290167008_9c751109` | [scene_info.json](3290167008_9c751109/scene_info.json) | [scene.usda](3290167008_9c751109/scene.usda) |
| `3363548676_9c751109` | [scene_info.json](3363548676_9c751109/scene_info.json) | [scene.usda](3363548676_9c751109/scene.usda) |
| `3585608244_9c751109` | [scene_info.json](3585608244_9c751109/scene_info.json) | [scene.usda](3585608244_9c751109/scene.usda) |
| `3594109827_9c751109` | [scene_info.json](3594109827_9c751109/scene_info.json) | [scene.usda](3594109827_9c751109/scene.usda) |
