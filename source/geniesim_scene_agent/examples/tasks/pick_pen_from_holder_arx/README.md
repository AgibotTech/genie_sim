# Pick a pen from its holder

Task: `pick_pen_from_holder_arx`. Robot: `ARX`.

## Task and scene references

- [Orchestration code](data_gen.py)
- [Robot and recording configuration](task_info.json)
- [Scene inventory and interface tags](manifest.json)
- [Complete API parameter reference](../../../skills/geniesim-task-orchestration/references/robot_api_reference.md)

## Interface tags

These tags are extracted from data_gen.py without executing it. They describe static calls
and literal capability lookups, not physical success or coverage of every branch.

### RobotAPI

`robot.get_object_pose` `robot.judge_arm_to_use` `robot.pick_actor`

### Client RPC

`robot.client.set_task_task_basic_info`

### Configuration classes

`GraspConfig`

## Scene and action tags

`keywords` `pick` `table_only` `unmovable_objects`

## Scene keyword sample

`pen` `pen_holder` `table` `container` `cylinder` `desktop_with_legs` `long` `stationery` `white` `workspace_table` `writing_tool`

## Scene instances

All instances below share the task code and interface tags above. Inspect their actual
metadata and USD before adapting target selection, poses, or checkers.

| Instance | Metadata | USD |
| --- | --- | --- |
| `1256778353_7a34fd7b` | [scene_info.json](1256778353_7a34fd7b/scene_info.json) | [scene.usda](1256778353_7a34fd7b/scene.usda) |
| `1259289431_7a34fd7b` | [scene_info.json](1259289431_7a34fd7b/scene_info.json) | [scene.usda](1259289431_7a34fd7b/scene.usda) |
| `1343765228_7a34fd7b` | [scene_info.json](1343765228_7a34fd7b/scene_info.json) | [scene.usda](1343765228_7a34fd7b/scene.usda) |
| `1481059574_7a34fd7b` | [scene_info.json](1481059574_7a34fd7b/scene_info.json) | [scene.usda](1481059574_7a34fd7b/scene.usda) |
| `1564596190_7a34fd7b` | [scene_info.json](1564596190_7a34fd7b/scene_info.json) | [scene.usda](1564596190_7a34fd7b/scene.usda) |
| `2646016536_7a34fd7b` | [scene_info.json](2646016536_7a34fd7b/scene_info.json) | [scene.usda](2646016536_7a34fd7b/scene.usda) |
| `2685273650_7a34fd7b` | [scene_info.json](2685273650_7a34fd7b/scene_info.json) | [scene.usda](2685273650_7a34fd7b/scene.usda) |
| `2865667366_7a34fd7b` | [scene_info.json](2865667366_7a34fd7b/scene_info.json) | [scene.usda](2865667366_7a34fd7b/scene.usda) |
| `3041800514_7a34fd7b` | [scene_info.json](3041800514_7a34fd7b/scene_info.json) | [scene.usda](3041800514_7a34fd7b/scene.usda) |
| `74300831_7a34fd7b` | [scene_info.json](74300831_7a34fd7b/scene_info.json) | [scene.usda](74300831_7a34fd7b/scene.usda) |
