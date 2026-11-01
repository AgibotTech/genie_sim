# Write an uppercase letter with a pen

Task ID: `write_uppercase_letter_with_pen_g2`.

## Scene-defined letter

The task has 26 numeric scene directories, `0` through `25`, corresponding to
uppercase `A` through `Z`. They share the shared writing layout. Each scene
contains `scene.usda`, `scene_info.json`, `generation.json`, and `validation.json`.
Initial workspace preview images are not bundled.

For example, `0/scene_info.json` contains:

```json
{
  "task_name": "write_uppercase_letter_with_pen_g2",
  "english_task_name": "Write an uppercase letter with a pen",
  "instance_id": "0",
  "target_letter": "A"
}
```

`data_gen.py` reads `robot.scene_info["target_letter"]` once per execution and
uses the same letter for the task instruction and contact trajectories. Missing,
lowercase, or unsupported values fail before any robot motion. There is no
module-level fixed target and no random letter override. To vary the letter,
select another scene instance in the collection runner.

## Orchestration

1. Locate the pen and paper by semantic keywords or asset IDs.
2. Build 144 body-local grasp candidates with local-Y translations and rotations.
3. Read the paper's live pose and dimensions from RobotAPI.
4. Convert the selected letter's normalized polylines into paper coordinates.
5. Sample stroke segments with a maximum spacing of 2 mm.
6. For every stroke, hover above the start, descend, follow the contact path,
   and lift before moving to another stroke.
7. Return the right arm to its origin.

The glyph is 6 cm wide and 8 cm high. Contact paths keep the pen tip 2 mm above
the paper's top surface, use 5 cm lifts, and provide three lateral placement
candidates. The robot faces world +X: visual up corresponds to paper +X,
and visual right to paper -Y. Letter trajectories fit inside the authored paper.

## Inspect and preview

From the repository root:

```bash
python3 examples/tasks/write_uppercase_letter_with_pen_g2/preview_letter_trajectory.py \
  --scene-info examples/tasks/write_uppercase_letter_with_pen_g2/0/scene_info.json \
  --output .cache/letter_a_trajectory.png
```

The shared [alphabet trajectory gallery](alphabet_trajectories.png) shows the
stroke shapes. It illustrates the planned paths, not ink drawn by a completed
robot episode.

The original USD includes an empty `/World/Objects/Ink` container used by the
writing runtime. It is declared in `auxiliary_prims`, separate from physical
asset roles. Object positions and the table asset ID in scene metadata have been
synchronized with the authored USD; USD geometry and physical settings are kept.

## Validation

The 26 instances share one physical layout and select different letter paths.
Run adapted code in the configured robot runtime to verify reachability and
physical execution.

The task manifest records the scene letters and current file hashes. Use
`python3 examples/validate.py` to check metadata, inventory, and USD dependencies.
