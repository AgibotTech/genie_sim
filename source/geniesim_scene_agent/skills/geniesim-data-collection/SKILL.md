---
name: geniesim-data-collection
description: Execute generated GenieSim data_gen.py tasks with the data-collection runtime, record camera videos directly with --render, inspect physical and recording results, and optionally replay recorded trajectories.
---

# Execute and record collection tasks

Use this skill when a task is ready for robot execution, a user wants to see
the actions, or approved episodes must be collected. Use
[task validation](../geniesim-task-validation/SKILL.md) for physical completion
criteria and [the complete workflow](../geniesim-scene-to-data/SKILL.md) when
the scene or task still needs design. Runtime commands, flags, output files,
and recovery notes are bundled in [the CLI reference](references/cli.md).

This is an optional downstream workflow. Scene generation and task code
preparation use scene-agent tools without this module or image. A request for
robot execution or an action demonstration needs this runtime even when no
bulk dataset is requested. Bulk collection is a separate scope decision.

## Prepare the runtime and task

Run the `geniesim data-collection` CLI on the host. It locates the collection
module and uses its configured image, which supplies Isaac Sim 6.2, cuRobo,
RobotAPI, gRPC, ROS recording, and conversion dependencies. The scene-agent
image supplies authoring and rendered scene previews; task execution uses the
collection runtime. Reuse an available compatible image/environment.
`data-collection install` registers the host locator and saves paths; it does
not install Isaac Sim or cuRobo. `data-collection build` builds the runtime.

Resolve the external asset root and a task library containing
`<task_name>/data_gen.py`, `task_info.json`, and scene folders with `scene.usda`
and `scene_info.json`. Use an English task name consistent across these files.
Keep generated tasks in a writable draft library; add accepted tasks to the
example library through the closeout workflow. Assets remain external.

Inspect `llm show` and the actual files. `ready` checks the robot configuration,
code presence and scene inventory. For execution, check the robot, initial pose/noise, scene fields,
asset interaction/grasp data, physical success checks, and supported cameras.
For direct videos, `recording_setting.camera_list` must contain existing camera
prim paths for this robot. Use the intended recording frequency, normally
30 Hz, and verify it against the installed runtime.

## Execute one fixed scene with direct recording

First select one explicit scene and one episode. Set the task library, task,
scene, assets, and a new output directory. Preview the command with `--dry-run`
when checking paths/options; a dry-run does not start simulation.

```bash
geniesim data-collection run "$TASK_NAME" \
  --task-root "$TASK_ROOT" --assets-root "$ASSETS_ROOT" \
  --scene "$SCENE_ID" --run-times 1 \
  --render --output-dir "$RUN_DIR"
```

**Direct recording with `--render` is the default user-facing demonstration.**
Recording and headless execution are enabled by default. This command executes
the task, captures the configured cameras, and converts the recording to
videos; replay is not required. The CLI copies the selected task into the run
directory before rebinding USD references, then manages server/client startup,
shutdown, logs, and episode reports. Keep the source task unchanged.

Add `--no-headless` for an interactive GUI with rendering; this requires a
working host DISPLAY and X11 access. A headless direct recording works on a
GPU server and can be shown to the user after completion. `--render` alone
does not open a GUI. `--no-record` disables saved episodes and videos; use it
only for an explicitly chosen physical/debug check.

## Verify and show the result

Read `run_result.json`, the per-episode report, `task_result.json`, server/client
logs, and the final physical checks. Investigate disagreement between an
action return, `execute_task_success`, metrics, and the observed final state.
Do not equate a process exit or a playable video with task completion.

For a requested recording, check `state.json` frames, `metadata.yaml`,
`recording_info.json`, action annotations, and `aligned_joints.h5`. Derive the
expected camera outputs from recording metadata, locate their MP4 files, and
verify streams, dimensions, nonzero frame counts, and plausible timing with
`ffprobe`. Direct collection does not perform the replay command's full video
verification automatically. Check the conversion log as well.

Show an informative configured camera video using the UI's local media
preview and an absolute output path. Preserve the complete original video;
label any excerpt and its time range. Report task/scene, robot, randomization,
actual validation level, and the first failing phase if any. Ask for feedback
on the demonstrated behavior before claiming user acceptance. Some failed
runs have no usable video; retain their reports and explain that limitation.

## Optional replay

Replay is useful for new camera observations, depth, or another rendered view
of an existing recording. It reuses recorded robot/object trajectories and
does not execute `data_gen.py` or replan the task. For one demonstration that
preserves appearance, use `--no-fuzz --max-concurrent-envs 1 --num-rounds 1`.
Read the CLI reference before using defaults or requesting depth. A task name
selects its latest successful recorded CLI run; an explicit episode path pins
the input. Verify `output_check.json` before presenting replay output.

## Approved collection and recovery

Record scene coverage, episodes per scene or total episodes, variation/noise,
camera settings, retries, and an output/resource budget before a batch. A
request for 1000 scene instances does not authorize 1000 episodes. Random scene
selection samples with replacement; it does not guarantee scene coverage.
Use explicit scene IDs for coverage and retain a per-attempt run/result index.

Apply the runtime's single-run restrictions: deformable tasks and tasks using
USD-attached behavior scripts require separate runs with `--run-times 1`.
Validate representative approved variants before scaling. Stop on a repeated
common failure, preserve evidence, and diagnose the affected phase instead of
silently retrying indefinitely or changing the collection definition.

For repeated IK failures, follow
[IK recovery](../geniesim-task-orchestration/references/ik-recovery.md); obtain
confirmation of specific layout edits before moving scene objects or the base.
After user acceptance, retain the task and representative scenes and apply
[task-driven self-improvement](../geniesim-task-orchestration/references/task-self-improvement.md).
