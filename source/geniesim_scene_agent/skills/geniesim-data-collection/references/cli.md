# Collection runtime CLI reference

All `geniesim` commands below run on the host with a compatible main CLI and
data-collection module available. The examples use the monorepo root as cwd.
No machine-specific checkout is required by this document. The module and
runtime are separate from the bundled scene-authoring tools. These are
optional installation/execution commands for robot tasks; scene generation
and rendered previews use the scene tools and their Isaac Sim renderer.

## Runtime and paths

```bash
geniesim data-collection install \
  --task-root "$TASK_ROOT" --assets-root "$ASSETS_ROOT"
geniesim data-collection config
geniesim data-collection llm list --task-root "$TASK_ROOT" --robot g2
geniesim data-collection llm show "$TASK_NAME" --task-root "$TASK_ROOT"
```

`install` performs an editable install of the lightweight locator and saves
defaults in `~/.config/geniesim/data-collection.json`. It accepts `--dry-run`.
Environment variables override saved defaults. Supply per-command roots to
select a specific task/asset version without changing global configuration.

Task discovery uses `GENIESIM_LLM_TASK_ROOT`, then `LLM_TASK_ROOT`, saved
settings, sibling scene-agent examples, installed scene-agent resources,
local task libraries, and configured asset-pack tasks. Asset discovery uses
`GENIESIM_ASSETS_PATH`/`SIM_ASSETS`, saved settings, installed asset packages,
and nearby relative layouts. `GENIESIM_ASSETS` is the authoring override;
pass its resolved value as `--assets-root` when using collection. Confirm that
authoring and execution use the intended same asset version. Runtime assets
must provide the compatible asset index and robot assets, not only USD files.

```bash
ASSETS_ROOT=$(python -m geniesim_scene_agent assets-root)
TASK_ROOT=/path/to/writable/task-library
TASK_NAME=example_task
SCENE_ID=0
RUN_DIR=/path/to/new/collection-run
```

Set these paths and names to prepared artifacts before executing commands.
For bundled references, the monorepo task root is
`source/geniesim_scene_agent/examples/tasks`; retained scenes can have opaque
IDs. Newly generated scenes use numeric IDs.

The default image is
`registry.agibot.com/genie-sim/geniesim3-data-collection:latest`, and the
default container name is `geniesim3-data-collection`.
`DATA_COLLECTION_IMAGE` or `--image` selects another compatible runtime.
Use `geniesim data-collection build` if the runtime must be built locally.
The host locator install does not install simulation dependencies.

## Collection

```bash
geniesim data-collection run "$TASK_NAME" \
  --task-root "$TASK_ROOT" --assets-root "$ASSETS_ROOT" \
  --scene "$SCENE_ID" --run-times 1 --render --output-dir "$RUN_DIR"
```

`llm run` is an equivalent explicit alias. `run --task-json FILE` selects
the legacy job-JSON launcher, whose option set and output layout differ.
The library's `task_info.json` is robot/task metadata, not a job JSON.

| Option | Default | Meaning |
| --- | --- | --- |
| `TASK` | Required | Exact task directory name; ambiguous substring matches fail |
| `--task-root PATH` | Discovered | Parent of task directories |
| `--assets-root PATH` | Discovered | External runtime asset package |
| `--scene ID` | `random` | Fixed scene folder or random selection for each episode |
| `--run-times N` | `1` | Serial episode attempts, not scene instance count |
| `--record` / `--no-record` | Recording on | Save trajectory/metadata and convert recorded data |
| `--render` | Off | Enable collection camera rendering and image/video conversion |
| `--headless` / `--no-headless` | Headless | No GUI / GUI requiring host DISPLAY and X11 access |
| `--output-dir PATH` | New timestamped run | Must not exist and must be outside the task library |
| `--image TAG` | Configured/default image | Collection image with compatible runtime |
| `--container-name NAME` | Default name | Choose a free name; an existing container blocks startup |
| `--server-start-timeout SEC` | `420` | Time allowed for the server to become ready |
| `--episode-timeout SEC` | `600` | Time allowed for the client episode process |
| `--task-result-timeout SEC` | `600` | Time allowed for recording conversion/final result |
| `--dry-run` | Off | Validate selection and print command; create no run or container |

Use `--render` with recording for direct MP4 output. `--render --no-headless`
adds a live viewport. `--no-record` cannot supply saved videos, regardless of
rendering. GUI mode may also require authorizing the container's X11 access
according to the host configuration; the CLI does not automatically do that.
There is no livestream option in this high-level CLI. A server livestream
flag alone is not an integrated browser viewing workflow.

Server physics settings come from the selected scene's `task_info.json`, or
the task-level file when no scene override exists:

| Field | Default | Effect |
| --- | --- | --- |
| `backend` | `cpu` | `cpu` or `gpu` physics; RTX rendering still requires a GPU |
| `physics_frequency` | `60` | Physics Hz, independent of recording fps |
| `enable_deformable` | `false` | Deformable support; requires GPU physics |
| `curobo_batch_size` | `10` | Planner/IK batch size |

A scene override must contain the complete required robot/task metadata.
Camera prims and recording fps come from `recording_setting`, not the physics
frequency. Use a compatible robot configuration and actual camera paths.
Changing noise, contact policy, or layout to obtain success requires preserving
the user's approved task; it is not a silent collection setting change.

## Outputs and verification

Default run root: `data_collection/recording_data/llm/<task>/<run>/`.
An explicit `--output-dir` gives the same structure under that selected root:

```text
<run>/
  run.log
  run_manifest.json
  run_result.json
  tasks/<task>/                 # isolated task copy; USD rebinding happens here
  report/
    server_config_1.json
    server_1.log
    client_1.log
    episode_1.json
    collection_result.json
  episodes/<episode>/
    state.json
    metadata.yaml
    recording_info.json
    frame_state.json
    task_result.json
    aligned_joints.h5
    observations/videos/*.mp4   # when cameras are configured and --render succeeds
```

Names inside an episode follow the recorder/converter; use metadata to locate
outputs rather than assuming a particular camera filename. More episodes
add corresponding numbered reports. Failed or non-recorded runs may not have
all episode artifacts. Keep the server configuration and first failing phase.

The CLI success summary checks process status and episode reports, including
the task result and required trajectory artifacts. It does not independently
prove task geometry or that every requested camera produced a valid video.
Apply [physical task validation](../../geniesim-task-validation/SKILL.md).

For each expected MP4, inspect the actual output path:

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames,avg_frame_rate:format=duration \
  -of json "$VIDEO_PATH"
```

Check playable nonempty streams, expected camera coverage, dimensions,
frame/timestamp consistency, and the conversion log. Do not require the
replay verifier's exact frame-count equality for direct collection without
checking its camera and trajectory sampling rules. Preserve metric failures
and return values; user acceptance is a separate recorded decision.

## Optional replay

```bash
geniesim data-collection replay "$EPISODE_DIR" \
  --assets-root "$ASSETS_ROOT" --no-fuzz \
  --max-concurrent-envs 1 --num-rounds 1 --output-dir "$REPLAY_RUN_DIR"
```

`EPISODE_DIR` contains one recorded episode. A task name instead selects its
latest successful recorded CLI run, including custom-output runs. An explicit
path prevents selecting another run after later collection.

| Option | Default | Meaning |
| --- | --- | --- |
| `--config FILE` | Preset defaults | Replay JSON configuration |
| `--preset data_collection\|teleop` | `data_collection` | Recording provenance; not selected by robot family |
| `--max-concurrent-envs N` | `3` for collection | Concurrent rendered scene clones |
| `--num-rounds N` | `1` | Finite replay rounds |
| `--render-hz N` | `30` | Simulation/replay advance frequency, not original recording fps |
| `--depth NONE\|ALL` | `NONE` | Depth subject to configured camera eligibility |
| `--no-fuzz` | Fuzzing on for collection | Disable random light/background/material changes |

Common collection flags also apply: assets, output, image, container name,
headless/GUI, and dry-run. Default clones/fuzzing serve data augmentation;
the explicit single-clone/no-fuzz command above serves user review and does
not add random backgrounds. Preserve any background already in the scene.
Replay copies input, renders recorded trajectories without re-executing the
policy, converts outputs, and checks them. Inspect `<replay_run>/output_check.json`
and videos under `output/<episode>/observations/videos/`.

## Coverage and concrete limitations

- Random scene selection is sampling with replacement. For one episode from
  every approved scene, enumerate explicit IDs and use a fresh output run per
  scene; record the selected scene, attempt, outcome, and episode directory.
- Deformable tasks reject `run-times > 1`. USD-attached behavior-script tasks
  also require one run, although the host's lightweight task catalog does not
  detect those asset scripts. Repeated collection uses separate single runs.
- The current multi-episode run stops after a failed episode. `run_times` is a
  requested attempt count, not a guaranteed successful episode count. Bounded
  retries and failure policy must be part of the approved collection plan.
- Each CLI run copies the entire selected task directory, including its scenes
  and previews. For a large library, budget this copy/storage cost or prepare
  a private single-scene input library with the task code/metadata, that scene,
  and all required local dependencies. Do not use writable USD hardlinks to
  the approved source. The current runtime starts/stops a server per episode,
  including episodes within `--run-times N`; do not assume server reuse.
- The default container name permits one active CLI run. Serialize runs;
  unique names alone do not establish safe GPU/resource concurrency.
- Metadata and scene seeds do not seed every runtime random source. Record
  robot pose/joint noise and the controls actually used for reproducibility.
- Additional task dependencies must exist in the execution image. Direct CLI
  Python startup does not promise to source task-owned shell `runtime_env`
  hooks used by another scheduler/launcher path.
- Legacy `client.llm_gen.launcher` needs an already running server, rewrites
  its input USD, and caches imports. In the inspected current version its
  recording flag defaults off; pass `--use_recording` explicitly when using
  it. Prefer the high-level isolated CLI workflow for user demonstrations.
