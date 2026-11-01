# Replay

Replay recorded robot and object trajectories in Isaac Sim, render new camera observations, and convert each output episode with the repository's `SimDataConverter`. Replay can create multiple scene clones and vary lighting, backgrounds, and materials between rounds.

This is trajectory replay: robot joints or link poses and object poses come from the recording. It does not execute the original task policy or replan motions. Robot selection supports the repository's G2, G1, ARX, and Aloha configurations; the corresponding robot and scene assets must be available.

The open-source replay provides RGB/depth observations and robot/object/cloth replay. LiDAR, point-cloud export, segmentation masks, bounding-box labels, per-frame robot TF export, and perception-specific camera processing are excluded.

## 1. Prepare the environment and input

Use the collection Docker image built with the
[Docker installation instructions](../README.md#docker-installation).
It includes Isaac Sim 5.1, Python 3.11, FFmpeg, and the replay dependencies.

For the host workflow, use `geniesim data-collection replay`; see
[Replay recordings](../README.md#replay-recordings). The rest of this guide
covers the direct replay entry **inside that Docker container**. Run from the
`data_collection` module directory, using `/isaac-sim/python.sh` as shown below.
Input, configuration, asset, and output paths must be accessible through the
container's mounts. The main README includes a complete `docker run` example.

The host CLI's `--depth`, `--no-fuzz`, and `--render-hz` flags prepare a JSON
configuration. They are not flags of the direct `/isaac-sim/python.sh -m replay`
entry; its arguments are listed in section 3.

`SIM_ASSETS` must contain the robot USDs, meshes, textures, and object assets referenced by the recording. There is no separate replay server to start.

Use **one episode directory** as input, rather than a parent containing many episodes:

```text
episode/
├── state.json             # Required: nonempty frames, scene reference, robot/objects
├── recording_info.json    # Recording metadata and object asset mappings
├── metadata.yaml          # Required by the final SimDataConverter stage
├── task_result.json       # Optional: task_name used to name output directories
├── frame_state.json       # Preserve original action annotations when present
├── scene.usda             # Example: the scene referenced by state.scene.scene_usd
└── deformable/            # Optional: recorded cloth/deformable NPZ frames
```

Keep the complete original episode whenever possible. The scene may be elsewhere if its reference resolves; the loader tries the recorded path, an episode-relative path, and matching filenames inside the episode. Missing `recording_info.json` falls back to empty metadata, which may omit information needed to rebuild the scene. Empty `state.json.frames` is rejected. Joint recordings use frame robot joints; link recordings use per-link world poses in frame objects.

Replay can **modify the input**: it rewrites scene asset references, smooths deformable NPZ files (backing up originals under `deformable_raw/`), and injects camera information into source `state.json`. Work on a copy if you need an unchanged source. Use a separate output directory.

## 2. Quick start

Default preset: three environments, one round, RGB video, with lighting/background/material fuzzing enabled and eight built-in HDR background candidates (their assets must exist under `SIM_ASSETS`).

Each HDR candidate uses `background/sphere_hdr_backgrounds/sphere_hdr_<name>/background.usda`.
Keep the referenced textures under `background/common/HDR/` in the same asset bundle.

```bash
/isaac-sim/python.sh -m replay \
  --input_dir /path/to/episode \
  --recording_root ./replay_output
```

`/isaac-sim/python.sh -m replay.run` uses the same arguments. To show the GUI:

```bash
/isaac-sim/python.sh -m replay --input_dir /path/to/episode --no-headless
```

To record depth as well, create a configuration file such as `replay_depth.json`:

```json
{
  "preset": "data_collection",
  "replay_config": {
    "max_concurrent_envs": 1,
    "num_rounds": 1,
    "record_depth": "ALL"
  }
}
```

```bash
/isaac-sim/python.sh -m replay \
  --input_dir /path/to/episode \
  --config_json replay_depth.json \
  --recording_root ./replay_depth_output
```

For trajectory inspection with unchanged lighting/materials and no cloth smoothing, save `replay_inspect.json`:

```json
{
  "preset": "data_collection",
  "fuzzing_config": {
    "enable_light_fuzz": false,
    "enable_background_fuzz": false,
    "enable_material_fuzz": false
  }
}
```

```bash
/isaac-sim/python.sh -m replay \
  --input_dir /path/to/episode \
  --config_json replay_inspect.json \
  --no-deformable-smoothing \
  --cloth-subdivision-levels 0 \
  --no-headless
```

This still records output. Set `replay_config.enable_recording` to `false` for preview without recording.

## 3. Command-line arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `--input_dir PATH` | Required | One existing episode directory. |
| `--config_json PATH` | `replay/config_default.json` | JSON configuration file; relative to the current working directory. |
| `--recording_root PATH` | `recording_data` | Output root. Relative paths resolve from the container module directory. Always overrides JSON `recording_root`, even when omitted on the command line. |
| `--headless` / `--no-headless` | Headless | Run without a GUI / show the GUI. |
| `--deformable-smoothing` / `--no-deformable-smoothing` | Follow JSON | Enable / skip temporal deformable smoothing. Overrides the inverse JSON flag `no_deformable_smooth`; enabled if that flag is false. Without deformable input, smoothing is a no-op. |
| `--cloth-subdivision-levels N` | Follow JSON (normally `2`) | Cloth render mesh subdivision: `0` disables it, `1`–`3` add detail and cost more memory/work. Overrides JSON. |
| `-h`, `--help` | — | Show arguments and exit before starting Isaac Sim. |

There are no CLI flags for individual JSON fields such as `render_hz`, `record_depth`, or `robot_config`; put those in the configuration file.

## 4. JSON configuration and presets

```json
{
  "preset": "data_collection",
  "render_mode": "rtx2",
  "sub_mode": false,
  "replay_config": {
    "max_concurrent_envs": 1,
    "num_rounds": 1,
    "robot_config": "G2_crsB_90D.json",
    "render_hz": 30
  },
  "fuzzing_config": {
    "seed": 42,
    "enable_material_fuzz": false
  }
}
```

Only set `robot_config` to a file matching your recorded robot. Leaving it empty enables automatic selection from `state.robot.name`, or a recorded `<robot>_base_link` key. An ambiguous or unknown robot name requires explicit selection from [`config/robot_cfg`](../config/robot_cfg).

| Top-level field | Default | Purpose |
| --- | --- | --- |
| `preset` | Absent in custom JSON; default config selects `data_collection` | Filename stem in `preset_configs/`. |
| `render_mode` | `"rtx2"` | `rtx2` adds RTX2 startup settings. Other values omit those settings; the application renderer remains `RealTimePathTracing`. This is not an arbitrary renderer selector. |
| `sub_mode` | `false` | Passed to the final data converter as its `is_teleop` flag. Set `true` for teleoperation conversion. Choosing a teleop preset alone does **not** set this flag. |
| `replay_config` | `{}` | Replay and recording options below. |
| `fuzzing_config` | `{}` | Appearance randomization options below. |

Merge order: dataclass defaults → preset `replay_config` / `fuzzing_config` → fields in your JSON → the documented CLI overrides. The merge is shallow: a provided list or dictionary replaces the corresponding preset value. Top-level fields are read from your JSON, not inherited from the preset. Unknown fields inside either config block are silently ignored, so check spelling.

### Available presets

| Preset | Environments × rounds | Depth | Light / background / material fuzzing | Notes |
| --- | --- | --- | --- | --- |
| `data_collection` | 3 × 1 | Off | On / on / on | Default. |
| `teleop` | 1 × 1 | Off | On / off / on | Set top-level `sub_mode: true` for teleop conversion. |

Depth and robot variants are configured through `replay_config.record_depth` and `replay_config.robot_config`; they do not need separate presets. For teleoperation conversion, set top-level `sub_mode: true`.

### Replay and recording parameters

Defaults below are from `ReplayConfig`, **before preset overrides**. In particular, the default `data_collection` preset inherits the default `3` environments and `1` round, while changing spacing to `999.0` and `filter_unmoved_objects` to `false`. The `teleop` preset explicitly selects `1` environment and `1` round.

| Field | Default | Purpose |
| --- | --- | --- |
| `max_concurrent_envs` | `3` | Scene clones rendered per round. Start with `1`; increase only when GPU memory allows. Use a positive integer. |
| `num_rounds` | `1` | Round limit used if `fuzzing_config.num_rounds` is unset. If both are null, replay continues until stopped; set a positive integer for finite runs. |
| `recording_root` | `"recording_data"` | Output root; the CLI value always takes precedence. |
| `clone_spacing` | `3.0` | Distance between cloned environments, in metres. |
| `robot_config` | `""` | Robot configuration filename in `config/robot_cfg`; empty enables automatic selection. |
| `enable_recording` | `true` | Capture observations and create output episodes. False runs replay without recording output. |
| `render_hz` | `30` | Isaac Sim stepping frequency during replay: `physics_dt = rendering_dt = 1 / render_hz`. It does not set the final dataset sampling interval or frequency; those follow the original recording. |
| `no_deformable_smooth` | `false` | Skip deformable temporal smoothing when true; overridden by CLI smoothing flags. |
| `cloth_subdivision_levels` | `2` | Integer `0`–`3`; controls cloth rendering subdivision. |
| `enable_object_compensation` | `false` | Apply local bounding-box centre compensation when placing/updating recorded objects. Enable only when the recording's pose convention needs it. |
| `filter_unmoved_objects` | `true` | Filter stationary objects when rebuilding objects from the recording's asset map; the default collection preset disables it. |
| `enable_unmoved_object_disappear` | `false` | Hide unmoved objects identified by the scene motion analysis. |
| `backgrounds` | `[]` | Static background USDs added during scene setup; separate from fuzzing candidates. See format below. |
| `add_default_light` | `true` | Add default dome/distant lights after scene setup. |
| `add_default_ground_plane` | `false` | Add a Z=0 collision plane when the scene lacks a ground plane. |
| `record_depth` | `"NONE"` | `"NONE"` disables depth; `"ALL"` captures eligible camera depth. Use these values only. |
| `record_all_camera_depth` | `false` | Include cameras normally skipped by name (`head_left_Camera`, `head_right_Camera`, `Fisheye`, `Top`) when depth is enabled. |
| `depth_realtime` | `false` | Encode depth during replay when true; otherwise capture depth files and convert them after replay. Image mode keeps image output instead. |
| `save_depth_in_uint16` | `true` | Save scaled, clipped uint16 PNG depth; false saves float32 NPZ depth. Current scaling is 1000 for camera names containing `head`, 10000 otherwise. |
| `image_mode` | `false` | Save camera images instead of encoding videos. **Also set `clean_cameras: false` to retain images.** |
| `clean_cameras` | `true` | Delete each completed output's `camera/` directory at the end, including raw images. Set false when retaining these outputs. |
| `rename_camera` | `true` | Map known camera prim names to the recorder's short names; false keeps original names. |
| `video_fps` | `30` | Fixed MP4 encoding FPS. |
| `video_preset` | `"ultrafast"` | Fixed video encoder preset. |
| `video_crf` | `24` | Fixed video quality setting. |
| `video_gop` | `8` | Fixed keyframe interval. |
| `video_pix_fmt` | `"yuv420p"` | Fixed pixel-format setting. |

The five video fields are fixed by `ReplayConfig`: conflicting values are logged and replaced. `convert_video` still appears in bundled presets but is not a supported `ReplayConfig` field and is ignored. Use `image_mode` to select images versus videos.

### What `render_hz` changes

**`render_hz` only sets the Isaac Sim advancement step during replay. It does not set the final dataset time step or sampling frequency; those are determined by the original recording.** The default is `30`, giving `physics_dt = rendering_dt = 1 / 30` seconds (about 33.3 ms). A value of `15` makes the Isaac Sim step about 66.7 ms.

The replay loop applies one original recorded frame, advances Isaac Sim once with `World.step(render=True)`, and captures one output frame. Robot and object trajectories are not resampled: changing `render_hz` does not interpolate, skip, or duplicate them. Source frame timestamps are retained in the copied state and converted data. Recording-frequency metadata uses the source `recording_info.json` FPS when available, independently of `render_hz`; older recordings without that field fall back to the robot recording configuration.

For example, data recorded at 25 Hz remains a 25 Hz trajectory even when Isaac Sim replay stepping is 30 Hz. The stored data time interval is 40 ms, while each replay simulation step advances about 33.3 ms.

The MP4 encoding rate is a separate setting, currently fixed at 30 FPS. For a non-30-Hz input, MP4 playback timing can therefore differ from the original data timestamps. Actual replay throughput depends on GPU load, cameras, and scene complexity. Duration/speed statistics may use `1 / render_hz` as a fallback or simulation-step reference; this does not rewrite dataset timestamps.

### Fuzzing parameters

| Field | Default | Purpose |
| --- | --- | --- |
| `enable_light_fuzz` | `true` | Vary light intensity and distant-light direction. |
| `enable_background_fuzz` | `true` | Cycle through configured background candidates. An empty candidate list produces no background replacement. |
| `enable_material_fuzz` | `true` | Apply scene or asset-config material variants. |
| `material_fuzz_mode` | `"both"` | `"scene_scan"`: choose scene peer materials; `"config_file"`: use asset `material_config.json` combinations; `"both"`: apply both strategies. |
| `no_fuzzing_object_id_list` | `[]` | Object IDs excluded from asset-config material collection. Do not rely on this list to exclude scene-scan material changes. |
| `num_rounds` | `null` | Overrides `replay_config.num_rounds` when provided. Use a positive integer. |
| `seed` | `null` | Seed for the fuzzer's local random generator. Does not guarantee deterministic rendering or every random operation in dependencies. |
| `backgrounds` | `[]` | Background replacement candidates, shuffled and cycled by the fuzzer. |

Both background lists use entries such as:

```json
{
  "usd": "background/room/room_1/background.usda",
  "position": [0, 0, 0],
  "rotation": [1, 0, 0, 0]
}
```

`usd` is relative to `SIM_ASSETS`; position is in metres and rotation is a quaternion in `[w, x, y, z]` order. The named asset must actually exist in your asset bundle.

## 5. Output and completion

Output directories are allocated under `--recording_root`, using `task_result.json.task_name` (or `replay`) and numeric suffixes when needed. Existing directories are skipped when choosing new names. Each recorded clone gets its own episode directory.

Typical output includes:

- `observations/videos/`: rendered RGB videos and depth videos when enabled.
- `camera/`: per-frame images when captured and retained with `clean_cameras: false`.
- New `recording_info.json`, plus copied source state, metadata, scene/action files, and your input config JSON.
- Converted episode data, including `aligned_joints.h5` and `data_info.json` when conversion succeeds.
- A `replay_done` marker.

The copied config is the original JSON, not a fully merged snapshot of preset/default/CLI values. The pipeline skips outputs with detected video failures, copies source files to successful recording directories, and attempts conversion. `replay_done` marks the end of replay processing. Check converter logs and expected converted files for conversion results. Missing `metadata.yaml` causes conversion to be skipped even if rendering succeeded.

## 6. Common issues

| Symptom | Check |
| --- | --- |
| Input/config not found | Run from the container module directory; pass an episode directory and an existing JSON file. |
| Robot selection is ambiguous | Set `replay_config.robot_config` to the matching config filename, including the correct gripper variant. |
| Missing scene/robot USD | Check `SIM_ASSETS`, scene references, and the asset bundle version. |
| GPU memory exhaustion | Reduce `max_concurrent_envs`, disable unneeded depth, or reduce cloth subdivision. |
| Replay runs indefinitely | Set a positive round limit; an explicit fuzzing limit takes precedence. |
| Images disappear | Set `clean_cameras: false`; this is required with image mode. |
| Video speed differs from input timing | MP4 encoding is fixed at 30 FPS. `render_hz` changes simulation stepping, not the MP4 FPS. |
| Conversion missing after rendering | Check `metadata.yaml`, converter logs, and video failures; do not rely on `replay_done` alone. |

