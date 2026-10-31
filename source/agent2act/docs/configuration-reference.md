# Configuration reference

This reference uses `examples/g2_llm_recipe.json` to describe fields, defaults, and the impact of
changes. Later sections cover observation switches, motion behavior, gripper presets, camera
calibration, and the gripper-center interface. Other embodiments share the same structure, with
differences in `robot`, `motion_backend`, and `simulator_codec`; see the
[Configuration guide](../examples/README.md).

Impact levels indicate how changes may affect task performance: **High** changes model inputs or
execution semantics directly; **Medium** affects replanning, fault handling, or timing;
**Low** affects diagnostics or overhead; **None** marks values matching code defaults.
A current value of `default` means that omitting the field is equivalent to retaining it.

## Top-level fields

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `version` | `1` | Required | Configuration format version; only `1` is accepted | High |
| `ranked_motion_candidates` | `true` | `false` | Generate multiple targets and select the first candidate passing local checks | High |
| `ranked_motion_candidate_count` | `5` | `default` | Candidate count; only `5` or `10` | High |

## `robot`

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `robot_type` | `G2_omnipicker` | Required | Protocol embodiment identifier; must match the `embodiment` preset | High |
| `left_joints` / `right_joints` | 7 each | Required | Joint ordering in observations and issued actions | High |
| `body_joints` | 5 torso joints | `[]` | Torso joint names; ordering indexes `body_limits` | High |
| `body_limits` | 5 limit pairs | `[]` | Torso position limits, checked at every step | High |
| `body_speed_limits` | `[1,1,1,1,1000]` | `[]` | Torso speed limits; the final `1000` is a very high waist-yaw limit | Medium |
| `dt_s` | `0.0333…` | Required | Action period (30 Hz), used for validation and issuance timing | High |
| `gripper_bounds` | `[[-0.785,0],[-0.785,0]]` | Required | Gripper command bounds; must contain preset command values | High |
| `supports_eef` | `true` | `false` | Allow EEF poses in addition to joint commands | High |
| `max_chunk_steps` | `256` | `default` | Maximum steps per action chunk; bounds execution prefix settings | Low |
| `enforce_speed_limits` | `false` | `true` | Enable speed validation and timing adjustments | High |

With `enforce_speed_limits: false`, speed validation is bypassed and compilers do not retime
trajectories to speed limits. Per-step displacements are not rejected for exceeding those limits;
IK and position-range checks remain separate. Before enabling it, configure `max_joint_speed`,
`max_eef_speed`, `max_eef_angular_speed`, and `max_gripper_speed` for the hardware. Omitting them
uses conservative defaults (`1.0 / 0.25 / 1.0 / 2.0`), which may reject otherwise feasible actions.

## `execution`

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `mode` | `motion` | `auto` | Execution mode | High |
| `allowed_modes` | `["motion"]` | All modes | Allowed routing modes; also constrains `serve --mode` | High |
| `execute_steps` | `10` | `8` | Maximum issued steps per turn (about 0.33 seconds) | High |
| `body_execute_steps` | `50` | `null` | Prefix size for torso motion; should cover a waist turn | High |
| `complete_hold_steps` | `1000` | `null` | Hold steps after a completion judgment, allowing simulator evaluation | Medium |
| `max_replans` | `2` | `default` | Additional replanning attempts after rejection | Medium |
| `memory_size` | `12` | `default` | Number of runtime events retained | Low |
| `need_depth` | `true` | `false` | Request depth; should match `planner.use_depth` | High |
| `history.interval` / `last_n` / `resolution` | `1` / `1` / See config | See schema | History sampling parameters echoed to the simulator, not directly added to model input | None |
| `max_motion_continuations` | `0` | `default` | Continue the same target from a fresh observation without querying the model; `0` queries every turn | Medium |
| `motion_frame` | `base_link` | `null` | Reference frame for model motion targets | High |
| `motion_frame_axes` | `forward/left/up` | `null` | Axis semantics of that frame | High |
| `terminal_on_planning_exhaustion` | `true` | `false` | Issue a hold and end the episode when replanning is exhausted, instead of raising | Medium |
| `retry_ik_exhaustion` | `true` | `false` | For pure IK exhaustion, issue a hold requesting replanning rather than terminate the episode | Medium |

`execution.history` is independent of `planner.history_frames`. The former is echoed into the
response JSON by `encode_result()` without driving planner logic; the latter controls historical
images actually attached to model input.

With `max_motion_continuations > 0`, excessive tracking/gripper command
error cancels continuation and returns control to the model.

## `planner`

### Connection and retries

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `type` | `embodied` | Required | Planner protocol implementation | High |
| `model` | `gpt-6-astra` | Required | Model identifier | High |
| `endpoint` | OpenAI Responses endpoint | Official OpenAI endpoint | Request URL | High |
| `api_key_env` | `AGENT2ACT_LIVE_LLM_KEY` | `OPENAI_API_KEY` | Environment variable containing the API key | High |
| `timeout_s` | `300` | `60` | Timeout per HTTP request | Medium |
| `max_output_tokens` | `16384` | `4096` | Output budget, including thinking tokens | High |
| `max_retries` | `20` | `2` | Maximum network retries (0–99) | Medium |
| `retry_backoff_s` | `1` | `default` | Initial exponential backoff in seconds | Low |
| `retry_max_delay_s` | `60` | `10` | Maximum retry delay (≤60 seconds) | Medium |

Backoff is `min(retry_max_delay_s, retry_backoff_s × 2^attempt)`, multiplied by a random factor
of 0.75–1.0. Retries cover HTTP 408, 429, 500, 502–504, 520–524, timeouts, connection resets,
temporary DNS failures, and interrupted response reads. Authentication failures, certificate errors,
permanent DNS errors, and other status codes fail immediately. Retries do not consume
`execution.max_replans`, but repeated requests may incur additional model charges.

Simulator receive waits must account for both budgets:

```text
(max_replans + 1) × [(max_retries + 1) × timeout_s + max_retries × retry_max_delay_s]
```

### Model inputs and images

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `json_mode` | `true` | `false` | Constrain JSON syntax, without guaranteeing field semantics | High |
| `history_frames` | `0` | `2` | Historical RGB frames attached to model input (0–8) | High |
| `image_max_side` | `640` | `null` | Maximum image side length (128–2048) | High |
| `image_quality` | `85` | `default` | JPEG quality (40–95) | Low |
| `use_depth` | `true` | `false` | Allow model access to depth | High |
| `send_raw_depth_image` | `false` | `true` | Attach depth as an image | High |
| `enable_raw_depth_query` | `true` | `false` | Expose `read_depth_pixels` for raw per-pixel depth in millimeters | High |
| `enable_depth_measurement` | `false` | `default` | Expose `measure_surface_points` for 3D surface measurements | High |

`enable_raw_depth_query` and `enable_depth_measurement` are **mutually exclusive**; enabling both
fails at startup. With `use_depth: true` and `send_raw_depth_image: false`, one of the tools must
be enabled. Otherwise the model has neither a depth image nor a query interface, and startup fails.

### Prompt trimming

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `compact_motion_context` | `true` | `false` | Trim joint, speed, and older event details in pure motion requests | Medium |
| `current_observation_only` | `true` | `false` | Send the current observation without historical images or previous poses | Medium |
| `concise_tracking_feedback` | `true` | `false` | Use compact tracking feedback in millimeters/radians | Medium |
| `decision_history_turns` | `6` | `0` | Number of recent issued decisions retained in the prompt (0–32) | Medium |
| `use_height_prior` | `false` | `default` | Include static table/base height priors | Medium |
| `height_calibration` | Unset | `null` | Height calibration, used only when enabled | Medium |

### Hardware semantics

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `use_grasp_geometry` | `true` | `false` | Supply fingertip/gripper geometry; requires preset calibration | High |
| `use_gripper_center` | `true` | Inherited | Convert between wrist EEF and nominal gripper center | High |
| `rgb_camera_calibration` | Three cameras | `null` | Intrinsics and mounting calibration for model pixel-to-3D calculations | High |

### Reasoning and diagnostics

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `reasoning_effort` | `high` | `null` | Reasoning effort | High |
| `reasoning_summary` | `auto` | `null` | Record public reasoning summaries returned by the API; not model input | Low |
| `diagnostic_mode` | `true` | `false` | Request an evidence report and record the input inventory | Medium |

`diagnostic_mode` appends instructions asking the model to report modality usage, referenced input
items, object pixel/3D positions, and expected arm waypoint projections. These fields do not generate
actions. They consume output budget in exchange for evidence that can be checked turn by turn.
See [Diagnostics](../README.md#diagnostics) for usage.

### `rgb_camera_calibration`

All three cameras (`head`, `hand_left`, `hand_right`) are required, with an exact field set:

| Field | Purpose |
|---|---|
| `mount_frame` | Camera mount frame: `head_link3` for `head`, `arm_{l,r}_end_link` for hand cameras |
| `kinematics_urdf` | Required only for `head`; updates extrinsics from current head-chain joints |
| `intrinsics` | Full OpenCV pinhole parameters: `width`, `height`, `fx`, `fy`, `cx`, `cy`, `k1`–`k6`, `p1`, `p2` |
| `camera_to_mount` | 4×4 homogeneous transform; checked for orthogonality and right-handedness |

Hand-camera extrinsics update from measured wrist poses each frame; head extrinsics update from
the head/waist joint chain. Calibration is specific to the robot and mounting and must be redone
when either changes.

## `motion` and `motion_backend`

`motion` defines timing parameters shared by motion backends. `motion_backend` determines how
targets become executable trajectories.

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `motion.speed_fraction` | `1.0` | `default` | Speed scale in (0,1], used only when speed limits are enabled | Medium |
| `motion.min_duration_s` | `0.0` | `default` | Minimum arm motion duration | Medium |
| `motion.min_gripper_duration_s` | `0.0` | `default` | Minimum gripper motion duration | Medium |
| `motion.target_hold_s` | `0.0` | `default` | Additional hold after reaching the target | Medium |
| `motion.profile` | Unset | `quintic` | Interpolation profile: `quintic` or `smooth_cruise` | Medium |
| `motion_backend.type` | `urdf_joint` | `local_interpolation` | Local numerical URDF IK followed by joint interpolation | High |
| `motion_backend.path` | Torso URDF | Required | Torso-chain URDF | High |
| `motion_backend.subtree_path` / `subtree_root` | `G2_NO_GRIPPER.urdf` / `arm_base_link` | — | **Required together** to replace the arm chain | High |
| `motion_backend.frame` | `base_link` | Required | IK reference frame | High |
| `motion_backend.left_link` / `right_link` | `arm_{l,r}_end_link` | Required | IK end-effector links | High |
| `motion_backend.waist_yaw_joint` | `idx05_body_joint5` | `null` | Enable waist yaw; must name a configured body joint | High |
| `motion_backend.accept_translation_m` / `accept_rotation_rad` | `0.01` / `0.02` | `null` | **Required together**; accepted residual bounds when IK stalls (≤0.01 m / ≤0.05 rad) | Medium |
| `motion_backend.waist_settle_s` | `1.5` | `0.0` | Extra hold after waist motion to allow torso settling (0–10 seconds) | Medium |

## `simulator_codec`

| Field | Current value | Purpose | Impact |
|---|---|---|---|
| `type` | `corobot_ik_frame` | Convert EEF actions from `base_link` to `arm_base_link` | High |
| `path` | `G2_NO_GRIPPER.urdf` | FK kinematics; must match `subtree_path` | High |
| `left_link` / `right_link` | `arm_{l,r}_end_link` | FK end-effector links | High |

The plugin options must be exactly `path`, `left_link`, and `right_link`. The codec fixes `frame`
internally to `arm_base_link`; do not include it in configuration. Each frame, it compares FK from
measured joints with simulator-reported EEF poses to derive the transform. Left/right disagreement
above 1 mm or 0.005 rad raises an error. EEF conversion is rejected when torso movement is requested.

The other codec is `corobot` (no conversion; the default, used by the offline demo).

## `embodiment`

| Field | Current value | Purpose | Impact |
|---|---|---|---|
| `preset` | `g2_omnipicker` | Load built-in TCP offsets, command polarity, and grasp geometry calibration | High |

The preset injects `tcp` (0.14845 m on both sides), `gripper_open_closed` (-0.785 / 0), and
`grasp_geometry_calibration`. Missing fields with `use_grasp_geometry: true` fail at startup,
as do command values outside `robot.gripper_bounds`.

## `server` and `logging`

| Field | Current value | Default | Purpose | Impact |
|---|---|---|---|---|
| `server.host` / `server.port` | `127.0.0.1` / `8999` | `default` | Bind address | None |
| `logging.format` | `text` | `json` | Terminal log format | Low |

## Cross-field constraints

| Constraint | Violation behavior |
|---|---|
| `enable_raw_depth_query` and `enable_depth_measurement` are mutually exclusive | Startup error |
| `use_depth` + `send_raw_depth_image: false` requires at least one depth tool | Startup error |
| `subtree_path` and `subtree_root` must both be present or both omitted | Startup error |
| `accept_translation_m` and `accept_rotation_rad` must be paired | Error |
| `waist_settle_s` requires `waist_yaw_joint` | Error |
| Embodiment `robot_type` must equal `robot.robot_type` | Startup error |
| Embodiment gripper commands must fall within `robot.gripper_bounds` | Startup error |
| `use_grasp_geometry: true` requires preset TCP and geometry calibration | Startup error |
| Execution prefix step limits must not exceed `robot.max_chunk_steps`; terminal holds have a separate contract | Startup error |
| `retry_backoff_s` ≤ `retry_max_delay_s` ≤ 60 | Startup error |
| `execution.need_depth` and `planner.use_depth` should agree | No error, but observation and input settings differ |

## Observation switches

The three switches are independent and supported by both `scripts/serve_llm.py` and
`agent2act serve/check/replay`. Omitted switches retain configuration values. Restart the service
after changes.

| Feature | CLI | Planner field | Behavior |
|---|---|---|---|
| Depth | `--depth` / `--no-depth` | `use_depth` | Also overrides `execution.need_depth`; when disabled, received depth is not sent to the model |
| Gripper center | `--gripper-center` / `--no-gripper-center` | `use_gripper_center` | Requires an existing `gripper_center_offset_m` when enabled |
| Height prior | `--height-prior` / `--no-height-prior` | `use_height_prior` | Requires `height_calibration`; disabling omits all height values |

`--height-calibration FILE.json` replaces calibration data only; it **does not enable the height
prior**. Enabling it with missing calibration, nonfinite values, or unsupported frames raises an
error rather than using another robot's constants.

```bash
# Disable all three: RGB and original wrist EEF
uv run --locked python scripts/serve_llm.py --config examples/g2_llm_recipe.json \
  --no-depth --no-gripper-center --no-height-prior

# Add static height calibration to the primary configuration
uv run --locked python scripts/serve_llm.py --config examples/g2_llm_recipe.json \
  --height-prior --height-calibration path/to/heights.json
```

Calibration may also be embedded in `planner.height_calibration`:

```json
{
  "frame": "base_link",
  "table_surface_world_z_m": 0.730005,
  "base_link_world_z_m": 0.0,
  "source": "Static asset calibration; see annotations for applicable scenes and file versions"
}
```

Base height means the **world Z coordinate of the `base_link` origin**, not the chassis top,
waist, or total robot height. This requires a fixed base, parallel upward world +Z and base_link +Z,
and forward/left/up `motion_frame_axes`. The model receives
`table_surface_z_m = table_surface_world_z_m - base_link_world_z_m`. Changing the table, scene,
or base pose requires recalibration. The prior contains table/base heights only, without finger-pad
geometry, grasp recipes, or scoring hints; it does not use object ground truth to select grasp points.

## Motion behavior

### Five-candidate IK

With `ranked_motion_candidates` enabled, motion for `status=continue` must contain only `frame`
and `candidates`, with exactly `ranked_motion_candidate_count` entries. Each candidate uses the
same waypoint fields as a single target. IK and full action validation run in model-ranked order,
and the first passing candidate is selected. Checking does not execute actions; the selected
candidate reuses the validated chunk.

```json
{
  "mode": "motion",
  "instruction": "Move toward the next target",
  "reason": "Five alternatives ordered by task suitability",
  "motion": {
    "frame": "base_link",
    "candidates": [
      {"right_pose": [0.40, -0.20, 0.80, 0, 0, 0, 1]},
      {"right_pose": [0.39, -0.20, 0.80, 0, 0, 0, 1]},
      {"right_pose": [0.38, -0.20, 0.80, 0, 0, 0, 1]},
      {"right_pose": [0.37, -0.20, 0.80, 0, 0, 0, 1]},
      {"right_pose": [0.36, -0.20, 0.80, 0, 0, 0, 1]}
    ]
  },
  "success_condition": "Inspect the next observation",
  "status": "continue"
}
```

Numbers illustrate the format only. If all candidates fail, the error includes each rejection
reason and enters the `max_replans` budget. The `ranked-motion-checks` trace records candidate
order, individual results, and the selected index. `complete`/`abandon` still use `motion=null`
and do not require candidates.

### Confirmation waits

When the model judges a local operation complete and wants to hold for environment confirmation,
it can provide a one-window `duration_s` while omitting arm, gripper, and waist targets. For example,
a single `{"duration_s": 1.0}` waypoint can be used at 30 Hz with `execute_steps=10`.
A hold is not a task-completion declaration. Native checks may run only every few steps;
a full window avoids repeated short holds and model calls. For two windows, request 2 seconds
and set `execution.max_motion_continuations=1`.

### Waist control

`motion_backend.waist_yaw_joint` identifies a joint in `robot.body_joints`. When enabled, waypoints
may contain `waist_yaw`: an absolute angle in radians, positive counterclockwise viewed from above,
subject to `robot.body_limits`.

- Omitting an arm pose holds that arm's joint angles, so it turns with the torso, useful when
  carrying an object during a waist turn.
- Providing a pose solves IK for the `base_link` target at the waypoint's new waist angle;
  the resulting path is not a straight line.
- Decisions without `waist_yaw` retain the backend's existing interpolation behavior.

`execution.body_execute_steps` extends the issued prefix when any body joint differs from its
measured value, preventing `execute_steps` from splitting a waist turn across multiple LLM calls.
`waist_settle_s` is included in this step budget. Model input includes
`context.execution.waist_control` and `states.waist_joint_states`, and the system prompt adds a
`WAIST CONTROL` section. Removing the joint setting disables waist control and rejects `waist_yaw` actions.

### Fixed-action diagnostics

`planner.type="motion_probe"` is a diagnostic planner that makes no model calls. It compares issued
commands and measured responses for the same target at different durations, using the initial
observation as a baseline. Each trial executes the target, holds, returns to baseline, and holds
again; measurements are recorded when the next request arrives. At completion it prints
`motion_probe_finished` and holds the baseline until manually stopped, without fabricating
`episode_done` or task success. `scripts/analyze_motion_probe.py` exports CSV comparisons of
requested duration, measured change, and residuals. It does not identify objects, grasp them,
or judge benchmark success.

## Gripper presets

Reuse a preset when changing tasks with the same gripper; explicitly select the corresponding
preset when changing grippers. Robot configuration still owns `robot_type`, URDF, motion backends,
and joint limits.

```json
{
  "embodiment": {"preset": "g2_omnipicker"},
  "planner": {"type": "embodied", "use_gripper_center": true, "use_grasp_geometry": true}
}
```

The sole data source for built-in `g2_omnipicker` is
[`g2_omnipicker.json`](../src/agent2act/adapters/embodiments/data/g2_omnipicker.json).
It ships in the wheel/sdist and does not depend on the working directory after installation.
Use `"embodiment": {"path": ...}` for a custom file; relative paths resolve against the configuration
file's directory.

Preset references cannot coexist with inline `gripper_center_offset_m`, `gripper_open_closed`,
`grasp_geometry_calibration`, or `pose_reference`. These fields are replaced completely; missing
fields never inherit values from a previous gripper. Definitions use `schema_version: 2` and contain
name, `revision`, `robot_type`, `source`, `tcp`, open/close commands, and calibration data.
`tcp.axes` currently supports only `wrist_parallel`; additional rotations raise an error.
Left/right commands can be described separately, but this does not imply arbitrary asymmetric
gripper support.

Validation rules:

- An embodiment requires an explicit `use_gripper_center`; loading offsets does not choose the
  experiment's enablement setting.
- `use_grasp_geometry=true` requires TCP enabled and complete calibration.
- `use_grasp_geometry=false` omits geometry and its instructions; TCP conversion may remain enabled.
- `use_gripper_center=false` omits TCP instructions and requires grasp geometry to be disabled.
- Startup checks preset compatibility with `robot_type`, gripper bounds, and numeric structure.
  Presets do not rewrite URDFs, limits, action budgets, or scoring.

After loading, `AppSettings.embodiment` stores preset contents, provenance, canonical JSON SHA-256,
and the description-generator version. Reloading does not require the original file. Hash mismatch
or independently modified hardware data raises an error. With tracing enabled, each session writes
hardware descriptions and a snapshot to `<trace_dir>/<session_id>/hardware.json`, without credentials;
this file is not model input.

## Camera calibration

Agent2Act accepts camera calibration in CoRobot observations. Depth must be enabled on the service,
for example with `--depth`, and the simulator must supply both depth and calibration (through the
container overlay on the GenieSim side). Both sides are required.

When depth is enabled, the configured input path supplies RGB, depth summaries or query access,
and calibration text. Depth images are not attached when `send_raw_depth_image=false`.
CoRobot RGB entries carry `calibration`; RGB and depth entries carry `snapshot_id` to associate
images and calibration from the same capture. A `snapshot_id` is a capture identifier, not a
simulation timestamp or render frame number.

- `fx/fy/cx/cy` are intrinsics at the original sensor resolution, accompanied by `width/height`.
- Distortion order is `k1,k2,p1,p2,k3,k4,k5,k6`; pinhole and OpenCV models are supported.
- `optical_to_frame` is a 4×4 column-vector transform: `p_base = T @ [x,y,z,1]`, targeting `base_link`.
- Optical axes are x right, y down, z forward. Depth is optical-z `distance_to_image_plane`,
  encoded as uint16 millimeters, not Euclidean range.
- `rgb_encoded_size` / `encoded_to_source_pixel` describe image resizing. Intrinsics retain
  the original resolution and must not be applied directly to thumbnails.

Extrinsics use `cameraViewTransform` from the same render product. Wrist extrinsics are read for
each capture rather than reused across frames. Unsupported lenses, missing data, resolution
mismatches, nonrigid transforms, or stale render poses return `available=false` with a reason.
Image noise, occlusion, or dropped-frame augmentation also conservatively disables measurement calibration.

Measurement tools are disabled by default: `use_depth` controls depth access but does not
automatically expose function calls. Depth grids contain regional medians, not point depths for
specific objects. Tools measure surfaces and do not guarantee grasp success.

## Gripper center

```json
"gripper_center_offset_m": {"left": [0, 0, 0.14845], "right": [0, 0, 0.14845]}
```

Both sides require finite 3D local offsets in meters. Model-facing `end_pose` and output
`left_pose` / `right_pose` represent a fixed nominal gripper center. `motion_frame` still selects
the reference frame, TCP axes remain parallel to wrist axes, and orientations use xyzw quaternions.
Observations use the current wrist rotation: `p_center = p_wrist + R * offset`. Commands invert
using the target's own rotation: `p_wrist_target = p_center_target - R_target * offset`.
The model does not add or subtract offsets itself. Omitting one side preserves its existing hold semantics.

Current observations, historical states/endpoints, tracking residuals, and measured displacements
all use center semantics. Rotation can move the center even when the wrist position is unchanged.
Raw `Observation`, runtime memory, CoRobot, and camera extrinsics retain wrist semantics.
The `gripper-center-execution-decision` trace stores converted wrist decisions separately from
raw model output. This interface currently supports pure motion only; proposal review and mixed
policy routing are rejected before requests. It cannot coexist with the legacy `wrist_to_center_m`
conversion tool.

Values come from the runtime USD's fixed `center_link`, not a live finger-pad contact center,
and do not apply to other uncalibrated robots.
