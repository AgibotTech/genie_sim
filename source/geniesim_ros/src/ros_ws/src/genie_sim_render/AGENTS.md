# genie_sim_render

Isaac Sim render node — publishes camera images from the USD stage over ROS 2.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_render/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE)

**Maintenance contract**: when you add a plugin, change the manifest schema
consumed here, or alter topic conventions, update this file in the same diff.

---

## Layout

```
genie_sim_render/
├── scripts/
│   └── isaacsim_render.py         ← Python Isaac Sim render node (render_isaacsim)
├── src/
│   ├── render_node.cpp            ← C++ OVRtx render node (render_ovrtx)
│   ├── main.cpp                   ← plugin registration
│   └── plugins/
│       └── ros_image_publisher_plugin.cpp ← RosImagePublisherPlugin
├── include/                       ← C++ headers
├── launch/
│   └── render.launch.py           ← standalone render launch
├── config/
│   └── cam_x4.perspective         ← default RViz2 camera perspective
└── plugins.xml                    ← pluginlib plugin description
```

---

## Two render backends

### `render_ovrtx` — C++ OVRtx node (`render_node.cpp`)

The production render backend. Runs as a separate process from the physics
node. Reads `manifest.json` (same schema as the physics node), opens the USD
stage, and publishes images via `RosImagePublisherPlugin`.

Manifest asset paths are loaded directly as ordinary USD files. Missing scene
or robot files fail validation; the renderer does not search for alternate files.

**First-run warmup.** OVRTX compiles shaders + JITs kernels on the first
`ovrtx_wait_op`, which can block for several minutes with blank images
(the observed G2 cold run took 224 s). `render_timer_callback` prints a
startup banner, then `OVRTX first frame complete — renderer is live.`
when the wait returns. Topics exist before the first frame, so topic
discovery alone does not prove delivery. Cached runs usually start faster.
The inline OVRTX visualizer has its own warmup progress reporting.

Activated by listing `render_ovrtx` in the launcher YAML `renders:` block.
Constructed by `utils.py:make_render_ovrtx_node()` in the bringup package.

Key subscriptions / publications:
- Subscribes `/tf_scene` — writes each incoming absolute prim path to
  its OVRTX prim. Under v2 wire protocol the payload is scene-body-only
  (robot links ride `/tf` via RSP). ``child_frame_id`` must be an
  absolute USD prim path (starts with ``/``); bare body names are
  rejected with a warn-once.
- Subscribes `/tf` + `/tf_static` — the tf2 buffer composes the URDF
  chain from RSP; every render tick, ``apply_robot_tf`` iterates
  ``manifest.robot["body_paths"]``, looks up each link's world pose
  from the buffer, and writes it to the OVRTX prim with
  ``omni:resetXformStack=1``.
- Subscribes `~/free_cam_pose` — moves the free-fly camera
- Publishes `<topic>/image_raw` + `<topic>/camera_info` per manifest camera

### `render_isaacsim` — Python Isaac Sim node (`isaacsim_render.py`)

Python replacement for the C++ OVRtx node. Uses the same `manifest.json`
schema and the same topic conventions. Activated by listing `render_isaacsim`
in the launcher YAML `renders:` block.

Constructed by `utils.py:make_render_isaacsim_node()` in the bringup package.

The tiled node (`scripts/isaacsim_tiled_render.py`) and fused renderer
(`scripts/_isaac_render_common.py::build_tiled_sensor_group`) use Isaac Sim
6.2's `isaacsim.sensors.experimental.rtx.TiledCameraSensor`. Both paths pass camera prim
paths, NumPy-order `(height, width)` resolution and `annotators=["rgb"]`;
they group cameras by height and scatter tiles into env-major SHM slots.
This import targets 6.2; tiled/fused rendering on 5.1/4.5 remains unverified.
`genie_sim_engine_isaacsim.kit.launch_config.prepare_launch_config` disables
6.2 multi-tick rendering before Kit starts so cameras can produce data after
stage replacement. Keep this helper shared with the converter and physics app.
The renderer declares the Isaac backend as a runtime dependency; the helper
itself has no SDK imports. Other SDK launch configs stay unchanged. Tiled/fused
GPU acceptance is separate from the single-world scene checks in
[the runtime verification record](../../../../docs/ISAACSIM_ACCEPTANCE.md).

`isaacsim_render.py` preserves each manifest camera's `depth_topic` and attaches
an additional ROS2CameraHelper of type `depth` to the same render product when
that topic is present. RGB, depth and CameraInfo use the manifest `frame_id`.
The Python renderer consumes `/tf_scene` world poses at absolute USD paths and
resets each prop's xform stack, matching the C++ v2 scene-transport contract. The bringup interpreter selection preserves system ROS
paths when Isaac runs in the launch interpreter; bundled runtimes still filter
incompatible native Python paths.

The single renderer authors `OmniLensDistortionOpenCvPinholeAPI` using manifest
fx/fy/cx/cy, distortion coefficients and image size. Its USD fallback frustum
uses separate horizontal/vertical aperture and principal-point offsets.
`ROS2CameraInfoHelper` forces fy=fx in 6.2, so the graph instead uses
`ROS2PublishCameraInfo` with the same manifest calibration and simulation time.
Do not fix only the published matrix: the live projection probe must verify an
off-centre camera with visibly unequal fx/fy as well as ROS CameraInfo.

---

## Manifest schema consumed

The fused node loads its instance-switch table from `manifest.scene_yaml`,
anchored at `base_path` (or the manifest directory), through
`scripts/_isaac_render_common.py::load_scene_config`. Only manifests without
that field use the sibling `rl_scene_cfg.json`; an explicitly missing
or malformed scene is an error, not a fallback to another task's table.

Instance-authored visual materials are stored under the scene config's
`instance_switch.table_materials`. Fused applies them inside `CMD_RESET`
before rendering its response; tiled merges them under reset randomization.
The shared `_instance_materials.py` verifies resolved USD bindings and leaves
purpose-specific PhysX material bindings untouched.

Both backends read the same `manifest.json` fields:

| Field | Purpose |
|---|---|
| `scene_usda` / `usd_path` | World USD to open |
| `robot_usda` | Robot USD to reference in |
| `robot_visual_usda` | Prebaked visual robot USD (`robot_source.prebaked` + `visual_only`). **`render_isaacsim` only**: when present it is attached for rendering *instead of* `robot_usda` (whose staged-URDF materials render flat white), and `robot.body_paths` is rebased onto its prim layout (`_remap_body_paths_to_visual`; the prebaked G2 asset is flat `/<robot_prefix>/<link>`). Unset → falls back to `robot_usda`. `render_ovrtx` (C++) ignores this field. |
| `render_layer_usda` | Render-product layer (cameras + RenderProducts) |
| `robot_prefix` | Prim namespace for the robot (`/<robot_prefix>`) |
| `free_cam_prim_path` | Path to the free-fly camera prim |
| `base_path` | Anchor for resolving relative paths |
| `cameras[]` | Per-camera: `topic`, `depth_topic`, `render_product_path`, `path`, `frame_id`, `width`, `height`, intrinsics |
| `robot.body_paths` | v2 wire protocol: URDF link name → composed OVRTX prim path. ``apply_robot_tf`` reads on every render tick. |
| `scene.body_paths` | v2 wire protocol: scene body name → absolute USD prim path. Populated for benchmark/logger consumers (renderer doesn't read; ``/tf_scene`` carries the prim path directly). |

---

## Plugin: `RosImagePublisherPlugin`

C++ pluginlib plugin (`genie_sim_render/RosImagePublisherPlugin`). Registered
in `plugins.xml`. Activated via the launcher YAML params block:

```yaml
render_ovrtx:
  ros__parameters:
    plugin:
      - genie_sim_render/RosImagePublisherPlugin
```

---

## Topic conventions

- RGB: `<camera_topic>/image_raw` (`sensor_msgs/Image`)
- Depth: `<camera_topic>/depth/image_raw` (`sensor_msgs/Image`, 32FC1 metres)
- Camera info: `<camera_topic>/camera_info` (`sensor_msgs/CameraInfo`)
- Free cam pose: `<node_ns>/free_cam_pose` (`geometry_msgs/PoseStamped`)
- TF scene: `/tf_scene` (`tf2_msgs/TFMessage`) — `child_frame_id` is either
  an absolute USD prim path or a backend body name resolved by the manifest
  compatibility map. Internal channel between `genie_sim_engine` and the
  render backends; not a substitute for `/tf` (which `robot_state_publisher`
  publishes with URDF-style frame names for general ROS consumers).

The default camera optical-frame TF contract is incomplete. See the
[README camera TF limitation](README.md#camera-tf-limitation) for current
backend behavior and spatial RGB-D acceptance requirements. An existing parent
link in `frame_id` does not establish the camera's calibrated optical frame.

---

## Routing rules

- OVRtx C++ render node → `src/render_node.cpp`
- Image publisher plugin → `src/plugins/ros_image_publisher_plugin.cpp`
- Python Isaac Sim render node → `scripts/isaacsim_render.py`
- Standalone launch → `launch/render.launch.py`
- Node construction (from bringup) → `genie_sim_bringup/launch/utils.py:make_render_ovrtx_node` / `make_render_isaacsim_node`

## Streaming timing diagnostics

`isaacsim_tiled_render.py` uses `_render_perf.py` for bounded-memory CPU timing
CSV output. `GENIESIM_RENDER_PERF_LOG=0` disables it. A successful reload closes
the old trace and opens a new unique file; shutdown also closes it. Timings cover
render API calls and host sensor copies, not separately synchronized GPU work.

## Fused timing and support

The fused node renders with zero Replicator timeline increment and uses
`sim.render()` for idle UI pumping. Only explicit physics steps (including reset
settling) advance simulation time. An invisible z=0 ground plane supplies the
same support as Newton's builder. Reset settling still evolves all environments;
partial reset only scopes the explicit teleport and material writes.

Fused validates `instance_switch.body_pose_version=1` and
`instance_body_poses` before starting PhysX. Rows are complete env-local
`[x,y,z,qw,qx,qy,qz]` body poses keyed by the full path below `World/Objects`.
Reset adds the env origin and writes only selected body indices and velocities;
it never reconstructs a body frame from a donor-instance locator delta.
`instances=-1` restores the currently selected instance. Locator-only
instance-switch artifacts must be rebuilt. See
`source/rlinf_geniesim/docs/INSTANCE_BODY_POSES.md` for the baking contract.

Superset slots an instance does not use (absent from its `name_alias`) are
parked away from the table with gravity off (`RigidPrim.disable_gravities`), so
they stay put instead of falling to the floor and rolling: at startup for the
build instance, and on every instance switch (`FusedPhysics.apply_instance`). A
slot coming back gets gravity again before its pose is written; a slot going
away is parked first, then loses gravity. Do not take them out of the
simulation (`disable_rigid_body_physics`): on GPU that broke the renderer's
start-up, the ray-tracing pipeline failed to build.

Background articulations (including `open_door`'s door and handle) have a separate
reset snapshot in `scripts/_fused_background_reset.py`. Fused captures PhysX root
poses, joint positions and drive targets immediately after `sim.reset()`, before
camera updates or stepping. Episode reset restores those values and clears root
and joint velocities and applied efforts only for the selected environments.
The views use PhysX's returned prim order. Background DOFs remain separate from
the robot action schema; a rebuilt stage captures a fresh snapshot.
Discovery covers both `env_i/World/Background` (room composed into the foreground,
as in the real open_door manifest) and `env_i/Background` (explicit/per-env room
references). Validate against `build_gridcloner_stage` using the actual manifest;
an isolated asset under only one of these roots does not cover both launch paths.
An articulation marker whose descendant rigid bodies are all explicitly disabled
in the composed stage is a static prop and is excluded from discovery (the
laboratory laptops in `pour_workpiece`). Other missing PhysX handles still fail.
The same views publish live positions on every STEP/RESET, including steps
without rendering. The fused schema's `scene_joint_paths` preserves PhysX DOF
order and the owning object's env-relative USD path; `FusedStateShm` carries
the separate `scene_joint_positions` array. `FusedIsaacBackend.get_scene_joint_states()`
exposes it to `ApiCoreShim`/`PushPull` for `open_door` scoring. The wire is
`FUSEDST3`; restart both the server and its Isaac subprocess after updating.

`scripts/_fused_point_instancers.py` expands dynamic rigid PointInstancers on the
runtime stage before PhysX starts. The installed GPU/Fabric path does not update
their visible instance poses. Expanded bodies retain prototype references and
use a separate tensor view for per-env reset; they do not enter the task object
schema. The view preserves instance scales and authored sleeping settings.

Fused honours reset seeds the way Newton does; both engines draw from
`genie_sim_engine_core.common.reset_randomization`. A reset with seed >= 0
applies `scene_cfg.randomization`: object xy offsets on top of the instance body
poses (every body of an object moves together; `exclude`, default `table`, never
moves), arm joint offsets added to the reset joint positions and drive targets,
and a robot base xy offset from the default base pose. Seed -1 is the verbatim
snapshot and puts the base back. When the instance has a layout
(`instance_switch.instance_layout`, recorded by the rest-state bake), object
offsets are constrained: no new overlaps, carried or touching objects move
together, objects on a surface stay on it. `randomization.joint_pd` per-env
multipliers scale arm, body and head gains once at startup, after
`apply_drive_gains`; a requested scale that cannot land raises.

`scripts/_render_randomize.py` applies render-side generalization (lights,
material, camera pose) and is shared by the tiled and fused nodes. The bridge side is
`rlinf_geniesim/render/bridges/_render_randomization.py`
(`RenderRandomizationMixin`, used by `IsaacTiledBackend` and
`FusedIsaacBackend`). Fused sends `randomize_cmd.json` before posting RESET and
checks `randomize_ack.json` after it; the node applies the command after the
joint reset and instance materials and before rendering the reset frame. Light
and camera baselines are cached once per stage. On a physics-alive stage
(fused), per-env background rooms are made scenery only after `stage.Load`
(`make_render_only`: de-instance, remove collider/rigid-body/articulation APIs,
disable joints), as background randomization only applies to decorative rooms.

`scripts/_fused_mimic_pose.py` synchronizes revolute PhysX mimic followers before
the initial joint teleport and reset-default snapshot. It resolves the authored
reference, gearing and angular offset; reset reuses that consistent snapshot.
Otherwise resetting only the master can swing a follower through nearby props
on the first physics tick (the left finger knocked over scoop_popcorn's bucket).

## Retained OVRtx build compatibility

CMake inspects the installed `ovrtx_config.h`: the newer SDK uses
`ovrtx_config_entry_motion_bvh(OVRTX_MOTION_BVH_ENABLE)` and the older SDK uses
`ovrtx_config_entry_enable_motion_bvh(true)`. Both enable motion BVH only when
lidar is active. Detect the API declaration, since OVRtx is independently
installed from Isaac. Both Python 3.10/Humble and Python 3.12/Jazzy builds are
required after changing this boundary.

`test/test_isaacsim_transport.py` is registered through ament pytest and covers
manifest depth/frame-ID wiring and nested scene world-pose updates without
starting Kit at module import. SDK-dependent USD checks skip where USD is absent.

## Frozen Newton snapshot worker

`isaacsim_snapshot_render.py` and `_isaac_snapshot_scene.py` receive trusted private Pipe snapshots, apply Newton geometry/body poses and render three independent cameras without a physics world or timeline advancement. Authored robot/background references retain materials while their physics is disabled. Snapshot sequence ACKs confirm image readback only. The snapshot worker applies the shared SDK-specific launch compatibility config before Kit starts, including legacy tick scheduling for explicit 6.2 captures. The owner closes worker, render products, temporary textures and video resources on failure/exit. `GENIE_SIM_RENDER_BUILD_OVRTX=OFF` permits Python-worker-only builds; ON preserves existing C++ targets. See [the worker protocol](docs/snapshot_renderer.md).

## Synchronized IPC snapshots and episode material plans

The single Python renderer's opt-in `sync_observation_topic` selects
`scripts/_sync_capture.py`: callbacks validate and enqueue immutable version-1
`xyz+wxyz` world-state snapshots, then the render loop applies and captures them
as one completed Replicator Writer batch. Live TF, free-camera and deformable
callbacks cannot overwrite the snapshot-owned stage. An empty topic retains the
existing live renderer and camera graph. `sync_render_on_demand=true` allows only
capture/warmup to perform Kit work; neither sync path advances physics.

`scripts/_material_generalization.py` consumes an explicit
`generalization.material.episode_plan` through the manifest's scene config.
It verifies the plan hash, authored visual targets and composed bindings, writes
an initialization receipt, and rechecks bindings for each completed synchronized
capture. Receipts associate snapshot/render identity and image hashes with the
plan. Physics-purpose bindings and original asset files are preserved.
See [the sync protocol](docs/SYNC_CAPTURE.md) for source/epoch/retry ownership,
world-point application, camera selection, completion metadata and failure cleanup.

For recorded libuipc sequences, the startup scene preloads each episode material
plan/receipt under `libuipc.episode_sequence`. A new correlated physical snapshot
selects its matching episode_index/reset epoch on the same render stage, applies
the plan before capturing RGB, and writes an independent receipt. Geometry and
physics-purpose bindings remain unchanged; stale/skipped episodes are rejected.

### Render-product readiness

New snapshot/synchronous camera products receive Kit update ticks before the
first blocking Replicator capture. On Isaac 6.2, requesting the synchronous
readback immediately after product creation can stall without a first image.
This initialization occurs only in the separate physics-free render stage;
Newton/IPC physics is paused independently and is never stepped by these ticks.
