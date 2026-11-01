# genie_sim_render

GPU rendering for the GenieSim simulator. Produces photoreal camera
images via OVRtx and publishes them on ROS 2 image topics.

## Two render modes

| Mode | Where it runs | When you'd pick it |
|---|---|---|
| **Inline OVRtx** | Inside the Newton-standalone physics process (`InlineOvrtxVisualizer` in `genie_sim_engine_newton`) | Newton-standalone runs (`launcher_newton_*`). Zero-copy `body_q → omni:xform` via Warp; no IPC. |
| **Standalone OVRtx node** | Separate ROS process (`render_ovrtx`, this package) | Isaac Sim runs (`launcher_ovrtx_isaac_*`). Physics and rendering get independent CPU/GPU budgets. |

Both modes consume the **same `manifest.json`** schema (cameras,
RenderProducts, robot prefix, free-cam prim path), so swapping
launcher_configs swaps render mode without scene-yaml edits.

A Python alternative — `render_isaacsim` — is also packaged for
Isaac-Sim-native rendering. Both `render_ovrtx` and `render_isaacsim`
publish the same topic shape.

For paused Newton evaluation, an independent Isaac6 snapshot worker accepts
the real Newton viewer geometry through a private Pipe and returns rendered
RGB arrays for independent sensor/observer cameras. An authored visual robot
reference replaces its Newton instance rows, retaining original MDL materials,
UVs and texture maps while fresh Newton body poses drive its links. Procedural
wok/stove/food geometry retains its source RGB; those objects originally have
no image textures. The worker does not run physics. See [snapshot renderer](docs/snapshot_renderer.md)
for its launch, protocol and cleanup contract.

Build only the Isaac Python workers without requiring an OVRTX SDK:

```bash
colcon build --packages-select genie_sim_render --cmake-args -DGENIE_SIM_RENDER_BUILD_OVRTX=OFF
```

The option defaults to `ON`, retaining the existing OVRTX C++ node/plugin
build. With `OFF`, Python entry points and docs are installed, and snapshot
pytest can still run when `BUILD_TESTING` is enabled.

## Topics

- `<camera_topic>/image_raw` — RGB (`sensor_msgs/Image`)
- `<camera_topic>/depth/image_raw` — depth in metres (32FC1)
- `<camera_topic>/camera_info` — intrinsics (`sensor_msgs/CameraInfo`)
- `<node_ns>/free_cam_pose` — interactive free-camera pose
  (`geometry_msgs/PoseStamped`)
- `/tf_scene` — per-body render transforms; absolute prim paths are used
  directly, while MuJoCo body names resolve through the manifest map

## Camera TF limitation

**The default camera interface does not establish a valid optical-frame TF
chain for RGB-D conversion into robot or world coordinates.** Image reception
and image-only visual inputs remain usable when the configured view is valid;
they do not establish spatial RGB-D acceptance.

The standalone `render_ovrtx` node writes the camera USD path into the RGB,
depth and CameraInfo `header.frame_id`. That path is not published as a camera
frame in the robot TF tree, so camera-to-robot/world tf2 queries fail. The
node's existing TF broadcasters publish lidar transforms, not camera mounts.
`/tf_scene` is an internal render transport, not a substitute for camera TF.

`render_isaacsim` uses the manifest `frame_id`, but that alone does not provide
a calibrated camera TF. Some scenes name the parent robot link rather than
a camera optical frame. Simply replacing a USD path with that parent link
would misrepresent the camera's position and axes.

Spatial RGB-D acceptance requires all of the following:

- RGB, depth and CameraInfo identify the same camera optical frame.
- That frame connects to the actual parent robot link with the calibrated
  mount transform, including the conversion from USD camera axes (looks -Z,
  +Y up) to ROS optical axes (+Z forward, +Y down). A fixed mount follows its
  parent through the robot's existing TF chain.
- tf2 resolves the optical frame into robot and world coordinates at image
  timestamps, including during head and chassis motion.
- A known object's RGB-D reconstruction agrees with its robot/world pose;
  a successful lookup alone does not validate calibration.

FR3/UR5 identity-extrinsic viewpoints are a separate configuration limitation:
valid TF cannot make a monochrome RGB image or all-zero depth a usable view.

## First-run warmup

OVRtx compiles shaders on first run; this can take 30–90 s with no
output. Both render paths print a 1 Hz heartbeat
("first frame: still compiling shaders / warming GPU (Ns elapsed)")
so operators know warmup vs hang. Subsequent runs hit the cache and
start instantly.

## When you'd touch this package

- Adding a new render plugin (`pluginlib` — see
  `RosImagePublisherPlugin`).
- Changing topic conventions or `manifest.json` schema.
- Modifying the `/tf_scene` protocol or free-cam pose handling.
- Working on the C++ OVRtx node (`src/render_node.cpp`) or the Python
  Isaac Sim node (`scripts/isaacsim_render.py`).

## Mechanics

See [AGENTS.md](AGENTS.md) for the manifest schema, plugin
registration, `/tf_scene` protocol details, and routing rules.
