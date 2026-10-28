# Optional paused Newton GL observations

The dedicated Wuji evaluation launcher now defaults to the separate
[Isaac Sim snapshot renderer](isaacsim_observation.md) for observations and
video. This document describes the optional `physics_engine_visualizer: newton`
path. The Wuji chef demo creates the wok and food inside its Newton plugin;
its smooth food surfaces are drawn by `on_render(viewer.log_mesh)`.
The GL path reads completed frames
through the public Newton **1.4.0** `ViewerGL.get_frame(render_ui=False)` API.
The upstream [versioned implementation](https://github.com/newton-physics/newton/blob/v1.4.0/newton/_src/viewer/viewer_gl.py)
defines the RGB readback and completed-frame boundary used here.
That API returns top-left RGB uint8 pixels; the adapter does not generate images
from state or duplicate a single viewport for multiple camera fields.

To select GL, copy the complete `launcher_newton_wuji_eval.yaml` to a local
file and change only `genie_sim_engine.ros__parameters.physics_engine_visualizer`
to `newton`. Launch the unchanged scene through normal bringup, preserving its
assembly and initial joint positions:

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_sp_chef_wuji \
  launcher_config:=/absolute/local-gl-launcher.yaml \
  headless:=false rviz:=false
```

The launcher preserves `mjxpbd`, the stock `physics_params.yaml`, 60 Hz physics,
24 particle substeps, four particle iterations and the original scene's robot,
limits, contact settings and food. It selects one-world lockstep with one tick
per command request. Existing engine command-arrival/latch limitations still
apply; this does not introduce a command-ID completion protocol.

## Cameras and configuration

These launcher ROS parameters belong to the Newton backend:

| Parameter | Meaning |
|---|---|
| `newton_rgb_views_json` | Nonempty JSON list enables capture; omitted/empty preserves the existing viewer behavior |
| `newton_rgb_width`, `newton_rgb_height` | Explicit positive framebuffer dimensions; the evaluation launcher uses 640 × 480 |

Each view specifies `name`, an absolute `topic` ending in `/image_raw`, `body`,
`eye`, `look_at`, `up` and `fov_degrees`. Eye and look-at points are metres in
the named body's frame; up is a direction in that frame. Camera translation,
forward direction and roll follow the measured Newton body transform. The
configured head and palm mounts are **virtual evaluation cameras**, not a claim
that their intrinsics/extrinsics reproduce a calibrated physical camera.
Validate their actual framing against the policy's training observations in a
GPU acceptance run before drawing policy-quality conclusions.

| View | Parent body | ROS topic |
|---|---|---|
| `head` | `head_link3` | `/wuji_eval/head/image_raw` |
| `hand_left` | `hand_l_base_link` | `/wuji_eval/hand_left/image_raw` |
| `hand_right` | `hand_r_base_link` | `/wuji_eval/hand_right/image_raw` |

All three are real separate renders of the same stopped state, including the
chef plugin's current geometry. The original UI camera is restored and rendered
after capture. The configured dimensions are enforced for the actual
offscreen render/readback, independently of the
native window size. Window-manager resize events cannot change a sensor's
resolution mid-capture. The current native window framebuffer, projection and
UI are restored afterwards; no image resampling is used.

## Pause and freshness

`pause()` runs at the loop's tick boundary. The Newton backend joins outstanding
GPU work, refreshes **both** slots of every joint/velocity/body/particle host
mirror, and joins the readback. With RGB configured, its backend-owned callback
then publishes the final joints/body state through the existing publish plan,
renders and reads all three mounted views, and publishes the complete image
batch. Only after this returns does the existing run loop advertise lifecycle
`inactive`. Any mirror, capture or publication failure propagates; it is not
logged and reported as a successful pause. No physics tick is integrated during
capture. Repeating a pause captures the same stopped state again.

Images are `sensor_msgs/Image`, `rgb8`, with one **identical ROS wall-clock
stamp** for the three-view batch and their configured view names as frame IDs.
The stamp is not simulation time or an engine tick. Publishers use RELIABLE,
TRANSIENT_LOCAL, KEEP_LAST(1), retaining the last real frame throughout a pause.
Image collectors should request matching durability to receive that retained
batch. The publisher owns only its publishers on the existing lifecycle node;
it does not shut down the shared rclpy context.

First pause and confirm `inactive`. Then record a wall-clock and state-sequence
baseline and request another pause/readback of that frozen state. After the
second refresh, still wait for fresh joint/body cache arrivals and all three
images with a matching stamp newer than the baseline. DDS delivery to a
subscriber can lag lifecycle acknowledgement. The retained previous batch must
never satisfy a fresh-observation request. Images update on explicit pause, not through a
timer that would relabel old pixels as new. Resume before issuing the next
action chunk; model/network latency then occurs while physics is held.

## Observer MP4 recording

Recording is off by default. In a **local copy** of the launcher, set an unused
output path and the playback/sample rate:

```yaml
genie_sim_engine:
  ros__parameters:
    newton_video_path: /absolute/run-artifacts/observer.mp4
    newton_video_fps: 30.0
```

Launch that local copy through `launcher_config:=/absolute/local-launcher.yaml`.
`newton_video_path` must end in `.mp4`; parent directories are created, existing
videos are never overwritten. `newton_video_fps` must be finite, positive and
no greater than `physics_hz`. An empty path disables all video readback and
encoding work. These are backend-only launcher parameters; they do not alter
the scene, solver, action wire or camera observations sent to the model.

The normal Newton GL observer view, including the live robot, wok and chef
plugin food surfaces, is rendered and sampled on **simulation time**. At the
shipped 60 Hz physics and 30 fps setting, one real observer frame is read every
two physics ticks. While inference holds physics paused, the recording adds no
repeated wall-clock frames. The video's duration is therefore simulated action
time rather than model/network latency. A scene reset starts the next video
segment without waiting for the old simulation timestamp to catch up.

The mounted head/palm sensor capture path never records frames, including its
final UI restoration render. The video reads only completed frames from the
ordinary observer render hook, so it cannot alternate into model camera views.
The observer keeps its normal scene-configured or operator-adjusted viewpoint;
sensor captures restore that viewpoint. Video shares the configured RGB render
dimensions (640 × 480 in the evaluation launcher), or the normal 1280 × 720
viewer default when RGB views are disabled. Each sample is really rendered at
that fixed size, while native window resizing remains available. The shared
Newton 1.4 FBO boundary isolates resize events and invalidates readback PBOs on
both size changes, then restores and redraws the native observer view. This can
add wall-clock render work, but integrates no extra physics tick.

The backend requires `ffmpeg` on PATH with the CPU `libx264` encoder. RGB pixels
are sent through a pipe to an owned encoder process; the output is H.264 in an
MP4 container, with encoder diagnostics in the adjacent `.ffmpeg.log`. Normal
shutdown and failure cleanup close/reap the encoder and finalize the file.
Missing encoders, broken pipes and unsuccessful finalization are errors. Video
readback/encoding can add wall time but do not integrate extra physics steps.

## Verification

```bash
python3 -m pytest \
  source/geniesim_ros/src/ros_ws/src/genie_sim_engine_newton/tests
```

CPU tests cover camera transforms, actual per-view render/readback ordering,
ownership, shared image stamps, construction-failure cleanup, mirror flushing
and failure-before-pause-acknowledgement. They do not validate OpenGL/CUDA interop,
ROS delivery, camera framing or a model's physical task success. Those require
the Newton/ROS/GPU environment and an actual three-view preview/closed-loop run.
Video tests additionally cover simulation-clock sampling, paused-frame exclusion,
observer/sensor routing and encoder cleanup. If ffmpeg/ffprobe are installed,
an optional CPU codec test checks a tiny synthetic MP4's rate and frame count;
its pixels are not a simulator recording or physical-task validation.
