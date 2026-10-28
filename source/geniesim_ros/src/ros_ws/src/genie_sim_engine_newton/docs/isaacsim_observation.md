# Paused Isaac Sim observations with Newton physics

`launcher_newton_wuji_eval.yaml` defaults to
`physics_engine_visualizer: isaacsim`. Newton remains the sole physics owner;
its process stays Kit-free. A separately owned Isaac Sim 6 worker renders
the three mounted views and optional observer video from completed Newton
scene snapshots. The launcher keeps `renders: []`: this worker is owned by
the backend, rather than the independent TF-driven Isaac render node.

The original `scene_flat_g2_sp_chef_wuji` physics, initial pose, robot limits,
stock physics profile, `mjxpbd`, 60 Hz, 24 particle substeps, four particle
iterations and RTF 0.5 are unchanged. One-world lockstep still uses one tick
per requested target. Rendering adds no physics ticks or autonomous wok control.

## Runtime and configuration

Use normal bringup so the assembled scene's initial joint pose reaches the
engine. The evaluator attaches with `spawn=False`:

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_sp_chef_wuji \
  launcher_config:=launcher_newton_wuji_eval \
  headless:=false rviz:=false
```

The engine's **current Python interpreter** must have the actual Isaac Sim 6
runtime, RTX/Replicator, the repository-pinned Newton/Warp and ROS dependencies. The parent
starts the worker with `sys.executable`; it does not select another Python
using `omni_python`. In a container with both pip Isaac6 and `/isaac-sim` 5.1,
activate the intended environment before launch. The bringup argument remains
`headless:=false` to enable this backend observation path, while the worker
itself creates a headless `SimulationApp`. A usable RTX GPU is required.

All settings below are backend ROS parameters in the launcher YAML. Scene and
physics configuration still comes only from the launcher and scene YAML.

| Parameter | Meaning / shipped value |
|---|---|
| `physics_engine_visualizer` | `isaacsim` selects the owned snapshot worker; `newton` selects optional GL |
| `isaac_render_log_path` | Unused worker stdout/stderr file; shipped `wuji_isaac_render.log` resolves against the launch working directory. Use a new absolute path in a local launcher copy for each run; existing files are rejected |
| `isaac_render_timeout_s` | Positive finite per-capture transport/render bound, shipped 120 seconds; includes sending the snapshot and receiving the complete response |
| `isaac_render_startup_timeout_s` | Positive finite worker readiness bound, shipped 600 seconds to allow Kit initialization; the first sensor render then uses the capture timeout |
| `isaac_render_required_meshes_json` | JSON list of mesh names that must actually be logged in every snapshot; chef requires `["/chef_noodles","/chef_pepper","/chef_carrot","/chef_ham"]` |
| `newton_rgb_views_json` | Required nonempty mounted-view configuration; existing names and field meanings are reused |
| `newton_rgb_width`, `newton_rgb_height` | Positive output dimensions, shipped 640 × 480 |
| `newton_video_path` | Empty disables video; an unused absolute `.mp4` path enables observer recording |
| `newton_video_fps` | Positive finite simulation-clock sample/playback rate, no higher than physics Hz; shipped 30 fps |

For example, copy the complete dedicated launcher, retain all its physics and
camera settings, and set these artifact fields in that local copy:

```yaml
genie_sim_engine:
  ros__parameters:
    physics_engine_visualizer: isaacsim
    isaac_render_log_path: /absolute/new-run/isaac-render.log
    newton_video_path: /absolute/new-run/observer.mp4
    newton_video_fps: 30.0
```

Pass the full copy as `launcher_config:=/absolute/new-run/launcher.yaml`.
Existing log/video files are never overwritten. The worker script is installed
by `genie_sim_render`; rebuild/source the current ROS overlay after adding it.

## Complete geometry and synchronous capture

`visualizers/scene_snapshot.py` subclasses Newton's renderer-independent
`ViewerBase`. Its original `set_model/log_state` methods produce the real
primitive/mesh geometry, instance transforms, colors, materials and visibility;
then the engine calls the scene plugin's original `on_render`. For chef, this
retains the actual dynamic wok and handle, stove, smooth noodle/vegetable tubes
and ham surface without rebuilding the food or its physics.

The robot uses the authored visual USD selected by
`robot_source.prebaked.usda: robot/G2_crsB_wuji/robot.usda` with
`visual_only: true`. Assembly writes `manifest.robot_visual_usda` as a reference
to that source, while `robot_usda` remains the URDF-based physics asset. This
preserves the original MDL materials, material/subset bindings, face-varying UVs
and texture references instead of approximating them through Newton colors.
The source robot includes `configuration/tex/{bc,met,nor,rgh}.tga` for the hand
glove surfaces; its other black/white/orange materials also retain their
authored roughness and metallic values.

Every snapshot includes an `authored_robot` asset descriptor and `body_poses`
mapping for the manifest's 88 robot body names. Poses come directly from the
completed Newton `body_q` as `xyz + quaternion xyzw`. The worker references the
visual asset once, disables its physics/sensor payloads and drives the visual
link Xforms from those poses. The parent removes only the corresponding robot
rows from Newton instance batches, including mixed robot/scene batches. The
remaining scene shapes stay in the snapshot, with no duplicate robot.

Chef's procedural objects originally define RGB colors, not texture maps or
authored PBR materials: the wok is dark grey, its handle uses brown/steel
colors, the stove/ring are grey, and each food group has its own RGB. They have
no original wood, metal or food image maps to restore. Their geometry and
colors continue through the mesh snapshot. An added visual PBR response for
these objects is a separate appearance setting. Authored environment assets can
provide that appearance through the scene configuration below.

### Authored environment visuals

The scene YAML's `newton.authored_visuals` optionally references an environment
asset for the owned Isaac worker. It never enters Newton's scene import or
changes its collisions, bodies, particles or scoring. `asset_path` resolves
against the assembled manifest's `base_path` (the assets root); an absolute
local path also works. Missing files and malformed descriptors fail before a
render worker is started. There is no additional environment-variable route.

```yaml
newton:
  authored_visuals:
    asset_path: scene/your_background/background.usda
    root_prim: /World
    hide_prims: [/World/objects]
    use_asset_lights: true
    body_visuals:
      - body: chef_wok
        source_prim: /World/objects/benchmark_pot_013
        reference_pose: [0.9949, 0, 1.09, 0, 0, 0, 1]
    hide_shape_prefixes: [chef_stove_]
```

`root_prim` selects the authored subtree, while `hide_prims` contains absolute
source prim paths to hide in the static environment reference. The worker
disables physics and sensor APIs on every referenced asset. When
`use_asset_lights` is true it keeps the asset's authored lights in place of its
default studio lights. Asset-relative material, texture and HDR references
remain relative to the original USD layers.

Each `body_visuals` entry references one source subtree separately and follows
the named Newton body's completed `body_q`. `reference_pose` is the Newton
world pose (`xyz + quaternion xyzw`) at which the asset is already aligned;
the worker applies the world-space delta from that reference to the current
pose while preserving the source subtree's own authored transforms. This does
not attach, teleport or control the physical body. The `authored_scene`
descriptor remains constant for the worker session; every capture carries
fresh `visual_body_poses`, including repeated paused captures.

The parent removes all procedural shape-instance rows belonging to a driven
body. `hide_shape_prefixes` additionally removes shape rows whose leaf label
starts with a configured prefix; each prefix must match an existing shape.
This field is consumed only by `SceneSnapshot`, never sent to the worker or
applied to the physical model. Filtering follows actual shape IDs even when a
primitive batch mixes robot, wok, stove and other scene shapes. Mesh prototypes
and required deforming-food meshes remain available.

For the fried-noodle asset, hiding `/World/objects` removes its original pot
and 42 static food objects from the background reference. The separate pot
reference drives the rendered wok; the original 46 food groups continue to use
their live Newton FEM surfaces and colors. The static background supplies the
table, stove, plates, materials and lights. Its source physics is disabled only
in the visual worker, while the original procedural physics and contact-based
lift rules remain authoritative.

The first snapshot includes all mesh prototypes. Later captures may omit
unchanged meshes, while every capture includes the complete remaining scene
instance batches and fresh authored body poses. Hidden collider geometry stays hidden.
The first frame also respects chef's suppression of the native raw FEM
surface, preventing it from covering the smooth food meshes.

The required-mesh check is per capture, before delta suppression is considered:
an unchanged mesh still has to be produced by this invocation of `on_render`.
A prior cached mesh cannot satisfy the check. This catches chef render-hook
failures even when the plugin catches and logs its own exception. Missing food
is an observation error rather than a successful partial image batch.

`visualizers/isaacsim.py` owns one worker and a private trusted multiprocessing
Pipe. Kit stdout/stderr go to the log, not the protocol. A capture carries a
monotonic sequence, the complete instance table, mesh deltas and distinct
world-space camera poses. The worker applies the snapshot onto a visual USD
stage with no solver/physics APIs. It renders separate RenderProducts using
Isaac6 Replicator with `delta_time=0`, `pause_timeline=True` and
`wait_for_render=True`, then copies each RGB annotator to CPU.
For generic snapshot meshes, the worker supports vertex `uvs` as USD `st` and
a local texture path or `uint8` RGB/RGBA image array. Embedded arrays become
owned temporary PNGs, connected through `UsdUVTexture` to the mesh material.
The authored robot retains its full existing material graph directly; it does
not pass its MDL normal/roughness/metallic maps through this simpler RGB texture
adapter. See the worker protocol for the distinction.

The parent accepts only an `isaacsim` response with the exact outstanding
integer sequence, exact camera-name set and `uint8 H×W×3` arrays. A stale
sequence, incomplete batch, invalid pixels, worker error, disconnect or timeout
fails capture. Transport/protocol failures poison and close the renderer;
construction failure and normal shutdown also close the Pipe and reap the
owned worker. Private deserialization is for this trusted child only and must
never be exposed to external senders. Worker details are documented in
[snapshot_renderer.md](../../genie_sim_render/docs/snapshot_renderer.md).

## Pause, image freshness and camera mounts

At the pause boundary Newton completes GPU work and refreshes both host-mirror
slots. The backend publishes final joints/body state, captures the stopped
scene and waits for the worker sequence/readback response before publishing
the RGB batch and completing the pause callback. The worker never infers this
snapshot by combining independently arriving TF/deformable topics.

RGB topics remain `/wuji_eval/{head,hand_left,hand_right}/image_raw`, `rgb8`,
RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(1). One capture has one identical ROS
wall-clock stamp across the three images. The stamp is neither simulation time
nor a physics tick, and retained images are never relabeled as new captures.
The publisher uses the existing lifecycle node and does not own its context.

The evaluator first pauses and confirms `inactive`, then records a baseline
and requests another refresh of the already frozen state. It still waits for
fresh joint/body arrivals and a matching three-image stamp after that baseline:
DDS delivery can lag the service response. Repeated captures integrate no
physics; model inference occurs while Newton and `/clock` remain held.
The rendering sequence ACK confirms this image request, not action completion:
the shared action contract remains `arrival_v1`, without command-ID K-tick
commit acknowledgement.

The launcher retains explicit local eye/look-at/up/FOV mounts on `head_link3`,
`hand_l_base_link` and `hand_r_base_link`. The up direction preserves wrist
roll, and FOV is vertical, matching Newton's camera projection. These are
virtual evaluation views, not physical-camera calibration. Verify framing
against the model's training inputs in the intended GPU runtime.

## Observer video and verification boundary

Video uses the same snapshot worker with an observer camera derived from
`viewer_camera_json`; its RGB is passed to the existing ffmpeg recorder.
It samples **simulation time**: at 60 Hz and 30 fps, once every two physics
ticks. Inference pauses add no repeated wall-clock frames. Mounted sensor
captures never enter the recorder. The observer and sensors use the configured
RGB dimensions, while the observer has its own pose and RenderProduct.

`ffmpeg` with CPU `libx264` is required. Normal engine shutdown closes its input,
waits for MP4 finalization, closes the renderer and reaps owned processes.
Encoder diagnostics remain beside the MP4 as `.ffmpeg.log`; worker diagnostics
use `isaac_render_log_path`. Missing encoders, broken pipes and finalization
failures are errors. Rendering/encoding can add wall time without extra ticks.

CPU tests cover mesh/instance snapshots and ownership, per-frame required food,
matching sequence and RGB validation, worker timeout/cleanup, in-memory USD
updates, camera transforms and simulation-clock video sampling. They start no
Kit/GPU/model services. Real Newton 1.4/Warp 1.15 CPU geometry checks do not
validate RTX output. On 2026-10-20 a separate GPU smoke run used Isaac Sim
6.0.0-rc.59 and Newton 1.4/Warp 1.15 on an RTX 4090 D: all three actual images
were inspected, with head/right views covering the robot, wok and food; the real model loop completed 12 requests
and 600 steps. All 12 inference windows held clock and joint state unchanged
under independent 20 ms monitoring. The observer MP4 contained 304 real frames
at 640x480/30 fps (10.133333 seconds, including startup). This validates that
working point's render/control link with the earlier basic RGB/roughness/metallic
materials; it did not validate the authored-robot material/texture path described
above.

On 2026-10-21 the same GPU runtime was rerun with the authored visual asset:
88 driven links, 11 materials and 114 bound meshes. Actual model JPEGs and
observer frames showed the original black/white shell and textured gloves,
without a duplicate robot. The final run completed 12 inference requests and
600 commands; all recorded full joint targets matched the returned actions,
and all 12 monitored inference windows held clock/joints unchanged. The final
MP4 again contained 304 frames at 640x480/30 fps. The physics robot USD hash
was unchanged. Procedural chef objects retained their original RGB appearance.
This GPU run covers the authored USD/MDL texture-reference path; generic
snapshot PNG/UV handling is covered separately by CPU USD tests. Neither
record establishes wok-task success.

For the optional OpenGL implementation and its framebuffer-specific behavior,
see [gl_observation.md](gl_observation.md).
