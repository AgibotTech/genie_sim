# OVRtx as Newton's in-process visualizer — zero-copy GPU sync, parallel render thread

> **Status**: implemented.
> **Code**: [`engine/newton/visualizers/ovrtx.py`](../genie_sim_engine_newton/engine/newton/visualizers/ovrtx.py)
> (`InlineOvrtxVisualizer`) and
> [`engine/newton/visualizers/ovrtx_kernels.py`](../genie_sim_engine_newton/engine/newton/visualizers/ovrtx_kernels.py).
> **Target OVRtx**: 0.4.x — the constructor raises `RuntimeError` on any other major/minor.
> **Sibling docs**: [engines.md](../../genie_sim_engine_core/docs/engines.md),
> [pipeline.md](../../genie_sim_engine_core/docs/pipeline.md),
> [perf.md](../../genie_sim_engine_core/docs/perf.md).
>
> This page describes the architecture — the threading contract, the CUDA
> stream handshake, and the attribute layout. For exact signatures
> (`__init__` parameters, `bind_newton_bodies`, `renderer.step`,
> `_publish_outputs`) read the two modules above; they are the
> ground truth and this page does not restate them line by line.

---

## Scope

The inline OVRtx visualizer lives inside `genie_sim_engine_newton` as the
`ovrtx` arm of `physics_engine_visualizer`. It is a separate path from the
cross-process renderer in `genie_sim_render`:

- The `genie_sim_render` render node (`genie_sim_render/src/render_node.cpp`)
  and its `RosImagePublisherPlugin` run as an independent ROS 2 process
  driven by `/tf_scene`. Nothing in `genie_sim_render/` is involved in the
  inline path; the two share code patterns, not compiled code.
- The two paths are mutually exclusive at runtime: a launcher either spawns
  the cross-process `render_ovrtx` node **or** sets
  `physics_engine_visualizer:=ovrtx` for inline OVRtx. The newton-standalone
  launchers (`launcher_newton_mjwarp.yaml`, `launcher_newton_fsvbd.yaml`)
  keep `launcher.renders: []`, so they are the home for the inline path.
- The cross-process renderer keeps its `/tf_scene`-driven workflow for
  render and physics in different processes — GPU isolation, separate
  failure domains, or one renderer driven by several physics sources.

---

## Context

In the newton-standalone path, body transforms never leave the physics
process — there is no separate-process renderer wired up
(`launcher_newton_mjwarp.yaml` has `launcher.renders: []`). The visualizer
surface is `none` (headless), `newton` (Newton GL OpenGL window), `ovrtx`
(this path), or `rerun` (a name reserved in `_VALID_VISUALIZERS`; the
dispatch warns and runs headless).

The cross-process node carries a different cost: body transforms travel
Newton GPU → physics CPU → `tf2_msgs::TFMessage` serialise → ROS middleware
→ deserialise → `ovrtx_set_xform_mat` (an inline wrapper around
`ovrtx_write_attribute` with `OVRTX_DATA_ACCESS_SYNC`, a CPU→GPU copy per
call). That cost is inherent to a cross-process design.

The inline path avoids it three ways:

1. **Zero CPU-side copy** of body transforms — a Warp kernel writes directly
   into OVRtx's internal Fabric buffer.
2. **No physics-thread blocking** — OVRtx runs on its own Python thread and
   CUDA stream.
3. **Single-process delivery** of camera images — published via
   `gsi::RosBridge` from the OVRtx thread.

The three properties that follow from the implementation:

- **Zero-copy GPU sync**: the Warp kernel writes into OVRtx's internal
  Fabric buffer through `binding.map(device=Device.CUDA)`.
- **Dedicated OVRtx render thread**: a Python `threading.Thread` owns the
  renderer, the per-frame sync kernel, the OVRtx step, and the ROS image
  publishes. The physics thread does no OVRtx work beyond recording one
  CUDA event per step.
- **Cross-stream CUDA handshake**: physics records `physics_step_event` on
  Newton's stream; the OVRtx thread's Warp stream waits on it before each
  frame's sync kernel. Physics never CPU-blocks; physics and OVRtx GPU work
  overlap.

---

## OVRtx conventions this design rests on

The OVRtx distribution ships skill docs under its own `skills/` directory;
they are the API source of truth for the patterns below.

| Convention | Value |
|---|---|
| Attribute name | `"omni:xform"` |
| Element type / shape | `float64`, `(4, 4)` |
| Memory layout | USD row-vector — translation in the last row, `m[3][0..2]` |
| Semantic | `OVRTX_SEMANTIC_XFORM_MAT4x4` |
| Zero-copy write path | `binding.map(device=Device.CUDA)` → Warp kernel → `unmap(stream=…)` |
| Render-output read path | `render_var.map(device=Device.CPU)` — ROS messages need CPU-resident buffers |

Warp's `wp.transform_to_matrix` produces the column-vector form, so
`sync_body_q_to_ovrtx_mat44d` transposes before storing.

The hot path is:

```python
with binding.map(device=Device.CUDA) as mapping:
    tensor = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
    wp.launch(kernel, dim=N, inputs=[tensor, ...], stream=stream)
    mapping.unmap(stream=stream.cuda_stream)
```

Zero copies: OVRtx's internal Fabric buffer is exposed as a DLPack tensor,
Warp wraps it without copying, the kernel writes into it directly, and
`unmap(stream=…)` signals OVRtx in stream order.

`unmap` is called **explicitly inside** the `with` block rather than left to
`__exit__`. The context manager's default `unmap()` takes no stream
argument, so the kernel could still be queued on `ovrtx_stream` when the C
unmap fires; passing the stream handle makes OVRtx serialise against the
kernel via CUDA stream order. On RTX hosts the `__exit__` path synchronises
with `cudaStreamLegacy`, which can fail or stall the render path.

The mapped tensor is valid only while the mapping is active. The kernel
launch is inside the `with` block; only the kernel's reference into the
buffer escapes, through Warp's stream-ordered queue.

---

## Selection

```bash
ros2 launch genie_sim_bringup app.launch.py \
    launcher_config:=launcher_newton_mjwarp \
    headless:=false \
    physics_engine_visualizer:=ovrtx
```

Selection rules:

- `headless:=true` (the launcher default — argv lacks `--gui`): no
  visualizer runs regardless of `physics_engine_visualizer`. The
  module-level guard in `genie_sim_engine_newton_node.py` downgrades to
  `none`.
- `headless:=false`: `physics_engine_visualizer` selects exactly one of
  `none` | `newton` | `ovrtx` | `rerun`. They are mutually exclusive, and
  the parameter is single-valued, so no fan-out logic exists.
- OVRtx is kitless and has no GUI window; it is treated as a non-headless
  visualizer so the single `headless` rule covers every visualizer the same
  way.
- No Kit involvement: newton-standalone never imports `omni.*` and never
  constructs a `SimulationApp`.

Per frame, the `ovrtx` arm: waits on the physics event → `binding.map(CUDA)`
→ `wp.launch(sync kernel)` → `mapping.unmap(stream=…)` → free-cam pose/zoom write
→ cloth points sync → `renderer.step(...)` → publish `sensor_msgs::Image`
from this thread via `gsi::RosBridge`.

`/tf_scene` publishes on the physics thread regardless of visualizer mode —
a debug/record channel (RViz TF view, `ros2 bag record /tf_scene`). Inline
OVRtx does not consume it; it reads `body_q` from Newton state through the
CUDA event handshake.

Latest-state-wins: if OVRtx falls behind, intermediate physics ticks are
skipped silently and physics never waits. With
`physics_engine_visualizer:=none` (or `headless:=true`) no OVRtx object is
constructed at all.

---

## Architecture

```
┌────────────────────────────── physics process ─────────────────────────────────┐
│                                                                                │
│  ┌────────────────────── Physics thread (main) ──────────────────────┐         │
│  │  EngineRunLoop.spin()                                             │         │
│  │   ├─ Newton step (wp.capture_launch) ─► writes state.body_q (GPU) │         │
│  │   ├─ wp.record_event(physics_step_event)  ◄── the only per-tick   │         │
│  │   │                                          OVRtx-related call   │         │
│  │   ├─ tick_extras (Fabric write — only if Kit viewport active)     │         │
│  │   ├─ publish_tick (clock, joints, odom, /tf_scene)                │         │
│  │   └─ sleep to next tick                                           │         │
│  └───────────────────────────────────────────────────────────────────┘         │
│                            │ (CUDA event handshake — non-blocking on CPU)      │
│                            ▼                                                   │
│  ┌──────────────────── OVRtx thread (threading.Thread) ─────────────┐          │
│  │  while not shutdown:                                             │          │
│  │   ├─ wait next render tick (timer at target FPS)                 │          │
│  │   ├─ ovrtx_stream.wait_event(physics_step_event)                 │          │
│  │   ├─ with binding.map(device=Device.CUDA) as mapping:            │          │
│  │   │     ovrtx_xforms = wp.from_dlpack(mapping.tensor,            │          │
│  │   │                                    dtype=wp.mat44d)          │          │
│  │   │     wp.launch(sync_body_q_to_ovrtx_mat44d,                   │          │
│  │   │               dim=N,                                         │          │
│  │   │               inputs=[ovrtx_xforms, body_indices, body_q],   │          │
│  │   │               stream=ovrtx_stream)                           │          │
│  │   │     mapping.unmap(stream=ovrtx_stream.cuda_stream)           │          │
│  │   ├─ free-cam pose write (when the scene has a FreeCam)          │          │
│  │   ├─ cloth points sync (when the scene has cloth)                │          │
│  │   ├─ authored Mesh reconstruction (scene-plugin FEM bindings)    │          │
│  │   ├─ outputs = renderer.step(render_products, delta_time)        │          │
│  │   └─ for cam in cameras:                                         │          │
│  │         with render_var.map(device=Device.CPU) as img:           │          │
│  │             _core.publish_camera_image_*(sim_time, topic, …)     │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Synchronization primitives

| Primitive | Producer | Consumer | Purpose |
|---|---|---|---|
| `physics_step_event` (`wp.Event`) | Physics thread after each step (`engine_base.py`) | OVRtx thread before each sync kernel | GPU-side cross-stream handshake — the sync kernel sees a committed `body_q`. Non-blocking on CPU. |
| `binding.map(...).tensor` (DLPack view into OVRtx's Fabric buffer) | OVRtx-internal | The Warp kernel | Zero-copy write target. Lifetime is the `with` block. |
| `ovrtx_stream` (`wp.Stream`) | OVRtx thread | — | Dedicated CUDA stream so OVRtx work doesn't serialise behind physics on Warp's default stream |
| `shutdown` (`threading.Event`) | `stop()` | OVRtx thread | Clean teardown; also the idle wait so Ctrl-C breaks out without burning CPU |
| `last_sim_time` (`std::atomic<double>` in `RosBridge`) | Physics thread (set in `publish_clock`) | OVRtx thread (read per frame) | Timestamp images without a mutex |

**No CPU mutex on `body_q`.** Physics is the only writer; OVRtx waits on
`physics_step_event`, so its kernel observes a fully-committed `body_q`.

**No backpressure on physics.** OVRtx renders the latest `body_q` whenever
it gets to it.

**Stream parallelism.** Physics kernels run on Newton's default stream; the
OVRtx sync kernel and render run on `ovrtx_stream`. Every `wp.launch` on the
OVRtx thread passes `stream=self._ovrtx_stream` explicitly — without it Warp
falls back to the default stream and re-serialises with physics, silently
losing the parallelism the design is built around.

---

## Components

### `InlineOvrtxVisualizer` (`engine/newton/visualizers/ovrtx.py`)

Owns the renderer, the persistent `omni:xform` binding, the render thread,
and the per-frame map/launch/unmap cycle. Lifecycle:

1. `__init__` — version guard, `ovrtx.Renderer` construction, USD layer
   loading, camera parsing.
2. `bind_newton_bodies(model)` — persistent rigid-transform, cloth-point, and
   scene-plugin deformable-Mesh bindings. A persistent binding pays prim
   resolution once and is reused every frame.
3. `attach_physics_event(event)` — receives the `wp.Event` the physics
   thread records after each step.
4. `start()` — spawns the render thread.
5. `_render_one_frame()` — per frame, on the render thread.
6. `stop()` — joins the thread, unbinds, destroys the renderer.

`RendererConfig` is constructed with `enable_spg=False`. OVRtx 0.4 enables
Sensor Processing Graphs by default; the inline visualizer publishes
geometric `DistanceToImagePlaneSD` depth, and SPG's depth-sensor model
clamps the usable range to roughly [0.3, 2.5] m. The standalone C++
renderer disables it for the same reason.

`ovrtx_root`, when set, is forwarded through
`ovrtx.register_schema_paths(binary_package_root=…)` **before** the
`Renderer` is constructed — it is not a `RendererConfig` field.

**Subscriber-aware skip.** When the scene has no FreeCam and no camera topic
has a subscriber, the whole frame is skipped — sync kernel, render, and
publishes all elided. When a FreeCam exists the frame renders
unconditionally, because the FreeCam *is* the viewport for this mode.

**First-frame banner.** The first `renderer.step()` compiles shaders and
JITs render kernels, typically 30–90 s on a cold shader cache. A one-shot
banner is logged before the thread blocks in the C extension; no in-process
heartbeat can tick during the compile because OVRtx's MDL/shader compile
holds the GIL.

**Cadence.** `next_tick += render_period`, and when the thread falls more
than one period behind it snaps forward rather than catching up. A slow
frame drops slots; it does not queue.

**Stats.** A 1 Hz log line reports frames/s, `skipped_no_subs`, `failed`,
and average ms/frame, and publishes the same numbers to module globals that
`engine/newton/stats.py` folds into the physics-thread 1 Hz line as
`ovrtx=NN.NHz`. The readout goes stale 3 s after the last update so a
transient stutter doesn't blank it.

**Failure logging.** The first frame failure is logged unconditionally so a
hung thread leaves a trace; later failures are counted and reported through
the 1 Hz stats line rather than logged per occurrence.

### The Warp kernels (`engine/newton/visualizers/ovrtx_kernels.py`)

- `sync_body_q_to_ovrtx_mat44d` — one thread per OVRtx-bound prim.
  `body_indices[i]` selects which Newton body feeds OVRtx slot `i`. Writes
  into the mapped tensor directly, with the transpose to row-vector form.
- `sync_particle_q_slice_to_points` — copies one cloth's contiguous
  `[start, end)` slice of `state.particle_q` into a per-cloth buffer that
  OVRtx's `bind_array_attribute("points", …)` consumes. No frame transform:
  the isaac_scene stage authors cloth prims at identity, so world-space
  points are correct in the prim's local frame.
- `reconstruct_tet_bound_points` — reconstructs authored-order Mesh vertices
  from four global Newton particle indices and barycentric weights per vertex.
  Scene-plugin meshes receive `omni:resetXformStack=True` plus an identity
  transform, so the world-space result is written directly to `points`.

### Scene-plugin deformable Meshes

A scene plugin may expose `get_deformable_render_bindings()` after
`on_build`. Each item identifies an exact `mesh_prim_path`, its contiguous
Newton particle range, and a fixed `tet_barycentric_v1` mapping in authored
vertex order. The engine node forwards these records to inline OVRtx.

At bind time the visualizer validates path uniqueness, array shapes, particle
bounds, and barycentric sums, then uploads the fixed maps once. Per frame it
launches one reconstruction kernel per Mesh and performs one asynchronous
OVRtx `points` write for the complete list. Material bindings and face topology
remain those of the source USDA; only point positions change. The resulting
world-space arrays are also the canonical input for the later NPZ writer.

### Engine hook (`engine/newton/engine_base.py`)

`attach_physics_event(event)` stores a `wp.Event`; the per-tick writeback
records it on Warp's current stream — the same stream the Newton CUDA graph
just launched on, so the OVRtx thread's stream-side wait sees a committed
`body_q`. Cost is microseconds, and it is the only OVRtx-related line on the
physics hot path. Passing `None` detaches.

### `RosBridge` surface (`genie_sim_engine_core/src/`)

The OVRtx thread publishes all render-derived sensor outputs. Camera RGB and
depth each get a `sensor_msgs::Image` and a `sensor_msgs::CameraInfo`. Lidar
is reserved in scene YAML (`lidars: []`) and is not wired into
newton-standalone.

`/clock`, `/joint_states`, and `/odom` continue at physics cadence on the
physics thread and are unaffected by the inline visualizer.

| Method | Purpose |
|---|---|
| `publish_camera_image_rgba8(sim_time, topic, frame_id, arr)` | One RGB frame; OVRtx hands back `(H, W, 4)` uint8 for `LdrColor` |
| `publish_camera_image_depth32f(sim_time, topic, frame_id, arr)` | One depth frame; OVRtx hands back `(H, W)` or `(H, W, 1)` float32 |
| `publish_camera_info(sim_time, topic, frame_id, h, w, K, P, R, D, model)` | Per-frame `CameraInfo`; intrinsics come from the manifest |
| `has_image_subscribers(topic) -> bool` | Subscriber-count gate for the frame-skip path |
| `last_sim_time() -> double` | Atomic read of the most recent physics-thread `publish_clock` time |
| `subscribe_camera_pose(topic)` | Free-cam pose subscription |
| `subscribe_camera_zoom(topic)` / `take_free_cam_zoom()` | Optional latest-value optical-zoom channel used only by inline OVRtx |

The pybind shims release the GIL around each C++ publish call, the same
idiom as the physics-thread publishes. `rclcpp::Publisher::publish` is
thread-safe, so no extra mutex is involved.

**Camera config source.** The manifest is the source of truth, not a
re-parse of scene YAML: `load_cameras_from_manifest` reads the `cameras`
array and `free_cam_prim_path` into `CameraCfg` records.

**Stamping.** Every output from a single OVRtx frame is stamped with one
`last_sim_time()` reading captured at the top of the frame, so RGB and depth
from the same frame share a stamp. That stamp can lag the most recent
physics commit by up to one render period (≈33 ms at 30 FPS).

**Free-cam pose.** RViz's `view_camera_pose_publisher_display` plugin
publishes to `/genie_sim_engine/viewer/camera_pose` by default; the pose is
applied every frame as a single `mat44d` write through the persistent
binding.

**Free-cam optical zoom.** `ovrtx_camera_zoom_topic` is empty by default.
`launcher_newton_mjxpbd.yaml` opts in on
`/genie_sim_engine/viewer/camera_zoom`, whose `std_msgs/Float64` value is a
0.25–8.0 focal-length multiplier. The render thread writes
`focalLength = 18 mm × zoom` and scales only the FreeCam's `fx`/`fy` in its
published `CameraInfo`; observation cameras are unchanged. Keeping zoom on a
separate channel preserves the existing pose contract for g2sim's external
`render_ovrtx` process.

The shipped `view_robot.rviz` enables `OVRtx Auto Zoom from Orbit`. The RViz
plugin observes changes to the active Orbit view controller's `Distance`
property (including mouse-wheel changes) and multiplies zoom by
`previous_distance / current_distance`. The first observed distance is only a
baseline, so startup never changes the authored 1.0 zoom.

---

## Failure modes & coupling

The threading split removes contention on the **CPU thread axis** — physics
never CPU-blocks on render finishing. Four shared resources remain:

| Resource | Shared? | Notes |
|---|---|---|
| GPU compute (SMs) | Yes — single device, two streams | CUDA scheduler fills idle SMs from either stream; truly parallel only with spare capacity |
| GPU memory bandwidth | Yes | Hardware-level; no software guard |
| Python GIL | Yes | Released during all C++ extension calls (Warp, OVRtx, rclcpp) |
| PCIe bandwidth | Yes | Mostly only OVRtx uses it (CPU image readback for the ROS publish) |
| CPU cores | No | Physics and OVRtx each have their own thread |
| `body_q` / mapped buffer | No race | CUDA event + stream-ordered unmap |

### Scenario A: physics runs slower than its target

Cause: MJWarp solver iterations grow on hard contact configurations;
`wp.capture_launch` takes longer.

Effect on render:

- The OVRtx thread's `ovrtx_stream.wait_event(...)` is a GPU-side wait
  enqueued onto the stream. The CPU returns immediately, so the Python loop
  is not blocked.
- The sync kernel cannot start until physics signals the event, so OVRtx's
  GPU pipeline throttles to the rate physics commits at.
- **Render rate is capped at the physics tick rate.** Physics dropping from
  200 Hz to 50 Hz leaves a 30 FPS render target unaffected; dropping below
  the render target takes render down with it.

Latency stays good: every rendered frame uses the most recently committed
`body_q`.

### Scenario B: render runs slower than its target

Cause: high resolution, many lights, complex materials — a 50 ms render
against a 33 ms budget.

Effect on render: the target FPS is missed. Frames don't queue; the loop
snaps to the next slot and drops the intermediate ones.

Effect on physics — where coupling does show up:

1. **GPU contention (dominant).** OVRtx render kernels and Newton step
   kernels share the device. Even on separate streams, CUDA schedules onto
   SMs by availability: if OVRtx takes 80% of SMs, Newton gets the rest and
   `wp.capture_launch` takes longer to drain. Symptom: `note_step_timing`
   reports growing tick wall time although physics CPU work didn't slow
   down. Invisible when the physics target has headroom; direct when
   physics is already GPU-saturated.
2. **GIL contention (small).** The OVRtx thread holds the GIL during Python
   orchestration (timer arithmetic, dict lookups, context-manager
   bookkeeping). The physics thread releases it during `wp.capture_launch`
   but holds it in Python-level publish helpers. Net: a few hundred
   microseconds of GIL waiting per tick — negligible at 200 Hz, possibly
   measurable at 1000 Hz.
3. **PCIe contention (small).** `render_var.map(device=Device.CPU)` does a
   GPU→CPU image copy per frame. Physics's CPU↔GPU traffic is the CUDA
   event signal and a small joint-state fetch. Unlikely to matter below very
   high camera resolutions.

Joint saturation, worked: physics 200 Hz × 5 ms/tick = 1000 ms/s of GPU
work; render 30 FPS × 20 ms/frame = 600 ms/s. Total demand 1600 ms/s against
a 1000 ms/s budget → both run at ~62% of target. Fully serialised they would
be worse, but parallel streams do not conjure capacity that isn't there.

### Symptoms by failure mode

| Failure | `note_step_timing` | `/clock` rate | camera image rate | Mitigation |
|---|---|---|---|---|
| Physics-bound | tick wall time ↑ | drops below target | drops, capped at physics rate | Reduce contact complexity, raise solver tolerance, drop physics Hz |
| Render-bound (GPU) | tick wall time ↑ slightly | drops slightly | below `render_hz` | Lower render resolution, lower `render_hz`, simpler materials |
| Render-bound (CPU/Python) | mostly unchanged | mostly unchanged | drops | Profile the OVRtx thread, check ROS topic backlog |
| Both saturated | ↑ | drops | below target | Split GPUs, lower targets |

### What the design guarantees

1. **The physics CPU thread never blocks on render finishing.** OVRtx work
   is on a different thread and a different CUDA stream — no `.join()`, no
   `wait()`, no `wp.synchronize_device()` on the physics path.
2. **Render always uses fresh physics state**, via the CUDA event handshake.
3. **No CPU-side mutex.** Physics never waits for OVRtx to release a buffer.
4. **Failure is graceful.** Slow render drops frames rather than queuing, so
   OVRtx does not accumulate latency.

### What it does not guarantee

1. **Independent throughput.** One GPU is shared; saturating one side
   throttles the other.
2. **Hard real-time physics.** GPU contention from OVRtx adds variance to
   tick wall time. Hard real-time needs OVRtx on a separate GPU.
3. **Render at exactly `render_hz`.** Latest-state-wins skips slow frames
   rather than queuing them.

### Mitigations

1. **Throttle render**: a lower `render_hz` cuts GPU time roughly
   proportionally.
2. **Smaller render output**: lower-resolution cameras in the scene YAML.
3. **Subscriber-aware rendering**: already in `_render_one_frame` — with no
   FreeCam and no subscriber on any camera topic, the entire frame is
   skipped, saving GPU, CPU, and PCIe cost.
4. **Realtime scheduling for physics**: `realtime_scheduler.cpp` pins the
   physics thread and sets SCHED_FIFO. The OVRtx thread runs at default
   priority so the kernel preempts it for physics when CPU is the
   bottleneck. Raising the OVRtx thread's priority defeats this.
5. **Multi-GPU split**: physics on GPU 0, OVRtx on GPU 1 removes GPU
   contention. Newton supports per-device placement and OVRtx accepts a
   device id at renderer creation.

For typical workloads (hundreds of bodies, 200 Hz physics, 30 FPS 720p
render, one RTX 4090-class GPU) there is enough GPU headroom that scenarios
A and B are rare, and the failure is one-sided throughput drop rather than
instability.

---

## Verification

1. **Version guard**: the constructor raises unless
   `ovrtx.__version__` is 0.4.x. Installed-header vs installed-wheel skew is
   the most common setup problem.

2. **Smoke test (kit-free newton + inline OVRtx)**:
   ```bash
   ros2 launch genie_sim_bringup app.launch.py \
       scene:=<scene> \
       launcher_config:=launcher_newton_mjwarp \
       headless:=false \
       physics_engine_visualizer:=ovrtx
   ```
   Confirm: no `omni.*` modules in the process; no `render_ovrtx` node in
   `ros2 node list`; camera image topics publish at the configured
   `render_hz`; physics ticks at target Hz (`note_step_timing` in
   `common/loop.py`).

3. **Physics throughput is unchanged**:
   - Baseline A: `physics_engine_visualizer:=none` (or `headless:=true`).
     Mean physics tick wall time over 60 s.
   - Test B: `physics_engine_visualizer:=ovrtx`, `headless:=false`, 30 Hz
     render.
   - Pass: B within 5% of A. A larger slowdown means physics is blocking on
     OVRtx — most likely a stream-scoping mistake (a `wp.launch` on the
     OVRtx thread without `stream=`).

4. **Headless invariant**: with `headless:=true` and
   `physics_engine_visualizer:=ovrtx`, the module-level guard downgrades to
   `none` and no renderer is constructed.

5. **Matrix layout**: place one body at translation `(5, 0, 0)`; after one
   step and one render, read back `omni:xform` for that prim and confirm
   `5.0` sits at `[3, 0]` (USD row-vector). At `[0, 3]` instead, the
   transpose in the kernel is wrong.

6. **CUDA stream parallelism** via `nsys profile -t cuda,nvtx`: Newton-graph
   kernels on one stream, OVRtx sync kernel and render on `ovrtx_stream`,
   with visible overlap on the timeline.

7. **Correctness vs the cross-process path**: SSIM > 0.99 on the same
   scripted trajectory at `t=5.0 s`.

8. **Latency**: measure physics-step-done → first-pixel-published. Inline
   beats the cross-process path by the ROS serialise + middleware +
   deserialise cost (0.5–2 ms typical).

9. **Stability**: a 60+ min run at 1000+ bodies. The 1 Hz stats line shows
   `failed=0`, no Warp graph-capture failures, and no unbounded memory
   growth.

10. **Shutdown**: Ctrl-C stops the OVRtx thread through `ovrtx_viz.stop()`
    after `session.run()` returns; the thread joins within 5 s; `nvidia-smi`
    shows no zombie context afterwards.

---

## Out of scope

- **`genie_sim_render/` edits**: the cross-process renderer operates
  independently. Code patterns are borrowed; files are not modified or
  linked in. The two paths are mutually exclusive at runtime.
- **Kit**: this entry point is newton-standalone. The Kit viewport is
  `physics_engine:=isaac_newton`, which runs
  `genie_sim_engine_isaacsim_node.py`.
- **Kit Fabric writes**: newton-standalone is Kit-free, so there is no USDRT
  stage to write into. OVRtx's mapping is into its own internal Fabric — a
  different system from Kit's USDRT Fabric.
- **Multi-visualizer fan-out**: exactly one of
  `none | newton | ovrtx | rerun` runs, enforced by the single-valued
  `physics_engine_visualizer` parameter.
- **Sensor types other than camera RGB / depth**: lidar, IMU, tactile,
  force/torque, and GPS are not wired into newton-standalone (`lidars: []`
  is reserved schema). GPU-derived sensors fit the OVRtx-thread model;
  pure-physics-derived sensors fit the physics-thread model.
- **Shared OVRtx code between the inline and cross-process paths**: the two
  re-implement the small overlap (renderer init, image serialisation)
  independently.
- **PhysX**: PhysX does not expose body transforms as Warp arrays.
- **OVRtx ≤ 0.3.x**: 0.4.x only.
- **Vulkan interop / `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`**: the ROS publish
  path uses `render_var.map(device=Device.CPU)`.
