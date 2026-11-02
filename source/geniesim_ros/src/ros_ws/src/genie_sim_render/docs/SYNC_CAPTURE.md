# Synchronized IPC observation capture

`isaacsim_render.py` can render one immutable IPC snapshot through the normal
manifest, calibrated OmniLens cameras and Kit launch configuration. Empty
`sync_observation_topic` preserves the ordinary live TF/deformable renderer,
CameraInfo, depth and frame IDs. Sync mode creates only requested Replicator RGB
products and disables the ordinary live camera graph and live state callbacks.

| ROS parameter | Default | Meaning |
| --- | --- | --- |
| `sync_observation_topic` | empty | Absolute prefix enabling snapshot ownership of the stage. |
| `sync_camera_topics` | `"{}"` | JSON map from response names to distinct manifest camera topics. |
| `sync_render_on_demand` | `false` | When true, idle work waits on ROS without Kit updates/rendering; capture owns Kit work. Otherwise idle uses `World.render()`. |

Neither sync path calls `World.step()`. The standard Wuji launcher selects all
three policy cameras and on-demand rendering. Physics remains independently
owned and continues while capture/inference waits.

## State identity and scene application

Snapshots arrive on `<prefix>/snapshot`; results use `/observation`; the physics
owner receives `/request`. Snapshot state uses RELIABLE, TRANSIENT_LOCAL,
depth-one QoS so a renderer finishing warmup can receive the latest frozen state.
Requests/actions remain volatile. The paired client owns one outstanding
observation; full identity checks prevent historical retained state from
satisfying another request. See the [wire protocol](../../genie_sim_client/IPC_CLIENT.md).

Version-1 snapshots contain `source_id`, `epoch`, `request_id`, `snapshot_id`,
`tick`, matching `cloth_tick`, `sim_time`, joints, bodies and deformables.
Body keys are absolute USD paths with world authored-root `xyz+wxyz` poses.
Offsets/quaternion conversion are already applied; the renderer resets xform
stacks and adds neither operation again. Deformables retain actual mesh IDs,
names, paths and full finite world points, including cords; their mesh xform
becomes identity while hierarchy/topology remain.

Robot/camera link resolution is restricted to the original/current robot
inventory. A prebaked visual must provide every required body as a mutable
Xformable; an incomplete visual logs missing bodies and falls back to assembled
`robot_usda` for the entire robot. This may change material appearance. A scene
prop with a matching basename retains its own absolute path.

Callbacks only validate and enqueue owned JSON. Capture drains prior GPU work,
validates the complete scene before writes, applies frozen state, and waits for
all selected cameras in one Writer callback. It neither spins ROS nor applies
live TF/particle updates during capture.

The renderer binds the first valid source for its lifetime; restart it when the
backend source changes. Increasing epochs cancel old pending work; old epochs
and foreign sources fail. Per-epoch request IDs retain their canonical payload
and serialized response. Identical retries reuse exact JPEG bytes; conflicting
payloads fail. Error replies preserve available correlation fields.

## GPU completion and cleanup

A batch requires owned uint8 RGB/RGBA buffers at each camera's configured
resolution, all selected products and a fresh rational `reference_time`.
`swhFrameNumber` is diagnostic because it may remain zero. Replies include local
`render_frame`, SDK frame number, reference time, completion stamp and JPEGs
from that batch. Physical tick/time remains separate from render reference time.

Isaac 6.2 submissions use zero timeline delta, with render-stage physics disabled
and World paused. On-demand idle waits on ROS and never calls Kit update/render;
even a paused update can submit competing GPU work. Capture/warmup retain their
GPU fences and Writer drains. Startup allows three warmup submissions within
300 seconds. Readiness requires a complete current-attempt callback; scheduled
reference intervals reject late prior batches. A pending Writer drains before
another snapshot is applied. Submission errors or missing schedule metadata
terminalize capture because late work may already be queued. Python budgets
cannot interrupt a blocking native SDK call.

Partial construction/shutdown attempts Writer detachment, render-product
destruction and ROS-handle cleanup even after an earlier failure. Protocol
validation alone does not establish real GPU image delivery or model outcomes.
GPU validation must inspect all three actual cameras in the prepared runtime.
