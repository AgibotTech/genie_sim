# IPC client and wire protocol

`IPCClient` owns transport only. The caller assembles the scene, starts normal
engine/renderer bringup, and owns episode shutdown. This client starts no world
or subprocess and imports no backend SDK, model, checker or application. ROS
imports are lazy. The Newton `SimulationClient` retains its separate
[arrival protocol](DESIGN.md).

```python
from genie_sim_client import IPCClient, IPCClientConfig

config = IPCClientConfig.from_files(scene_yaml, launcher_yaml)
with IPCClient(config) as client:
    identity = client.identity
    observation = client.observe()
    positions = {name: observation["snapshot"]["joints"][name]
                 for name in identity["joint_names"]}
    completion = client.step(positions, duration_s=2 / identity["physics_hz"])
```

Scene YAML supplies distinct absolute `libuipc.action_sequence.topic` and
`libuipc.observation_snapshot.topic` prefixes, without a trailing slash. Launcher
YAML supplies `physics_hz`; enabled CPU barrier settings supply the rate and
substep count. The client validates these settings without overriding them.
`ready_timeout_s` (600 s) and `wait_timeout_s` (120 s) are transport budgets.
`event` receives owned plain request/reply packets for caller archiving.
`on_accepted` runs once for a correlated accepted action or a validated completed
reply. Calls are serialized; returned packets are immutable. `packet_dict()`
returns a mutable independent copy.

## Discovery and actions

All channels use `std_msgs/String` JSON. `<action prefix>/identity` uses RELIABLE,
TRANSIENT_LOCAL, depth-one QoS and contains version 1, `source_id`, `epoch`,
`tick`, `sim_time`, `physics_hz`, controlled `joint_names`,
`reported_joint_names`, `status` and `ready`. Reported names can include fixed
joints. Wait for ready identity before sending. Failure/closure, an unexpected source/epoch change, or changed joint layout
terminalizes an attached client. An explicit configured sequence reset is the
only allowed epoch transition.

Action `/request` contains exactly:

```json
{"request_id":"action-1","source_id":"engine-id","epoch":0,
 "names":["controlled_joint"],"positions":[0.25],"duration_s":0.2}
```

Every controlled joint must occur once, with finite positions. Duration must be
a positive integer number of IPC ticks. `/done` correlates request/source/epoch
and uses `accepted`, `completed`, `rejected` or `cancelled`. One action may be
pending/active. The engine freezes its full target vector; concurrent joint
messages cannot replace it. Per-epoch repeated IDs with identical payloads
return the cached reply; changed payloads conflict. The client exposes no
automatic retry or replay after an uncertain timeout.

Completion validates committed IPC count, executed simulation duration, final
tick/time and configured CPU action/total counts. A solve commits only after
native readback, post-step/archive work and snapshot servicing succeed. This
proves integration completion, not target reachability. Foreign/delayed replies
cannot complete the current request. Tick/time share the first latched epoch's
clock offset and YAML rate. CPU totals are independent monotonic counters and
may advance during uncommanded ticks: physics keeps stepping between actions.

## Snapshots and images

Snapshot `/request` contains `request_id`, `source_id` and `epoch`.
`/snapshot` returns version 1, a unique `snapshot_id`, matching tick/cloth tick,
`sim_time`, capture time, finite named joints, absolute authored-root body poses
and full named deformable point lists. `pose_schema` is `xyz+wxyz`; offsets and
quaternion conversion are already applied. Deformable points are world-space.
Generic rigid-only scenes may have an empty deformable list; task callers must
validate their own complete inventory.

The snapshot publisher/subscriber use RELIABLE, TRANSIENT_LOCAL, depth-one QoS
for delayed renderer discovery. Requests/actions remain volatile. Retained
state cannot execute a command or satisfy a request with a different identity.
Renderer `/observation` must embed that exact snapshot and full
source/epoch/request/snapshot/tick/time identity, all selected JPEGs (Wuji:
`head`, `hand_left`, `hand_right`), a completed render frame/reference time and
completion stamp. Partial/conflicting state or images fail the client.

`observe()` first discovers the renderer publisher within its capture timeout,
then requests and joins new state/images; discovery alone is not completion.
`snapshot()` returns physical state without requiring a renderer. See the
[synchronized capture contract](../genie_sim_render/docs/SYNC_CAPTURE.md).

## Recorded episode sequence

When the scene YAML supplies `libuipc.episode_sequence`, the client exposes
`begin_episode()`, `finish_episode()` and `reset(next_index)`. Each call sends one
JSON request on `<action prefix>/episode/request` and waits for a correlated
`/episode/done`. Requests contain version 1, request_id, source_id, old epoch,
operation (`begin`, `finish`, `reset`) and configured index. They carry no paths,
material plans or physics settings. Completion includes the committed ready
identity; reset requires exactly old epoch + 1 with the same source/joint names.
Only the next YAML index is allowed, after finish. Exact duplicate IDs return
cached replies without repeating native state recovery. There is no automatic
send retry, reset-after-failure, or replay.

Finish certifies a sealed physical archive and pauses the owner while scoring
runs. Reset restores native and CPU state, opens a new archive at tick/time zero,
and leaves it paused for correlated material capture. Begin resumes physics.
State snapshots contain `episode_index` so the renderer applies the matching
preloaded plan. Diagnostic identity fields engine_pid/cpu_worker_pid establish
process reuse; they are not credentials or control inputs.

After an uncertain action/capture/reset failure, close
the client and start a fresh caller-owned session. `close()` releases its node,
executor, thread and private context, preserves the host's global rclpy context,
and sends no process signals. [Wuji evaluation](../../../../../geniesim_benchmark/README.md#deformable-evaluation)
owns policy adaptation, processes, recording and scoring.
