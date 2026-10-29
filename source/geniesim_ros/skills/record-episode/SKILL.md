---
name: record-episode
description: >
  Capture a Genie Sim RT Engine episode while a scene runs — record
  the canonical ROS 2 topics (`/joint_states`, `/joint_command`,
  `/tf`, `/tf_static`, `/clock`, and selected scene, control and camera topics)
  with `ros2 bag`; check that the required streams reached the bag. Teleop and
  benchmark loops have their own per-episode output hooks.
  Trigger: When the user asks to "record an episode",
  "save the run", "dump rosbags", "capture trajectories", or wants
  to persist the world state during a `launch-scene` / teleop /
  benchmark run for later replay or training.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.1"
prerequisites:
  - geniesim_ros:launch-scene
inputs:
  - name: output_dir
    desc: Bag output directory
    required: false
    default: ./runs/<timestamp>
  - name: topics
    desc: Select required robot topics and optional scene, control and camera topics
    required: false
outputs:
  - desc: "`./runs/<dir>/` with `metadata.yaml` + chunked `.mcap` files; `ros2 bag play --clock` replays at sim-time"
---

## When to Use

- A scene is up (`launch-scene` skill) and the user wants to persist
  topic streams for later replay or dataset assembly.
- Teleop is running (`run-teleop` skill in `geniesim_teleop`) and the
  user wants ROS-side recording on top of the teleop loop's own
  per-episode artifacts.
- Benchmark run wants extra raw-topic capture beyond what
  `--benchmark.record=true` writes.

Do **not** use for:
- Replaying an already-recorded bag → just use `ros2 bag play`.
- Recording from a stopped sim — `ros2 bag` needs live publishers.

## Critical Patterns

1. **`use_sim_time:=true`** is set engine-side, so every bag captures
   `/clock` and replays cleanly at simulation rate. Don't override
   the topic with wall-clock.
2. **Always record `/clock`** in addition to your topic list — replay
   without it falls back to wall-clock pacing and timing breaks for
   downstream learners.
3. **Camera topics are heavy.** A G2 scene with three cameras at 30 Hz
   easily clocks 200+ MB/min. Either narrow the topic list or use
   `--max-bag-size` to roll over bags.
4. **One recording per run.** Restart the scene between recordings —
   the engine's `init_*` blocks run only at startup, and a fresh
   reset is the only way to guarantee a deterministic t0.
5. **`ros2 bag` is the canonical path.** There is no dedicated
   "recorder" distribution in the stack — the teleop and benchmark
   loops have their own per-episode writers (see below), and
   everything else goes through `ros2 bag`.
6. **Robot and scene transforms are separate.** `/tf` and `/tf_static` carry
   the robot tree. `/tf_scene` carries dynamic non-robot rigid bodies; a launcher
   using `/tf_render` publishes renderer body transforms on that channel instead.
   Select the active channel when scene-body motion is part of the dataset.

## What to record

| Topic | Why |
|---|---|
| `/clock` | Sim time — required for deterministic replay |
| `/tf`, `/tf_static` | Robot world pose and complete robot link tree |
| `/tf_scene` or `/tf_render` | Active scene-body transform channel selected by the launcher |
| `/joint_states` | Robot state from the engine |
| `/joint_states_scene` | Scene articulation state, when articulated objects are present |
| `/joint_command` | Whatever a teleop / planner pushed in |
| `/odom` | Mobile base (if applicable) |
| `/cmd_4ws` | Mobile-base commands when the platform uses 4WS control |
| `/genie_sim/teleport` | Discrete object teleport commands, when used |
| Configured IMU topic | Inertial observations, when enabled |
| Configured image and CameraInfo topics | RGB, depth, fisheye and calibration; select exact names from the ROS graph |
| `<deformable-prefix>/{info,mapping,state,rigid_state}` | Record all four channels together for deformable replay |

## Workflow

### Step 1 — Confirm the scene is up

```bash
ros2 topic list --show-types
ros2 topic hz /clock                       # stop with Ctrl-C after checking pacing
ros2 topic echo --once /joint_states
```

Confirm feedback contains the expected joints and advancing timestamps. Topic
discovery alone does not prove that messages are being delivered.

### Step 2 — Pick a topic list

In the recording shell, select the required robot streams and add the optional
streams needed by the episode. Use the table above to decide what is required
for that dataset:

```bash
REQUIRED_TOPICS=(
  /clock
  /tf /tf_static
  /joint_states
)

OPTIONAL_TOPICS=()
# Append only topics present in the active ROS graph, for example:
# OPTIONAL_TOPICS+=(/tf_scene /joint_states_scene /joint_command /odom /cmd_4ws)

# Discover exact camera/calibration and IMU names; do not guess `/camera/*`.
ros2 topic list --show-types | rg 'sensor_msgs/msg/(Image|CompressedImage|CameraInfo|Imu)'
CAMERA_TOPICS=()
# Add the selected Image/CameraInfo names to CAMERA_TOPICS before recording.

TOPICS=("${REQUIRED_TOPICS[@]}" "${OPTIONAL_TOPICS[@]}" "${CAMERA_TOPICS[@]}")
```

### Step 3 — Start the bag

```bash
# Inside the container, in the same shell as the topic selection above:
source devel/setup.bash
ros2 bag record \
  --output ./runs/$(date +%Y%m%d_%H%M%S)_episode \
  --max-bag-size 1073741824 \
  --storage mcap \
  "${TOPICS[@]}"
```

Stop with Ctrl-C. The bag dir contains `metadata.yaml` + chunked
`.mcap` files.

### Step 4 — Verify

```bash
ros2 bag info ./runs/<dir>
ros2 bag play ./runs/<dir> --clock          # replay sim-time
```

Before accepting the recording, check `ros2 bag info` for a nonzero message
count on every selected required stream, including scene, action and camera
topics the dataset depends on. Check `/tf_static` even when recording starts
after the robot publisher: its retained transforms must reach the bag. Confirm
the bag duration covers the action interval and images retain matching
CameraInfo. Missing optional streams must not be reported as captured data.

### Step 5 — (alternative) Let the loop record for you

For teleop:

```bash
geniesim teleop run --device_type=pico --record-dir ./runs
# writes per-episode artifacts under ./runs/<episode>/
```

For benchmark:

```bash
geniesim benchmark run <CONFIG> --infer-host=<IP>:<PORT> \
  --benchmark.record=true
# output_dir comes from the config; check the run banner
```

## Commands (copy-paste summary for the user)

```bash
# Inside the container, alongside a running rigid scene, after Step 2:
source devel/setup.bash
ros2 bag record \
  --output ./runs/$(date +%Y%m%d_%H%M%S)_demo \
  --storage mcap \
  "${TOPICS[@]}"

# Inspect, then replay:
ros2 bag info ./runs/<dir>
ros2 bag play ./runs/<dir> --clock
```

## Notes

- `--storage mcap` is preferred over the legacy `sqlite3` storage —
  faster random access, smaller files, and the standard for ROS 2
  Jazzy.
- If you plan to feed the bag to a dataset pipeline, also pin the
  `manifest.json` (under `assets/scenes/<scene>/`) alongside the
  bag — it records the exact USD + robot variant the bag was
  captured against.
- For dataset-scale recording, the `geniesim_generator` skills
  (`generate-scene`, `search-assets`) plus the benchmark
  `record=true` flag are the closer fit — they write
  episode-indexed artifacts, not raw bags.

## Resources

- **Teleop recording hooks**: [source/geniesim_teleop/skills/run-teleop/SKILL.md](../../../geniesim_teleop/skills/run-teleop/SKILL.md)
- **Benchmark recording flag**: [source/geniesim_benchmark/skills/run-benchmark/SKILL.md](../../../geniesim_benchmark/skills/run-benchmark/SKILL.md)
- **ros2 bag (Jazzy)**: https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html
