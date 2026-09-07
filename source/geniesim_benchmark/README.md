# geniesim_benchmark — Benchmark tasks, scoring, LLM eval 🧪

Declarative task configs + a runtime that loads a scene, drives a
robot, evaluates a policy, and records scores. The canonical entry is
the `geniesim benchmark` CLI verb (owned by
[`geniesim_cli`](../geniesim_cli/)).

License: [Mozilla Public License Version 2.0](LICENSE)
Agent doc: see [`../../.agent/geniesim_benchmark.md`](../../.agent/geniesim_benchmark.md)
Skills: [`skills/`](skills/)

---

## 📦 Install

```bash
pip install -e source/geniesim_benchmark/
```

Pulled in automatically by `geniesim bootstrap`. Heavy runtime deps
(Isaac Sim, MuJoCo, open3d, …) come from this package.

---

## 🛠️ What you can do

### Run a task against an inference server

```bash
geniesim benchmark run g2op_if_pick_block_color \
  --infer-host=<IP>:8999
```

### Probe an inference server before sinking minutes into a sim launch

```bash
geniesim benchmark check-inference \
  --infer-host=<IP>:8999 --arch=corobot
```

### Discover tasks

```bash
geniesim benchmark categories         # show category counts
geniesim benchmark robots             # show robot counts
geniesim benchmark list --robot=g2op --category=if
```

### Batch-evaluate a sweep

```bash
geniesim benchmark batch --category=if --robot=g2op
```

### Convert collected datasets between formats

The benchmark stack ships dataset utilities under
`geniesim_benchmark.dataset.*`. The first converter goes from
**agibot v1 → LeRobot v2.1** (parquet + HEVC/PNG-encoded MP4s):

```bash
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir ./agibot \
  --output-dir ./lerobot_out
```

The `--agibot-dir` argument accepts either a single-episode dir
(contains `aligned_joints.h5` directly) or a parent dir of multiple
episode subdirs — auto-detected at runtime. Pass
`--lerobot-ref-dir <path>` to fill missing fisheye / head_back
extrinsic columns from a reference dataset; omit it to leave those
columns empty. Requires **ffmpeg on `PATH`** (RGB → HEVC, depth → PNG).

---

## 🔌 Inference interface (`corobot` policy)

The benchmark talks to your inference server over a WebSocket, one msgpack
request per action chunk. Two halves: the **observation** it sends and the
**control** envelope it expects back.

Authoritative source:
[`benchmark/policy/corobotpolicy.py`](src/geniesim_benchmark/benchmark/policy/corobotpolicy.py)
— `get_payload()` builds the request, `_parse_result()` / `_post_process_action()`
consume the response.

### Observation — what the benchmark sends

```jsonc
{
  "method": "infer",
  "params": {
    "timestamps": { "head": <ns int>, "states": <ns int> },
    "images": {                       // JPEG bytes; decode BGR then convert to RGB
      "head":       { "encoding": "JPEG", "image_data": <bytes>, "height": <int>, "width": <int> },
      "hand_left":  { ... },
      "hand_right": { ... },
      "head_depth":       { "encoding": "RAW_UINT16", "image_data": <bytes>, "height": <int>, "width": <int> }, // optional, see below
      "hand_left_depth":  { ... },                                                                              // optional
      "hand_right_depth": { ... }                                                                               // optional
    },
    "states": {
      "head_joint_states":  [ ... ],  // 0 dims on G2_omnipicker (`obs_extra_joints`)
      "arm_joint_states":   [ ...14 ],// left_arm(7) + right_arm(7)
      "waist_joint_states": [ ...5  ],
      "gripper_states":     [ ...2  ],// [left, right]
      "end_pose":           { ... }   // observed EEF pose, both frames — see below
    },
    "prompt":        "<natural-language task instruction>",
    "robot_type":    "G2_omnipicker",
    "task_name":     "<sub_task_name>",
    "episode_idx":   0,
    "episode_done":  false,
    "task_progress": [ { "name": "...", "score": 0, "status": "PENDING" } ],
    "history":       { "interval": <int>, "images": [ ... ] }   // optional, see below
  }
}
```

Joint dims follow the robot's entry in
[`utils/name_utils.py`](src/geniesim_benchmark/utils/name_utils.py) (`ROBOT_CONFIGS`);
the numbers above are `G2_omnipicker`.

#### `states.end_pose` — EEF observation in **both** frames

Forward kinematics gives the end-effector pose in `arm_base_link`; the runtime
also reframes it into `base_link` and ships **both at once**, so a server
trained in either frame reads the one it expects without any config:

```jsonc
"end_pose": {
  "base_link":     { "left_arm": { "position": [x,y,z], "orientation": [qx,qy,qz,qw] },
                     "right_arm": { ... } },
  "arm_base_link": { "left_arm": { ... }, "right_arm": { ... } }
}
```

Quaternions are **xyzw**. The `base_link` half is derived from the
`arm_base_link → base_link` transform, re-read from the scene on every
observation, so it stays correct as the torso/waist moves. The field is `null`
when FK or that transform is unavailable (e.g. robots without IK config) —
treat it as optional.

#### `params.history` — rolling past frames (server-driven, opt-in)

The policy can attach a buffer of past camera frames (all three: `head`,
`hand_left`, `hand_right`), letting temporal models see what happened while
the previous action chunk was replaying. Nothing to configure on the
benchmark side — **the server controls it per response** via a `history`
block in `result`:

```jsonc
"history": {
  "interval":   5,                  // required to enable capture; > 0 = capture stride
  "last_n":     10,                 // optional; keep only the N most-recently captured frames
  "resolution": {                   // optional; per-camera resize for history frames only
    "head":       [320, 240],
    "hand_left":  [160, 120],
    "hand_right": [160, 120]
  }
}
```

- **`interval`** (int) `> 0` enables capture and sets the sampling stride (a
  frame every N chunk-replay steps); `0` / omitted disables it. For backward
  compatibility, a top-level `hist_frame_interval` in `result` is still read
  when `history.interval` is absent or `0`.
- **`last_n`** (int, optional) — when attaching the buffer to the next
  request, send only the most recently captured `last_n` frames instead of
  everything captured since the last inference. Omitted / `0` sends every
  captured frame (previous behavior).
- **`resolution`** (dict, optional) — per-camera `[width, height]` to resize
  **history** frames to before sending, keyed by payload camera name (`head`,
  `hand_left`, `hand_right`); a camera not listed (or the whole field omitted)
  keeps the default: `head` at native resolution, `hand_left` / `hand_right`
  downscaled to half size (the previous behavior). This only affects frames in
  `params.history.images` — the live per-inference `params.images` are always
  sent at native resolution, unaffected by this field. Resizing is
  downscale-only: any size larger than a camera's native resolution is clamped
  to the source, so a bad value can't inflate the payload or the sim-side
  work.

The captured frames ride along on the *next* request as
`params.history = {"interval": N, "images": [...]}`. The first request of an
episode (before the server has opted in) carries no history.

A server that never returns any of these fields behaves exactly as before.

#### `images.*_depth` — depth maps (server-driven, opt-in)

The three depth cameras (`head_depth`, `hand_left_depth`, `hand_right_depth`,
paired with `head`/`hand_left`/`hand_right` respectively) are `RAW_UINT16`
distance-to-camera maps in **millimeters**, uniformly across all three
cameras, clipped to the `uint16` range.

- **The first request of every episode always includes depth**, so the
  server has a real frame to decide from.
- From then on, **the server controls it per response** — return a truthy
  `need_depth` in `result` to keep depth in the *next* request; omit it (or
  return `false`) and the next request drops all three `*_depth` keys.
- A server that never returns `need_depth` gets depth on the first request of
  each episode only, and none after that.
- `need_depth` should be a real boolean; a JSON-ish string (`"false"`, `"no"`,
  `"0"`) is also tolerated and parsed as `false`, but don't rely on that —
  send a native `true`/`false`.
- Turning depth off isn't just a smaller payload: the runtime skips the
  depth camera readback entirely on the sim side once it's off, so leaving
  it off when you don't need it also saves render/IO cost on the benchmark
  side.

### Control — what your server returns

```jsonc
{
  "result": {
    "left_arm":  { "kind": "JOINT_ABS", "values": [[ ...7 ], ...H], "base_link": "base_link" },
    "right_arm": { "kind": "JOINT_ABS", "values": [[ ...7 ], ...H], "base_link": "base_link" },
    "left_effector":  [[ ...1 ], ...H],
    "right_effector": [[ ...1 ], ...H],
    "waist":  { "kind": "JOINT_ABS", "values": [[ ...5 ], ...H] },  // optional
    "head":   [[ ... ], ...H],                                      // optional
    "history": { "interval": 0, "last_n": 0, "resolution": {} },    // optional, see below
    "hist_frame_interval": 0,                                       // optional, legacy alias for history.interval
    "need_depth": false                                             // optional; see images.*_depth above
  }
}
```

`H` is the action horizon; the sim buffers the whole chunk and asks again when
it drains. A top-level `{"error": "..."}` is treated as a fatal server error.

**`kind`** — `left_arm.kind` and `right_arm.kind` must match:

| `kind` | `values` per step | Meaning |
|---|---|---|
| `JOINT_ABS` | 7 joint positions | Absolute arm joint targets, applied directly |
| `EEF_ABS` | `[x, y, z, qx, qy, qz, qw]` | Absolute EEF pose, IK-solved sim-side |

**`base_link` — declaring the EEF control frame.** For `EEF_ABS`, this field
names the frame your poses are expressed in. It is read per-arm
(`left_arm.base_link`) with a top-level `result["base_link"]` fallback:

| Value | Behavior |
|---|---|
| `"arm_base_link"` | Poses go straight to IK (which solves in this frame) |
| `"base_link"` | Poses are transformed into `arm_base_link` first, using the live scene transform |
| *absent* | Defaults to `"base_link"` |

Any other value raises `ValueError`. Because the default is `base_link`, a
server emitting `arm_base_link` poses **must declare the frame explicitly** —
otherwise its poses get an unwanted transform applied. `JOINT_ABS` ignores the
field entirely.

**Optional `head` / `waist`.** Both are driven when you send them and simply
left uncontrolled when you don't — omit the key and those joints hold their
reset pose. To drive them, send a full `H`-step chunk (bare list or
`{"kind", "values"}`; `waist` additionally needs `kind` `JOINT_ABS` or
`ABS_JOINT`, both spellings accepted). Values are matched to the robot's
`head_joints` / `waist_joints` in order, clamped to whichever is shorter.

Announcing one of these keys without a usable payload — empty `values`, a
missing `values`, `null`, or fewer steps than the arm chunk — is **not** an
error: that group stays uncontrolled for the chunk and the runtime logs a
warning naming the key and the reason, so a half-wired action head shows up in
the log instead of killing the episode.

**Other caveats:**

- The response is unpacked with plain `msgpack` (`raw=False`), **not
  `msgpack_numpy`**. Convert every array to native Python lists (`.tolist()`)
  before packing, or `np.array(values)` breaks on ext-encoded garbage.
- `EEF_ABS` in `base_link` requires the scene transform to be available; if the
  runtime could not read it, the step raises rather than silently mis-solving.

---

## 🤖 Skills

| Skill | Purpose |
|---|---|
| [run-benchmark](skills/run-benchmark/SKILL.md) | Launch a benchmark task locally against a user-provided inference server |
| [check-inference](skills/check-inference/SKILL.md) | Probe a model inference WebSocket server and validate the response |

---

## 📂 Layout

```
src/geniesim_benchmark/
├── app/app.py            # runtime entry, called by `geniesim benchmark run`
├── config/               # *.yaml task configs (the work-list)
├── dataset/              # dataset utilities (format conversion, …)
│   └── convert/
│       └── agibot_to_lerobot.py   # public convert_agibot_to_lerobot() + convert_cli()
└── …
```

`config/*.yaml` is the source of truth for what's a benchmark task —
robot, scene, policy, scoring rule. The runtime is config-driven; new
tasks land as new yaml files, not new code.

`dataset/` is the home for off-line data utilities (format converters,
schema inspectors). Each converter exposes a plain-Python API plus a
`convert_cli(argv)` wrapper used by the `geniesim dataset convert …`
dispatcher — `argparse` only lives in the wrapper, the API is usable
from notebooks.

---

## 🔗 Pointers

- 🗺️ Module map: [`../README.md`](../README.md)
- 🏠 Repo root: [`../../README.md`](../../README.md)
- 🤖 Agent dispatcher: [`../../.agent/geniesim_benchmark.md`](../../.agent/geniesim_benchmark.md)
- 🏆 Leaderboard / public scores: [`../../README.md`](../../README.md) § Genie Sim Benchmark Leaderboard
