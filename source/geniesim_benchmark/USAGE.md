# Run the Genie Sim Benchmark locally

How to run a packaged benchmark task on your own machine against your own
inference server — the same tasks and scoring contract used by
[RoboColiseum](https://robocoliseum.ai/),
so a policy that passes locally behaves the same when scored remotely.

Run everything **inside the Genie Sim Docker container** (`geniesim docker up` → `geniesim docker into`),
where Isaac Sim / `omni_python` is already on the host.

## Driving tasks via SKILLs

Drive any packaged benchmark task via its SKILL — agent-ready or human-readable:

```bash
# inside the container:
geniesim status                 # verify the stack is healthy
cat skills/run-benchmark/SKILL.md
cat skills/check-inference/SKILL.md
```

## Common commands

The whole benchmark is driven by the `geniesim benchmark` CLI verb:

```bash
# 1. Discover what's available
geniesim benchmark categories                       # category counts (instruction / manipulation / spatial / …)
geniesim benchmark robots                           # robot/embodiment counts
geniesim benchmark list --robot=g2op --category=if

# 2. Probe your inference server BEFORE launching a sim (catches protocol / NaN issues early)
geniesim benchmark check-inference --infer-host=<IP>:8999

# 3. Run a single task against a live inference server (IP:PORT)
geniesim benchmark run g2op_if_pick_block_color --infer-host=<IP>:8999
# ...with pass-through ParameterServer overrides (headless + persist per-episode logs):
geniesim benchmark run g2op_if_pick_block_color --infer-host=<IP>:8999 \
  --app.headless=true --benchmark.record=true --benchmark.num_episode=20 --benchmark.seed=0

# 4. Batch-evaluate a whole sweep (one category × robot)
geniesim benchmark batch --category=if --robot=g2op
```

## What your inference server must speak

Full wire contract — observation in, control out — lives in
[`README.md` § Inference interface](README.md#-inference-interface-corobot-policy).
The short version:

- **Observation** (`params`): three JPEG cameras (`head`, `hand_left`,
  `hand_right`), joint states, the task `prompt`, and `states.end_pose` — the
  observed EEF pose delivered in **both** `base_link` and `arm_base_link` at
  once, so you read whichever frame your model was trained in.
- **Control** (`result`): per-arm `{"kind", "values"}` action chunks.
  `kind: "JOINT_ABS"` for joint targets, `kind: "EEF_ABS"` for EEF poses
  (IK-solved sim-side). For `EEF_ABS`, declare your frame with
  `"base_link": "base_link" | "arm_base_link"` — **it defaults to `base_link`**,
  so `arm_base_link` poses must say so explicitly.
- Pack arrays as native Python lists (`.tolist()`); the response is read with
  plain `msgpack`, not `msgpack_numpy`.

> 🖼️ **History image observations (corobot):** the runtime will feed a rolling
> buffer of past camera frames back to your inference server whenever the
> server opts in — return a `history: {"interval": N, ...}` block in the
> response `result` (`interval > 0` enables it and sets the sampling stride,
> `0`/omitted disables it; a legacy top-level `hist_frame_interval` is also
> still read). Optional `history.last_n` caps the buffer to the most recently
> captured frames, and `history.resolution` lets you resize each of the three
> history cameras independently — both default to "send everything, keep the
> existing resize behavior" when omitted. The frames arrive on the next
> request as `params.history`. Servers that don't return any of these fields
> are unaffected. See [`README.md`](README.md) for the full contract.

> 📏 **Depth images (corobot):** every request's `params.images` can carry
> `head_depth` / `hand_left_depth` / `hand_right_depth` (`RAW_UINT16`
> distance maps) alongside the RGB frames. The first request of each episode
> always includes them; after that, return a truthy `need_depth` in the
> response `result` to keep receiving them on the next request, or
> omit/return `false` to stop. See [`README.md`](README.md) for the full
> contract.

> 🚧 `geniesim_benchmark` is the **legacy** benchmark runtime — it drives Isaac Sim directly and is **independent and parallel to the RT Engine**. The roadmap is to refactor it into a benchmark layer on top of `geniesim_ros`; until then, treat the two as separate paths.

## More

- Task catalogue and scoring contract: [`README.md`](README.md)
- Submit a debugged policy to the leaderboard: the agent-friendly one-click SKILLs under [`skills/robocoliseum/`](skills/robocoliseum/)
