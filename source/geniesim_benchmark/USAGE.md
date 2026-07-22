# Run the Genie Sim Benchmark locally

How to run a packaged benchmark task on your own machine against your own
inference server — the same tasks and scoring contract used by the
[AgiBot World Challenge: Open-Session](https://agibot-world.com/challenge/open-session/),
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
geniesim benchmark list --robot=g2op --category=instruction_following

# 2. Probe your inference server BEFORE launching a sim (catches protocol / NaN issues early)
geniesim benchmark check-inference --infer-host=<IP>:8999

# 3. Run a single task against a live inference server (IP:PORT)
geniesim benchmark run g2op_if_pick_block_color --infer-host=<IP>:8999
# ...with pass-through ParameterServer overrides (headless + persist per-episode logs):
geniesim benchmark run g2op_if_pick_block_color --infer-host=<IP>:8999 \
  --app.headless=true --benchmark.record=true --benchmark.num_episode=20 --benchmark.seed=0

# 4. Batch-evaluate a whole sweep (one category × robot)
geniesim benchmark batch --category=instruction_following --robot=g2op
```

> 🖼️ **History image observations (corobot):** the runtime will feed a rolling
> buffer of past head-camera frames back to your inference server whenever the
> server opts in — return an integer `hist_frame_interval` in the response
> `result` (`> 0` enables it and sets the sampling stride, `0`/omitted disables
> it). The frames arrive on the next request as `params.history`. Servers that
> don't return the field are unaffected. See [`README.md`](README.md) for the
> full contract.

> 🚧 `geniesim_benchmark` is the **legacy** benchmark runtime — it drives Isaac Sim directly and is **independent and parallel to the RT Engine**. The roadmap is to refactor it into a benchmark layer on top of `geniesim_ros`; until then, treat the two as separate paths.

## More

- Task catalogue and scoring contract: [`README.md`](README.md)
- Submit a debugged policy to the leaderboard: the agent-friendly one-click SKILLs under [`skills/agibot-world-challenge/`](skills/agibot-world-challenge/)
