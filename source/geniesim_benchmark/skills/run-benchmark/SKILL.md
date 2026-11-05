---
name: run-benchmark
description: >
  Launch a geniesim_benchmark task locally (typically inside the GUI Docker
  container) against a user-provided inference server, using the
  `geniesim benchmark run` CLI verb.
  Trigger: When the user asks to "run geniesim", "本地跑仿真", "启动仿真任务",
  "run a benchmark", "launch <some>_<config>.yaml", or wants to execute a
  benchmark task config (anything under `geniesim_benchmark/config/*.yaml`)
  against a remote inference host (ip:port).
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_cli:fresh-machine-setup   # see source/geniesim_cli/AGENTS.md § 0
inputs:
  - name: config
    desc: Task config (basename, full path, or unique substring)
    required: true
  - name: infer_host
    desc: "IP:PORT of the inference server; required for live policy execution, omitted for preview/setup or supported hold checks"
    required: false
  - name: headless
    desc: GUI off (required on remote / batch hosts)
    required: false
    default: "false"
  - name: num_episode
    desc: Override episode count
    required: false
  - name: seed
    desc: RNG / instance-sampling seed
    required: false
  - name: record
    desc: Persist episode logs to output_dir
    required: false
    default: "false"
outputs:
  - desc: Per-episode pass/fail summary; logs under `output_dir` if `record=true`
---

## When to Use

- User wants to run a `geniesim_benchmark` task on their workstation (not the Challenge platform).
- User has an inference server already running somewhere reachable and provides its `ip:port`.
- User references any task config under `source/geniesim_benchmark/src/geniesim_benchmark/config/`.

Do **not** use for:
- Submitting jobs to the Challenge platform → `challenge-submit-job`.
- Verifying an inference server is healthy → use the `check-inference` skill.
- Adding a new benchmark task → `add-benchmark-task`.

## Critical Patterns

1. **Resolve the task and requested validation level first.** Collect inference
   IP/port for live policy execution. Preview, setup checks and supported hold
   checks do not require a real policy endpoint.
2. **Select the runtime from the task config.** Ordinary Isaac tasks use bundled
   5.1 `omni_python`; Newton/deformable tasks use system Python 3.12 / Isaac 6.2,
   and Rubik selects its prepared native MuJoCo interpreter. Follow the canonical
   [quick start and smoke guide](../../benchmark_test_docs.md#ai-smoke-testing) for build,
   source-path checks, backend-specific commands and result locations.
   Do not forward Isaac preview flags to every `dexmanip` entry.
   To skip building the default `geniesim3` combo image, follow the Docker Hub
   pull and retag instructions in [issue #201](https://github.com/AgibotTech/genie_sim/issues/201),
   then run `geniesim docker up` and `geniesim docker into` on the host.
3. **Working directory**: anywhere under the repo root works — the CLI
   walks up to find `scripts/` and uses `find_spec` to locate the
   benchmark package.
4. **Confirm before launching.** The task spawns a full simulator and
   typically holds a GPU; ask before kicking it off if there's any ambiguity.

## Workflow

### Step 1 — Collect inputs

If the user hasn't named a config, list candidates:

```bash
geniesim benchmark categories      # show category counts
geniesim benchmark robots          # show robot counts
geniesim benchmark list --robot=<R> --category=<C>
```

Resolve any missing task or live-policy inputs from the user's request:
- **Task config**: free-text (use the basename — e.g. `g2op_if_pick_block_color`).
- **Inference host** as `ip:port`.

### Step 2 — Probe inference (optional but recommended)

Before sinking minutes into Isaac Sim startup, sanity-check the server (uses the
bundled corobot payload):

```bash
geniesim benchmark check-inference --infer-host=<IP>:<PORT>
```

See the `check-inference` skill to override the payload.

### Step 3 — Run the task

Inside the GUI container (`geniesim docker into`):

```bash
geniesim benchmark run <CONFIG> --infer-host=<IP>:<PORT>
```

Example:

```bash
geniesim benchmark run g2op_if_pick_block_color --infer-host=<IP>:<PORT>
```

### Step 4 — Pass-through overrides (when asked)

`geniesim benchmark run` forwards any unknown `--key=value` to the
benchmark's `ParameterServer`. Common ones:

| Flag | Meaning |
|---|---|
| `--app.headless=true` | No GUI (required on remote / batch hosts) |
| `--benchmark.num_episode=N` | Override episode count |
| `--benchmark.seed=N` | RNG / instance-sampling seed |
| `--benchmark.record=true` | Persist episode logs to `output_dir` |
| `--benchmark.policy_class=…` | Use a different policy class |

Full schema: `source/geniesim_benchmark/src/geniesim_benchmark/config/params.py`.

## Commands (copy-paste summary for the user)

```bash
# Host — start the container (GUI by default; add --headless on remote/batch hosts)
cd /path/to/main
geniesim docker up

# Host — drop into a shell inside the running container
geniesim docker into
# inside container:
geniesim status                                       # verify the stack is healthy
geniesim benchmark check-inference --infer-host=<IP>:<PORT>
geniesim benchmark run <CONFIG> --infer-host=<IP>:<PORT>
```

## Notes

- If `geniesim` isn't on `$PATH` (the launcher wasn't installed), substitute `python3 -m geniesim_cli benchmark …` — same args, same behaviour.
- `<CONFIG>` accepts the bare basename (`g2op_if_pick_block_color`), a full path, or a unique substring.
- For batch evaluations, prefer `geniesim benchmark batch --category=… --robot=…` over a shell loop — it forwards extras consistently and prints a per-config pass/fail summary.
- Batch extras must be supported by every selected backend. Use per-entry commands
  for mixed-runtime `dexmanip`; CLI exit success alone is not E2E task success.
- The new CLI replaces the older ad-hoc `omni_python app/app.py --config …` invocation. The new form normalizes interpreter selection, host shorthand, and config resolution.
