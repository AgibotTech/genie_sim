# geniesim_cli — `geniesim` command-line dispatcher 🧞

Lightweight command-line front-end for the Genie Sim stack. Published as
the `geniesim_cli` distribution; sole owner of the `geniesim` console
script.

License: [Mozilla Public License Version 2.0](LICENSE)
Agent doc: [`AGENTS.md`](AGENTS.md) · Deep-dive: [`../../.agent/geniesim_cli.md`](../../.agent/geniesim_cli.md)

---

## 📦 Install

```bash
pip install -e source/geniesim_cli/
geniesim version
```

The CLI is a **standalone PEP 517 / PEP 621 wheel** with no heavy
runtime deps (no USD, no Isaac Sim, no MuJoCo at import time), so it
runs even on a control node where the rest of the SDK isn't installed.

---

## 🛠️ Command surface

| Command | Purpose |
|---|---|
| `geniesim version` | Print versions of all distributions + Python |
| `geniesim status` | Per-distribution health probe |
| `geniesim doctor` | Diagnose & repair (status + rosdep + env) |
| `geniesim bootstrap` | Install required peers from local sources in dependency order |
| `geniesim env` | Show `GENIESIM_*` env vars |
| `geniesim completion bash\|zsh` | Generate shell completion |
| `geniesim docker {build,up,down,into,logs}` | Manage the Genie Sim container |
| `geniesim docker5.1 …` / `docker6.2 …` / `docker4.5 …` | Per–Isaac-Sim variants |
| `geniesim docker --profile post-train …` / `--profile post-train-teleop …` | post-training simulation server / optional teleop layer; see [Docker profiles](../../docker/README.md#post-training-profiles) |
| `geniesim post-train server start\|status\|stop\|logs …` | Operate post-training server processes inside the prepared runtime; see [the server workflow](../rlinf_geniesim/README.md#start-and-initialize) |
| `geniesim post-train teleop …` | Connect Pico debugging; see teleoperation |
| `geniesim ros build {dev,release,cleanup}` | `colcon build` the ROS 2 workspace |
| `geniesim ros doctor` | Repair rosdep |
| `geniesim tool {deps-dag,ros-dag,docs} [--fix]` | Repo-maintenance audits (DAG + doc-coverage) |
| `geniesim deploy [MODULE]` | Build pure-Python wheel(s) into `./deploy/` |
| `geniesim benchmark {run,batch,list,…}` | Drive `geniesim_benchmark` tasks |
| `geniesim teleop run` | Launch the teleop loop |

Run `geniesim -h` for the full list.


---

## 🚀 Fresh-machine setup

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
pip install -e source/geniesim_cli/   # 1. CLI first (no heavy deps)
geniesim bootstrap                    # 2. installs required local peers; reports missing asset sources
geniesim status                       # 3. all-green check
geniesim doctor                       # 4. repair if not
```

See [`AGENTS.md` § 0 — Fresh-machine setup](AGENTS.md) for the install-order contract.

---

## 🏗️ Architecture (in one line)

`cli.py` dispatches public commands: every subcommand lives in
`commands/<name>.py` and is loaded lazily so the import cost stays near
zero. Heavy deps (USD, MuJoCo, anything `geniesim.*`) must NEVER be
imported at module top-level — see [`AGENTS.md`](AGENTS.md) §4.

---

## 🔗 Pointers

- 🗺️ Module map: [`../README.md`](../README.md)
- 🏠 Repo root: [`../../README.md`](../../README.md)
- 📚 Full command surface + style rules: [`../../.agent/geniesim_cli.md`](../../.agent/geniesim_cli.md)
- 🤖 Agent guide: [`AGENTS.md`](AGENTS.md)

## Independent data collection module

From a main checkout with the CLI installed:

```bash
geniesim data-collection install
geniesim data-collection build
# Optional: geniesim data-collection build --china
geniesim data-collection up
```

Image builds share the repository mirror presets: `--mirrors NAME` or `--china`
selects a preset; explicit Docker build arguments override it. Without a preset,
settings inherit the parent image. Use `--dry-run` to inspect the build command.

The installer registers `source/data_collection` as an editable locator. Its
Isaac Sim 5.1/cuRobo v2/ROS runtime is supplied by its independent Docker image.
These host commands work without automatically bootstrapping the other peers.
Install persists resolved task and asset roots in
`~/.config/geniesim/data-collection.json`; `--task-root PATH` and
`--assets-root PATH` override discovery. Environment variables override saved
settings. The task default is `source/geniesim_scene_agent/examples/tasks`.

```bash
geniesim data-collection config
geniesim data-collection llm list --robot g2
geniesim data-collection llm show place_long_block_across_two_supports_g2
geniesim data-collection run place_long_block_across_two_supports_g2 --scene 2026121696_166a0404
geniesim data-collection replay place_long_block_across_two_supports_g2
geniesim data-collection run --task-json tasks/my_job.json --headless
```

`run` defaults to LLM tasks; JSON jobs require `--task-json`. The LLM workflow
preserves the task library by using a per-run copy. Replay accepts a task name
(latest successful CLI recording) or an episode directory, and verifies saved
HDF5 and video outputs. See [the collection guide](../data_collection/README.md)
for options, run locations, and replay.

LLM server startup reads `backend`, `physics_frequency`, and
`enable_deformable` from the selected scene's `task_info.json`, falling back to
task-level metadata. Deformable tasks reject `--run-times` greater than 1.
Tasks whose USD assets have attached behavior scripts also require
`--run-times 1`; this asset condition requires manual inspection.

Newton Wuji evaluation uses `geniesim benchmark run g2wuji_dexmanip_wok_flip`. Portable scene layouts can be generated with `benchmark generate-layouts` and evaluated with `benchmark run-layouts`; see the evaluation guide.

Commands for optional modules require a matching consumer checkout. Post-training
commands require the server launchers; Newton layout commands require the
benchmark's Newton tools; collection workflows require the collection helper
scripts. Missing consumers are reported by the command. Help and Docker dispatch
remain available without importing those runtimes.
