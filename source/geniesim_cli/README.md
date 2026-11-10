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
| `geniesim docker {build,up,down,into,logs}` | Manage the default Isaac Sim 6.2 container (alias of `docker6.2`) |
| `geniesim docker5.1 …` / `docker6.2 …` / `docker4.5 …` | Per–Isaac-Sim variants |
| `geniesim docker --profile post-train …` / `--profile post-train-teleop …` | post-training simulation server / optional teleop layer; see [Docker profiles](../../docker/README.md#post-training-profiles) |
| `geniesim post-train server start\|status\|stop\|logs …` | Operate post-training server processes inside the prepared runtime; see [the server workflow](../rlinf_geniesim/README.md#start-and-initialize) |
| `geniesim post-train teleop …` | Connect Pico debugging; see [teleoperation](../rlinf_geniesim/TELEOP_USAGE.md) |
| `geniesim ros build {dev,release,cleanup}` | `colcon build` the ROS 2 workspace |
| `geniesim ros doctor` | Repair rosdep |
| `geniesim tool deps-dag` / `ros-dag` `[--fix]` | Verify or regenerate dependency diagrams |
| `geniesim tool docs` | Audit documentation coverage, indexes, local links and package metadata |
| `geniesim deploy [MODULE]` | Build pure-Python wheel(s) into `./deploy/` |
| `geniesim benchmark {run,batch,list,…}` | Drive `geniesim_benchmark` tasks |
| `geniesim teleop run` | Launch the teleop loop |

Run `geniesim -h` for the full list.


---

## Installation tiers

Tier 1 contains `geniesim_cli`, `geniesim_assets`, `geniesim_benchmark` and
`geniesim_ros`. Install this CLI first; bootstrap installs or repairs the
remaining required peers and the `geniesim` umbrella. Assets are acquired separately.

Tier 2 contains optional `geniesim_teleop`, `geniesim_scene_agent` and
`geniesim_world`; default bootstrap skips these packages. Dream is separately
installed outside the umbrella extras. See the [module tiers](../README.md#installation-tiers)
and each module's guide for environment requirements.

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

## Release version

The eight first-class `geniesim` / `geniesim_*` distributions share the root
[`VERSION`](../../VERSION), currently **3.5.0**. Each package's local `VERSION`
is a symlink to that file. Separate runtime environments do not change the
release version. Independent modules, service packages, data formats, and
third-party dependencies keep their own versions.

The 3.5.0 benchmark release uses the `geniesim3:latest` Isaac Sim 5.1 + 6.2
combo. The dedicated `geniesim4:latest` / `docker6.2` variant remains the
separate 4.0 line; see [Docker release lines](../../docker/README.md#release-lines).

To skip building the explicit `docker5.1` combo image, pull and retag the prebuilt Docker Hub
image using [issue #201](https://github.com/AgibotTech/genie_sim/issues/201).
See the [copyable pull commands](../../docker/README.md#use-the-prebuilt-combo-image).

From a source checkout:

```bash
geniesim version --check            # check package links and owned ROS manifests
geniesim version --bump 3.5.0       # set the release and synchronize metadata
geniesim version --sync             # synchronize after editing root VERSION
```

`geniesim version` reports installed distribution metadata. Rebuild wheels or
refresh editable installations after a release bump to update installed metadata.

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
Isaac Sim 6.2/cuRobo v2/ROS 2 Jazzy runtime is supplied by its independent Docker image,
which incrementally extends the main GenieSim image.
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


## Automatic native libuipc installation

Jazzy base image builds select host GPU 0 and install its bundled SM89/SM120
artifact during construction. `--build-arg LIBUIPC_SM=89|120` overrides detection;
unsupported/unreadable capability fails before Docker. Humble and derived
post-training builds retain their existing paths.

`geniesim bootstrap` additionally checks/installs optional native libuipc using
the current interpreter and the checkout's `docker/install_libuipc.py --if-needed`.
`geniesim bootstrap --libuipc-only` skips peer probes, installs and prompts.
Use system CPython 3.12 for native wheels, including in the 5.1 combo image.
Required-peer source resolution, ordering and failure status remain intact.
Native failures warn without blocking required peers or container startup, and
completion output distinguishes that warning from full native success.
Installer progress streams to the caller; a 300-second timeout bounds the whole
process group, with 15 seconds for graceful shutdown before forced cleanup.
No GPU, physics or rendering dependency is imported into the CLI.

CPU coverage: `tests/test_libuipc_bootstrap.py` and the existing Docker/mirror,
installation-health and process-boundary suites. A build-command test does not
qualify a completed image build or a GPU solve.
