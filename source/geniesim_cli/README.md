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
| `geniesim docker5.1 …` / `docker6.2 …` / `docker4.5 …` | Isaac Sim variants; `docker4.5` and `docker5.1` are ⚠️ E.O.L. / deprecated |
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

Isaac Sim **4.5**, standalone **5.1**, and the **5.1 + 6.2 combo** are
**deprecated (E.O.L.)**. The `docker4.5` (`geniesim2`) and `docker5.1`
(`geniesim3`) command handlers remain until code retirement.
Their commands print an advisory and remain available without an extra confirmation.
Use `geniesim docker` / `docker6.2` for new deployments.


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
git submodule update --init -- source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model
geniesim bootstrap                    # 2. installs required local peers; reports missing asset sources
geniesim status                       # 3. all-green check
geniesim doctor                       # 4. repair if not
```

See [`AGENTS.md` § 0 — Fresh-machine setup](AGENTS.md) for the install-order contract.

### Skip the Docker build

Follow the [prebuilt-image instructions](../../docker/PREBUILT_IMAGE.md) to use
the dedicated Isaac Sim 6.2 image without building locally. Then start the
container with `geniesim docker up` and enter it with `geniesim docker into`.

---

## Release version

The eight first-class `geniesim` / `geniesim_*` distributions share the root
[`VERSION`](../../VERSION), currently **3.6.0**. Each package's local `VERSION`
is a symlink to that file. Separate runtime environments do not change the
release version. Independent modules, service packages, data formats, and
third-party dependencies keep their own versions.
ROS packages maintained as Git submodules, including `genie_sim_robot_model`,
also own their versions and are excluded from parent release synchronization.

Benchmark commands use system Python 3.12 and Isaac Sim 6.2 in the dedicated
`geniesim4:latest` / `docker6.2` runtime. This image retains the 4.0 runtime
line; package versions still follow the root `VERSION`. See
[Docker release lines](../../docker/README.md#release-lines).

From a source checkout:

```bash
geniesim version --check            # check package links and owned ROS manifests
geniesim version --bump 3.6.0       # set the release and synchronize metadata
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


## Asset-backed native libuipc installation

The default is a direct prebuilt-wheel installation. Optional source builds use
the official pinned commit and ordered patches in
[the source build guide](../../3rdparty/libuipc/README.md).

`geniesim bootstrap` checks the optional native runtime using the current
interpreter and `docker/install_libuipc.py --if-needed`. `--libuipc-only` skips
peer probes and prompts. Required peers and optional failure behavior are unchanged;
bootstrap still bounds the installer process group to 300 seconds.

Wheels now live under the asset repository's `wheels/libuipc/`, with pinned
sizes/hashes and hosting-source identities in `docker/libuipc-artifacts.json`.
The standalone installer checks a durable verified cache, then local
`GENIESIM_ASSETS_SRC`, then the selected HF/ModelScope publication. It resolves the provider metadata to a real commit before transfer and uses that
commit throughout retries. ModelScope failures automatically switch to HuggingFace,
resolving HF’s own commit; a revision override applies only to the primary host.
Explicit HuggingFace selection does not retry ModelScope. Release bytes remain pinned by size and SHA256;
publication requires no code edits. An explicit commit override remains optional.
Only Linux x86_64 system CPython 3.12, SM89/SM120 and CUDA 12.8 are supported.

Dedicated 6.2 Docker builds select GPU 0 or explicit `LIBUIPC_SM`. The CLI forwards
the local `wheels/` directory as named context `libuipc_assets`, and forwards
`GENIESIM_LIBUIPC_SOURCE`/`GENIESIM_LIBUIPC_REVISION` as build arguments; explicit
arguments/contexts win. Images retain `GENIESIM_LIBUIPC_CACHE` for offline startup
checks. The CLI enables this build path only for recipes declaring `LIBUIPC_SM`.
The deprecated (E.O.L.) 5.1 combo omits native libuipc from image construction and automatic startup;
it does not probe GPU architecture or forward native asset build inputs.
Humble and derived profiles retain their existing paths.

Usage, publication and cache contracts are documented in
[asset-backed installation](../../docker/README.md#asset-backed-native-libuipc-installation).
CPU tests live in `docker/tests/test_libuipc_download.py`,
`docker/tests/test_install_libuipc.py` and `tests/test_libuipc_bootstrap.py`.
Real archive QA additionally uses a hydrated `GENIESIM_ASSETS_SRC`; CPU tests do
not establish an image build, native load or GPU solve.
