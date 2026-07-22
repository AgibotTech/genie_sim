# data_collection — Agent Development Guide

Standalone, host-orchestrated **automated trajectory data-collection** module:
Isaac Sim 5.1 + cuRobo, a gRPC **client/server** pair that drives a robot
through declarative task templates and records full **agibot-format episodes**.
This is the open-source incarnation of the automated production code behind the
data-collection pipeline — episodes carry the same `aligned_joints*.h5` +
`observations/` + `data_info.json` layout used downstream.

Source: [source/data_collection/](.)
License: [MPL-2.0](LICENSE.txt)
Human docs: [README.md](README.md) (install/run) · [TASK_CONFIG_GUIDE.md](TASK_CONFIG_GUIDE.md) (task JSON schema)
CLI: `geniesim autocollect` — owned by
[geniesim_cli/commands/data_collection.py](../geniesim_cli/src/geniesim_cli/commands/data_collection.py)
Skills (Claude Code / opencode): [skills/](skills/)

> **Status** — [source/AGENTS.md](../AGENTS.md) lists this as a *legacy /
> non-geniesim* module: it ships its own Docker image + conda path and the
> **runtime** code is not a Python package. A thin **locator shim**
> ([`pyproject.toml`](pyproject.toml) + [`__init__.py`](__init__.py)) makes the
> distribution `geniesim-data-collection` `pip install -e`-able so the CLI can
> `find_spec("data_collection")` this dir (parity with `geniesim_benchmark`).
> The CLI **never imports the runtime tree** and **falls back** to a cwd /
> `$GENIESIM_REPO_ROOT` walk when the shim isn't installed. To fully revert:
> `pip uninstall geniesim-data-collection` and delete `pyproject.toml` +
> `__init__.py`.

> **Maintenance contract** — when you change a script's CLI flags, the task
> JSON schema, the gRPC interface, the `data_filter` rules, or the Docker
> image/entrypoint, **update this file in the same diff**. Agents read this as
> the source of truth.

---

## 1. The CLI surface

`geniesim autocollect <subcommand>`. If the `geniesim`
console script isn't on `$PATH`, substitute `python3 -m geniesim_cli
autocollect …` — same dispatch.

| Subcommand | What it does |
|---|---|
| `list [--robot=R] [--task=T] [SUBSTR]` | List task templates under `tasks/` |
| `tasks` | Distinct tasks (families) + counts |
| `robots` | Distinct robots (`g1` / `g2`) + counts |
| `run <TASK> [flags]` | Collect one task — **host-orchestrated** (see §2) |
| `build [--image=TAG] [docker args…]` | Build the image (`docker build -f dockerfile`; base `geniesim3:latest` must exist) |
| `up` / `into` / `down` `[--container-name=N]` | Interactive GUI container: create+enter / enter running / stop+remove (wraps `start_gui.sh`) |

`run` accepts **only** the flags below. Unlike `benchmark run`, the underlying
script is a strict parser and does **not** forward arbitrary `--key=value`:

| Flag | Effect |
|---|---|
| `--headless` | No GUI; required on unattended / no-X11 hosts |
| `--no-record` | Disable recording (drops `--publish_ros` + `--use_recording`) |
| `--standalone` | Logs to files only, no terminal echo |
| `--container-name=N` | Override container name (default `data_collection_open_source`) |
| `--dry-run` | Resolve the task + print the command, **don't** launch |

### `<TASK>` resolution

First hit wins: **literal path** (abs / relative to cwd / to the module dir) →
**exact basename** (stem, `.yaml`-style auto-suffix `.json`) → **unique
substring** against task stems. Ambiguous substring → error with candidates.

---

## 2. How an agent drives this module (read first)

**`autocollect run` is NOT the same shape as `benchmark run`.** `benchmark
run` execs a single `app.py` *inside* an already-running container.
`autocollect run` is a **host-side orchestrator**: it shells out to
[`scripts/run_data_collection.sh`](scripts/run_data_collection.sh), which does
`docker run -d` against this module's **locally built** image
(`registry.agibot.com/genie-sim/geniesim3-data-collection:latest`), and the
in-container entrypoint launches **two** processes (Isaac Sim server + task
client) over gRPC.

> **cuRobo distribution and license boundary.** The repository does not vendor
> cuRobo, and AgiBot does not publish a prebuilt data-collection image containing
> it. A user-initiated Docker build fetches `v0.7.6` directly from the official
> NVIDIA cuRobo repository; local deployments must use the same version because
> this runtime depends on the cuRobo 0.7.x API. cuRobo is not MPL-2.0-covered
> GenieSim source: `v0.7.6` remains under the
> [NVIDIA cuRobo license](../../THIRD_PARTY_LICENSES/LICENSE.CUROBO), including
> its non-commercial research/evaluation limitation.

- The **agent runs on the host**; the checkout is bind-mounted into the
  container at `/geniesim/main/data_collection`, so host edits are live inside.
- **Prefer `run --headless --standalone`** for unattended runs: detached
  container + file logs, no X11. The interactive two-terminal flow (§5) is for
  human debugging and maps poorly onto an agent's non-persistent shell.
- **Outputs** land in `recording_data/[{TASK}_{INDEX}]/` — one dir per episode,
  **~1.5 GB each** (`camera/` raw mcap + `observations/videos/*` +
  `aligned_joints*.h5` + `state.json` + `data_info.json` with action labels).
  Per-run logs land in `logs/{TASK}/` (`data_collector_server.log`,
  `run_data_collection.log`, …).
- The container is **ephemeral** — `run_data_collection.sh` traps EXIT and
  `docker stop/rm`s it. State survives only in the mounted dirs above.

> ✅ **Unattended / no-tty.** `run_data_collection.sh` grants the container's
> uid 1234 access to the bind-mounted dirs. It **prefers** `sudo setfacl` when
> passwordless sudo is available and **degrades to `chmod -R a+rwX`** otherwise,
> so headless/background runs no longer hard-fail when sudo can't cache its
> tty-keyed credentials (`timestamp_type=tty`). Caveat: the fallback makes
> `recording_data/` / `saved_task/` world-writable on the host — tighten
> afterward if that matters.

> **`geniesim_assets` must be pip-installed (editable)** on the host — the CLI
> discovers it via `find_spec` (same as `geniesim docker`), bind-mounts it to
> `/geniesim_assets`, and the entrypoint editable-installs it + sets
> `SIM_ASSETS=/geniesim_assets` (from which it copies the cuRobo robot assets).

---

## 3. Architecture (client ↔ server)

Two processes, one container, gRPC between them:

| Side | Entry | Key flags | Internals |
|---|---|---|---|
| **server** | [`scripts/data_collector_server.py`](scripts/data_collector_server.py) | `--enable_physics --enable_curobo --publish_ros [--headless]` | `server/grpc_server.py`; `controllers/` (kinematics, parallel_gripper, ruckig); `motion_generator/` (cuRobo reacher); `recording/`; `ros_publisher/` (camera/lidar/imu) |
| **client** | [`scripts/run_data_collection.py`](scripts/run_data_collection.py) | `--task_template <json> [--use_recording]` | `client/layout/` (layout/task gen) → `client/planner/` (grasp/place/insert/rotate/stage) → `client/agent/omniagent.py` → `client/robot/` (gRPC client) |

[`scripts/data_collection_entrypoint.sh`](scripts/data_collection_entrypoint.sh)
maps the high-level flags onto the two processes: `--headless` → server
`--headless`; recording on → server `--publish_ros` **and** client
`--use_recording`; `--task` → client `--task_template`. It starts the server,
waits ~10–15 s for Isaac Sim, starts the client, monitors both, and exits 0 on
client `job done`.

**`--publish_ros` is required for recording** — the ROS publishers are the
recording source. `--no-record` drops both it and `--use_recording`.

---

## 4. Tasks & config naming

```
tasks/<collection>/<task>/<robot>/<name>.json
# e.g. tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json
```

- `<robot>` is `g1` / `g2`; `<task>` is the task (type) — the dir above the
  robot, grouping its variant files. **Task dir names differ from the variant
  stems** (task `sort_fruit` vs files `sort_the_fruit_…`); `list <SUBSTR>`
  matches against `task/robot/name` so either works. Filter with `--task=<dir>`.
- Each JSON's top-level `task` is the episode/task name. Full schema (objects,
  scene, robot, stages, checkers, `recording_setting`, `task_metric`) is in
  [TASK_CONFIG_GUIDE.md](TASK_CONFIG_GUIDE.md), parsed by
  `client/layout/task_generate.py`.
- Data-quality checkers / filter rules live in
  [`common/data_filter/`](common/data_filter/) — see its
  [README](common/data_filter/README.md).

---

## 5. Path map & workflows

| Artifact | Location |
|---|---|
| Task templates | [`tasks/geniesim_2025/<task>/<g1\|g2>/*.json`](tasks/) |
| Server / client entries | [`scripts/data_collector_server.py`](scripts/), [`scripts/run_data_collection.py`](scripts/) |
| Orchestrators | [`scripts/run_data_collection.sh`](scripts/) (one-shot), [`scripts/start_gui.sh`](scripts/) (interactive), [`scripts/*entrypoint*.sh`](scripts/) |
| Robot / cuRobo configs | [`config/robot_cfg/{G1,G2}*.json`](config/), [`config/curobo/configs/`](config/curobo/) |
| aimdk protocol | [`common/aimdk/protocol/`](common/aimdk/) |
| Recording / conversion | [`server/recording/`](server/recording/) (`extract_ros_bag.py`, `sim_data_converter.py`) |
| Outputs | `recording_data/[{TASK}_{INDEX}]/` · logs `logs/{TASK}/` |
| Docker image | built from [`dockerfile`](dockerfile) → `registry.agibot.com/genie-sim/geniesim3-data-collection:latest` |

### Collect one task (recommended, unattended)

```bash
pip install -e /path/to/geniesim_assets   # once on the host (editable)
geniesim autocollect run sort_the_fruit_into_the_box_apple_g2 --headless --standalone
# preview without launching:
geniesim autocollect run apple_g2 --dry-run
```

### Discover tasks

```bash
geniesim autocollect list                       # all
geniesim autocollect list --robot=g2 sort_fruit
geniesim autocollect tasks
geniesim autocollect robots
```

### Interactive (human debugging — two terminals)

```bash
pip install -e /path/to/geniesim_assets   # once on the host (editable)
./scripts/start_gui.sh run my_container          # Terminal A: create container
./scripts/start_gui.sh exec my_container         # Terminal B: enter; then inside:
python scripts/data_collector_server.py --enable_physics --enable_curobo --publish_ros
# Terminal C (same container):
python scripts/run_data_collection.py --task_template tasks/.../<name>.json --use_recording
```

---

## 6. Environment variables

| Var | Effect |
|---|---|
| `GENIESIM_ASSETS_SRC` / `SIM_ASSETS` | CLI auto-discovers the editable-installed `geniesim_assets` (`find_spec`) → `GENIESIM_ASSETS_SRC`; launch scripts bind-mount it to `/geniesim_assets`; the entrypoint sets `SIM_ASSETS=/geniesim_assets` and copies cuRobo assets from there. No manual export needed (just `pip install -e geniesim_assets`). |
| `GENIESIM_REPO_ROOT` | Override repo-root detection used by the CLI to locate `source/data_collection`. |
| `TORCH_CUDA_ARCH_LIST` | cuRobo build arch (image default `8.9` = RTX 4090D). |
| ROS (`ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION`, `LD_LIBRARY_PATH` → isaacsim ros2 bridge) | Set by the entrypoint / README's local-deploy steps. |

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `sudo: a terminal is required` then early exit | Handled — `run_data_collection.sh` now falls back to `chmod` when sudo is unusable (§2). If you still hit it, your copy predates that fix. |
| CLI errors `geniesim_assets is not pip-installed (editable)` | Run `pip install -e /path/to/geniesim_assets` on the host first (the CLI discovers it via `find_spec`). |
| `recording_data/` stays empty | Recording needs `--publish_ros` (server) **and** `--use_recording` (client); via the CLI just **don't** pass `--no-record`. |
| cuRobo install/import fails | GPU arch mismatch — image is built for `TORCH_CUDA_ARCH_LIST=8.9` (4090D); 50-series (SM_120) may be unsupported by cuRobo. |
| `Ambiguous task 'X'` | Substring matched ≥2 task stems — use a longer substring or full basename. |
| Container left running after a crash | `run_data_collection.sh` traps EXIT to clean up; if killed hard, `docker rm -f data_collection_open_source`. |

---

## 8. Do not

- Don't pin a host/secret inside a task JSON — keep them runtime args.
- Don't bypass `data_filter` quality checks when producing deliverable data.
- Don't change the gRPC interface on one side only — `server/grpc_server.py`
  and `client/robot/client.py` must stay in sync.
- Don't assume cuRobo builds on any GPU — the image targets 4090D (SM 8.9).
- Don't treat `run` as an in-container exec like `benchmark run` — it brings up
  its own container from the host (§2).

---

## 9. Output format

Episodes produced here are **agibot-format** (`.h5` + videos + `data_info.json`
with pick/place action labels). This tree is the data *generation* side.
