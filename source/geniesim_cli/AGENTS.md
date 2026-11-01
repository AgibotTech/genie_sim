# geniesim_cli — Agent Development Guide

Lightweight command-line front-end for the **geniesim** stack. Published as the
[`geniesim_cli`](./pyproject.toml) PyPI distribution; sole owner of the
`geniesim` console script.

**Maintenance contract** (read before editing): whenever you change command
dispatch, add/remove a subcommand, change colour/emoji conventions, or alter
how a sibling distribution is detected, **update this file in the same diff**.
This document is the agent's source of truth for what `geniesim_cli` does and
how it talks to its siblings.

---

## 0. Fresh-machine setup

The CLI is the **first thing** installed on a new machine — it ships the
`geniesim bootstrap` verb that installs the required local peers.

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
pip install -e source/geniesim_cli/      # 1. CLI first (no heavy deps)
geniesim bootstrap                       # 2. installs required peers; reports missing asset sources
geniesim status                          # 3. all-green check
geniesim doctor                          # 4. repair if not
```

**Install-order contract** (do not violate):

1. **`geniesim_cli` first.** If the CLI itself isn't installed, the `geniesim` console script doesn't exist
   and you can't call `bootstrap` at all. This is why `geniesim_cli` is
   **deliberately excluded** from `_TARGETS` — re-installing the
   running CLI mid-execution is a footgun (see §7a).
2. **Dependency order is the CLI's job.** `_TARGETS` in
   `commands/bootstrap.py` follows required peers from the umbrella metadata,
   with the umbrella last. Bypassing the CLI and
   running `pip install` by hand in the wrong order leaves dangling deps.
3. **Healthy stacks default to no changes.** Bootstrap checks imports,
   runtime dependencies and submodules, then asks whether to reinitialize
   a healthy stack (default No).
4. **Asset pack is a separate download.** Python distributions are tiny;
   the 3D assets are several GB on HuggingFace + ModelScope. `bootstrap`
   prints `_DOWNLOAD_SOURCES` and editable-install instructions when the
   asset source cannot be found; it does not download the pack.
5. **`geniesim` and `geniesim_assets` are NOT on PyPI.** Always go through
   `geniesim bootstrap`; never suggest `pip install geniesim` to the user.

---

## 1. Why this distribution exists

The CLI ships as a separate, **dependency-light** wheel so that:

- `pip install geniesim_cli` works in any environment, even one without USD,
  MuJoCo, or Isaac Sim.
- The console script does not collide with — and is not blocked by — the heavy
  SDK installations.
- Operators can run `geniesim status`, `geniesim version`, `geniesim deploy`
  even on a control node where only the CLI is installed.

The CLI, umbrella and asset package have separate responsibilities; see the
[module map](../README.md) for the complete peer list:

| Distribution | Role | Heavy deps? |
|---|---|---|
| `geniesim_cli` (this) | Dispatcher + ops commands | **No** — `tomli` only |
| `geniesim` | Umbrella metadata: required peers and optional extras | Through peers |
| `geniesim_assets` | USD↔MJCF converters, asset compile | Medium (USD, trimesh, Pillow) |

The CLI has no SDK install extra. Required peers and optional extras are
declared in the [umbrella metadata](../geniesim/pyproject.toml); bootstrap
installs required peers, while optional peers need their own setup.

---

## 2. Layout

```
source/geniesim_cli/
├── AGENTS.md              ← this file
├── pyproject.toml         ← `geniesim` console script entry point lives here
├── setup.py               ← thin shim, find_packages(where="src")
└── src/geniesim_cli/
    ├── __init__.py        ← `__version__`
    ├── __main__.py        ← `python -m geniesim_cli`
    ├── _style.py          ← canonical ANSI / NO_COLOR / TTY-detect constants
    ├── _lazy.py          ← lazy imports and distribution source lookup
    ├── _tiers.py         ← required / optional peers from umbrella metadata
    ├── _status_spec.py   ← derived health probe specifications
    ├── _workspace.py     ← ROS workspace discovery
    ├── _isaacsim.py      ← base-image / wheel resolution
    ├── _mirrors.py       ← Docker mirror presets and build-argument merging
    ├── commands/         ← per-command implementations
    └── cli.py            ← public dispatcher and bootstrap gate
```

`_style.py` is the **canonical home** of the ANSI helpers for this package.
Honor its NO_COLOR and TTY detection.
**Never** redefine ANSI escapes inline in this package — always import from
[`geniesim_cli._style`](./src/geniesim_cli/_style.py).

---

## 3. Command surface

All commands route through [`main()`](./src/geniesim_cli/cli.py) at the bottom
of `cli.py`. Keep the top-of-file usage docstring, [`_print_usage()`](./src/geniesim_cli/cli.py),
and the dispatcher in `main()` in sync.

| Command | Purpose | Heavy deps required? |
|---|---|---|
| `geniesim version` | Print distribution versions + Python | No |
| `geniesim status` | Per-distribution health probe + console-script + env | No |
| `geniesim bootstrap` | Install / reinitialize required local peers, umbrella last (interactive `pip install`) | No (but invokes `pip`) |
| `geniesim ros build dev` | `colcon build` with RelWithDebInfo + `--symlink-install` | Needs colcon on PATH |
| `geniesim ros build release` | `colcon build` with Release | Needs colcon on PATH |
| `geniesim ros doctor` | Diagnose and repair rosdep dependencies | Needs ROS tooling |
| `geniesim ros build cleanup` | Remove `devel*`, `build/`, `install/`, `log/` (interactive prompt) | No |
| `geniesim deploy [MODULE]` | Build wheel(s) for deploy candidates (or one named MODULE) into `./deploy/` | Needs `python -m build` |
| `geniesim deploy list` | List deploy candidates | No |
| `geniesim deploy upload` | `curl -F` upload of `./deploy/*.whl` to file server | Needs `curl` |
| `geniesim doctor` | Diagnose & repair (status + rosdep + more) | No (probes tools) |
| `geniesim docker` / `docker4.5` / `docker5.1` / `docker6.2 <build\|up\|down\|into\|logs>` | Manage the GenieSim container across Isaac Sim variants | Needs docker on host |
| `geniesim docker --profile post-train\|post-train-teleop <build\|up\|down\|into\|logs>` | Manage the post-training simulation server or optional teleop layer | Needs docker on host |
| `geniesim post-train server <start\|status\|stop\|logs>` | Operate selected post-training simulation server ports | Start: prepared simulation runtime; status/logs: Bash and checkout |
| `geniesim post-train teleop …` | Connect Pico debugging to an initialized server | Prepared teleop runtime |
| `geniesim env [--all\|--unset]` | Show `GENIESIM_*` env vars | No |
| `geniesim completion bash\|zsh` | Generate shell completion script | No |
| `geniesim teleop run` / `teleop bridge` | Run the VR loop or image bridge | Prepared teleop runtime |
| `geniesim benchmark <run\|list\|batch\|categories\|robots\|check-inference>` | Run / inspect benchmark tasks (see [geniesim_benchmark/README.md](../geniesim_benchmark/README.md)) | `run`/`batch`: Yes (Isaac Sim); rest: No |
| `geniesim data-collection <install\|config\|llm\|list\|tasks\|robots\|run\|replay\|build\|up\|into\|down>` | List / run / build / manage data_collection tasks & container (see [data_collection/AGENTS.md](../data_collection/AGENTS.md)) | `run`/`up`: Docker + GPU; `build`: Docker; rest: No |
| `geniesim tool deps-dag` | Verify the module-dependency DAG in `source/README.md` matches every peer's `pyproject.toml` (CI hook) | No |
| `geniesim tool deps-dag --fix` | Regenerate the DAG in-place between `<!-- AUTOGEN:deps-dag -->` markers | No |
| `geniesim tool ros-dag` | Verify the ROS-package DAG in `source/geniesim_ros/README.md` matches every workspace `package.xml` (CI hook). | No |
| `geniesim tool ros-dag --fix` | Regenerate the ROS-package DAG in-place between `<!-- AUTOGEN:ros-dag -->` markers | No |
| `geniesim tool docs` | Repo-wide doc-coverage audit: coverage × index × links × metadata, across cli + ros scopes (CI hook) | No |
| `geniesim tool docs --scope ros` | ROS-workspace-only audit | No |
| `geniesim tool docs --check links` | Only run the broken-link audit across every AGENTS/README/SKILL | No |
| `geniesim tool docs --check metadata` | Only run the ROS `package.xml` metadata audit (maintainer / license / LICENSE file) | No |
| `geniesim dataset convert agibot-to-lerobot --agibot-dir … --output-dir …` | Convert agibot v1 → LeRobot v2.1. Dispatcher only — logic lives in [`geniesim_benchmark.dataset.convert.agibot_to_lerobot`](../geniesim_benchmark/src/geniesim_benchmark/dataset/convert/agibot_to_lerobot.py). | Yes (`geniesim_benchmark` + ffmpeg on PATH) |

---

`docker6.2` uses the internal variant key `""` for Docker registry API calls.
`geniesim docker` maps to `docker5.1`, which keeps bundled Isaac 5.1 plus
system-Python Isaac 6.2.
The `docker6.2` variant resolves the exact base spec `6.2.0.1` to
`nvcr.io/nvidia/isaac-sim:6.2.0.1`, paired explicitly with the available
`isaacsim==6.2.0.0` cp312 wheel. The Dockerfile carries the same offline
fallback. `_isaacsim._TAG_WHEEL_PAIRS` owns this image-respin exception;
other four-component tags must have an equal-version wheel. Verify the
resolver and Docker build dispatch with
`PYTHONPATH=src python3 -m pytest tests/test_isaacsim.py` from this package.
See [runtime selection](../../docker/README.md) and the
platform roadmap for qualification work.

## 4. Architectural rules (do not violate)

`Variant`, `VARIANTS`, `get_variant` and `variant_env` in
`commands/docker.py` form the public container-configuration API. Variants
and their environment mappings are immutable; keep positional construction
compatible when adding optional fields.

The ROS metadata audit requires one maintainer named `Genie Sim Team` with
email `yangwendisheng@agibot.com`, one license entry with
`file="LICENSE"` and text `Mozilla Public License Version 2.0`, and a
`LICENSE` file beside each source `package.xml`.

Docker profiles reuse the `Variant` value type through `PROFILES` in
`commands/docker.py`; the public Isaac-version `VARIANTS`, `get_variant`, and
`variant_env` API stays unchanged. `--profile` precedes the verb and is accepted
only by `geniesim docker`. Profiles select system Python 3.12/Jazzy, an 8 GB
shared-memory segment, and the image's post-training entrypoint. Existing containers must
match checkout/profile ownership labels before any lifecycle command proceeds.
The host needs only this CLI; `GENIESIM_ASSETS_SRC` can point directly to the
asset package without installing its dependencies. Docker process-boundary
tests live in `docker/tests/test_docker_cli.py`.

Profile-specific argument checks, readiness waiting, and path normalization
must not change the behavior of `docker` without `--profile` or the existing
Isaac Sim variant commands.

Docker build mirrors come from `docker/mirrors.toml`. Base builds select
`upstream` by default; derived profiles inherit their parent unless a preset is
explicitly selected. `--china` aliases `--mirrors china`; `--mirrors upstream`
resets inherited APT/pip defaults. `_mirrors.py` filters preset settings to
declared Dockerfile arguments and merges explicit `--build-arg` values last,
emitting each argument once. Keep provider URLs out of the dispatcher. Bare
build arguments retain Docker's environment forwarding. Isaac metadata
resolution remains upstream and independent of mirror selection.

The base image installs `geniesim-configure-mirrors` and sets
`PIP_CONFIG_FILE=/etc/pip.conf`; derived recipes reuse that helper before
installing dependencies. Empty helper values preserve inherited settings.
See [Docker build arguments](../../docker/README.md#build-arguments).

`post-train` skips automatic bootstrap. Its dispatcher executes existing checkout
scripts without importing simulator packages: lifecycle commands use
`start_sim_servers.sh`, logs use `start_sim_server.sh --logs`, and teleop uses
`host_run_teleop.sh`. The shell launcher owns slot paths, process ownership,
ROS builds, and rollback. Keep argument boundaries, signals, and exit codes.
Reject lifecycle flags that conflict with the selected CLI action.

The runtime checkout is selected by `GENIESIM_REPO_ROOT`, then
`GENIESIM_REPO_PATH` / `SIM_REPO_ROOT`, then the current directory's ancestors.
An invalid explicit path is an error. These commands require a server checkout;
the standalone client wheel does not include the launchers. Help, status, and
logs remain usable without simulator imports. Tests live in
`source/geniesim_cli/tests/test_post_train.py`.

1. **No top-level heavy imports.** `usd-core`, `mujoco`, `trimesh`, `Pillow`,
   `coacd`, anything from `isaacsim.*`, anything from `geniesim.*`,
   `geniesim_assets.*` — **all** must be imported lazily, inside the function
   that uses them, via [`import_or_die(module_path, hint_pkg)`](./src/geniesim_cli/_lazy.py).
   Verify by:
   ```bash
   python -c "import geniesim_cli.cli"   # must succeed in a venv with no extras
   ```
2. **Every coloured f-string ends with `{RST}`.** No exceptions.
3. **No inline ANSI codes.** Import from [`_style`](./src/geniesim_cli/_style.py).
4. **Unknown subcommands print usage and exit non-zero.** See the bottom of
   `main()`. Don't silently fall through.
5. **Lazy-import errors must be friendly.** `import_or_die` already handles
   this — direct missing private distributions to `geniesim bootstrap`, never let an
   `ImportError` traceback leak.
6. **The `geniesim` console script is owned here.** If a sibling distribution
   ever declares `[project.scripts] geniesim = …` again, that's a packaging
   bug; resolve in their `pyproject.toml`, not here.
7. **`status` must never raise on missing siblings.** Treat absent
   distributions as a *finding*, not an error. The probe lives in
   [`inspect()`](./src/geniesim_cli/commands/status.py) and uses
   `importlib.metadata.distribution()` + `importlib.import_module()` wrapped
   in `try/except`.

---

## 5. Locating sibling source trees

[`distribution_source_root(name)`](./src/geniesim_cli/_lazy.py) resolves the
on-disk source tree of an installed distribution via `importlib.util.find_spec`.
This is what `geniesim deploy` uses to feed `python -m build`. Two consequences:

- Discovery follows the installed package rather than an assumed sibling path.
- A deploy candidate must be **installed (editable counts)** for `deploy` to
  find it. Run `geniesim bootstrap` before deploying from a fresh checkout;
  optional peers need their own installation.

The ROS workspace root for `geniesim ros build *` is resolved by
[`ros_workspace_root()`](./src/geniesim_cli/_workspace.py). It honors
`$GENIESIM_WORKSPACE`, then the current directory if it is a colcon workspace,
then the workspace bundled with the installed `geniesim_ros` peer.
Build outputs remain rooted in the invocation directory: `devel`, `devel_build`
and `devel_log` for development; `install`, `build` and `log` for release.

---

## 6. Style rules

- Format every changed `.py` with `black --line-length 120` before finishing.
- Colour conventions:
  - `GREEN` + ✅ — success
  - `RED` + ❌ — error
  - `YELLOW` + ⚠️ — warning / not-installed
  - `CYAN` + relevant emoji — informational / progress
  - `DIM` — debug, secondary detail, metadata
  - `BOLD` — file paths, tool names, key values
  - `MAGENTA` + 🧞 — geniesim brand / banner
  - `WHITE` — list items
- Do **not** add comments unless the user asked for them. Use docstrings on
  non-trivial functions instead (see `commands/status.py:inspect` and `run`).

---

## 7. How `status` decides "healthy"

[`run()`](./src/geniesim_cli/commands/status.py) calls `inspect` for each
distribution and reports the following facts.
[`_status_spec.py`](./src/geniesim_cli/_status_spec.py) derives the probe spec
from package metadata and installed layout; `commands/status.py` exposes it as
`DISTRIBUTIONS`.

1. **installed** — `importlib.metadata.distribution(name)` resolves.
2. **version** — `dist.version`.
3. **editable** — read PEP 610 `direct_url.json` from the dist-info; check
   `dir_info.editable`. Returns `None` (→ "unknown mode") when the file is
   absent (e.g. `PYTHONPATH` shims, system packages without direct_url).
4. **import_ok** — actually `importlib.import_module(top_module)`. Captures
   any exception and surfaces `type(exc).__name__: exc`.
5. **runtime deps** — for each `(pip_name, import_name)` pair declared on
   the spec, attempt `importlib.import_module(import_name)` via
   [`probe_module`](./src/geniesim_cli/_lazy.py). A failed dep flips the
   overall verdict to ⚠️ and prints `fix with: geniesim bootstrap`.
6. **submodules** — for each discovered public sub-package,
   import-probe it. Useful for catching partial
   installs or namespace collisions.
7. **optional extras** — for each component extra derived from metadata,
   probe its import names. Missing extras are reported as
   skipped (⏭️ `DIM`), **not** as failures — they don't affect the verdict.
8. **ROS 2 sourcing** — separate `🤖 ROS 2` block. Considered "sourced"
   iff both `ROS_DISTRO` and `ROS_VERSION` are present in `os.environ`.
   Each of `ROS_DISTRO` / `ROS_VERSION` / `ROS_PYTHON_VERSION` /
   `ROS_LOCALHOST_ONLY` is rendered individually (✅ with value, or ⏭️
   unset). `AMENT_PREFIX_PATH` is rendered as the first colon-separated
   entry plus a `(+N more)` suffix. Not-sourced state prints a
   ⚠️ banner with the actionable
   `source /opt/ros/<distro>/setup.bash` hint. ROS state is a
   *finding*, not a hard failure: it does **not** flip the overall
   verdict — host-only commands do not require a sourced ROS environment.

Verdict: `🎉 All systems go!` requires every required distribution to be
installed, with healthy imports, dependencies and submodules for installed
peers. Missing optional peers are skipped; broken installed optional peers
still affect health. Missing required siblings or failed deps flip the verdict to
`⚠️ Some components need attention.` In that case, the report ends with
`Run geniesim bootstrap to install the missing distributions.` — the redirect is
intentional: `status` reports, `bootstrap` installs.

When extending `status` (e.g. probing optional features inside an installed
package), keep it cheap and non-fatal. Update peer dependencies / extras in
package metadata and required / optional tiers in the umbrella metadata;
`_tiers.py` reads source metadata first, then installed metadata. Keep
`_status_spec.py` import mappings and discovery rules aligned with new packages.

---

## 7a. How `bootstrap` decides what to install

[`run()`](./src/geniesim_cli/commands/bootstrap.py) reuses `status.inspect`
for each target in `_TARGETS`. Required peers come from `_tiers.py`, followed
by the `geniesim` umbrella. `geniesim_cli` is **deliberately excluded**:
re-installing the running CLI mid-execution is a footgun.

- **Healthy stack** → prompt `Re-initialize anyway? [y/N]` (default
  **No**, per the user contract "ask to re-init or do nothing(default)").
- **Unhealthy stack** → enumerate findings (missing dist / failed import /
  bad runtime dep / missing submodule), build an install plan, prompt
  `Proceed with the install plan above? [Y/n]` (default **Yes** — the user
  reached this branch by invoking `bootstrap` to fix things).

The install command per dist is chosen by
[`_install_command`](./src/geniesim_cli/commands/bootstrap.py): if the dist is
importable from a local source tree (i.e. editable-installed somewhere or
on `PYTHONPATH`), the command becomes `pip install -e <source_root>`.
Otherwise it searches for sibling source trees, including local asset sources.
**There is no PyPI fallback** — `geniesim` and `geniesim_assets` are
private distributions and must be installed from a local checkout. When
no local source tree can be located, `_install_command` returns
`None`; `run()` then surfaces a friendly ⚠️ block listing the
unresolvable targets and asks the user to clone the source repo (or
place it on `PYTHONPATH`) and re-run `geniesim bootstrap`.

**Hard rule for any new code in this CLI**: do **not** add hints of the
form `pip install geniesim` or `pip install geniesim_assets` (with or
without extras like `[coacd]`). Those distributions are not on PyPI.
Always redirect to `geniesim bootstrap`, optionally with the extra context
"installs from local checkout" — see [`run()`](./src/geniesim_cli/commands/status.py)
for the canonical phrasing.

When a private distribution has a public **dataset / asset bundle**
mirror (e.g. `geniesim_assets` is published on HuggingFace + ModelScope),
register the download URLs in
[`_DOWNLOAD_SOURCES`](./src/geniesim_cli/commands/bootstrap.py). `run()` reads
that table when it can't find a local source tree and prints a
two-step recipe (`1. Download from <URL>` → `2. pip install -e`) per
unresolved target. The canonical source of these URLs is
[`main/README.md`](../../README.md); whenever the README's badges /
links change, update `_DOWNLOAD_SOURCES` in the same diff.

---

## 8. Verification recipes

| Goal | Command |
|---|---|
| Lint check this package | `black --line-length 120 --check src/ setup.py` |
| Smoke-test `python -m geniesim_cli` | `PYTHONPATH=src python3 -m geniesim_cli` |
| Smoke-test `version` | `PYTHONPATH=src python3 -m geniesim_cli version` |
| Smoke-test `status` | `PYTHONPATH=src python3 -m geniesim_cli status` |
| Smoke-test `bootstrap` (interactive) | `PYTHONPATH=src python3 -m geniesim_cli bootstrap` |
| Smoke-test `bootstrap` non-interactive (decline) | `printf 'n\n' \| PYTHONPATH=src python3 -m geniesim_cli bootstrap` |
| Confirm zero heavy imports | `PYTHONPATH=src python3 -c "import geniesim_cli.cli"` (in a venv with **only** `geniesim_cli`) |
| CLI regression tests | `PYTHONPATH=src python3 -m pytest tests` |
| Documentation audits | `geniesim tool docs`, `geniesim tool deps-dag`, `geniesim tool ros-dag` |
| Editable install | `pip install -e .` from this directory |
| Build wheel | `python -m build --wheel --outdir ../../deploy .` |

---

## 9. Canonical home

This directory is the **canonical home** of `geniesim_cli`. The package
lives here, gets edited here, and is built from here. Treat this tree as
the source of truth.

## Public data collection installation and image builds

`geniesim data-collection install [--task-root PATH] [--assets-root PATH] [--dry-run]` registers the editable standalone
module locator from the current checkout. Data collection is exempt from automatic
tier-1 bootstrap, as it orchestrates its own image on the host. Do not add its
Isaac Sim, cuRobo, or ROS dependencies to the lightweight CLI.


`geniesim data-collection build` builds the complete image using
`source/data_collection/docker/Dockerfile`, starting from official Isaac Sim 5.1.
It accepts `--image=TAG`, `--dry-run`, `--mirrors NAME` (or `--china`), and Docker build arguments, including
`--build-arg ISAAC_SIM_BASE_IMAGE=IMAGE`. `DATA_COLLECTION_IMAGE` selects the
application image; the default is
`registry.agibot.com/genie-sim/geniesim4-data-collection:latest`.
Default container name is `geniesim4-data-collection`; `--container-name`
overrides it. Dependency installation and application copies share a single
Dockerfile. Use this complete-image build path.

The shared `_module_image.py` helper consumes mirror options using
`docker/mirrors.toml`, filters presets to declared Dockerfile arguments, and
forwards explicit build arguments with precedence over presets. BuildKit adds
`geniesim_mirrors` as a named context without widening the module build context.
Without a preset, mirror settings inherit the parent image. Dry runs need no
Docker daemon and print the resolved build command.

`data-collection run/up` resolve assets through the module's standard-library
`scripts/resolve_assets.py`. Forward `_env.assets_path()` using the canonical
environment name and export the resolved `SIM_ASSETS` for the launch scripts.
An editable asset-package install is optional; relative asset directories also
work. Resolve external task libraries through `scripts/resolve_llm_task_root.py`;
forward `_env.llm_task_root()` and export both configured task-root names. The
JSON/GUI launchers remap them to `/llm_tasks` inside the container.
Install persists roots in `~/.config/geniesim/data-collection.json` (override:
`GENIESIM_DATA_COLLECTION_CONFIG`); explicit environment roots take precedence.
`config` displays saved/effective settings. `llm list/show` inspects external tasks.
`run <TASK>` defaults to LLM collection; `run --task-json PATH` selects the JSON job launcher. `llm run` is an alternate alias. New LLM workflows copy tasks to
`recording_data/llm/<task>/<run>/tasks` and mount that working copy at `/run/tasks`.
`replay <TASK_OR_EPISODE>` isolates input and verifies newly converted H5/video
outputs; task names resolve the latest successful CLI recording. Do not add
Isaac/ROS dependencies to the host CLI. Keep direct scripts on the shared
resource discovery path. Before launching LLM collection, apply the module
`llm_task_policy.py` guard: deformable tasks cannot use `run_times > 1`.
USD assets with attached behavior scripts also require one run; the README
documents this restriction because host catalog inspection does not compose USD.


## Scene agent peer

`geniesim_scene_agent` is optional via
`source/geniesim/[scene_agent]` after installing the local peer. Its distribution
and import name match; deploy and shell completion use `geniesim_scene_agent`.
The peer owns its scene/asset tool commands and bundles its skills/examples in
wheels. Default bootstrap and the shared tier-1 container install remain unchanged.
See [the scene-agent guide](../geniesim_scene_agent/AGENTS.md).
Its independent image builder reuses `_module_image.py` and shared mirror
presets from the checkout; its runtime dependencies stay in its own environment.

The scene agent optionally consults `_env.assets_path()` for external assets,
then locates the installed asset package without importing it, and falls back
to relative `assets/` layouts. It does not require a CLI dependency. Existing
example USD references can be bound with the module’s auto-discovery helper.

`geniesim tool ros-dag` excludes generated `devel`, `devel_build` and
`devel_log` trees alongside build/install/external, so stale colcon package.xml
symlinks cannot shadow source packages or break the generator.

## Newton benchmark dispatch

`benchmark run` reads the task YAML `runtime` before selecting an interpreter: omitted/isaac retains the existing app; newton executes `geniesim_benchmark.newton.runner` in the prepared ROS Python. `--manifest` and task-runner options are forwarded. `benchmark generate-layouts` and `run-layouts` dispatch to the benchmark-owned Newton modules; the CLI imports no ROS, physics or rendering runtime. See Wuji evaluation.

### Benchmark task categories

`benchmark list`, `batch` and `categories` recognize the second token in
`<robot>_<category>_<task>.yaml`. New task categories `finemanip`, `dexmanip`,
`crsemb` and `agent` are registered alongside `if`, `wbc`, `spatial`, `manip`
and `s2r`. Use the single-token `g2op` prefix for G2 Omnipicker task entries.
Task renaming changes config resolution only; YAML runtime contents and the
published G2 track entries remain unchanged. Control and validation utility
configs retain their separate names.
