# docker/ — GenieSim container variants

All Isaac Sim variants live here. One shared `entrypoint.sh` and `start.sh`
handle all of them; variant-specific behaviour is driven entirely by env vars
injected by the CLI (`docker.py` → `start.sh` → `entrypoint.sh`).

## Variants

| File | Isaac Sim | Ubuntu | ROS 2 | Python | Image | Status |
|------|-----------|--------|-------|--------|-------|--------|
| `Dockerfile` | 6.X — exact spec `6.2.0.1`, paired with wheel `6.2.0.0` | 24.04 Noble | Jazzy | system `python3` 3.12 (pip-installed isaacsim) | geniesim4 | Dedicated 6.2 runtime; see acceptance |
| `Dockerfile.5.1` | **5.1 + 6.2** (combo; tag `5.1.0`, wheel `6.2.0.0`) | 24.04 Noble | Jazzy | `omni_python` (5.1 bundled) + system `python3` 3.12 (6.2 pip-installed) | geniesim3 | ✅ default |
| `Dockerfile.4.5` | 4.5 (tag + wheel `4.5.0`) | 22.04 Jammy | Humble | system `python3` 3.10 (pip-installed isaacsim) | geniesim2 | ⚠️ E.O.L. |

## Image naming — one tag per family, no minor versions

**Canonical form: `registry.agibot.com/genie-sim/geniesim<N>:latest`**, where
`<N>` is the family major only — `geniesim2`, `geniesim3`, `geniesim4`. Image
tags drop the minor version on purpose: `geniesim3:latest` follows whatever the
current 5.1-variant build is, so version bumps don't require renaming images,
containers, or CLI dispatch.

**Never write a minor-suffixed image name**, such as `geniesim4.0`. A tag that differs from downstream `FROM` references does not update those downstream builds.

Rules, in this directory and downstream:

- **Build to the unsuffixed tag.** The `docker build` lines in each
  `Dockerfile` header are copy-paste sources, so they must name
  `geniesim<N>:latest` and agree with the `image` field in `docker.py`'s
  `VARIANTS`. `start.sh`'s `GENIESIM_IMAGE` fallback must agree too — the CLI
  normally injects it, so a wrong default only bites direct `start.sh` callers,
  which is exactly the case nobody tests.
- **Consume the unsuffixed tag.** Downstream images take
  `FROM registry.agibot.com/genie-sim/geniesim<N>:latest` — never a minor
  suffix, never an Isaac version in the image name. `Dockerfile.post-train`
  selects that base through `ARG BASE_IMAGE`; `Dockerfile.post-train-teleop`
  inherits the server layer through `ARG SERVER_IMAGE`.
- **Keep the Isaac minor out of image names entirely.** The release is carried
  by the `ISAACSIM_TAG` / `ISAACSIM_PIP_VERSION` build args — see "Three
  strings, one release" below — not by the tag.
- **Derived images keep the family prefix and add a role, not a version:**
  `geniesim3-data-collection:latest`. Intermediate images never published to
  the registry use their own short name (`rlinf-geniesim:cloud`,
  `rlinf-geniesim:teleop`) and still pin their base to an unsuffixed tag.
- **Container names follow the same families** (`geniesim2`/`geniesim3`/
  `geniesim4`, via `GENIESIM_CONTAINER`; see the env var contract below), so
  prose that names a container — comments included — uses the family name.

This must print nothing. The two files that quote forbidden spellings in order
to forbid it are excluded:

```bash
git grep -nE 'geniesim[0-9]+\.[0-9]' -- . ':!AGENTS.md' ':!docker/AGENTS.md'
```

`docker/tests/test_image_naming.py` enforces that scan and the invariants
around it — every `VARIANTS` image is a family-major tag, each `Dockerfile`
header builds the tag the CLI actually runs, and the downstream rlinf `FROM`
lines pin unsuffixed bases. Run it with the existing CPU workflow:

```bash
python3 -m pytest -q docker/tests
```

The Isaac 6.2 command is `geniesim docker6.2 {build,up,into,down,logs}`.
The internal variant registry key remains
`""` for downstream callers of `get_variant` / `variant_env`.

The dedicated 6.2 variant pins exact base spec `6.2.0.1` with wheel `6.2.0.0`.

**Three strings, one release — two resolved, one guarded.** `ISAACSIM_TAG` is
the NGC base image tag (three components, or four for a respin);
`ISAACSIM_PIP_VERSION` is the PyPI wheel version (four); and `mujoco` in `source/geniesim_ros/pyproject.toml`
(`~=3.12.0`) must name the MuJoCo line that release's Newton ships with — 6.2
ships Newton 1.6.1, which itself requires `mujoco~=3.12.0` (`requires_dist`,
`sim` extra), so that pin is Newton's specifier copied verbatim rather than our
choice. The base/wheel mapping and Newton dependency line must agree.

**The first two are now derived from one declaration.** A variant carries an
`isaacsim_series` spec (`"6.2.0.1"`) in `docker.py`'s `VARIANTS`;
`geniesim_cli._isaacsim.resolve()` turns it into the concrete pair and
`_do_build` passes both as `--build-arg`. Resolution happens host-side before Docker build; `FROM` requires a concrete tag.

Two things to know about that resolver:

- **It requires an available wheel, including explicit respin pairings.**
  `_isaacsim._TAG_WHEEL_PAIRS` pairs image `6.2.0.1` with wheel `6.2.0.0`.
  Both the exact spec and `6.X` resolve to that pair for cp312. No wheel named
  `6.2.0.1` is required or published. Unknown four-component tags must still
  match an equal-version wheel; do not infer new respin pairings.
- **It never fails a build.** Offline, HTTP error, or no match → one warning and
  the Dockerfile's literal `ARG` defaults apply, so `ISAACSIM_TAG=6.2.0.1` /
  `ISAACSIM_PIP_VERSION=6.2.0.0` in `Dockerfile` stay meaningful and must be
  kept current as the pinned fallback. `--no-resolve` skips the network;
  `--isaacsim <spec>` overrides the variant's spec; `GENIESIM_ISAACSIM_TAG` +
  `GENIESIM_ISAACSIM_PIP_VERSION` pin both verbatim.

`Dockerfile.5.1` deliberately has **no** spec: it is a 5.1-base/6.2-wheel combo,
a pairing the correspondence rule cannot express. It keeps both literal pins.
`geniesim docker` stays mapped to this variant until `docker6.2` passes its
runtime acceptance gates. Keep bundled 5.1 and the system-Python 6.2 co-install.

System and bundled Python require separate acceptance checks. Build the combo image
with the normal `geniesim docker5.1 build` command and select `docker6.2 up`
with the combo image override to use system Python. See [the runtime guide](README.md#system-python-runtime).

**The third string cannot be resolved the same way, so it is checked instead.**
The `isaacsim` meta-wheel's `Requires-Dist` names only `isaacsim-kernel` plus
`isaacsim-*` extras — newton appears nowhere, and the subpackage that does
declare it is a multi-GB wheel, so there is nothing cheap to read host-side.
Inside the image there is: Isaac Sim 6.X pip-installs `newton` as an ordinary
distribution, so `docker/check_newton_pins.py` reads Newton's own
`mujoco` / `mujoco-warp` requirement from installed metadata, compares it to
what the dependency layer actually installed, and **fails the build** on
mismatch. Both `Dockerfile` and `Dockerfile.5.1` run it right after their deps
layer; it no-ops where Newton is not pip-visible.

The 6.2 runtime contract describes assembly, physics and rendering requirements. Runtime acceptance records image-specific evidence. Open release work belongs in the roadmap.

**`SKIP_EXACT` is not a version constraint.** It removes a direct requirement from collection while transitive resolution can still replace the package. Use installed SDK constraints and the Newton guard to preserve runtime compatibility.

## Isaac/colcon dependency constraints

`collect_runtime_constraints.py` reads requirements from installed `isaacsim*`
distributions and `colcon-core` in system Python. It evaluates platform markers,
activates Isaac's `all`/`extscache` extras and follows dependency extras such as
`newton[sim]` and `etils[epath]`. It writes plain pip constraints plus explicit
repair requirements for missing or incompatible installed packages.

Both `Dockerfile` and `Dockerfile.5.1` apply these constraints to the system-Python
GenieSim dependency layer without `--ignore-installed`. This preserves compatible
SDK packages and replaces incompatible versions with normal pip uninstall/install.
The generated file remains at `/opt/geniesim/docker/isaac-constraints.txt` for
later deliberate installs with `-c`; it is not a global `PIP_CONSTRAINT`, because
derived profiles can declare their own constraints. `collect_deps.py` continues to
skip runtime-owned packages; skipping alone is not version protection.

The Isaac profile runs `python3 -m pip check` after the final build layers and
retains `check_newton_pins.py` for Newton's optional sim requirements. The bundled 5.1 interpreter keeps its existing legacy
dependency install and NumPy/SciPy pins; system-Python constraints do not apply to it.

The collector tests live in `docker/tests/test_collect_runtime_constraints.py`.
Run them with the existing `python3 -m pytest -q docker/tests` CPU workflow.

## Variant-specific build notes

### Post-training profiles

`geniesim docker --profile post-train <verb>` uses `Dockerfile.post-train` on top of an
existing GenieSim base (`BASE_IMAGE`). `--profile post-train-teleop` uses
`Dockerfile.post-train-teleop` on top of the built server (`SERVER_IMAGE`). These are
simulation-side layers; the RLinf training/client image remains separate.
Keep post-training-specific dependency constraints in these derived images.
Do not change shared base-image Torch or physics pins to satisfy a profile.

The public post-training workflow builds the default base `registry.agibot.com/genie-sim/geniesim4:latest`
locally with `geniesim docker6.2 build`; `Dockerfile.post-train` uses it as its
default `BASE_IMAGE` without registry access. The base must include bundled Isaac Sim 6.2.x
at `/isaac-sim`. Both profile Dockerfiles check `/isaac-sim/VERSION`.
The default 5.1 variant has a different bundled runtime and is not a compatible
parent, even when its system Python also has Isaac Sim 6.2.

`PROFILES` in `commands/docker.py` owns their image/container defaults and runtime
environment. Both use system Python 3.12/Jazzy, 8 GB shared memory, and
`entrypoint.post-train.sh`, then delegate to the common `entrypoint.sh`. The batch
server launcher builds ROS as the mapped runtime user; the entrypoint must not
build ROS as root. The server image installs `posix_ipc` in both system and
bundled Isaac Python. Keep the base Newton/MuJoCo/Warp versions together.
`post-train-runtime.txt` pins the added transport/decomposition packages and
constrains the bundled Python install.

The profile entrypoint creates the bundled Kit's `cache`, `config`, `data`, and
`logs` directories for the mapped runtime UID/GID. Limit ownership changes to
these runtime directories. Profiles disable the base image's Kit-log healthcheck:
the container may be idle or run only a teleop client. Use the runtime CLI to
inspect simulation server processes.

The image selects its baked `entrypoint.post-train.sh`; rebuild it when that
wrapper changes. The common `entrypoint.sh` remains bind-mounted from the checkout.
For post-training profiles, `start.sh` applies `GENIESIM_SHM_SIZE` and labels
containers with checkout/profile ownership.
Post-training lifecycle commands require both labels to match. Assets may be passed
directly through `GENIESIM_ASSETS_SRC`. Entrypoint readiness has a configurable
`GENIESIM_START_TIMEOUT` (300 seconds); timeout/exit must return failure.
These checks apply only to post-training profiles; retain the original startup
and repeated-`up` behavior for the Isaac Sim variants.
Use `python3 -m pytest docker/tests/test_docker_cli.py` for CLI/startup coverage.

**Dockerfile (6.2)**
- Includes Pico/Wuji and recording dependencies from the teleop package metadata,
  with `WUJI_SDK_VERSION=2026.8.17`. ROS controls use apt NumPy/SciPy/OpenCV
  through `/usr/lib/python3/dist-packages`; the simulator uses system `python3`
  without this control-only overlay. Preserve the Isaac dependency constraints.
- The mounted MC runtime and built ROS controller overlay require session preflight.
- Base `6.2.0.1`, system-Python wheel `6.2.0.0`, ROS 2 Jazzy.
- `PIP_BREAK_SYSTEM_PACKAGES=1` permits Ubuntu 24.04 system installs.
- `OMNI_KIT_ACCEPT_EULA=Y` and the Kit EULA marker prepare noninteractive startup.
- `GENIESIM_EXTRA_CACHE_DIRS="kit pip numba"` creates runtime cache directories.
- See acceptance for measured coverage; no simulator-wide pass follows from an image build.

**Dockerfile.5.1 (combo: 5.1 + 6.2)**
- Installs the same Pico/Wuji and recording dependencies as `Dockerfile` into
  system Python 3.12, including `WUJI_SDK_VERSION=2026.8.17` and the apt
  NumPy/SciPy/OpenCV control overlay. The bundled Python 3.11 retains its
  simulator dependency selection; Wuji SDK input runs under system Python.
- The recorded clean local rebuild qualifies system Python 3.12/Isaac 6.2 with the `isaac` dependency profile. Bundled Python 3.11 has separate open acceptance work in the roadmap.
- Isaac Sim 5.1 is pre-installed in the NVIDIA base image at `/isaac-sim`.
- `omni_python` wrapper (`/usr/local/bin/omni_python`) execs `/isaac-sim/python.sh` so `$0` stays correct.
- `chmod o+rx /isaac-sim` — NVIDIA base ships `/isaac-sim` as `drwxr-x---`; opened for the remapped user.
- Isaac Sim 6.2 is **also** pip-installed into system python3.12 (same block as `Dockerfile`), pinned `==6.2.0.0` via `ISAACSIM_PIP_VERSION`.
- Both EULA vars are set: `ACCEPT_EULA=Y` (5.1) and `OMNI_KIT_ACCEPT_EULA=Y` (6.X).
- `typing_extensions.py` patch applied for the 6.X install.
- GeneSim deps installed into **both** interpreters at runtime (entrypoint.sh).
- numpy/scipy pinned for `omni_python`: `numpy==1.26.4`, `scipy==1.13.1`.
- Runtime default interpreter (`GENIESIM_PY_CMD`) is `omni_python`; use `python3` for 6.X features.

**Dockerfile.4.5**
- Isaac Sim installed via `pip install "isaacsim[all,extscache]==4.5.0"` into system python3.10.
- Two pre-conditions must be met before the ROS repo is added (see Dockerfile comments):
  1. `libbrotli1` must be downgraded to `1.0.9-2build6` — the base image ships `1.1.0` from sury.org which conflicts with `libbrotli-dev` pulled by `ros-humble-desktop`.
  2. `libfreetype6-dev` / `libfontconfig1-dev` must be installed from Ubuntu repos first to avoid "held broken packages" once the ROS repo is present.
- `ACCEPT_EULA=Y` is the EULA env var.

## Build mirror contract

`mirrors.toml` is the provider-preset source. The CLI defaults base builds to
`upstream`; `--china` aliases `--mirrors china`. Derived images inherit by
default and can explicitly select `--mirrors upstream` or another preset.
Explicit `--build-arg` values override the preset and are emitted once.

All base recipes install `configure_mirrors.py` as
`/usr/local/bin/geniesim-configure-mirrors`. It runs with bundled Python before
the first apt install, then with system Python after `add-apt-repository`.
It updates Ubuntu sources, preserves third-party repositories, and remembers
the selected Ubuntu mirror in `/etc/geniesim/apt-mirror` for later switching.
`APT_MIRROR=upstream` restores the official archive/security sources; a URL
selects a mirror; an empty value inherits. A requested APT change fails if no
managed Ubuntu sources can be identified.

Pip defaults are written to `/etc/pip.conf`, with `PIP_CONFIG_FILE` selecting
that file for root, runtime users and derived images. Other global pip options
are preserved. CLI `--index-url` and environment `PIP_INDEX_URL` remain explicit
overrides. Derived Dockerfiles declare `ARG APT_MIRROR` and `ARG PIP_INDEX_URL`
without defaults and run the inherited helper before dependency installation.
Do not force a regional primary index in individual install commands. Keep
package-specific NVIDIA/PyTorch indexes explicit.

CPU coverage: `python3 -m pytest docker/tests/test_configure_mirrors.py` plus
the CLI variant/process tests.

## Env var contract

`start.sh` always passes both `ACCEPT_EULA=Y` and `OMNI_KIT_ACCEPT_EULA=Y` — harmless when not applicable.

`start.sh` passes these into the container; `entrypoint.sh` reads them:

| Var | Purpose | 6.X (6.2) | 5.1 (combo) | 4.5 |
|-----|---------|-----|-------------|-----|
| `ROS_DISTRO` | ROS distro for sourcing and bashrc | jazzy | jazzy | humble |
| `GENIESIM_PY_CMD` | default interpreter for editable pip install | python3 | omni_python | python3 |
| `GENIESIM_BREAK_SYSTEM_PKGS` | add `--break-system-packages` to system pip calls | 1 | 0 | 1 |
| `GENIESIM_CHOWN_KIT_PATH` | path to chown for EULA acceptance (empty = skip) | `.../python3.12/.../isaacsim/kit` | `.../python3.12/.../isaacsim/kit` | `.../python3.10/.../isaacsim/kit` |
| `GENIESIM_EXTRA_CACHE_DIRS` | extra subdirs to create under `cache/main/` | `kit pip numba` | `kit pip numba` | *(empty)* |
| `GENIESIM_ISAACSIM_KIT_CACHE_PATH` | in-package kit/cache path for named volume (empty = skip) | `.../python3.12/.../isaacsim/kit/cache` | `.../python3.12/.../isaacsim/kit/cache` | `.../python3.10/.../isaacsim/kit/cache` |
| `GENIESIM_OVRTX_CACHE_PATH` | in-package ovrtx cache path for named volume (empty = skip) | `.../python3.12/.../ovrtx/bin/cache` | `.../python3.12/.../ovrtx/bin/cache` | `.../python3.10/.../ovrtx/bin/cache` |

`start.sh` also reads `GENIESIM_VARIANT_LABEL` (e.g. `docker5.1`) for user-facing messages.

## Design goals

### Shared standalone dependency installation

Docker owns image/entrypoint profiles and dependency installation. Backend tools
and examples live in the owning engine package. Do not add task menus, duplicate
container managers, render consumers, recording controls or per-task cache
ownership repair here. Use the existing Docker CLI and normal ROS build/launch.

Both Jazzy image recipes include teleop dependencies through
`collect_deps.py --include-peer geniesim_teleop` plus the pinned Wuji SDK
in system Python; this does not alter the SDK-owned Newton, MuJoCo or Warp
versions.


`install_ros_build_deps.py` reads build SDK requirements from package metadata
and installs with --no-deps. Preserve the mounted OVRtx cache during upgrades;
overlay the wheel and retire superseded dist-info only after pip succeeds. Do
not unmount caches to work around uninstall failures.

Build/assemble/launch as the intended user. Existing generated caches must be
writable by that user. Ownership repair is an explicit owner/admin operation on
a specific cache; it must not recurse into source assets or unrelated scenes.

- **Single source of truth per concern.** Shell logic lives in `start.sh` /
  `entrypoint.sh` once. Dockerfiles are separate because their build-time
  differences are structural and would make a single parameterized Dockerfile
  unreadable.

- **No variant-specific directories.** Everything is under `docker/`. The CLI
  (`VARIANTS` in `commands/docker.py`) is the single place that maps a variant name to
  its Dockerfile and env vars.

- **`collect_deps.py` is shared.** Base Dockerfiles `COPY docker/collect_deps.py`;
  Post-training layers reuse the base dependencies.

- **Build contexts are recipe-specific.** `Dockerfile.dockerignore` and
  `Dockerfile.5.1.dockerignore` admit only these recipes' COPY inputs, excluding
  the large source asset tree and generated Python caches. Keep both inclusion
  lists aligned when adding a COPY input. Other recipes retain their own context.

- **Entrypoint is bind-mounted at runtime.** `start.sh` mounts
  `docker/entrypoint.sh` over `/usr/local/bin/geniesim-entrypoint` so changes
  to the entrypoint take effect without rebuilding the image.

- **Shader caches use named Docker volumes.** In-package kit/cache and
  ovrtx/bin/cache are mounted as named volumes (seeded from image on first
  create, persisted across `docker down/up`). Bind-mounting an empty host dir
  would shadow the bundled `shaderFolderHash.bin` / `version` files and break
  the renderer.

- **Warp JIT cache is per runtime UID.** `entrypoint.sh` exports
  `WARP_CACHE_PATH=/tmp/geniesim-newton-warp-<uid>` so containers that still run as
  image UID 1234 cannot poison the remapped developer user's kernel cache.

- **5.1 combo image.** `Dockerfile.5.1` is a superset of `Dockerfile`: Isaac
  Sim 5.1 (via `omni_python`) and Isaac Sim 6.X (via `python3`) coexist in the
  same container. All geniesim packages are importable from both interpreters.

## Adding a new variant

1. Add `docker/Dockerfile.X.Y` (copy the closest existing one, adjust ROS
   distro and Python strategy, and set **both** pins — `ISAACSIM_TAG` for the
   NGC base image and `ISAACSIM_PIP_VERSION` for the wheel). These stay the
   offline fallback even when the variant declares a series spec, so pin them to
   the release you expect that spec to resolve to.
2. Add an entry to `VARIANTS` in
   `source/geniesim_cli/src/geniesim_cli/commands/docker.py` with the correct
   `dockerfile` and `env` values — plus `isaacsim_series="X.Y"` (or `"X.X"` for
   a whole major) if the two pins should be resolved rather than fixed, and an
   entry in `_VARIANT_PYTHON_TAG` naming the base distro's CPython tag so a
   wrong-interpreter wheel is refused host-side. Verify with
   `python3 -m geniesim_cli._isaacsim X.X --python-tag cpNNN` before building.
3. Add `dockerX.Y` to the completion lists in
   `source/geniesim_cli/src/geniesim_cli/commands/completion.py`.
4. Add the CLI dispatch in
   `source/geniesim_cli/src/geniesim_cli/cli.py` (search for `docker5.1`).
5. Name the image `registry.agibot.com/genie-sim/geniesim<N>:latest` — family
   major only, per "Image naming" above. The same string goes in the new
   `Dockerfile` header's `docker build -t`, the `VARIANTS` `image` field, and
   the variants table at the top of this file. Then confirm the
   `git grep` check under "Image naming" still prints nothing.

## Checkout dependency gate

The constrained image builds require the ROS project's MuJoCo requirement to
agree with the selected Isaac SDK. Isaac Sim 6.2 supplies Newton 1.6.1 and
MuJoCo 3.12; a checkout declaring `mujoco<3.9.0` cannot satisfy that SDK
constraint. Resolve that declaration in the ROS module before building this
runtime. Do not drop the SDK guard or silently override the project metadata.

Post-training profiles also require their server scripts and ROS packages in
the mounted checkout. CLI dispatch and Docker process tests do not establish
that those consumers are available or that simulator acceptance has passed.
