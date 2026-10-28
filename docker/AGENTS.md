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
