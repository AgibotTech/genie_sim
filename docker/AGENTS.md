# docker/ — GenieSim container variants

All Isaac Sim variants live here. One shared `entrypoint.sh` and `start.sh`
handle all of them; variant-specific behaviour is driven entirely by env vars
injected by the CLI (`docker.py` → `start.sh` → `entrypoint.sh`).

## Variants

| File | Isaac Sim | Ubuntu | ROS 2 | Python | Image | Status |
|------|-----------|--------|-------|--------|-------|--------|
| `Dockerfile` | 6.0 (`6.0.1`) | 24.04 Noble | Jazzy | system `python3` 3.12 (pip-installed isaacsim) | geniesim4 | 🚧 incoming, **not implemented** |
| `Dockerfile.5.1` | **5.1 + 6.0** (combo) | 24.04 Noble | Jazzy | `omni_python` (5.1 bundled) + system `python3` 3.12 (6.0 pip-installed) | geniesim3 | ✅ default |
| `Dockerfile.4.5` | 4.5 (`4.5.0`) | 22.04 Jammy | Humble | system `python3` 3.10 (pip-installed isaacsim) | geniesim2 | ⚠️ E.O.L. |

Image tags drop the minor version on purpose — `geniesim3:latest` follows
whatever the current 5.1-variant build is, so version bumps don't require
renaming images, containers, or CLI dispatch.

## Variant-specific build notes

**Dockerfile (6.0)** — 🚧 *incoming, not implemented*
- The `Dockerfile` here is a placeholder: the image (`geniesim4`) is not
  published, and `geniesim docker6.0 …` exits with a "not implemented" error
  rather than dispatching.
- Planned shape (kept here so the eventual implementation can match):
  - Isaac Sim installed via `pip install "isaacsim[all,extscache]==6.0.1"` into system python3.12.
  - `PIP_BREAK_SYSTEM_PACKAGES=1` lifts Ubuntu 24.04's PEP 668 restriction.
  - `OMNI_KIT_ACCEPT_EULA` is the EULA env var.
  - A `typing_extensions.py` patch is applied post-install to fix a `Sentinel` import error in pydantic.
  - Extra cache subdirs `kit pip numba` are created under `cache/main/`.

**Dockerfile.5.1 (combo: 5.1 + 6.0)**
- Isaac Sim 5.1 is pre-installed in the NVIDIA base image at `/isaac-sim`.
- `omni_python` wrapper (`/usr/local/bin/omni_python`) execs `/isaac-sim/python.sh` so `$0` stays correct.
- `chmod o+rx /isaac-sim` — NVIDIA base ships `/isaac-sim` as `drwxr-x---`; opened for the remapped user.
- Isaac Sim 6.0 is **also** pip-installed into system python3.12 (same block as `Dockerfile`).
- Both EULA vars are set: `ACCEPT_EULA=Y` (5.1) and `OMNI_KIT_ACCEPT_EULA=Y` (6.0).
- `typing_extensions.py` patch applied for the 6.0 install.
- GeneSim deps installed into **both** interpreters at runtime (entrypoint.sh).
- numpy/scipy pinned for `omni_python`: `numpy==1.26.4`, `scipy==1.13.1`.
- Runtime default interpreter (`GENIESIM_PY_CMD`) is `omni_python`; use `python3` for 6.0 features.

**Dockerfile.4.5**
- Isaac Sim installed via `pip install "isaacsim[all,extscache]==4.5.0"` into system python3.10.
- Two pre-conditions must be met before the ROS repo is added (see Dockerfile comments):
  1. `libbrotli1` must be downgraded to `1.0.9-2build6` — the base image ships `1.1.0` from sury.org which conflicts with `libbrotli-dev` pulled by `ros-humble-desktop`.
  2. `libfreetype6-dev` / `libfontconfig1-dev` must be installed from Ubuntu repos first to avoid "held broken packages" once the ROS repo is present.
- `ACCEPT_EULA=Y` is the EULA env var.

## Env var contract

`start.sh` always passes both `ACCEPT_EULA=Y` and `OMNI_KIT_ACCEPT_EULA=Y` — harmless when not applicable.

`start.sh` passes these into the container; `entrypoint.sh` reads them:

| Var | Purpose | 6.0 | 5.1 (combo) | 4.5 |
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

- **Single source of truth per concern.** Shell logic lives in `start.sh` /
  `entrypoint.sh` once. Dockerfiles are separate because their build-time
  differences are structural and would make a single parameterized Dockerfile
  unreadable.

- **No variant-specific directories.** Everything is under `docker/`. The CLI
  (`_VARIANTS` in `docker.py`) is the single place that maps a variant name to
  its Dockerfile and env vars.

- **`collect_deps.py` is shared.** All Dockerfiles `COPY docker/collect_deps.py`.

- **Entrypoint is bind-mounted at runtime.** `start.sh` mounts
  `docker/entrypoint.sh` over `/usr/local/bin/geniesim-entrypoint` so changes
  to the entrypoint take effect without rebuilding the image.

- **Shader caches use named Docker volumes.** In-package kit/cache and
  ovrtx/bin/cache are mounted as named volumes (seeded from image on first
  create, persisted across `docker down/up`). Bind-mounting an empty host dir
  would shadow the bundled `shaderFolderHash.bin` / `version` files and break
  the renderer.

- **5.1 combo image.** `Dockerfile.5.1` is a superset of `Dockerfile`: Isaac
  Sim 5.1 (via `omni_python`) and Isaac Sim 6.0 (via `python3`) coexist in the
  same container. All geniesim packages are importable from both interpreters.

## Adding a new variant

1. Add `docker/Dockerfile.X.Y` (copy the closest existing one, adjust base tag,
   ROS distro, Python strategy).
2. Add an entry to `_VARIANTS` in
   `source/geniesim_cli/src/geniesim_cli/commands/docker.py` with the correct
   `dockerfile` and `env` values.
3. Add `dockerX.Y` to the completion lists in
   `source/geniesim_cli/src/geniesim_cli/commands/completion.py`.
4. Add the CLI dispatch in
   `source/geniesim_cli/src/geniesim_cli/cli.py` (search for `docker5.1`).
